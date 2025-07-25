###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers
from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank

stimer = StragglerDetector()

from .trainer import MegatronTrainer


class MegatronPretrainTrainer(MegatronTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "pre_trainer"
        super().__init__(*args, **kwargs)

    def get_batch(self, data_iterator):
        """Generate a batch."""

        # TODO: this is pretty hacky, find a better way
        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            return None, None, None, None, None

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        # slice batch along sequence dimension for context parallelism
        args = get_args()
        if args.context_parallel_size > 1 and args.enable_primus_turbo and args.use_turbo_attention:
            try:
                from primus.backends.megatron.core.utils import (
                    produce_attention_sharder,
                    shard_batch_on_this_cp_rank,
                )
            except:
                raise ImportError("Module 'primus_turbo' may not installed. Please install it")
            sharder = produce_attention_sharder(args.cp_comm_type)
            batch = shard_batch_on_this_cp_rank(sharder, batch)
        else:
            batch = get_batch_on_this_cp_rank(batch)

        return batch.values()

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            loss_mask (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context="loss",
                ),
                message="Spiky loss",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
        # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
        # on loss[0] fixes this
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {"lm loss": (reporting_loss[0], reporting_loss[1])},
        )

    def forward_step(self, data_iterator, model: GPTModel, return_schedule_plan: bool = False):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        get_args()
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers("batch-generator", log_level=2).start()
        global stimer
        with stimer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
        timers("batch-generator").stop()

        with stimer:
            # output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            if return_schedule_plan:
                assert (
                    args.overlap_moe_expert_parallel_comm
                ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(self.loss_func, loss_mask)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

        return output_tensor, partial(self.loss_func, loss_mask)
