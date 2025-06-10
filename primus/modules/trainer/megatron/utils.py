###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""megatron utils"""

import inspect
import os

import megatron
import torch
from megatron.core import parallel_state

from primus.core.utils import logger


######################################################log after torch distributed initialized
def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(msg):
    """If distributed is initialized, print only on last rank."""
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if torch.distributed.is_initialized():
        if is_last_rank():
            log_func(msg, module_name, function_name, line)
    else:
        log_func(msg, module_name, function_name, line)


def set_wandb_writer_patch(args):  # monkey patch
    """
    This function is adapted from the original Megatron implementation, with an additional
    wandb argument `entity` be added.
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    megatron.training.global_vars._ensure_var_is_not_initialized(
        megatron.training.global_vars._GLOBAL_WANDB_WRITER, "wandb writer"
    )

    if getattr(args, "wandb_project", "") and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == "":
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, "wandb")
        wandb_kwargs = {
            "dir": save_dir,
            "name": args.wandb_exp_name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "config": vars(args),
        }
        os.makedirs(wandb_kwargs["dir"], exist_ok=True)
        wandb.init(**wandb_kwargs)
        megatron.training.global_vars._GLOBAL_WANDB_WRITER = wandb


def validate_manual_split(args):
    """
    The use of decoder_pipeline_manual_split_list is to relax the divisibility
    restriciton of the current (interleaved) 1f1b pipeline schedule. The layer
    split or number of each pp rank is
    decoder_pipeline_manual_split_list[pp_rank*vp_size:(pp_rank+1)*vp_size] or
    decoder_pipeline_manual_split_list[pp_rank] when interleaved pipeline is
    used or not. For example, the split list could be "[2,3,2,2,2,2,2,1]"
    in layer16-pp4-vpp2 config, where the vpp split of
    pp_rank0/pp_rank1/pp_rank2/pp_rank3 is [2,3]/[2,2]/[2,2]/[2,1].

    """

    if (
        args.num_layers_per_virtual_pipeline_stage is not None
        or args.decoder_first_pipeline_num_layers is not None
        or args.decoder_last_pipeline_num_layers is not None
        or args.account_for_embedding_in_pipeline_split
        or args.account_for_loss_in_pipeline_split
    ):
        raise ValueError(
            "decoder_pipeline_manual_split_list is not compatible "
            "with num_layers_per_virtual_pipeline_stage/"
            "decoder_first_pipeline_num_layers/"
            "decoder_last_pipeline_num_layers/"
            "account_for_embedding_in_pipeline_split/"
            "account_for_loss_in_pipeline_split yet"
        )

    num_layers = args.num_layers
    pp_size = args.pipeline_model_parallel_size
    vp_size = args.virtual_pipeline_model_parallel_size
    pp_split = args.decoder_pipeline_manual_split_list

    if pp_size <= 1:
        raise ValueError(
            f"pipeline_model_parallel_size={pp_size} should be larger "
            f"than 1 when decoder_pipeline_manual_split_list is used"
        )

    if not isinstance(pp_split, list):
        raise ValueError(f"decoder_pipeline_manual_split_list={pp_split} should be a list")

    split_size = pp_size if vp_size is None else pp_size * vp_size
    if len(pp_split) != split_size:
        raise ValueError(
            f"the size of decoder_pipeline_manual_split_list="
            f"{pp_split} should be {split_size} "
            f"given pipeline_model_parallel_size={pp_size} and "
            f"virtual_pipeline_model_parallel_size={vp_size}"
        )

    if not all(x > 0 for x in pp_split):
        raise ValueError(
            f"layer numbers in decoder_pipeline_manual_split_list={pp_split} should all be larger than 0"
        )

    if sum(pp_split) != num_layers:
        raise ValueError(
            f"the sum of decoder_pipeline_manual_split_list="
            f"{pp_split} is {sum(pp_split)} and "
            f"should be equal to num_layers={num_layers}"
        )

    return True


def set_manual_pipeline_split_patch(args):
    """
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    megatron.core.transformer.TransformerConfig.decoder_pipeline_manual_split_list = (
        args.decoder_pipeline_manual_split_list
    )

    # patch get_num_layers_to_build
    def get_num_layers_to_build_patch(config, vp_stage):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_size = config.virtual_pipeline_model_parallel_size
        pp_idx = pp_rank if vp_size is None else pp_rank * vp_size + vp_stage
        num_layers_to_build = config.decoder_pipeline_manual_split_list[pp_idx]
        return num_layers_to_build

    megatron.core.transformer.transformer_block.get_num_layers_to_build = get_num_layers_to_build_patch
    megatron.core.models.gpt.gpt_layer_specs.get_num_layers_to_build = get_num_layers_to_build_patch

    # patch get_transformer_layer_offset
    def get_transformer_layer_offset_patch(config, vp_stage):
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = config.pipeline_model_parallel_size
        vp_size = config.virtual_pipeline_model_parallel_size

        if not parallel_state.is_inside_encoder():
            pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
            if pp_decoder_start is not None:
                pp_rank = pp_rank - pp_decoder_start

        offset = 0
        if vp_stage is not None:
            for _ in range(vp_stage):
                for pp_idx in range(pp_size):
                    offset += config.decoder_pipeline_manual_split_list[pp_idx * vp_size]
            for pp_idx in range(pp_rank):
                offset += config.decoder_pipeline_manual_split_list[pp_idx * vp_size + vp_stage]
        else:
            offset = sum(config.decoder_pipeline_manual_split_list[:pp_rank])
        return offset

    megatron.core.transformer.transformer_layer.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.transformer.transformer_block.get_transformer_layer_offset = (
        get_transformer_layer_offset_patch
    )
    megatron.core.models.gpt.gpt_layer_specs.get_transformer_layer_offset = get_transformer_layer_offset_patch
