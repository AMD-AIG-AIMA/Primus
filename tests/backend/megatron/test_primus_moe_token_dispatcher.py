from dataclasses import dataclass

import megatron.core.parallel_state as ps
import torch
import torch.distributed as dist
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.training.global_vars import set_args
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from primus.backends.megatron.core.transformer.moe.token_dispatcher import (
    PrimusMoEAll2AllTokenDispatcher,
)

from .utils import PrimusMoEModelTestContainer


@dataclass
class HackArgsForTokenDispatcher:
    seq_length: int = 1024
    micro_batch_size: int = 32
    context_parallel_size: int = 1


def run_moe_layer(hidden_states: torch.Tensor, moe_layer):
    probs, routing_map = moe_layer.router(hidden_states)

    (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = (
        moe_layer.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
    )

    expert_output = permuted_local_hidden_states

    restored_hidden_states, _ = moe_layer.token_dispatcher.token_unpermutation(expert_output)

    return permuted_local_hidden_states, tokens_per_expert, permuted_probs, restored_hidden_states


@instantiate_parametrized_tests
class PrimusMoETokenDispatcherTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(123)

    @skip_if_lt_x_gpu(2)
    @parametrize("num_of_experts", [32])
    @parametrize("moe_router_topk", [2])
    @parametrize("ep_etp_size", [(8, 1)])
    @parametrize("capacity_factor", [1.0])
    def test_dispatcher(
        self,
        num_of_experts,
        moe_router_topk,
        ep_etp_size,
        capacity_factor,
        moe_permute_fusion=False,
        **kwargs
    ):
        self._init_process()

        ep_size, etp_size = ep_etp_size
        ps.initialize_model_parallel(
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
        )

        num_local_experts = num_of_experts // ep_size
        moe_test_container = PrimusMoEModelTestContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            cp_size=1,
            moe_tp_size=etp_size,
            num_moe_experts=num_of_experts,
            moe_router_topk=moe_router_topk,
            moe_token_dispatcher_type="alltoall",
            moe_expert_capacity_factor=capacity_factor,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=moe_permute_fusion,
        )

        arg = HackArgsForTokenDispatcher(seq_length=8, micro_batch_size=32, context_parallel_size=1)
        set_args(arg)

        moe_layer, config = moe_test_container.new_moe_layer()
        hidden_states = torch.randn(
            (arg.micro_batch_size, arg.seq_length, moe_layer.config.hidden_size), dtype=torch.bfloat16
        )
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True

        output_grad = torch.randn_like(hidden_states)
        hidden_states_ref = hidden_states.clone().detach().requires_grad_()

        # run_refs
        internal_tensor_refs = run_moe_layer(hidden_states_ref, moe_layer)
        restored_hidden_states_ref = internal_tensor_refs[-1]
        restored_hidden_states_ref.backward(output_grad)

        # run patched dispatcher
        comm_group = get_default_model_comm_pgs()
        moe_layer.token_dispatcher = PrimusMoEAll2AllTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=[],
            config=config,
            model_comm_pgs=comm_group,
        )

        internal_tensor = run_moe_layer(hidden_states, moe_layer)
        restored_hidden_states = internal_tensor[-1]
        restored_hidden_states.backward(output_grad)

        for i in range(len(internal_tensor)):
            torch.testing.assert_close(
                internal_tensor[i].to(device="cpu"),
                internal_tensor_refs[i].to(device="cpu"),
            )

        torch.testing.assert_close(
            hidden_states.grad,
            hidden_states_ref.grad,
        )
