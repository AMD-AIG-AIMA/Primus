from typing import Optional

import torch
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager

from primus.backends.megatron.core.fusions.fused_indices_converter import (
    fused_indices_to_multihot,
)

try:
    from primus_turbo.pytorch.deep_ep import Buffer

    HAVE_TURBO_BACKEND = True
except ImportError:
    HAVE_TURBO_BACKEND = False


_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0

    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_token_per_expert_use_cpu: bool = True,
        previous_event=None,
    ):
        global _groupgemm_backend
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event

        if num_token_per_expert_use_cpu:
            device = "cpu"
        else:
            device = x.device
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, device=device)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, previous_event=None):
        """Forward pass of fused combine."""
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x, handle=handle, async_finish=False, previous_event=None, allocate_on_comm_stream=False
        )
        ctx.handle = handle
        ctx.group = group

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, None, None


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    num_token_per_expert_use_cpu: bool = True,
    previous_event=None,
):
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event

    Returns:
        Result of FusedDispatch
    """
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        num_token_per_expert_use_cpu,
        previous_event,
    )


def fused_combine(x, group, handle, previous_event=None):
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event

    Returns:
        Result of FusedCombine
    """
    return FusedCombine.apply(x, group, handle, previous_event)


class PrimusDeepepManager(_DeepepManager):

    def __init__(
        self,
        *args,
        groupgemm_backend: Optional[str] = "native",
        **kwargs,
    ):
        global _groupgemm_backend

        if not HAVE_TURBO_BACKEND:
            raise NotImplementedError("DeepEP not support for mori backend, please install mori!")
        super().__init__(*args, **kwargs)

        if groupgemm_backend not in ["native", "ck"]:
            raise NotImplementedError(f"unkown groupgemm backend: {groupgemm_backend}")

        self.num_token_per_expert_use_cpu = groupgemm_backend == "native"

    def set_deepep_num_cus(self, num_cus: int):
        assert HAVE_TURBO_BACKEND
        Buffer.set_num_sms(num_cus)

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=torch.sum(self.tokens_per_expert),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = fused_combine(hidden_states, self.group, self.handle)
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            self.num_token_per_expert_use_cpu,
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states
