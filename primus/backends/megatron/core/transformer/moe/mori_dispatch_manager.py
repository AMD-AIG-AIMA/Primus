from typing import Optional

import torch
import torch.distributed as dist
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.moe.token_dispatcher import _DispatchManager

from primus.backends.megatron.core.fusions.fused_indices_converter import (
    fused_indices_to_multihot,
)

try:
    import mori
    from mori.ops import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    HAVE_MORI_BACKEND = True
except ImportError:
    HAVE_MORI_BACKEND = False


_handle: EpDispatchCombineOp = None
_handle_buffer_bytes: int = None
_handle_group: None
_handle_num_cus: int = 64
_num_gpus_per_node = 8


def get_mori_buffer_size(cfg: EpDispatchCombineConfig):
    num_token_recv = (
        cfg.world_size
        * cfg.max_num_inp_token_per_rank
        * min(cfg.num_experts_per_token, cfg.num_experts_per_rank)
    )
    token_size = num_token_recv * cfg.hidden_dim * cfg.data_type.itemsize
    staging_topk_size = num_token_recv * (
        cfg.hidden_dim * cfg.data_type.itemsize
        + 8 * cfg.num_experts_per_token
        + cfg.scale_dim * cfg.scale_type_size
    )

    weight_size = num_token_recv * cfg.num_experts_per_token * 4
    scale_size = num_token_recv * cfg.scale_dim * cfg.scale_type_size
    index_size = num_token_recv * cfg.num_experts_per_token * 4

    return 2 * staging_topk_size + token_size + 2 * weight_size + 2 * scale_size + 2 * index_size


def set_num_cus(num_cus: int):
    global _handle_num_cus
    _handle_num_cus = num_cus


def get_mori_handle(
    group, hidden_states: torch.Tensor, indices: torch.Tensor, num_experts: int
) -> EpDispatchCombineOp:
    global _handle, _handle_group, _handle_buffer_bytes, _num_gpus_per_node, _handle_num_cus

    assert len(hidden_states.shape) == 2, hidden_states.shape
    assert len(indices.shape) == 2

    num_tokens, hidden_size = hidden_states.shape
    _, router_topk = indices.shape
    rank = group.rank()
    world_size = group.size()

    num_nodes = world_size // _num_gpus_per_node
    num_local_experts = num_experts // world_size

    cfg = EpDispatchCombineConfig(
        data_type=hidden_states.dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_size,
        scale_dim=0,
        scale_type_size=0,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        warp_num_per_block=16,
        block_num=_handle_num_cus,
        max_token_type_size=hidden_states.dtype.itemsize,
        kernel_type=(
            EpDispatchCombineKernelType.InterNode if num_nodes > 1 else EpDispatchCombineKernelType.IntraNode
        ),
    )

    buffer_bytes = get_mori_buffer_size(cfg)

    if _handle is None or _handle_group != group or _handle_buffer_bytes < buffer_bytes:
        _handle = EpDispatchCombineOp(cfg)
        _handle_group = group
        _handle_buffer_bytes = buffer_bytes

    return _handle


def get_engine():
    pass


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group):
        """Forward pass of fused dispatch."""
        handle = get_mori_handle(group, x, token_indices, num_experts)
        recv_x, recv_token_probs, _, recv_token_indices = handle.dispatch(x, token_probs, None, token_indices)
        ctx.handle = handle
        num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
        for i in range(num_experts):
            num_tokens_per_expert[i] = (token_indices == i).sum()
        dist.all_reduce(num_tokens_per_expert, group=group)
        return (recv_x, recv_token_indices, recv_token_probs, num_tokens_per_expert[group.rank()], handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch."""


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, dispatch_indices, handle):
        """Forward pass of fused combine."""
        combine_x, _, _ = handle.combine(x, None, dispatch_indices)
        ctx.handle = handle
        return combine_x

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
):
    return FusedDispatch.apply(x.contiguous(), token_indices, token_probs, num_experts, group)


def fused_combine(x, group, handle):

    return FusedCombine.apply(x, group, handle)


class MoriDeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    Mori backend.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: Optional[float] = None,
        num_experts: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
    ):

        ranks = list(range(group.size()))
        gloo_group = dist.new_group(ranks, backend="gloo")
        self.group = gloo_group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if not HAVE_MORI_BACKEND:
            raise NotImplementedError("DeepEP not support for mori backend, please install mori!")

        torch._C._distributed_c10d._register_process_group("default", group)
        mori.shmem.shmem_torch_process_group_init("default")

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]

        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = fused_dispatch(
            hidden_states, self.token_indices, self.token_probs, self.num_experts, self.group
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = fused_combine(hidden_states, self.group, self.handle)
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

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
            num_out_tokens=sum(self.tokens_per_expert),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        return hidden_states
