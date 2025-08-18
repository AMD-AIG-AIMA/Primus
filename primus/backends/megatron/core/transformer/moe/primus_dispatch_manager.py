import torch
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager

from primus.backends.megatron.core.fusions.fused_indices_converter import (
    fused_indices_to_multihot,
)

try:
    HAVE_TURBO_BACKEND = True
except ImportError:
    HAVE_TURBO_BACKEND = False


class PrimusDeepepManager(_DeepepManager):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if not HAVE_TURBO_BACKEND:
            raise NotImplementedError("DeepEP not support for mori backend, please install mori!")
        super().__init__(*args, **kwargs)

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.permute_fusion:
            print("used fused_indices_to_multihot")
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
