###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


import torch
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args


class BalancedTopKRouter(TopKRouter):
    """Balanced route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)

    def routing(self, logits: torch.Tensor):
        scores, routing_map = super().routing(logits)

        # profile for moe
        args = get_args()
        if args.moe_router_force_load_balancing:
            indices = (
                torch.arange(routing_map.sum(), device=routing_map.device).view(
                    routing_map.size(0), self.topk
                )
                % self.num_experts
            )
            row = torch.arange(routing_map.size(0), device=routing_map.device).repeat_interleave(self.topk)
            col = indices.view(-1)
            routing_map = torch.zeros_like(routing_map, dtype=torch.bool).index_put_(
                (row, col), torch.ones(1, device=routing_map.device, dtype=torch.bool)
            )

        return scores, routing_map
