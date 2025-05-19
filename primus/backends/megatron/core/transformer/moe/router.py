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

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)

    def routing(self, logits: torch.Tensor):
        scores, routing_map = super().routing(logits)

        # profile for moe
        args = get_args()
        if args.moe_router_force_load_balancing:
            # indices = (
            #     torch.arange(routing_map.sum(), device=routing_map.device).view(
            #         routing_map.size(0), self.topk
            #     )
            #     % self.num_experts
            # )
            # print(indices)
            # row = torch.arange(routing_map.size(0), device=routing_map.device).repeat_interleave(self.topk)
            # col = indices.view(-1)
            # routing_map = torch.zeros_like(routing_map, dtype=torch.bool).index_put_(
            #     (row, col), torch.ones(1, device=routing_map.device, dtype=torch.bool)
            # )
            num_tokens = routing_map.size(0)
            topk = self.topk
            num_experts = self.num_experts
            ep_size = 8 
            #args.ep_size
            experts_per_rank = num_experts // ep_size

            assert topk == ep_size, "This routing strategy requires topk == ep_size (1 expert per rank)"

            # Step 1: 每个 rank 中随机选择 1 个 expert
            # Shape: [ep_size] → 每个元素是该 rank 的一个 expert
            selected_experts = torch.tensor([
                torch.randint(i * experts_per_rank, (i + 1) * experts_per_rank, (1,), device=logits.device).item()
                    for i in range(ep_size)
            ], device=logits.device)

            # Step 2: 对每个 token，都路由到这 8 个 expert 上
            # Shape: [num_tokens, topk]，复制 selected_experts
            chosen_experts = selected_experts.repeat(num_tokens, 1)  # [num_tokens, 8]
            
            # print(chosen_experts)

            # Step 3: 构造 routing_map
            row = torch.arange(num_tokens, device=logits.device).repeat_interleave(topk)  # [0,0,...,1,1,...]
            col = chosen_experts.view(-1)

            routing_map = torch.zeros_like(routing_map, dtype=torch.bool).index_put_(
                (row, col),
                torch.ones(1, device=logits.device, dtype=torch.bool)
            )


        return scores, routing_map
