##############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
##############################################################################

from typing import Optional, Tuple

import torch
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import (
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
)
from megatron.training import get_args
from primus.backends.megatron.core.tensor_parallel.mappings import fp8_all_to_all


class PrimusMoEAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.
        Optimized for Primus. Support float8 AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
                - Permuted probs of each token produced by the router.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_1", tokens_per_expert)
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            hidden_states,
            routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_ep_alltoall", tokens_per_expert)

        args = get_args()
        if self.config.fp8 and args.use_turbo_token_dispatcher_fp8_alltoall:
            extra_args = {
                "fwd_quant": True,
                "bwd_quant": False,
            }

            global_input_tokens = fp8_all_to_all(
                self.ep_group,
                permutated_local_input_tokens,
                self.output_splits,
                self.input_splits,
                **extra_args,
            )
        else:
            global_input_tokens = all_to_all(
                self.ep_group,
                permutated_local_input_tokens,
                self.output_splits,
                self.input_splits,
            )
        global_probs = all_to_all(self.ep_group, permuted_probs, self.output_splits, self.input_splits)
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_2", tokens_per_expert)
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

        return global_input_tokens, tokens_per_expert, global_probs

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            # The precision of TP reduce_scatter should be the same as the router_dtype
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
                input_split_sizes=input_split_sizes,
            ).to(hidden_states.dtype)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        args = get_args()
        if self.config.fp8 and args.use_turbo_token_dispatcher_fp8_alltoall:
            extra_args = {
                "fwd_quant": False,
                "bwd_quant": True,
            }

            permutated_local_input_tokens = fp8_all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits, **extra_args
            )
        else:
            permutated_local_input_tokens = all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits
            )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output, None
