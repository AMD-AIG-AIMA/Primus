###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import List, Optional

import torch
import torch.distributed._symmetric_memory as symm_mem
from megatron.core.tensor_parallel import all_to_all
from megatron.core.transformer.moe.moe_utils import (
    maybe_move_tensor_to_cpu,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.training import get_args

from primus.backends.megatron.third_party.on_device_all2all import OnDeviceAllToAllV


class PrimusMoEAll2AllTokenDispatcher(MoETokenDispatcher):
    """Dispatcher for MoE tokens using torchtitan's OnDeviceAllToAllV."""

    token_send_buf: Optional[torch.Tensor] = None
    probs_send_buffer: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None

    def __init__(self, config, num_local_experts: int, local_expert_indices: List[int], model_comm_pgs=None):
        super().__init__(config, model_comm_pgs)
        args = get_args()
        self.seq_length = args.seq_length
        self.micro_batch_size = args.micro_batch_size
        if args.cp_size > 1:
            self.seq_length = args.seq_length // args.cp_size
        self.moe_router_topk = config.moe_router_topk

        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.tp_size == 1  # tp currently not support
        if PrimusMoEAll2AllTokenDispatcher.token_send_buf is None:
            self._init_symmetric_memory_buffers()

        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else "cpu"

        self.cuda_sync_point = "no_sync"
        self.cuda_dtoh_point = "before_permutation_1"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_stream = torch.cuda.Stream()

        input_chunk_idxs = torch.arange(self.num_experts * self.tp_size, device=self.permute_idx_device)

        self.sort_input_by_local_experts = input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel()

    def _init_symmetric_memory_buffers(self):
        """Initialize symmetric buffers for token dispatching."""
        overflow = 2
        OnDeviceAllToAllV.max_output_len = (
            self.seq_length * self.micro_batch_size * self.moe_router_topk * overflow
        )
        symm_mem.enable_symm_mem_for_group(self.ep_group.group_name)
        # Input buffer for DP-to-EP shuffle
        # print(f"debug yc seq-len {self.seq_length} topk {self.moe_router_topk} hidden_sz {self.config.hidden_size}")
        PrimusMoEAll2AllTokenDispatcher.token_send_buf = symm_mem.empty(
            self.seq_length
            * self.micro_batch_size
            * self.moe_router_topk
            * overflow,  # seq len * top k (flattened)
            self.config.hidden_size,  # hidden dim
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

        # Input buffer for EP-to-DP shuffle
        PrimusMoEAll2AllTokenDispatcher.token_gather_buf = symm_mem.empty(
            self.seq_length
            * self.micro_batch_size
            * self.moe_router_topk
            * overflow,  # seq len * top k (flattened)
            self.config.hidden_size,  # hidden dim
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

        PrimusMoEAll2AllTokenDispatcher.probs_send_buffer = symm_mem.empty(
            self.seq_length
            * self.micro_batch_size
            * self.moe_router_topk
            * overflow,  # seq len * top k (flattened)
            1,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if self.cuda_sync_point_priority[point] < self.cuda_sync_point_priority[self.cuda_sync_point]:
            self.cuda_sync_point = point

    def _maybe_dtoh_and_synchronize(self, point: str, tokens_per_expert: torch.Tensor = None) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.

                    # Will removed if group-gemm do not need it
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )

                    # do not need if no capacity set
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )

                    # do not need if local expert == 1
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )

            if point == self.cuda_sync_point:
                # Synchronize with the dtoh stream at self.cuda_sync_point.
                self.cuda_dtoh_stream.synchronize()

        return tokens_per_expert

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """Preprocess routing map for token dispatching."""
        if self.drop_and_pad:
            raise NotImplementedError("Not implemented yet")
        tokens_per_expert = routing_map.sum(dim=0)
        # num_outs_tokens = tokens_per_expert.sum()

        if self.config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            self.num_out_tokens = tokens_per_expert.sum()
            # A synchronization is needed before the first permutation
            # to get the `num_out_tokens` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # Dropless
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        with torch.no_grad():
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            torch.distributed.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(dim=1)
            if self.num_local_experts > 1:
                # [e1,e2,  e1,e2,  e1,e2,  e1,e2] ep_size = 4; local_expert = 2
                self.num_global_tokens_per_local_expert = tokens_per_expert_group.view(self.ep_size, -1)
                self.output_range = self.num_global_tokens_per_local_expert.sum()

                if not self.config.moe_permute_fusion:
                    # A synchronization is needed before permutation 2
                    # to get the `num_global_tokens_per_local_expert` CPU value.
                    self._maybe_update_cuda_sync_point("before_permutation_2")
        tokens_per_local_expert = tokens_per_expert_group.view(self.ep_size, -1).sum(dim=0)
        self._maybe_update_cuda_sync_point("before_finish")
        return tokens_per_local_expert, input_splits, output_splits

    def get_token_send_buf(self):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_probs_send_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.probs_send_buffer.grad = None
        return self.probs_send_buffer.detach()

    def get_token_gather_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def token_permutation(self, hidden_states, probs, routing_map):
        """Forward pass for MoE token dispatching."""
        # exchange local experts info
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert hidden_states.dtype == torch.bfloat16, "token permutation only support bfloat16 yet"
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        tokens_per_expert, input_splits, output_splits_ref = self.preprocess(routing_map)

        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_1", tokens_per_expert)
        self.hidden_shape_before_permute = hidden_states.shape
        # permute local logits by routing_map
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

        # all2all logits for local experts
        token_send_buf = self.get_token_send_buf()
        assert permutated_local_input_tokens.dim() == 2

        out_splits_cpu = output_splits_ref.to(torch.device("cpu"), non_blocking=False).numpy()
        in_splits_cpu = input_splits.to(torch.device("cpu"), non_blocking=False).numpy()
        self.in_splits_cpu = in_splits_cpu
        self.output_splits_cpu = out_splits_cpu
        self.local_probs_num = permutated_local_input_tokens.shape[0]

        if True:
            print(
                f"debug1 {permutated_local_input_tokens.shape} | {permutated_local_input_tokens.dtype} | {input_splits} | {self.local_probs_num}"
            )
            token_send_buf[: self.local_probs_num].copy_(permutated_local_input_tokens)
            global_input_tokens, output_splits = OnDeviceAllToAllV.apply(
                token_send_buf,
                input_splits,
                self.ep_group,
            )
            print(f"debug3 {output_splits}")
        else:
            global_input_tokens = all_to_all(
                self.ep_group, permutated_local_input_tokens, out_splits_cpu, in_splits_cpu
            )

        global_input_tokens = global_input_tokens[: self.output_range]
        self.output_splits = output_splits_ref
        self.in_splits = input_splits

        # all2all probs
        if False:
            probs_send_buffer = self.get_probs_send_buf()
            permuted_probs = permuted_probs.unsqueeze(-1).contiguous()
            probs_send_buffer[: self.local_probs_num].copy_(permuted_probs)

            global_probs, _ = OnDeviceAllToAllV.apply(
                probs_send_buffer,
                input_splits,
                self.ep_group,
                1,
                1024,
            )
            global_probs = global_probs[: self.output_range].squeeze(-1).contiguous()
        else:
            global_probs = all_to_all(self.ep_group, permuted_probs, out_splits_cpu, in_splits_cpu)

        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_2", tokens_per_expert)
        # permute local chunks if local_experts > 1 (use te)
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                raise NotImplementedError("Not implemented yet")
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

    def token_unpermutation(self, hidden_states, bias=None):
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
        if True:
            torch.cuda.current_stream().synchronize()
            torch.distributed.barrier(self.ep_group)
            # all2all logits for local experts
            processed_tokens = self.get_token_gather_buf()
            # Move into Symmetric Memory for the return shuffle
            print("debug_hidden_states_shape", hidden_states.shape)
            self.hs_0 = hidden_states.shape[0]
            processed_tokens[: self.hs_0].copy_(hidden_states)

            print(
                f"debug2 {hidden_states.shape} | {hidden_states.dtype} | {self.output_splits} | {self.hs_0}"
            )

            # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
            # The input/output splits are just a reverse of the previous shuffle.
            token_return_buf, tmp_out_splits = OnDeviceAllToAllV.apply(
                processed_tokens,
                self.output_splits,
                self.ep_group,
            )
            torch.cuda.current_stream().synchronize()
            torch.distributed.barrier(self.ep_group)

            print(f"debug4 {tmp_out_splits}")  # the answer in unstable
            token_return_buf = token_return_buf[: self.local_probs_num]
        else:
            token_return_buf = all_to_all(
                self.ep_group, hidden_states, self.in_splits_cpu, self.output_splits_cpu
            )

        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(token_return_buf)
            self.shared_experts.post_forward_comm()
        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            token_return_buf,
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
