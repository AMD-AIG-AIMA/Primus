# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple

import torch
import transformer_engine.pytorch.triton.permutation as triton_permutation
from transformer_engine.pytorch.float8_tensor import Float8Tensor


class _moe_chunk_sort(torch.autograd.Function):
    """functional MoE chunk permute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            return inp

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert split_sizes.is_cuda, "TransformerEngine needs CUDA."
        assert sorted_idxs.is_cuda, "TransformerEngine needs CUDA."

        num_tokens, hidden_size = inp.shape
        num_splits = split_sizes.size(0)
        assert num_splits == sorted_idxs.size(0)

        fp8 = isinstance(inp, Float8Tensor)
        if fp8:
            fp8_dtype = inp._fp8_dtype
            fp8_scale_inv = inp._scale_inv
            inp = inp._data
        output, row_id_map = triton_permutation.sort_chunks_by_idx(
            inp,
            split_sizes,
            sorted_idxs,
            num_tokens,
            hidden_size,
            num_splits,
        )
        if fp8:
            output = Float8Tensor(data=output, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv)

        ctx.save_for_backward(row_id_map)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None

        act_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            fp8 = isinstance(permuted_act_grad, Float8Tensor)
            if fp8:
                fp8_dtype = permuted_act_grad._fp8_dtype
                fp8_scale_inv = permuted_act_grad._scale_inv
                permuted_act_grad = permuted_act_grad._data
            act_grad = triton_permutation.sort_chunks_by_map(
                permuted_act_grad,
                row_id_map,
                ctx.num_tokens,
                ctx.hidden_size,
            )
            if fp8:
                act_grad = Float8Tensor(data=act_grad, fp8_dtype=fp8_dtype, fp8_scale_inv=fp8_scale_inv)
        return act_grad, None, None
