# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Permutation kernels written with OpenAI Triton."""

from typing import Union

import torch
import triton
import triton.language as tl


@triton.jit
def _row_id_map_pass_1_kernel(
    # pointers
    routing_map_ptr,
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int64)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)


@triton.jit
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * num_tokens + offset, mask=(offset < num_tokens), other=0
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id,
        mask=(offset < num_tokens),
    )


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    # pylint: disable=missing-function-docstring
    row_id_map = torch.empty((num_experts, num_tokens), dtype=torch.int64, device="cuda")
    block_size = 256
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int64, device="cuda")
    # block cumsum
    _row_id_map_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        block_size,
    )
    # cumsum all and process the mask
    _row_id_map_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )
    return row_id_map


@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    scale_ptr,
    permuted_probs_ptr,
    permuted_scale_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    scale_hidden_dim,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_scale_token,
    stride_scale_hidden,
    stride_permuted_probs_token,
    stride_permuted_scale_token,
    stride_permuted_scale_hidden,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    PERMUTE_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_pos = 0
    while cur_pos < hidden_size:
        cur_off = cur_pos + tl.arange(0, BLOCK_SIZE)
        mask = cur_off < hidden_size
        input_off = pid * stride_input_token + cur_off * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        if PERMUTE_SCALE:
            mask_scale = cur_off < scale_hidden_dim
            scale_off = pid * stride_scale_token + cur_off * stride_scale_hidden
            scale = tl.load(scale_ptr + scale_off, mask=mask_scale)
        for expert_idx in range(num_experts):
            dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if dst_row != -1:
                output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
                tl.store(output_ptr + output_off, inp, mask=mask)
                if PERMUTE_SCALE:
                    permuted_scale_off = (
                        dst_row * stride_permuted_scale_token + cur_off * stride_permuted_scale_hidden
                    )
                    tl.store(permuted_scale_ptr + permuted_scale_off, scale, mask=mask_scale)
                if PERMUTE_PROBS:
                    if cur_pos == 0:
                        prob_off = pid * stride_probs_token + expert_idx * stride_probs_expert
                        prob = tl.load(probs_ptr + prob_off)
                        permuted_prob_off = dst_row * stride_permuted_probs_token
                        tl.store(permuted_probs_ptr + permuted_prob_off, prob)
        cur_pos += BLOCK_SIZE


try:
    _permute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["hidden_size"],
    )(_permute_kernel)
except RuntimeError:
    pass


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    scale: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None

    if scale is not None:
        permuted_scale = torch.empty((num_out_tokens, scale_hidden_dim), dtype=scale.dtype, device="cuda")
    else:
        permuted_scale = None

    grid = (num_tokens,)
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        scale,
        permuted_probs,
        permuted_scale,
        num_tokens,
        num_experts,
        hidden_size,
        scale_hidden_dim,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        scale.stride(0) if scale is not None else None,
        scale.stride(1) if scale is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        permuted_scale.stride(0) if permuted_scale is not None else None,
        permuted_scale.stride(1) if permuted_scale is not None else None,
        PERMUTE_PROBS=probs is not None,
        PERMUTE_SCALE=scale is not None,
    )
    return output, permuted_scale, permuted_probs


@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    unpermuted_probs_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # metas
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        for expert_idx in range(num_experts):
            src_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if src_row != -1:
                input_off = src_row * stride_input_token + current_offset * stride_input_hidden
                inp = tl.load(input_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                if WITH_MERGING_PROBS:
                    merging_prob_off = (
                        pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                    )
                    merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                    inp *= merging_prob
                accumulator += inp
            if PERMUTE_PROBS:
                if current_start == 0:
                    unpermuted_prob_off = (
                        pid * stride_unpermuted_probs_token + expert_idx * stride_unpermuted_probs_expert
                    )
                    if src_row != -1:
                        permuted_prob_off = src_row * stride_permuted_probs_token
                        prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
                    else:
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, 0.0)
        accumulator = accumulator.to(data_type)
        output_off = pid * stride_output_token + current_offset * stride_output_hidden
        tl.store(output_ptr + output_off, accumulator, mask=mask)
        current_start += BLOCK_SIZE


try:
    _unpermute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["hidden_size"],
    )(_unpermute_kernel)
except RuntimeError:
    pass


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Union[torch.Tensor, None],
    permuted_probs: Union[torch.Tensor, None],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if permuted_probs is not None:
        unpermuted_probs = torch.empty((num_tokens, num_experts), dtype=permuted_probs.dtype, device="cuda")
    else:
        unpermuted_probs = None
    grid = (num_tokens,)
    _unpermute_kernel[grid](
        inp,
        output,
        row_id_map,
        merging_probs,
        permuted_probs,
        unpermuted_probs,
        num_tokens,
        num_experts,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
    )
    return output, unpermuted_probs


@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    merging_probs_grad_ptr,
    row_id_map_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    for expert_idx in range(num_experts):
        dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
        if dst_row != -1:
            prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
            current_start = 0
            while current_start < hidden_size:
                current_offset = current_start + tl.arange(0, BLOCK_SIZE)
                mask = current_offset < hidden_size
                input_off = (
                    pid * stride_fwd_output_grad_token + current_offset * stride_fwd_output_grad_hidden
                )
                inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                merging_prob_off = pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                output = inp * merging_prob
                output = output.to(data_type)
                output_off = (
                    dst_row * stride_fwd_input_grad_token + current_offset * stride_fwd_input_grad_hidden
                )
                tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

                fwd_input_off = dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
                fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
                prob_grad_accum += fwd_input.to(compute_type) * inp
                current_start += BLOCK_SIZE
            probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
            probs_grad_off = (
                pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
            )
            tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)
        else:
            probs_grad_off = (
                pid * stride_merging_probs_grad_token + expert_idx * stride_merging_probs_grad_expert
            )
            tl.store(merging_probs_grad_ptr + probs_grad_off, 0.0)


try:
    _unpermute_bwd_with_merging_probs_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["hidden_size"],
    )(_unpermute_bwd_with_merging_probs_kernel)
except RuntimeError:
    pass


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    act_grad = torch.empty((num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device="cuda")
    merging_probs_grad = torch.empty((num_tokens, num_experts), dtype=merging_probs.dtype, device="cuda")
    grid = (num_tokens,)
    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_tokens,
        num_experts,
        hidden_size,
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        act_grad.stride(0),
        act_grad.stride(1),
        fwd_input.stride(0),
        fwd_input.stride(1),
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
    )
    return act_grad, merging_probs_grad


@triton.jit
def _sort_chunks_by_idxs_kernel(
    # pointers
    input_ptr,
    split_sizes_ptr,
    sorted_indices_ptr,
    output_ptr,
    dst_rows_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    num_splits,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    IDX_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    sorted_indices = tl.load(sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits)

    # get chunk idx of the current token in the input tensor
    input_chunk_idx = -1
    in_chunk_offset = tl.zeros([], dtype=tl.int64)
    acc_chunk_sizes = tl.zeros([], dtype=tl.int64)
    cursor = 0
    while cursor < num_splits:
        cur_chunk_size = tl.load(split_sizes_ptr + cursor).to(tl.int64)
        acc_chunk_sizes += cur_chunk_size
        if input_chunk_idx == -1 and acc_chunk_sizes > pid:
            input_chunk_idx = cursor
            in_chunk_offset = pid - (acc_chunk_sizes - cur_chunk_size)
        cursor += 1

    # get chunk idx of the current token in the output tensor
    output_chunk_idx = 0
    cursor = 0
    while cursor < num_splits:
        cur_input_idx = tl.load(sorted_indices_ptr + cursor)
        if cur_input_idx == input_chunk_idx:
            output_chunk_idx = cursor
        cursor += 1

    # make row_id_map
    output_split_sizes = tl.load(split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits).to(
        tl.int64
    )
    output_pre_split_sizes = tl.where(load_split_offset < output_chunk_idx, output_split_sizes, 0)
    dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset
    tl.store(dst_rows_ptr + pid, dst_row)

    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = pid * stride_input_token + current_offset * stride_input_hidden
        output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
        inp = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, inp, mask=mask)
        current_start += BLOCK_SIZE

    if PERMUTE_PROBS:
        prob_off = pid * stride_probs_token
        prob = tl.load(probs_ptr + prob_off)
        permuted_prob_off = dst_row * stride_permuted_probs_token
        tl.store(permuted_probs_ptr + permuted_prob_off, prob)


try:
    _sort_chunks_by_idxs_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["hidden_size"],
    )(_sort_chunks_by_idxs_kernel)
except RuntimeError:
    pass


def sort_chunks_by_idx(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_splits: int,
):
    # pylint: disable=missing-function-docstring
    row_id_map = torch.empty((num_tokens,), dtype=torch.int64, device="cuda")
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _sort_chunks_by_idxs_kernel[grid](
        inp,
        split_sizes,
        sorted_indices,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_splits,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return output, row_id_map, permuted_probs


@triton.jit
def _sort_chunks_by_map_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_row = tl.load(row_id_map_ptr + pid)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = dst_row * stride_input_token + current_offset * stride_input_hidden
        output_offsets = pid * stride_output_token + current_offset * stride_output_hidden
        inp = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, inp, mask=mask)
        current_start += BLOCK_SIZE
    if PERMUTE_PROBS:
        prob_off = dst_row * stride_probs_token
        prob = tl.load(probs_ptr + prob_off)
        permuted_prob_off = pid * stride_permuted_probs_token
        tl.store(permuted_probs_ptr + permuted_prob_off, prob)


try:
    _sort_chunks_by_map_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["hidden_size"],
    )(_sort_chunks_by_map_kernel)
except RuntimeError:
    pass


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _sort_chunks_by_map_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
    )
    return output, permuted_probs
