###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import Optional, Tuple, Union

import primus_turbo.pytorch as pt
import torch
from transformer_engine.pytorch.cpp_extensions.gemm import _empty_tensor

import primus.backends.transformer_engine.transformer_engine_torch as ptex


def is_fp8(dtype: torch.dtype):
    return dtype in [pt.float8_e4m3, pt.float8_e5m2]


def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    out_index=None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    ub_algo: tex.CommOverlapAlgo = None,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    extra_output_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    if not use_bias:
        bias = None

    empty_tensor = _empty_tensor()
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None
    assert_dim_for_fp8_exec(A)
    assert_dim_for_fp8_exec(B)
    assert A.dtype == torch.uint8
    assert B.dtype == torch.uint8

    if A_scale_inv.numel() > 0:
        A_scale_inv = A_scale_inv[A_fp8_tensor]
    if B_scale_inv.numel() > 0:
        B_scale_inv = B_scale_inv[B_fp8_tensor]

    if out is None:
        out = torch.empty(
            B.shape[0],
            A.shape[0],
            dtype=out_dtype,
            device="cuda",
        )
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    if gelu:
        gelu_input = torch.empty_like(out, dtype=bias_dtype)
    else:
        gelu_input = empty_tensor

    if gelu or accumulate:
        raise NotImplementedError(f"not impl for async tp, gelu: {gelu}, accumulate: {accumulate}")

    out_dtype = out.dtype if D_dtype is None else D_dtype

    D_scale = None if out_index is None else fp8_meta_tensor.scale[out_index]

    A = ptex.comm_overlap.view_as_torch_dtype(A, A_dtype)
    B = ptex.comm_overlap.view_as_torch_dtype(B, B_dtype)

    out_dtype = ptex.comm_overlap.te_to_torch_dtype(out_dtype)
    out = out.view(out_dtype)

    args = (B, A.T)
    kwargs = {
        "scale_a": B_scale_inv,
        "scale_b": A_scale_inv,
        "scale_result": D_scale,
        "out_dtype": out_dtype,
        "bias": bias,
        "use_fast_accum": not use_split_accumulator,
    }

    fn = torch._scaled_mm

    if ub_algo is not None:
        assert ub is not None, "ub object is None!"
        args = args + ("NN", out)
        if ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            args = tuple(args + (ptex.CommOverlapType.AG, kwargs))
        elif ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            args = tuple(args + (ptex.CommOverlapType.RS, kwargs))
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag
            args = tuple(args + (extra_output_tensor, A_dtype, kwargs))
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs

        elif (
            ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS or ub_algo == tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P
        ):
            raise NotImplementedError("not impl!")
        fn(*args)
    else:
        fn(*args, out=out, **kwargs)

    return out, gelu_input


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    gelu_input: Optional[torch.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    ub_algo: ptex.CommOverlapAlgo = None,
    ub: Union[ptex.CommOverlap, ptex.CommOverlapP2P] = None,
    extra_output_tensor: torch.Tensor = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    empty_tensor = _empty_tensor()

    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    if gelu or accumulate or use_bias:
        raise NotImplementedError

    assert (
        A.dtype == dtype and B.dtype == dtype
    ), f"Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}"

    B = B.T if layout[1] == "T" else B
    if (ub_algo is not None) and (ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS):
        layout = "N" + layout[0]
    else:
        A = A.T if layout[0] == "T" else A
        layout = "NN"

    args = (B, A)
    if ub_algo is not None:
        assert ub is not None, "ub object is None!"

        args = args + (layout, out)
        if ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            args = tuple(args + (ptex.CommOverlapType.AG,))
        elif ub_algo == ptex.CommOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            args = tuple(args + (ptex.CommOverlapType.RS,))
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P:
            fn = ub.split_overlap_ag
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert extra_output_tensor is not None, "SPLIT_PIPELINED_RS requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs
            assert extra_output_tensor is not None, "SPLIT_PIPELINED_RS_P2P requires extra output tensor"
            args = tuple(args + (extra_output_tensor,))
        fn(*args)

    else:
        fn = torch.mm
        if use_bias:
            fn = torch.addmm
            args = args + (bias,)

        fn(*args, out=out)
    return out, grad_bias, gelu_input
