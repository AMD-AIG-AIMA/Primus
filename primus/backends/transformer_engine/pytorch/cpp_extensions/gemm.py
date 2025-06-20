###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


from typing import Optional, Tuple, Union

import torch
from transformer_engine.pytorch.cpp_extensions.gemm import _empty_tensor

import primus.backends.transformer_engine.transformer_engine_torch as ptex


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

    A = A.T if layout[0] == "T" else A
    B = B.T if layout[1] == "T" else B

    args = (B, A)
    if ub_algo is not None:
        assert ub is not None, "ub object is None!"

        args = args + ("NN", out)
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
        elif ub_algo == ptex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P:
            fn = ub.split_overlap_rs
        fn(*args)

    else:
        fn = torch.mm
        if use_bias:
            fn = torch.addmm
            args = args + (bias,)

        fn(*args, out=out)
    return out, grad_bias, gelu_input
