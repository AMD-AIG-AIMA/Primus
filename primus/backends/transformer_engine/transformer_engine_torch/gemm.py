from megatron.core.utils import is_te_min_version
if is_te_min_version("2.0"):
    from typing import Iterable, Optional, Tuple, Union, List
    import torch
    from transformer_engine.pytorch.tensor.quantized_tensor import (
        Quantizer,
        QuantizedTensor,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8Quantizer,
        Float8Tensor,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    # from transformer_engine.pytorch.constants import TE_DType
    import primus.backends.transformer_engine.transformer_engine_torch as ptex

    from typing import List, Tuple
    import operator
    from functools import reduce


    def product(shape: Tuple[int], start: int, end: int) -> int:
        """Product of shape[start:end]"""
        return reduce(operator.mul, shape[start:end], 1)


    def get_gemm_output_shape(
        A_shape: Tuple[int], transa: bool, B_shape: Tuple[int], transb: bool
    ) -> List[int]:

        # Flatten outer dims (i.e., batch dims) to 2D matrices
        A0 = product(A_shape, 0, len(A_shape) - 1)
        A1 = A_shape[-1]
        B0 = product(B_shape, 0, len(B_shape) - 1)
        B1 = B_shape[-1]

        # Check matrix dim compatibility
        dim_A = A1 if transa else A0
        dim_B = B0 if transb else B1

        assert dim_A == dim_B, (
            f"Invalid GEMM shapes: A=({A0}, {A1}), transa={transa}, "
            f"B=({B0}, {B1}), transb={transb}"
        )

        # Build output shape
        output_shape = []

        if transb:
            output_shape.append(B1)
        else:
            # Copy B's batch dims (flattened previously)
            output_shape.extend(B_shape[:-1])

        if transa:
            output_shape.append(A0)
        else:
            output_shape.append(A1)

        return output_shape


    def get_fp8_meta(inp: torch.Tensor, need_transpose: bool = False):
        scale_inv = None
        if not isinstance(inp, QuantizedTensor):
            return (inp if not need_transpose else inp.T, scale_inv)

        if isinstance(inp, Float8Tensor):
            scale_inv = inp._scale_inv
            if not need_transpose:
                return inp._data, scale_inv
            if not inp._transpose_invalid and inp._transpose is not None:
                return inp._transpose, scale_inv
            return inp.T, scale_inv
        if isinstance(inp, MXFP8Tensor):
            if not need_transpose:
                if inp._rowwise_data is not None and inp._rowwise_scale_inv is not None:
                    return inp._rowwise_data, inp._rowwise_scale_inv
                quantizer = inp._quantizer
                init_rowwise_usage, init_columnwise_usage = (
                    quantizer.rowwise_usage,
                    quantizer.columnwise_usage,
                )
                quantizer.set_usage(rowwise=True, columnwise=False)
                inp = quantizer(inp.dequantize())
                quantizer.set_usage(
                    rowwise=init_rowwise_usage, columnwise=init_columnwise_usage
                )
                return inp._rowwise_data, inp._rowwise_scale_inv

            if inp._columnwise_data is not None and inp._columnwise_scale_inv is not None:
                return inp._columnwise_data, inp._columnwise_scale_inv
            quantizer = inp._quantizer
            init_columnwise_usage = quantizer.columnwise_usage
            quantizer.set_usage(columnwise=True)
            inp = quantizer(inp.dequantize())
            quantizer.set_usage(columnwise=init_columnwise_usage)
            return inp._columnwise_data, inp._columnwise_scale_inv
        raise ValueError(f"quantized tensor inp's type not suppoted to get_fp8_meta")


    def generic_gemm(
        A: torch.Tensor,
        transA: bool,
        B: torch.Tensor,
        transB: bool,
        D: torch.Tensor,
        quantizer: Quantizer,
        output_dtype: Optional[torch.dtype],
        bias: Optional[torch.Tensor],
        bias_type: torch.dtype,
        gelu: bool,
        gelu_in: torch.Tensor,
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Union[ptex.CommOverlap, ptex.CommOverlapP2P] = None,
        comm_type: ptex.CommOverlapType = None,
        extra_output: torch.Tensor = None,
        bulk_overlap: bool = False,
    ) -> Iterable[Optional[torch.Tensor]]:
        if is_te_min_version("2.1"):
            from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
            per_tensor_quantizers = (Float8Quantizer, Float8CurrentScalingQuantizer)
        else:
            per_tensor_quantizers = Float8Quantizer
        assert A is not None and B is not None, "Tensor A or B has not been provided"
        assert not isinstance(A, MXFP8Tensor) and not isinstance(
            B, MXFP8Tensor
        ), "async tp does not support MXFP8"
        assert (
            len(A.shape) >= 1 and len(B.shape) >= 1
        ), "Tensor A and B need to have at least 1 dimension"

        D_shape = get_gemm_output_shape(A.shape, transA, B.shape, transB)
        if D is None:
            out_dtype = output_dtype if output_dtype is not None else A.dtype
            out_dtype = ptex.comm_overlap.te_to_torch_dtype(out_dtype)
            if quantizer is not None and isinstance(A, QuantizedTensor):
                D = quantizer.make_empty(D_shape, dtype=out_dtype, device="cuda")
            else:
                D = torch.empty(
                    tuple(D_shape),
                    dtype=out_dtype,
                    device="cuda",
                )
        else:
            if len(D.shape) != len(D_shape):
                raise ValueError(
                    f"Gemm output has invalid dims(expected {D_shape}, got {D.shape})"
                )
            for i in range(len(D_shape)):
                if D_shape[i] != D.shape[i]:
                    raise ValueError(
                        f"Gemm output has invalid dims(expected {D_shape}, got {D.shape})"
                    )
            if output_dtype is not None and output_dtype != D.dtype:
                raise ValueError(
                    f"Gemm output has invalid dtype(expected {output_dtype}, found {D.dtype})"
                )

        use_bias = bias is not None and bias.numel() > 0
        bias_grad = None
        if use_bias and grad:
            bias_grad = torch.empty(B.shape[-1], dtype=D.dtype, device="cuda")
            bias = bias_grad
        elif use_bias and not bias.is_contiguous():
            bias = bias.contiguous()

        if isinstance(A, QuantizedTensor) or isinstance(B, QuantizedTensor):
            gelu_dtype = bias_type
        else:
            gelu_dtype = D.dtype

        if gelu and not grad:
            pre_gelu_out = torch.empty_like(D, dtype=gelu_dtype)
        else:
            pre_gelu_out = None

        if extra_output is not None and extra_output.numel() <= 0:
            extra_output = None

        if gelu or accumulate:
            raise NotImplementedError(f"Not impl for async TP, {gelu=}, {accumulate=}")

        if isinstance(A, QuantizedTensor) and isinstance(B, QuantizedTensor):
            A, A_scale_inv = get_fp8_meta(A, transA)
            B, B_scale_inv = get_fp8_meta(B, transB)
            D_scale = (
                quantizer.scale
                if quantizer is not None
                and isinstance(quantizer, per_tensor_quantizers)
                else None
            )
            layout = "NN"
            kwargs = {
                "scale_a": B_scale_inv,
                "scale_b": A_scale_inv,
                "scale_result": D_scale,
                "out_dtype": out_dtype,
                "bias": bias,
                "use_fast_accum": not use_split_accumulator,
            }
        elif not isinstance(A, QuantizedTensor) and not isinstance(B, QuantizedTensor):
            B = B.T if transB else B
            layout = "NT" if transA else "NN"
            kwargs = None
        else:
            raise ValueError("Async tp does not support only A or B is QuantizedTensor")

        args = (B, A)

        if comm_overlap:
            if use_bias:
                raise NotImplementedError(f"Not impl for async TP, {use_bias=}")
            assert comm_type is not None, "Async TP needs comm_type is not None"
            args = args + (layout, D)
            if bulk_overlap is True:
                fn = comm_overlap.bulk_overlap
                args = tuple(args + (comm_type, kwargs))
            elif comm_type == ptex.CommOverlapType.AG:
                fn = comm_overlap.split_overlap_ag
                args = tuple(args + (extra_output, kwargs))
            elif comm_type == ptex.CommOverlapType.RS:
                fn = comm_overlap.split_overlap_rs
                assert extra_output is not None, "split_overlap_rs requires extra output"
                args = tuple(args + (extra_output,))
            else:
                raise ValueError(
                    f"TP comm overlap on, but provided {bulk_overlap=} and {comm_type=} are invalid"
                )
            fn(*args)
        else:
            if kwargs is not None:
                fn = torch._scaled_mm
                fn(*args, out=D, **kwargs)
            else:
                fn = torch.mm
                if use_bias:
                    fn = torch.addmm
                    args = args + (bias,)

                fn(*args, out=D)
        return D, bias_grad, pre_gelu_out, extra_output
