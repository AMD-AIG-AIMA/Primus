###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import operator
from functools import reduce
from typing import Dict, List, Optional, Union

import primus_turbo.pytorch as pt
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from hip import hip
from triton_dist.utils import HIP_CHECK

from .comm_overlap_type import CommOverlapType

_backend_streams: Dict[int, List[torch.cuda.Stream]] = {}


def get_backend_stream(size=1, priority=0, prefix=""):
    global _backend_streams

    key = (priority, prefix)
    if key not in _backend_streams or len(_backend_streams[key]) < size:
        _backend_streams[key] = [torch.cuda.Stream(priority=priority) for _ in range(size)]

    return _backend_streams[key][:size]


class CommOverlapBase:
    def __init__(self, buffer_shape: List[int], buffer_dtype: torch.dtype, group_name: str, tp_size: int):

        group = c10d._resolve_process_group(group_name)
        assert tp_size == group.size(), f"tp_size {tp_size} is difference with group size: {group.size()}"

        alloc_size = reduce(operator.mul, buffer_shape, 1) * buffer_dtype.itemsize
        self.buf = torch.empty((alloc_size,), dtype=torch.uint8, device="cuda")
        self.buf_size = self.buf.nbytes
        self.tp_size = tp_size
        self.group = group
        self.rank = group.rank()
        self.buf_dtype = buffer_dtype
        self.buf_shape = buffer_shape
        self.group_name = group_name

    def is_atomic_gemm(self) -> bool: ...

    def is_p2p_overlap(self) -> bool: ...

    def is_fp8_ubuf(self) -> bool: ...

    def copy_input_to_ubuf(self, input: torch.Tensor, comm_type: Union[bool, int]) -> None:
        """copy input to local buffer

        Args:
            input (torch.Tensor): ...
            comm_type (int): 0 or 1

            if comm_type is CommOverlapType.AG, copy input to tp_size chunk of local buffer;
            if comm_type is CommOverlapType.RS, copy input to local_buffer
        """
        comm_type = CommOverlapType(int(comm_type))

        if comm_type == CommOverlapType.AG:
            if (
                input.numel() * self.tp_size != self.buf.nbytes // self.buf_dtype.itemsize
                or input.element_size() != self.buf_dtype.itemsize
            ):
                raise ValueError(f"input and ubuf size do not match!")
        else:
            if (
                input.numel() != self.buf.nbytes // self.buf_dtype.itemsize
                or input.element_size() != self.buf_dtype.itemsize
            ):
                raise ValueError(f"input and ubuf size do not match!")

        local_chunk = comm_type == CommOverlapType.AG

        self.copy_into_buffer(input, local_chunk=local_chunk)

    def copy_into_buffer(self, input, local_chunk: bool = False):
        buf = self.get_buffer(local_chunk=local_chunk)
        HIP_CHECK(
            hip.hipMemcpyAsync(
                buf.data_ptr(),
                input.data_ptr(),
                input.nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                torch.cuda.current_stream().cuda_stream,
            )
        )

    def get_buffer(self, local_chunk: bool = False, shape=None):
        out_shape = shape or self.buf_shape

        if shape is None and local_chunk:
            out_shape = [out_shape[0] // self.tp_size] + list(out_shape)[1:]

        request_size = reduce(operator.mul, out_shape, 1) * self.buf_dtype.itemsize

        if local_chunk:
            buf = self.buf.chunk(self.tp_size)[self.rank]
        else:
            buf = self.buf

        buffer = buf[0:request_size].view(self.buf_dtype).view(*out_shape)
        return buffer

    def get_ubuf_output(self, comm_type: int) -> torch.Tensor:
        """return local buffer as output.
        Args:
            comm_type (int): CommOverlapType.AG or CommOverlapType.RS

        Returns:
            torch.Tensor: if comm_type is CommOverlapType.AG, return the total buffer as output;
                          if comm_type is CommOverlapType.RS, return the tp_size chunk of local buffer as output;
        """
        buffer = self.get_buffer(local_chunk=comm_type == 0)
        return buffer

    def bulk_overlap(
        self, A: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor, comm_type: CommOverlapType
    ):

        with torch.profiler.record_function("torch_native_bulk_overlap"):
            output = self.get_ubuf_output(comm_type.value)
            local_buf = self.get_buffer(local_chunk=comm_type == CommOverlapType.AG)

            if comm_type == CommOverlapType.AG:
                handle = dist.all_gather_into_tensor(output, local_buf, group=self.group, async_op=True)
            else:
                handle = dist.reduce_scatter_tensor(output, local_buf, group=self.group, async_op=True)

            A = A.T if layout[0] == "T" else A
            B = B.T if layout[1] == "T" else B

            torch.mm(A, B, out=D)

            handle.wait()


class CommOverlap(CommOverlapBase):
    def __init__(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        group_name: str,
        tp_size: int,
        num_splits: int = 2,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
    ):

        super().__init__(buffer_shape, buffer_dtype, group_name, tp_size)

        self.num_splits = num_splits
        self.atomic_gemm = atomic_gemm

    def is_atomic_gemm(self) -> bool:
        return self.atomic_gemm

    def is_p2p_overlap(self) -> bool:
        return False

    def is_fp8_ubuf(self) -> bool:
        return False

    def split_overlap_rs(self, A_out: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor):
        raise NotImplementedError("not support for now!")

    def split_overlap_ag(
        self,
        A_out: torch.Tensor,
        B: torch.Tensor,
        layout: str,
        D: torch.Tensor,
        A_copy: Optional[torch.Tensor] = None,
    ):
        """
        virtual void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main)
        """
        local_A = self.get_buffer(local_chunk=True)
        gemm_streams = [torch.cuda.current_stream()]
        comm_streams = get_backend_stream(size=self.tp_size - 1, priority=0, prefix="comm")

        copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")
        if A_copy is not None:
            if A_copy.shape != local_A.shape:
                raise ValueError("A_copy shape is difference with local_A")
            copy_streams[0].wait_stream(torch.cuda.current_stream())
            HIP_CHECK(
                hip.hipMemcpyAsync(
                    A_copy.data_ptr(),
                    local_A.data_ptr(),
                    A_copy.nbytes,
                    hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                    copy_streams[0].cuda_stream,
                )
            )

        pt.ops.fused_all_gather_matmul(
            local_A,
            [B],
            [layout],
            gather_dim=0,
            group_name=self.group_name,
            gemm_streams=gemm_streams,
            comm_streams=comm_streams,
            copy_streams=copy_streams,
            comm_method="pipeline",
            num_splits=self.num_splits,
            skip_copy_local_A=True,
            return_A=True,
            A_out=A_out,
            outputs=[D],
        )


class CommOverlapP2P(CommOverlapBase):
    def __init__(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        group_name: str,
        tp_size: int,
        comm_type: CommOverlapType,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ):
        super().__init__(buffer_shape, buffer_dtype, group_name, tp_size)

    def is_atomic_gemm(self) -> bool: ...

    def is_p2p_overlap(self) -> bool:
        return True

    def is_fp8_ubuf(self) -> bool:
        return False

    def split_overlap_rs(self, A_out: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor):
        raise NotImplementedError("not support for now!")

    def split_overlap_ag(self, A_out: torch.Tensor, B: torch.Tensor, layout: str, D: torch.Tensor):
        raise NotImplementedError("not support for now!")

    def copy_input_to_ubuf(self, input: torch.Tensor, comm_type: int) -> None:
        raise NotImplementedError("not support for now!")
