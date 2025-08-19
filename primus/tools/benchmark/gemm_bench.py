###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import datetime
import math
import os
import socket

import torch
import torch.distributed as dist

from primus.tools.utils import get_current_device

CACHE_ROTATING_BUFFER_BYTES = 2 * 1024 * 1024 * 1024  # 2GB rotating buffer


def add_gemm_parser(parser: argparse.ArgumentParser):
    """
    Register GEMM arguments under a given parser.
    Now supports direct M/N/K input.
    """
    parser.add_argument("--M", type=int, default=4096, help="GEMM M dimension (default: 4096)")
    parser.add_argument("--N", type=int, default=4096, help="GEMM N dimension (default: 4096)")
    parser.add_argument("--K", type=int, default=4096, help="GEMM K dimension (default: 4096)")
    parser.add_argument("--trans_a", action="store_true", help="Transpose A matrix")
    parser.add_argument("--trans_b", action="store_true", help="Transpose B matrix")
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Data type for GEMM computation."
    )
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--output", default="benchmark_result", help="Directory to save GEMM markdown results"
    )
    parser.add_argument("--tag", default=None, help="Optional tag for result filename")
    return parser


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def profile_gemm(m, n, k, dtype, trans_a, trans_b):
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"

    device = get_current_device()

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = math.ceil(CACHE_ROTATING_BUFFER_BYTES / mem_size_bytes) + 1
    num_run = 100

    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.randn((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    # Warm-up
    for i in range(num_rotations):
        a = maybe_transpose(a_list[i], trans_a)
        b = maybe_transpose(b_list[i], trans_b)
        c_list[i] = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        for i in range(num_rotations):
            a = maybe_transpose(a_list[i], trans_a)
            b = maybe_transpose(b_list[i], trans_b)
            c_list[i] = torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()

    # result
    avg_time_s = start_event.elapsed_time(end_event) / 1000 / (num_rotations * num_run)
    tflop = 2 * m * n * k / 1e12
    tflops = tflop / avg_time_s
    bandwidth = mem_size_bytes / 1e9 / avg_time_s
    return {
        "m": m,
        "n": n,
        "k": k,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "dtype": str(dtype),
        "avg_time_s": avg_time_s,
        "tflop": tflop,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth,
    }


def run_gemm_benchmark(args):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    m, n, k = args.M, args.N, args.K
    trans_a, trans_b = args.trans_a, args.trans_b
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    result = profile_gemm(m, n, k, dtype, trans_a, trans_b)

    hostname = socket.gethostname()
    result["hostname"] = hostname

    # Gather results
    gathered_results = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(result, gathered_results, dst=0)

    if rank == 0:
        # os.makedirs(args.output, exist_ok=True)
        tag = args.tag or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(args.output, f"gemm_M{m}_N{n}_K{k}_{tag}.md")

        print(f"[Rank 0] GEMM benchmark results: {output_path}")

        with open(output_path, "w") as f:
            f.write("| Rank | Hostname | M | N | K | dType |Time (s) | TFLOP | TFLOPS | BW (GB/s) |\n")
            f.write("|------|----------|---|---|---|-------|---------|-------|--------|-----------|\n")
            for r, res in enumerate(gathered_results):
                f.write(
                    f"| {r} | {res['hostname']} | {res['m']} | {res['n']} | {res['k']} | {res['dtype']} | "
                    f"{res['avg_time_s']:.6f} | {res['tflop']:.2f} | {res['tflops']:.2f} | {res['bandwidth_gbps']:.2f} |\n"
                )

        print(f"[Rank 0] GEMM benchmark results saved to {output_path}")


def build_gemm_parser() -> argparse.ArgumentParser:
    """
    Build a standalone parser for local execution.
    """
    parser = argparse.ArgumentParser(description="GEMM benchmark")
    add_gemm_parser(parser)
    return parser


if __name__ == "__main__":
    parser = build_gemm_parser()
    args = parser.parse_args()
    run_gemm_benchmark(args)
