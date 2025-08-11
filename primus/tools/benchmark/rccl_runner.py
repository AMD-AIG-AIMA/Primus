###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse

import torch

from primus.tools.utils import allreduce_once


def add_rccl_parser(subparsers=None):
    """
    Register the 'rccl' benchmark subcommand.
    If subparsers is None, build a standalone parser (for direct run).
    """
    if subparsers is not None:
        parser = subparsers.add_parser("rccl", help="RCCL collectives benchmark.")
    else:
        parser = argparse.ArgumentParser(description="RCCL collectives benchmark.")

    parser.add_argument(
        "--collective",
        choices=["all_reduce", "all_gather", "reduce_scatter", "broadcast"],
        default="all_reduce",
        help="Collective to benchmark.",
    )
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length for the benchmark tensor.")
    parser.add_argument("--hidden-size", type=int, default=8192, help="Hidden size for the benchmark tensor.")
    parser.add_argument(
        "--topk", type=int, default=5, help="Top-K values to average for performance summary."
    )
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds.")

    return parser


def bench_allreduce(
    mbs: int, seq: int, hidden: int, dtype, iters: int, local_rank: int, world_size: int, warmup: int = 5
):
    """Return (avg_time, avg_bw, times[], bws[]) for a single (mbs, seq, hidden)."""
    device = torch.device(f"cuda:{local_rank}")
    shape = (mbs, seq, hidden)
    tensor = torch.ones(shape, dtype=dtype, device=device)

    # warmup
    for _ in range(warmup):
        allreduce_once(tensor)
    dist.barrier()
    torch.cuda.synchronize(device)

    # bytes per rank for ring all-reduce: 2 * N * (p-1)/p
    elem_bytes = tensor.element_size()
    size_bytes = tensor.numel() * elem_bytes
    per_rank_bytes = 2 * size_bytes * (world_size - 1) / world_size

    times, bws = [], []
    for _ in range(iters):
        start = time.time()
        allreduce_once(tensor)
        torch.cuda.synchronize(device)
        end = time.time()
        dt = end - start
        times.append(dt)
        bws.append(per_rank_bytes / dt / 1e9)  # GB/s

    avg_t = sum(times) / len(times)
    avg_bw = sum(bws) / len(bws)
    return avg_t, avg_bw, times, bws


def benchmark_rccl_allreduce_no_agg(
    seq: int,
    hidden_size: int,
    mbs: int,
    dtype: str = "bf16",
    iters: int = 100,
    per_iter: bool = False,
):
    """Each rank runs and prints its own results (no cross-rank aggregation)."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    hostname = os.uname().nodename

    torch.manual_seed(42 + rank)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    _dtype = dtype_map[dtype]

    # Header
    print(
        f"# RCCL AllReduce (no-agg) | rank={rank} host={hostname} ws={world_size} dev=cuda:{local_rank} dtype={dtype}"
    )
    print(f"## Shape: mbs={mbs}, seq={seq}, hidden={hidden_size}, iters={iters}\n")

    avg_time, avg_bw, times, bws = bench_allreduce(
        mbs=mbs,
        seq=seq,
        hidden=hidden_size,
        dtype=_dtype,
        iters=iters,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Summary (Markdown)
    print("| MBS | Avg Time(s) | Avg GB/s |")
    print("|---:|------------:|---------:|")
    print(f"| {mbs} | {avg_time:.6f} | {avg_bw:.2f} |")

    if per_iter:
        print(f"\n<details><summary>Per-iteration details</summary>\n")
        print("| Iter | Time(s) | GB/s |")
        print("|----:|--------:|-----:|")
        for i, (t, bw) in enumerate(zip(times, bws), 1):
            print(f"| {i} | {t:.6f} | {bw:.2f} |")
        print("\n</details>")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
