###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse

# from datetime import time
import datetime
import os
import time

import torch
import torch.distributed as dist

SUPPORTED_COLLECTIVES = {
    "all_reduce": dist.all_reduce,
    "broadcast": dist.broadcast,
    "reduce_scatter": dist.reduce_scatter,
    "all_gather": dist.all_gather,
}


def rccl_benchmark_once(tensor, collective: str, rank: int):
    if collective == "broadcast":
        SUPPORTED_COLLECTIVES[collective](tensor, src=0)
    elif collective == "all_gather":
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        SUPPORTED_COLLECTIVES[collective](output, tensor)
    elif collective == "reduce_scatter":
        input = [tensor.clone() for _ in range(dist.get_world_size())]
        SUPPORTED_COLLECTIVES[collective](tensor, input)
    else:
        SUPPORTED_COLLECTIVES[collective](tensor)


def add_rccl_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--collective",
        choices=["all_reduce", "all_gather", "reduce_scatter", "broadcast"],
        default="all_reduce",
        help="Collective to benchmark.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8_388_608,  # 8MB
        help="Shape of data to communicate.",
    )
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--output", default="benchmark_result", help="Directory to save GEMM markdown results"
    )
    parser.add_argument("--tag", default=None, help="Optional tag for result filename")

    return parser


def profile_rccl(
    bytes: int,
    collective: str,
    duration: int = 60,
    warmup: int = 5,
    tag: str = None,
    output: str = None,
):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    hostname = os.uname().nodename

    from primus.tools.utils import get_current_device

    device = get_current_device()
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    tensor = torch.empty(bytes, dtype=torch.uint8, device=device)

    # Bandwidth calculation for ring-based collectives
    ring_bytes = 2 * bytes * (world_size - 1) / world_size

    # warmup
    for _ in range(warmup):
        rccl_benchmark_once(tensor, collective, rank)
    dist.barrier()

    times = []
    bws = []
    start_time = time.time()

    while (time.time() - start_time) < duration:
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()

        rccl_benchmark_once(tensor, collective, rank)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        times.append(dt)
        bws.append(ring_bytes / dt / 1e9)

    # Gather to rank0
    local_data = {"rank": rank, "host": hostname, "bws": bws, "times": times}
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_data, gathered, dst=0)

    # Rank 0 summary
    if rank == 0:
        lines = []
        lines.append(f"# RCCL Benchmark | {collective.upper()} | Size = {bytes} bytes\n")
        lines.append("| Rank | Host     | Avg Time(s) | Avg BW (GB/s) |")
        lines.append("|------|----------|-------------|----------------|")

        for d in gathered:
            avg_time = sum(d["times"]) / len(d["times"])
            avg_bw = sum(d["bws"]) / len(d["bws"])
            lines.append(f"| {d['rank']} | {d['host']} | {avg_time:.6f} | {avg_bw:.2f} |")

        markdown = "\n".join(lines)
        print("\n" + markdown)

        os.makedirs(output, exist_ok=True)
        tag = tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output, f"rccl_{collective}_{tag}.md")

        if output_path:
            with open(output_path, "w") as f:
                f.write(markdown)
            print(f"\n[rank0] Benchmark result written to: {output_path}")
        # print(f"\n# RCCL Benchmark | {collective.upper()} | Size = {bytes} bytes")
        # print("| Rank | Host     | Avg Time(s) | Avg BW (GB/s) |")
        # print("|------|----------|-------------|----------------|")
        # for d in gathered:
        #     avg_time = sum(d["times"]) / len(d["times"])
        #     avg_bw = sum(d["bws"]) / len(d["bws"])
        #     print(f"| {d['rank']} | {d['host']} | {avg_time:.6f} | {avg_bw:.2f} |")


def run_rccl_benchmark(args):
    profile_rccl(
        bytes=args.size,
        collective=args.collective,
        duration=args.duration,
        warmup=5,
        tag=args.tag,
        output=args.output,
    )
