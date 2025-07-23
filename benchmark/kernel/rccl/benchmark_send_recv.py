###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# [Seq, HiddenSize]
MODEL_PARAMS_TABLE = {
    "llama-2-7B": (4096, 4096),
    "llama-2-70B": (4096, 8192),
    "llama-3-8B": (8192, 4096),
    "llama-3-70B": (8192, 8192),
    "deepseek-v2-lite": (4096, 2048),
    "deepseek-v2": (4096, 5120),
    "deepseek-v3": (8192, 7168),
    "mitral-8x22B": (8192, 6144),
}
MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
ITERS = 100


def test_p2p(mbs, seq, hidden, dtype, rank, local_rank):
    shape = (mbs, seq, hidden)
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.randn(shape, dtype=dtype, device=device)

    # Warm-up
    if rank == 0:
        for _ in range(5):
            dist.send(tensor, dst=1)
    elif rank == 1:
        for _ in range(5):
            dist.recv(tensor, src=0)
    dist.barrier()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(ITERS):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        elif rank == 1:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=0)
    torch.cuda.synchronize()
    end = time.time()

    size_bytes = tensor.numel() * tensor.element_size()
    avg_time = (end - start) / ITERS / 2
    bandwidth = size_bytes / avg_time / 1e9

    return avg_time, bandwidth


def main(output_csv_path):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    assert world_size == 2, "This script requires exactly 2 processes."

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    dist.barrier()
    torch.manual_seed(42 + rank)

    #
    benchmark_results = []
    for model_name in MODEL_PARAMS_TABLE.keys():
        for mbs in MBS_LIST:
            for dtype in [torch.float16]:
                avg_time, bandwidth = test_p2p(
                    mbs,
                    MODEL_PARAMS_TABLE[model_name][0],
                    MODEL_PARAMS_TABLE[model_name][1],
                    dtype,
                    rank,
                    local_rank,
                )
                if rank == 0:
                    result = {
                        "Model": model_name,
                        "MBS": mbs,
                        "Seq": MODEL_PARAMS_TABLE[model_name][0],
                        "HiddenSize": MODEL_PARAMS_TABLE[model_name][1],
                        "DataType": dtype,
                        "Time(s)": avg_time,
                        "Bandwidth(GB/s)": bandwidth,
                    }
                    benchmark_results.append(result)

    if rank == 0:
        fieldnames = list(benchmark_results[0].keys())
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in benchmark_results:
                writer.writerow(result)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-csv-path", type=str)
    args = parser.parse_args()

    main(args.report_csv_path)
