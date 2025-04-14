import os
import time
import socket
import torch
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

WARMUP = 10
ITERATION = 50


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)


def cleanup():
    dist.destroy_process_group()


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def create_dir(dir):
    # Create a Path object
    path = Path(dir)
    try:
        # Recursively create the dir. If it already exists, do nothing.
        path.mkdir(parents=True, exist_ok=True)
        print(f"Dir {dir} created successfully or already exists.")
    except PermissionError:
        print(f"Permission denied to create dir {dir}.")
    except Exception as e:
        print(f"An error occurred while creating dir {dir}: {e}")


def run_gemm():
    sizes = [1024 * (2**i) for i in range(4)]  # 1024, 2048, 4096, 8192
    sizes = [1024, 2048, 4096, 8192, 10240]
    latency_results = {}
    flops_results = {}
    for size in sizes:
        a = torch.randn((size, size), device=f"cuda:{LOCAL_RANK}", dtype=torch.bfloat16)
        b = torch.randn((size, size), device=f"cuda:{LOCAL_RANK}", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERATION):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        t = (end - start) / ITERATION
        tflops = 2 * size * size * size / (t * 1e12)
        latency_results[f"{size}x{size}x{size}"] = t
        flops_results[f"{size}x{size}x{size}"] = tflops
    all_latency_results = [None for _ in range(WORLD_SIZE)]
    all_tflops_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
    dist.gather_object(flops_results, all_tflops_results if RANK == 0 else None, dst=0)
    if RANK == 0:
        log("=======GEMM Latency (us)=======")
        sizes_sorted = flops_results.keys()
        formatted_sizes = [f"{size:<14}" for size in sizes_sorted]
        log(f"{'Hostname':<55} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_latency_results):
            hostname = socket.gethostname()
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]*1000000:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<55} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
        log("=======GEMM TFLOPS=======")
        sizes_sorted = flops_results.keys()
        formatted_sizes = [f"{size:<14}" for size in sizes_sorted]
        log(f"{'Hostname':<55} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_tflops_results):
            hostname = socket.gethostname()
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<55} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")

        dump_path = f"output/preflight"
        create_dir(dump_path)
        for size_key in sizes_sorted:
            values = [r[size_key] for r in all_tflops_results]
            plt.figure(figsize=(10, 4))
            bars = plt.bar(range(WORLD_SIZE), values)
            plt.xlabel("Rank")
            plt.ylabel("TFLOPS")
            plt.title(f"GEMM TFLOPS for {size_key}")
            plt.xticks(range(WORLD_SIZE))
            plt.grid(True, axis="y")

            # plt value
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plt.savefig(f"{dump_path}/gemm_tflops_{size_key.replace('x', '_')}.png")
            plt.close()
        # Bar chart visualization for rank 0
        rank_0_values = [all_tflops_results[0][size_key] for size_key in sizes_sorted]
        plt.figure(figsize=(10, 4))
        bars = plt.bar(sizes_sorted, rank_0_values)
        plt.xlabel("Size")
        plt.ylabel("TFLOPS")
        plt.title("GEMM TFLOPS for Rank 0")
        plt.grid(True, axis="y")

        # plt value
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(f"{dump_path}/gemm_tflops_rank_0.png")
        plt.close()


def create_pg_for_local():
    return dist.new_group(
        ranks=[
            i
            for i in range(RANK - RANK % LOCAL_WORLD_SIZE, RANK - RANK % LOCAL_WORLD_SIZE + LOCAL_WORLD_SIZE)
        ]
    )


def create_pg_for_peer_nodes(rank_a, rank_b):
    ranks = list(range(rank_a * LOCAL_WORLD_SIZE, (rank_a + 1) * LOCAL_WORLD_SIZE)) + list(
        range(rank_b * LOCAL_WORLD_SIZE, (rank_b + 1) * LOCAL_WORLD_SIZE)
    )
    return dist.new_group(ranks=ranks)


def run_local_comm():
    pg = create_pg_for_local()
    device = torch.device(f"cuda:{LOCAL_RANK}")
    results = {}
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    groups = {
        "4gpu-allreduce": list(range(4)),
        "8gpu-allreduce": list(range(8)),
        "2gpu-alltoall": list(range(2)),
        "4gpu-alltoall": list(range(4)),
        "8gpu-alltoall": list(range(8)),
    }
    for size in sizes:
        for name, ranks in groups.items():
            if LOCAL_RANK >= len(ranks):
                continue
            group_ranks = [RANK - LOCAL_RANK + r for r in ranks]
            group = dist.new_group(ranks=group_ranks)
            tensor = torch.rand(size // 4, dtype=torch.float32, device=device)
            dist.barrier()
            for _ in range(WARMUP):
                if "allreduce" in name:
                    dist.all_reduce(tensor, group=group)
                else:
                    dist.all_to_all_single(tensor, tensor, group=group)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(ITERATION):
                if "allreduce" in name:
                    dist.all_reduce(tensor, group=group)
                else:
                    dist.all_to_all_single(tensor, tensor, group=group)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / ITERATION
            gbps = size / elapsed / 1e9
            results[f"{name}_{size//1024//1024}MB"] = gbps
    all_tflops_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(results, all_tflops_results if RANK == 0 else None, dst=0)
    if RANK == 0:
        log("Local Comm Bandwidth (GB/s)")
        keys = sorted(list(results.keys()))
        log("\t" + "\t".join(keys))
        for i, r in enumerate(all_tflops_results):
            log(f"rank-{i}\t" + "\t".join([f"{r.get(k, 0):.2f}" for k in keys]))


def run_inter_node_comm():
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    results = {}
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    my_node = RANK // LOCAL_WORLD_SIZE

    # All nodes allreduce
    for size in sizes:
        tensor = torch.rand(size // 4, dtype=torch.float32, device=device)
        dist.barrier()
        for _ in range(WARMUP):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERATION):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / ITERATION
        gbps = size / elapsed / 1e9
        results[f"all_node_allreduce_{size//1024//1024}MB"] = gbps

    # 2-node alltoall & send/recv (adjacent pairs)
    if my_node % 2 == 0 and my_node + 1 < num_nodes:
        peer_node = my_node + 1
        pg = create_pg_for_peer_nodes(my_node, peer_node)
        for size in sizes:
            # alltoall
            tensor = torch.rand(size // 4, dtype=torch.float32, device=device)
            dist.barrier(group=pg)
            for _ in range(WARMUP):
                dist.all_to_all_single(tensor, tensor, group=pg)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(ITERATION):
                dist.all_to_all_single(tensor, tensor, group=pg)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / ITERATION
            gbps = size / elapsed / 1e9
            results[f"2node_alltoall_{my_node}/{peer_node}_{size//1024//1024}MB"] = gbps

            # send/recv
            peer_rank = peer_node * LOCAL_WORLD_SIZE + LOCAL_RANK
            dist.barrier(group=pg)
            for _ in range(WARMUP):
                dist.send(tensor, dst=peer_rank)
                dist.recv(tensor, src=peer_rank)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(ITERATION):
                dist.send(tensor, dst=peer_rank)
                dist.recv(tensor, src=peer_rank)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / ITERATION
            gbps = size / elapsed / 1e9
            results[f"2node_sendrecv_{my_node}/{peer_node}_{size//1024//1024}MB"] = gbps

    all_tflops_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(results, all_tflops_results if RANK == 0 else None, dst=0)
    if RANK == 0:
        log("Inter-node Comm Bandwidth (GB/s)")
        keys = sorted(set(k for r in all_tflops_results for k in r.keys()))
        log("\t" + "\t".join(keys))
        for i, r in enumerate(all_tflops_results):
            log(f"rank-{i}\t" + "\t".join([f"{r.get(k, 0):.2f}" for k in keys]))


def main():
    setup()
    run_gemm()
    # run_local_comm()
    # run_inter_node_comm()
    cleanup()


if __name__ == "__main__":
    main()
