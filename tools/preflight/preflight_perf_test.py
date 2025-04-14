import argparse
import os
import socket
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

WARMUP = 10
ITERATION = 50


def setup():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def create_dir(dir):
    path = Path(dir)
    try:
        # Recursively create the dir. If it already exists, do nothing.
        path.mkdir(parents=True, exist_ok=True)
        print(f"Dir {dir} created successfully or already exists.")
    except PermissionError:
        print(f"Permission denied to create dir {dir}.")
    except Exception as e:
        print(f"An error occurred while creating dir {dir}: {e}")


def gather_hostnames():
    hostname = socket.gethostname()
    if RANK == 0:
        all_hostnames = [None for _ in range(WORLD_SIZE)]
        dist.gather_object(hostname, all_hostnames, dst=0)
        return all_hostnames
    else:
        dist.gather_object(hostname, None, dst=0)
        return None


def run_square_gemm(args):
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
    hostnames = gather_hostnames()

    if RANK == 0:
        max_len = max(len(s) for s in hostnames) + 2
        sizes_sorted = flops_results.keys()
        formatted_sizes = [f"{size:<14}" for size in sizes_sorted]

        log("=======Square GEMM Latency (us)=======")
        log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_latency_results):
            hostname = hostnames[rank]
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]*1000000:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")

        log("=======Square GEMM TFLOPS=======")
        log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_tflops_results):
            hostname = hostnames[rank]
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")

        log("=======Plot Square GEMM TFLOPS=======")
        dump_path = f"{args.dump_path}/square_gemm_tflops"
        create_dir(dump_path)
        for size_key in sizes_sorted:
            values = [r[size_key] for r in all_tflops_results]
            plt.figure(figsize=(10, 4))
            bars = plt.bar(range(WORLD_SIZE), values)
            plt.xlabel("Rank")
            plt.ylabel("TFLOPS")
            plt.title(f"Square GEMM TFLOPS for {size_key}")
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
            plt.savefig(f"{dump_path}/square_gemm_tflops_{size_key.replace('x', '_')}.png")
            plt.close()
        # Bar chart visualization for rank 0
        rank_0_values = [all_tflops_results[0][size_key] for size_key in sizes_sorted]
        plt.figure(figsize=(10, 4))
        bars = plt.bar(sizes_sorted, rank_0_values)
        plt.xlabel("Size")
        plt.ylabel("TFLOPS")
        plt.title("Square GEMM TFLOPS for Rank 0")
        plt.grid(True, axis="y")

        # plt value
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(f"{dump_path}/square_gemm_tflops_rank_0.png")
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


def run_local_comm(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    groups = {
        "allreduce": [2, 4, 8],
        "alltoall": [2, 4, 8],
    }

    for comm, parallel in groups.items():
        for num_procs in parallel:
            tflops_results = {}
            latency_results = {}
            case_name = f"{comm}-{num_procs}gpu"

            assert LOCAL_WORLD_SIZE % num_procs == 0
            assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
            num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
            num_groups_per_node = LOCAL_WORLD_SIZE // num_procs
            group = None
            for i_node in range(num_nodes):
                for i_group in range(num_groups_per_node):
                    group_ranks = [
                        i_node * LOCAL_WORLD_SIZE + i_group * num_procs + r for r in range(num_procs)
                    ]
                    tmp_group = dist.new_group(ranks=group_ranks)
                    if RANK in group_ranks:
                        assert group is None
                        group = tmp_group
            assert group is not None

            for size in sizes:
                tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
                for _ in range(WARMUP):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(ITERATION):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / ITERATION
                comm_size = 2 * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                tflops_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])

            # destroy this parallel group
            dist.destroy_process_group(group)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_tflops_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(tflops_results, all_tflops_results if RANK == 0 else None, dst=0)
            hostnames = gather_hostnames()

            if RANK == 0:

                def extract_number(key):
                    return int(key.rstrip("MB"))

                keys = sorted(
                    list({k for r in all_tflops_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in hostnames) + 2

                log(f"=======LocalComm - {case_name} (us)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_latency_results):
                    hostname = hostnames[rank]
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

                log(f"=======LocalComm - {case_name} (GB/s)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_tflops_results):
                    hostname = hostnames[rank]
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")


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


def main(args):
    setup()
    run_square_gemm(args)
    run_local_comm(args)
    # run_inter_node_comm(args)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, default="output/preflight")
    # parser.add_argument("--num-devices", type=int, default=1)
    args = parser.parse_args()

    main(args)
