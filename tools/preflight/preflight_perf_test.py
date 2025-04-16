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
HOST_NAMES = None


def setup():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl")
    global HOST_NAMES
    HOST_NAMES = gather_hostnames()


def cleanup():
    dist.destroy_process_group()


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def extract_number(key):
    return int(key.rstrip("MB"))


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


def extract_first_middle_last(lst):
    if not lst:
        return []

    n = len(lst)
    if n == 1:
        return [lst[0]]
    elif n == 2:
        return [lst[0], lst[1]]
    else:
        return [lst[0], lst[n // 2], lst[-1]]


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

    if RANK == 0:
        max_len = max(len(s) for s in HOST_NAMES) + 2
        sizes_sorted = flops_results.keys()
        formatted_sizes = [f"{size:<14}" for size in sizes_sorted]

        log("=======Square GEMM Latency (us)=======")
        log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_latency_results):
            hostname = HOST_NAMES[rank]
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]*1000000:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")

        log("=======Square GEMM TFLOPS=======")
        log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
        for rank, result in enumerate(all_tflops_results):
            hostname = HOST_NAMES[rank]
            node_id = rank // LOCAL_WORLD_SIZE
            formatted_values = [f"{result[size]:<14.2f}" for size in sizes_sorted]
            log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")

        if not args.plot:
            return

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
        log(f"")


def create_pg_for_peer_nodes(rank_a, rank_b):
    ranks = list(range(rank_a * LOCAL_WORLD_SIZE, (rank_a + 1) * LOCAL_WORLD_SIZE)) + list(
        range(rank_b * LOCAL_WORLD_SIZE, (rank_b + 1) * LOCAL_WORLD_SIZE)
    )
    return dist.new_group(ranks=ranks)


def run_intra_node_comm(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    cases = {
        "allreduce": [2, 4, 8],
        "alltoall": [2, 4, 8],
    }

    for comm, parallel in cases.items():
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

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_tflops_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in HOST_NAMES) + 2

                log(f"=======IntraNodeComm - {case_name} (us)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_latency_results):
                    hostname = HOST_NAMES[rank]
                    if rank % num_procs != 0:
                        continue
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

                log(f"=======IntraNodeComm - {case_name} (GB/s)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_tflops_results):
                    hostname = HOST_NAMES[rank]
                    if rank % num_procs != 0:
                        continue
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

                if not args.plot:
                    continue

                log(f"=======Plot IntraNode {case_name} TFLOPS=======")
                dump_path = f"{args.dump_path}/infra_node_comm/{comm}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                first_rank_tflops_results = [
                    all_tflops_results[i] for i in range(len(all_tflops_results)) if i % num_procs == 0
                ]
                num_print_ranks = len(first_rank_tflops_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_tflops_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("TFLOPS")
                    plt.title(f"Intra Node {case_name} TFLOPS for {size_key}")
                    xtick_labels = [f"{i*num_procs}" for i in range(num_print_ranks)]
                    plt.xticks(range(num_print_ranks), xtick_labels)
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
                    plt.savefig(f"{dump_path}/intra_node_{case_name}_tflops_{size_key.replace('x', '_')}.png")
                    plt.close()

                # Bar chart visualization for rank 0
                rank_0_values = [all_tflops_results[0][size_key] for size_key in keys]
                plt.figure(figsize=(10, 4))
                bars = plt.bar(keys, rank_0_values)
                plt.xlabel("Size")
                plt.ylabel("TFLOPS")
                plt.title(f"Intra Node {case_name} TFLOPS for Rank 0")
                plt.grid(True, axis="y")

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                plt.tight_layout()
                plt.savefig(f"{dump_path}/intra_node_{case_name}_tflops_rank_0.png")
                plt.close()
                log(f"")


def run_inter_node_comm(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    RANK // LOCAL_WORLD_SIZE

    if num_nodes <= 1:
        log(f"Skip inter node comm benchmark, {num_nodes=}")
        return

    # N-node allreduce & alltoall (adjacent pairs)
    # 2-node allreduce, pair nodes: [0, 1], [2, 3], ...
    # 4-node allreduce, pair nodes: [0, 1, 2, 3], [4, 5, 6, 7]...
    cases = {
        "allreduce": list(set([2, 4] + [num_nodes])),
        "alltoall": list(set([2, 4] + [num_nodes])),
    }

    for comm, adjacent_node_list in cases.items():
        for adjacent_nodes in adjacent_node_list:
            if adjacent_nodes > num_nodes:
                continue

            case_name = f"{comm}-{adjacent_nodes}nodes"
            latency_results = {}
            tflops_results = {}

            num_procs = adjacent_nodes * LOCAL_WORLD_SIZE
            num_adjacent_groups = num_nodes // adjacent_nodes
            adjacent_group = None
            for i_group in range(num_adjacent_groups):
                group_ranks = [
                    i_group * adjacent_nodes * LOCAL_WORLD_SIZE + r
                    for r in range(adjacent_nodes * LOCAL_WORLD_SIZE)
                ]
                tmp_group = dist.new_group(ranks=group_ranks)
                if RANK in group_ranks:
                    assert adjacent_group is None
                    adjacent_group = tmp_group
            if RANK < num_adjacent_groups * adjacent_nodes:
                assert adjacent_group is not None

            for size in sizes:
                if adjacent_group is None:
                    break

                tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
                dist.barrier(group=adjacent_group, device_ids=[torch.cuda.current_device()])
                for _ in range(WARMUP):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=adjacent_group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=adjacent_group)
                    else:
                        assert False
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(ITERATION):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=adjacent_group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=adjacent_group)
                    else:
                        assert False
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / ITERATION
                comm_size = 2 * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                tflops_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])
            if adjacent_group is not None:
                dist.destroy_process_group(adjacent_group)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_tflops_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(tflops_results, all_tflops_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_tflops_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in HOST_NAMES) + 2

                log(f"=======InterNodeComm - {case_name} (us)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_latency_results):
                    hostname = HOST_NAMES[rank]
                    if rank % num_procs != 0:
                        continue
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

                log(f"=======InterNodeComm - {case_name} (GB/s)=======")
                formatted_keys = [f"{key:<6}" for key in keys]
                log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                for rank, r in enumerate(all_tflops_results):
                    hostname = HOST_NAMES[rank]
                    if rank % num_procs != 0:
                        continue
                    node_id = rank // LOCAL_WORLD_SIZE

                    formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
                    log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

                if not args.plot:
                    continue

                log(f"=======Plot IntraNode {case_name} TFLOPS=======")
                dump_path = f"{args.dump_path}/inter_node_comm/{comm}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                first_rank_tflops_results = [
                    all_tflops_results[i] for i in range(len(all_tflops_results)) if i % num_procs == 0
                ]
                num_print_ranks = len(first_rank_tflops_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_tflops_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("TFLOPS")
                    plt.title(f"Inter Node {case_name} TFLOPS for {size_key}")
                    xtick_labels = [f"{i*num_procs}" for i in range(num_print_ranks)]
                    plt.xticks(range(num_print_ranks), xtick_labels)
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
                    plt.savefig(f"{dump_path}/inter_node_{case_name}_tflops_{size_key.replace('x', '_')}.png")
                    plt.close()

                # Bar chart visualization for rank 0
                rank_0_values = [all_tflops_results[0][size_key] for size_key in keys]
                plt.figure(figsize=(10, 4))
                bars = plt.bar(keys, rank_0_values)
                plt.xlabel("Size")
                plt.ylabel("TFLOPS")
                plt.title(f"Inter Node {case_name} TFLOPS for Rank 0")
                plt.grid(True, axis="y")

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                plt.tight_layout()
                plt.savefig(f"{dump_path}/inter_node_{case_name}_tflops_rank_0.png")
                plt.close()
                log(f"")

            # if RANK == 0:
            #     keys = sorted(
            #         list({k for r in all_tflops_results for k in (r or {}).keys()}), key=extract_number
            #     )
            #     max_len = max(len(s) for s in HOST_NAMES) + 2

            #     log(f"=======InterNodeComm - {case_name} (us)=======")
            #     formatted_keys = [f"{key:<6}" for key in keys]
            #     log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            #     for adjacent_group_id, r in enumerate(all_latency_results):
            #         rank = adjacent_group_id * adjacent_nodes * LOCAL_WORLD_SIZE
            #         hostname = HOST_NAMES[rank]
            #         node_id = rank // LOCAL_WORLD_SIZE

            #         formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
            #         log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")

            #     log(f"=======InterNodeComm - {case_name} (GB/s)=======")
            #     formatted_keys = [f"{key:<6}" for key in keys]
            #     log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            #     for adjacent_group_id, r in enumerate(all_tflops_results):
            #         rank = adjacent_group_id * adjacent_nodes * LOCAL_WORLD_SIZE
            #         hostname = HOST_NAMES[rank]
            #         node_id = rank // LOCAL_WORLD_SIZE

            #         formatted_keys = [f"{r.get(key, 0):<6.2f}" for key in keys]
            #         log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_keys)}")
            #     log(f"")

    # N-node allreduce & alltoall (adjacent pairs)
    # 2-node p2p
    #   pair nodes: [0, 1]
    #        ranks: [0, 8], [1, 9], [2, 10], ...
    #   pair nodes: [2, 3]
    #        ranks: [16, 24], [17, 25], [18, 26], ...

    # if my_node % 2 == 0 and my_node + 1 < num_nodes:
    #     peer_node = my_node + 1
    #     pg = create_pg_for_peer_nodes(my_node, peer_node)
    #     for size in sizes:
    #         # alltoall
    #         tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
    #         dist.barrier(group=pg)
    #         for _ in range(WARMUP):
    #             dist.all_to_all_single(tensor, tensor, group=pg)
    #         torch.cuda.synchronize()
    #         start = time.time()
    #         for _ in range(ITERATION):
    #             dist.all_to_all_single(tensor, tensor, group=pg)
    #         torch.cuda.synchronize()
    #         elapsed = (time.time() - start) / ITERATION
    #         gb_per_sec = size / elapsed / 1e9
    #         results[f"2node_alltoall_{my_node}/{peer_node}_{size//1024//1024}MB"] = gb_per_sec

    #         # send/recv
    #         peer_rank = peer_node * LOCAL_WORLD_SIZE + LOCAL_RANK
    #         dist.barrier(group=pg)
    #         for _ in range(WARMUP):
    #             dist.send(tensor, dst=peer_rank)
    #             dist.recv(tensor, src=peer_rank)
    #         torch.cuda.synchronize()
    #         start = time.time()
    #         for _ in range(ITERATION):
    #             dist.send(tensor, dst=peer_rank)
    #             dist.recv(tensor, src=peer_rank)
    #         torch.cuda.synchronize()
    #         elapsed = (time.time() - start) / ITERATION
    #         gb_per_sec = size / elapsed / 1e9
    #         results[f"2node_sendrecv_{my_node}/{peer_node}_{size//1024//1024}MB"] = gb_per_sec


def main(args):
    setup()
    # run_square_gemm(args)
    # run_intra_node_comm(args)
    run_inter_node_comm(args)
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, default="output/preflight")
    parser.add_argument("--disable-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    main(args)
