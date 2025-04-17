import argparse
import os
import socket
import time
from pathlib import Path

import markdown2
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from weasyprint import HTML

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


def remove_file(file_path):
    if RANK == 0:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted.", flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])


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


def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks", "footnotes"])

    # Add CSS to ensure that images do not overflow the page width
    css = """
    <style>
        img {
            max-width: 100%;
            height: auto;
        }
        table {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            word-wrap: break-word;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 4px;
            font-size: 10px;
            word-wrap: break-word;
        }
    </style>
    """

    # Combine the CSS and HTML content
    html_with_css = css + html

    HTML(string=html_with_css, base_url=os.path.dirname(md_path)).write_pdf(pdf_path)
    print(f"✅ PDF Report saved to: {pdf_path}")


def get_first_ib_unidirectional_bandwidth():
    ib_path = "/sys/class/infiniband"

    # Check if the InfiniBand path exists
    if not os.path.exists(ib_path):
        log("No InfiniBand device found.")
        return 0

    # List available InfiniBand devices
    ib_devs = os.listdir(ib_path)
    if not ib_devs:
        log("No InfiniBand devices detected.")
        return 0

    # Use the first detected InfiniBand device
    ib_dev = ib_devs[0]

    # Get the ports directory for this device (typically contains "1")
    port_path = os.path.join(ib_path, ib_dev, "ports")
    port = sorted(os.listdir(port_path))[0]

    # Read the rate of the port from the 'rate' file
    # Bidirectional Bandwidth
    rate_path = os.path.join(port_path, port, "rate")
    with open(rate_path) as f:
        rate_str = f.read().strip()  # e.g., "400 Gb/sec (4X EDR)"

    # Extract the numeric part of the rate and convert from Gb/s to GB/s
    gbps = float(rate_str.split()[0])
    # Unidirectional Bandwidth
    GBps = gbps / 8 / 2
    return GBps


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

        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# Square Gemm Perf\n\n")
            f.write(f"=======Square GEMM Latency (us)=======\n")
            log("=======Square GEMM Latency (us)=======")

            # f.write(f"| Hostname | Node | Rank |\n")
            # f.write(f"|----------|----------|----------|\n")
            f.write(f"| Hostname | Node | Rank | {' | '.join(formatted_sizes)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(formatted_sizes)}|\n")
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
            for rank, result in enumerate(all_latency_results):
                hostname = HOST_NAMES[rank]
                node_id = rank // LOCAL_WORLD_SIZE
                formatted_values = [f"{result[size]*1000000:<14.2f}" for size in sizes_sorted]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

            f.write(f"=======Square GEMM TFLOPS =======\n")
            log("=======Square GEMM TFLOPS=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(formatted_sizes)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(formatted_sizes)}|\n")
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
            for rank, result in enumerate(all_tflops_results):
                hostname = HOST_NAMES[rank]
                node_id = rank // LOCAL_WORLD_SIZE
                formatted_values = [f"{result[size]:<14.2f}" for size in sizes_sorted]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

        if not args.plot:
            return

        log("=======Plot Square GEMM TFLOPS=======")
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======Plot Square GEMM TFLOPS=======\n")
        plot_case = f"square_gemm_tflops"
        dump_path = f"{args.dump_path}/{plot_case}"
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

            png_file = f"square_gemm_tflops_{size_key.replace('x', '_')}.png"
            plt.tight_layout()
            plt.savefig(f"{dump_path}/{png_file}")
            plt.close()
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
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

        png_file = f"square_gemm_tflops_rank_0.png"
        plt.tight_layout()
        plt.savefig(f"{dump_path}/{png_file}.png")
        plt.close()
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
            f.write(f"\n")
        log(f"")


def run_intra_node_comm(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    cases = {
        "allreduce": [2, 4, 8],
        "alltoall": [2, 4, 8],
    }

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# IntraNode Comm Perf\n")

    for comm, parallel in cases.items():
        if RANK == 0:
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"## IntraNode - {comm}\n")
        for num_procs in parallel:
            bandwidth_results = {}
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
                scale = 2 if comm == "allreduce" else 1
                comm_size = scale * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])

            # destroy this parallel group
            dist.destroy_process_group(group)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in HOST_NAMES) + 2

                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======IntraNodeComm - {case_name} (us)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (us)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_latency_results):
                        hostname = HOST_NAMES[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    f.write(f"=======IntraNodeComm - {case_name} (GB/s)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (GB/s)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_bandwidth_results):
                        hostname = HOST_NAMES[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    if not args.plot:
                        continue

                log(f"=======Plot IntraNode {case_name} Bandwidth=======")
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======Plot InterNode {case_name} Bandwidth=======\n")
                plot_case = f"intra_node_comm/{comm}"
                dump_path = f"{args.dump_path}/{plot_case}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                first_rank_bandwidth_results = [
                    all_bandwidth_results[i] for i in range(len(all_bandwidth_results)) if i % num_procs == 0
                ]
                num_print_ranks = len(first_rank_bandwidth_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_bandwidth_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("Bandwidth")
                    plt.title(f"Intra Node {case_name} bandwidth for {size_key}")
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

                    png_file = f"intra_node_{case_name}_bandwidth_{size_key.replace('x', '_')}.png"
                    plt.tight_layout()
                    plt.savefig(f"{dump_path}/{png_file}")
                    plt.close()
                    with open(args.markdown_file, "a", encoding="utf-8") as f:
                        f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")

                # Bar chart visualization for rank 0
                rank_0_values = [all_bandwidth_results[0][size_key] for size_key in keys]
                plt.figure(figsize=(10, 4))
                bars = plt.bar(keys, rank_0_values)
                plt.xlabel("Size")
                plt.ylabel("Bandwidth")
                plt.title(f"Intra Node {case_name} bandwidth for Rank 0")
                plt.grid(True, axis="y")

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                png_file = f"intra_node_{case_name}_bandwidth_rank_0.png"
                plt.tight_layout()
                plt.savefig(f"{dump_path}/{png_file}")
                plt.close()
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
                    f.write(f"\n")
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

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# InterNode Comm\n")

    for comm, adjacent_node_list in cases.items():
        if RANK == 0:
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"## InterNode - {comm}\n")
        for adjacent_nodes in adjacent_node_list:
            if adjacent_nodes > num_nodes:
                continue

            case_name = f"{comm}-{adjacent_nodes}nodes"
            latency_results = {}
            bandwidth_results = {}

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
            if RANK < num_adjacent_groups * adjacent_nodes * LOCAL_WORLD_SIZE:
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
                scale = 2 if comm == "allreduce" else 1
                comm_size = scale * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])
            if adjacent_group is not None:
                dist.destroy_process_group(adjacent_group)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in HOST_NAMES) + 2

                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======InterNodeComm - {case_name} (us)=======\n")
                    log(f"=======InterNodeComm - {case_name} (us)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")

                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_latency_results):
                        hostname = HOST_NAMES[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    f.write(f"=======InterNodeComm - {case_name} (GB/s)=======\n")
                    log(f"=======InterNodeComm - {case_name} (GB/s)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_bandwidth_results):
                        hostname = HOST_NAMES[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                if not args.plot:
                    continue

                log(f"=======Plot IntraNode {case_name} Bandwidth=======")
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======Plot InterNode {case_name} Bandwidth=======\n")
                plot_case = f"inter_node_comm/{comm}"
                dump_path = f"{args.dump_path}/{plot_case}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                first_rank_bandwidth_results = [
                    all_bandwidth_results[i] for i in range(len(all_bandwidth_results)) if i % num_procs == 0
                ]
                num_print_ranks = len(first_rank_bandwidth_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_bandwidth_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("Bandwidth")
                    plt.title(f"Inter Node {case_name} Bandwidth for {size_key}")
                    xtick_labels = [f"{i*num_procs}" for i in range(num_print_ranks)]
                    plt.xticks(range(num_print_ranks), xtick_labels)
                    plt.grid(True, axis="y")

                    # Add roofline
                    roofline_bandwidth = args.ib_bw
                    plt.axhline(
                        y=roofline_bandwidth,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"IB BW Roofline: {roofline_bandwidth} GB/s",
                    )
                    plt.legend()

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

                    png_file = f"inter_node_{case_name}_bandwidth_{size_key.replace('x', '_')}.png"
                    plt.tight_layout()
                    plt.savefig(f"{dump_path}/{png_file}")
                    plt.close()
                    with open(args.markdown_file, "a", encoding="utf-8") as f:
                        f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")

                # Bar chart visualization for rank 0
                rank_0_values = [all_bandwidth_results[0][size_key] for size_key in keys]
                plt.figure(figsize=(10, 4))
                bars = plt.bar(keys, rank_0_values)
                plt.xlabel("Size")
                plt.ylabel("Bandwidth")
                plt.title(f"Inter Node {case_name} Bandwidth for Rank 0")
                plt.grid(True, axis="y")
                # Add roofline
                roofline_bandwidth = args.ib_bw
                plt.axhline(
                    y=roofline_bandwidth,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"IB BW Roofline: {roofline_bandwidth} GB/s",
                )
                plt.legend()

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                png_file = f"inter_node_{case_name}_bandwidth_rank_0.png"
                plt.tight_layout()
                plt.savefig(f"{dump_path}/{png_file}")
                plt.close()
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
                    f.write(f"\n")
                log(f"")


def run_inter_node_comm_p2p(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    RANK // LOCAL_WORLD_SIZE

    if num_nodes <= 1:
        log(f"Skip inter node comm benchmark, {num_nodes=}")
        return
    # 2-node p2p
    #   pair nodes: [0, 1]
    #        ranks: [0, 8], [1, 9], [2, 10], ...
    #   pair nodes: [2, 3]
    #        ranks: [16, 24], [17, 25], [18, 26], ...
    comm = "p2p"
    adjacent_nodes = 2
    case_name = f"{comm}-{adjacent_nodes}nodes"
    latency_results = {}
    bandwidth_results = {}

    num_adjacent_groups = num_nodes // adjacent_nodes
    p2p_group = None
    is_src_rank = ((RANK // LOCAL_WORLD_SIZE) % 2) == 0
    peer_rank = RANK + LOCAL_WORLD_SIZE if is_src_rank else RANK - LOCAL_WORLD_SIZE
    assert peer_rank >= 0 and peer_rank < WORLD_SIZE
    for i_group in range(num_adjacent_groups):
        for i_r in range(LOCAL_WORLD_SIZE):
            group_ranks = [
                i_group * adjacent_nodes * LOCAL_WORLD_SIZE + i_r,
                i_group * adjacent_nodes * LOCAL_WORLD_SIZE + i_r + LOCAL_WORLD_SIZE,
            ]
            tmp_group = dist.new_group(ranks=group_ranks)
            if RANK in group_ranks:
                assert p2p_group is None
                p2p_group = tmp_group
    if RANK < num_adjacent_groups * adjacent_nodes * LOCAL_WORLD_SIZE:
        assert p2p_group is not None

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"## InterNode - P2P\n")

    for size in sizes:
        if p2p_group is None:
            break

        tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
        dist.barrier(group=p2p_group, device_ids=[torch.cuda.current_device()])
        for _ in range(WARMUP):
            if is_src_rank:
                dist.send(tensor, dst=peer_rank, group=p2p_group)
            else:
                dist.recv(tensor, src=peer_rank, group=p2p_group)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERATION):
            if is_src_rank:
                dist.send(tensor, dst=peer_rank, group=p2p_group)
            else:
                dist.recv(tensor, src=peer_rank, group=p2p_group)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / ITERATION
        comm_size = size
        gb_per_sec = comm_size / elapsed / 1e9
        latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
        bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

    dist.barrier(device_ids=[torch.cuda.current_device()])
    if p2p_group is not None:
        dist.destroy_process_group(p2p_group)

    all_latency_results = [None for _ in range(WORLD_SIZE)]
    all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
    dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

    if RANK == 0:
        keys = sorted(list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number)
        max_len = max(len(s) for s in HOST_NAMES) + 2

        # result of src ranks will be print
        src_ranks = []
        peer_ranks = []
        src_rank_latency_results = []
        src_rank_bandwidth_results = []
        for rank, r in enumerate(all_bandwidth_results):
            is_src_rank = ((rank // LOCAL_WORLD_SIZE) % 2) == 0
            peer_rank = rank + LOCAL_WORLD_SIZE if is_src_rank else rank - LOCAL_WORLD_SIZE
            assert peer_rank >= 0 and peer_rank < WORLD_SIZE
            if not is_src_rank:
                continue
            src_ranks.append(rank)
            peer_ranks.append(peer_rank)
            src_rank_latency_results.append(all_latency_results[rank])
            src_rank_bandwidth_results.append(r)

        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======InterNodeComm - {case_name} (us)=======\n")
            log(f"=======InterNodeComm - {case_name} (us)=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")

            formatted_keys = [f"{key:<6}" for key in keys]
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            for i_r in range(len(src_ranks)):
                rank = src_ranks[i_r]
                hostname = HOST_NAMES[rank]
                node_id = rank // LOCAL_WORLD_SIZE

                formatted_values = [f"{src_rank_latency_results[i_r].get(key, 0):<6.2f}" for key in keys]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

            f.write(f"=======InterNodeComm - {case_name} (GB/s)=======\n")
            log(f"=======InterNodeComm - {case_name} (GB/s)=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
            formatted_keys = [f"{key:<6}" for key in keys]
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            for i_r in range(len(src_ranks)):
                rank = src_ranks[i_r]
                hostname = HOST_NAMES[rank]
                node_id = rank // LOCAL_WORLD_SIZE

                formatted_values = [f"{src_rank_bandwidth_results[i_r].get(key, 0):<6.2f}" for key in keys]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

        if not args.plot:
            return

        log(f"=======Plot InterNode {case_name} Bandwidth=======")
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======Plot InterNode {case_name} Bandwidth=======\n")
        plot_case = f"inter_node_comm/{comm}"
        dump_path = f"{args.dump_path}/{plot_case}"
        create_dir(dump_path)
        print_keys = extract_first_middle_last(keys)

        for size_key in print_keys:
            values = [r[size_key] for r in src_rank_bandwidth_results]
            plt.figure(figsize=(10, 4))
            bars = plt.bar(range(len(src_ranks)), values)
            plt.xlabel(f"RankPair (rank-i <-> rank-i+{LOCAL_WORLD_SIZE})")
            plt.ylabel("Bandwidth")
            plt.title(f"Inter Node {case_name} Bandwidth for {size_key}")
            xtick_labels = [f"r-{src_ranks[i]}/{peer_ranks[i]}" for i in range(len(src_ranks))]
            plt.xticks(range(len(src_ranks)), xtick_labels)
            plt.grid(True, axis="y")
            # Add roofline
            roofline_bandwidth = args.ib_bw
            plt.axhline(
                y=roofline_bandwidth,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"IB BW Roofline: {roofline_bandwidth} GB/s",
            )
            plt.legend()

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

            png_file = f"intra_node_{case_name}_bandwidth_{size_key.replace('x', '_')}.png"
            plt.tight_layout()
            plt.savefig(f"{dump_path}/{png_file}")
            plt.close()
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")

        # Bar chart visualization for rank 0
        rank_0_values = [all_bandwidth_results[0][size_key] for size_key in keys]
        plt.figure(figsize=(10, 4))
        bars = plt.bar(keys, rank_0_values)
        plt.xlabel("Size")
        plt.ylabel("Bandwidth")
        plt.title(f"Inter Node {case_name} Bandwidth for Rank 0")
        plt.grid(True, axis="y")
        # Add roofline
        roofline_bandwidth = args.ib_bw
        plt.axhline(
            y=roofline_bandwidth,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"IB BW Roofline: {roofline_bandwidth} GB/s",
        )
        plt.legend()

        # plt value
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        png_file = f"inter_node_{case_name}_bandwidth_rank_0.png"
        plt.tight_layout()
        plt.savefig(f"{dump_path}/{png_file}")
        plt.close()
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
            f.write(f"\n")
        log(f"")


def main(args):
    setup()

    if RANK == 0:
        bw = get_first_ib_unidirectional_bandwidth()
        log(f"=======IB Bandwidth roofline (GB/s)=======")
        log(f"Bandwidth of first IB device of Node 0 : {bw:.2f} GB/s")
        args.ib_bw = bw

        if not os.path.isdir(args.dump_path):
            log(f"mkdir {args.dump_path}")
            os.makedirs(args.dump_path)

    args.markdown_file = f"{args.dump_path}/{args.report_file_name}.md"
    args.pdf_file = f"{args.dump_path}/{args.report_file_name}.pdf"
    remove_file(args.markdown_file)

    run_square_gemm(args)
    run_intra_node_comm(args)
    run_inter_node_comm(args)
    run_inter_node_comm_p2p(args)

    if RANK == 0 and args.save_pdf:
        md_to_pdf(args.markdown_file, args.pdf_file)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, default="output/preflight")
    parser.add_argument("--report-file-name", type=str, default="preflight_report")
    parser.add_argument("--disable-pdf", dest="save_pdf", action="store_false")
    parser.add_argument("--disable-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    main(args)
