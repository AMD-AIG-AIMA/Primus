###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import itertools
import math
import os
import socket
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def add_gemm_parser(subparsers=None):
    """
    Register the 'gemm' benchmark subcommand for CLI or return a standalone parser.

    Args:
        subparsers: argparse subparsers object from main CLI (optional).
                    If None, a standalone parser will be created.

    Returns:
        argparse.ArgumentParser: parser object with GEMM benchmark options.
    """
    if subparsers is not None:
        parser = subparsers.add_parser("gemm", help="Run GEMM benchmark (dense / attention-like shapes).")
    else:
        parser = argparse.ArgumentParser(description="GEMM benchmark (dense / attention-like shapes).")

    # Shape configuration
    parser.add_argument("--mbs", type=int, default=1, help="Micro-batch size to test, e.g. --mbs 1")
    parser.add_argument(
        "--seq", type=int, default=4096, help="Sequence length for the benchmark input tensor."
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=8192,
        help="Hidden size (H) of the model. Used in shape generation.",
    )

    # Execution control
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Data type for GEMM computation."
    )
    parser.add_argument("--device", default="cuda", help="Device for benchmarking. Default is 'cuda'.")

    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--output", default="benchmark_result", help="Directory to save GEMM markdown results"
    )

    return parser


CACHE_ROTATING_BUFFER_BYTES = 512 * (1024**2)  # 512 MB

CACHE_ROTATING_BUFFER_BYTES = 512 * (1024**2)
MBS_LIST = [1]

MODEL_CONFIGS = [
    {
        "model": "Llama2_7B",
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    {
        "model": "Llama2_70B",
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    {
        "model": "Llama3.1_8B",
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    {
        "model": "Llama3.1_70B",
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    {
        "model": "Llama3.1_405B",
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    {
        "model": "Mistral_8x7B",
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    {
        "model": "Mistral_8x22B",
        "seqlen": 4096,
        "hidden_size": 6144,
        "intermediate_size": 16384,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
]


DENSE_MODELS = [
    "Llama2_7B",
    "Llama2_70B",
    "Llama3.1_8B",
    "Llama3.1_70B",
    "Llama3.1_405B",
    "Mistral_8x7B",
    "Mistral_8x22B",
]
DEEPSEEK_MODELS = ["Deepseek_V2_Lite", "Deepseek_V2", "Deepseek_V3"]
MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


def init_distributed_if_needed():
    """Init only when launched with >1 processes via torchrun."""
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def gather_all_results(obj):
    """
    Gather arbitrary Python object(s) from all ranks.
    - If `obj` is a list, returns a single flattened list (concatenate).
    - If `obj` is a dict (your per-rank summary), returns a list[dict].
    - If `obj` is None, it's skipped.
    """
    # Single-process fallback
    if not dist.is_initialized():
        if obj is None:
            return []
        # keep list as-is; wrap non-list as list
        return obj if isinstance(obj, list) else [obj]

    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)

    # Filter out None from any rank that returned nothing
    gathered = [g for g in gathered if g is not None]

    # If ranks each returned a list, concatenate
    if len(gathered) > 0 and isinstance(gathered[0], list):
        out = []
        for g in gathered:
            out.extend(g)
        return out

    # Otherwise, assume ranks returned single objects (e.g., dict summaries)
    return gathered


# def is_rank_0() -> bool:
#     return not dist.is_initialized() or dist.get_rank() == 0
def is_rank_0() -> bool:
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized, cannot determine rank.")
    return dist.get_rank() == 0


def write_markdown_summary(all_summaries, markdown_path, model_name):
    """
    Write aggregated GEMM benchmark results to a Markdown table.

    This function is typically called on rank 0 after all ranks have
    completed benchmarking and results have been gathered with
    `gather_all_results`.

    Args:
        all_summaries (list[dict]): A list of per-rank summary dictionaries.
            Each dict should contain:
                - rank (int): The rank ID.
                - hostname (str): Hostname of the machine.
                - count (int): Number of GEMM ops measured.
                - t_mean / t_p95 / t_max / t_topk (float): TFLOPS statistics.
                - b_mean / b_p95 / b_max / b_topk (float): Bandwidth statistics.
        markdown_path (str): Path to the output `.md` file.
        model_name (str): Name of the model being benchmarked.

    Notes:
        - Non-dict entries in `all_summaries` are ignored for robustness.
        - Output table is sorted by (hostname, rank) for deterministic ordering.
        - Numeric values are formatted to two decimal places for readability.
    """
    # Filter only dict entries to avoid type errors (robustness)
    safe = [s for s in all_summaries if isinstance(s, dict)]

    # Sort by hostname, then rank
    safe.sort(key=lambda s: (str(s.get("hostname", "")), int(s.get("rank", 0))))

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_name} GEMM Benchmark Summary\n\n")
        f.write(
            "| Rank | Hostname | Count | Mean TFLOPS | P95 TFLOPS | Max TFLOPS | Top-5 TFLOPS | "
            "Mean GB/s | P95 GB/s | Max GB/s | Top-5 GB/s |\n"
        )
        f.write(
            "|-----:|----------|------:|------------:|-----------:|-----------:|-------------:|"
            "---------:|---------:|---------:|------------:|\n"
        )

        for s in safe:
            f.write(
                f"| {s.get('rank','N/A')} | {s.get('hostname','N/A')} | {s.get('count',0)} "
                f"| {s.get('t_mean',0):.2f} | {s.get('t_p95',0):.2f} | {s.get('t_max',0):.2f} | {s.get('t_topk',0):.2f} "
                f"| {s.get('b_mean',0):.2f} | {s.get('b_p95',0):.2f} | {s.get('b_max',0):.2f} | {s.get('b_topk',0):.2f} |\n"
            )


# def write_markdown_summary(all_summaries, markdown_path, model_name):
#     """
#     Write aggregated benchmark results to a Markdown table, including Rank.
#     """
#     with open(markdown_path, "w", encoding="utf-8") as f:
#         f.write(f"# {model_name} GEMM Benchmark Summary\n\n")
#         f.write("| Rank | Hostname | Count | Mean TFLOPS | P95 TFLOPS | Max TFLOPS | Top-5 TFLOPS | "
#                 "Mean GB/s | P95 GB/s | Max GB/s | Top-5 GB/s |\n")
#         f.write("|-----:|----------|------:|------------:|-----------:|-----------:|-------------:|"
#                 "---------:|---------:|---------:|------------:|\n")

#         all_summaries = sorted(all_summaries, key=lambda s: (s.get("hostname",""), s.get("rank", 0)))

#         for s in all_summaries:
#             rank_str = str(s.get("rank", "N/A"))
#             host_str = str(s.get("hostname", "N/A"))
#             f.write(
#                 f"| {rank_str} | {host_str} | {s['count']} "
#                 f"| {s['t_mean']:.2f} | {s['t_p95']:.2f} | {s['t_max']:.2f} | {s['t_topk']:.2f} "
#                 f"| {s['b_mean']:.2f} | {s['b_p95']:.2f} | {s['b_max']:.2f} | {s['b_topk']:.2f} |\n"
#             )


def get_local_rank() -> int:
    """Return local rank for this node (torchrun/Slurm/fallback)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() % max(1, torch.cuda.device_count())
    if "LOCAL_RANK" in os.environ:  # torchrun
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_current_device() -> torch.device:
    """Set and return the current CUDA device for this rank."""
    lr = get_local_rank()
    torch.cuda.set_device(lr)
    return torch.device(f"cuda:{lr}")


def profile_gemm(m, n, k, dtype, transA, transB):
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"

    device = get_current_device()

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = math.ceil(CACHE_ROTATING_BUFFER_BYTES / mem_size_bytes) + 1
    num_run = 100

    # Shape and Tensor
    a_shape = (k, m) if transA else (m, k)
    # In PyTorch, weights are typically stored as [n, k] rather than [k, n].
    b_shape = (n, k) if transB else (k, n)
    a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.randn((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    # Warm-up
    for i in range(num_rotations):
        a = maybe_transpose(a_list[i], transA)
        b = maybe_transpose(b_list[i], transB)
        c_list[i] = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        for i in range(num_rotations):
            a = maybe_transpose(a_list[i], transA)
            b = maybe_transpose(b_list[i], transB)
            c_list[i] = torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()

    # result
    avg_time_s = start_event.elapsed_time(end_event) / 1000 / (num_rotations * num_run)
    tflop = 2 * m * n * k / 1e12
    tflops = tflop / avg_time_s
    bandwidth = mem_size_bytes / 1e9 / avg_time_s
    return (m, n, k, transA, transB, dtype, avg_time_s, tflop, tflops, bandwidth)


def profile_gemm_fwd(m, n, k, dtype):
    return profile_gemm(m, n, k, dtype, False, True)


def profile_gemm_wgrad(m, n, k, dtype):
    return profile_gemm(n, k, m, dtype, True, False)


def profile_gemm_dgrad(m, n, k, dtype):
    return profile_gemm(m, k, n, dtype, False, False)


# def calc_summary_stats(hostname: str, rank: int, tflops_list, bandwidth_list, top_k=5):
#     """
#     Calculate aggregated GEMM performance statistics for a single rank.

#     Args:
#         hostname (str): Hostname of the current rank's node.
#         tflops_list (List[float]): TFLOPS values for all GEMM ops in this rank.
#         bandwidth_list (List[float]): Bandwidth values (GB/s) for all GEMM ops in this rank.
#         top_k (int): Number of top results to average for "Top-K Mean".

#     Returns:
#         dict: Aggregated statistics for this rank.
#     """
#     def _stats(values):
#         if not values:
#             return (0, 0, 0, 0)
#         mean_val = statistics.mean(values)
#         p95_val = statistics.quantiles(values, n=100)[94]
#         max_val = max(values)
#         topk_mean = statistics.mean(sorted(values, reverse=True)[:top_k])
#         return (mean_val, p95_val, max_val, topk_mean)

#     t_mean, t_p95, t_max, t_topk = _stats(tflops_list)
#     b_mean, b_p95, b_max, b_topk = _stats(bandwidth_list)

#     return {
#         "hostname": hostname,
#         "rank": rank,            # 👈 带上 rank
#         "count": len(tflops_list),
#         "t_mean": t_mean,
#         "t_p95":  t_p95,
#         "t_max":  t_max,
#         "t_topk": t_topk,
#         "b_mean": b_mean,
#         "b_p95":  b_p95,
#         "b_max":  b_max,
#         "b_topk": b_topk,
#     }


def calc_summary_stats(hostname: str, rank: int, tflops_list, bandwidth_list, top_k: int = 5):
    """
    Calculate aggregated GEMM performance statistics for a single rank.

    Notes:
      - P95 is computed via numpy.percentile(values, 95), which is the conventional definition.
      - Top-K Mean uses the top-K values (descending) with bounds checking.
    """

    def _p95(values):
        # Standard 95th percentile; fallback for older NumPy
        if len(values) < 2:
            return float(values[0]) if values else 0.0
        try:
            return float(np.percentile(values, 95, method="linear"))
        except TypeError:
            # NumPy < 1.22 uses 'interpolation' instead of 'method'
            return float(np.percentile(values, 95, interpolation="linear"))

    def _topk_mean(values, k):
        if not values:
            return 0.0
        k = max(1, min(k, len(values)))
        return float(statistics.mean(sorted(values, reverse=True)[:k]))

    def _stats(values):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        mean_val = float(statistics.mean(values))
        p95_val = _p95(values)
        max_val = float(max(values))
        topk_val = _topk_mean(values, top_k)
        # Defensive clamp so P95 never exceeds Max due to numeric quirks
        if p95_val > max_val:
            p95_val = max_val
        return mean_val, p95_val, max_val, topk_val

    t_mean, t_p95, t_max, t_topk = _stats(tflops_list)
    b_mean, b_p95, b_max, b_topk = _stats(bandwidth_list)

    return {
        "hostname": hostname,
        "rank": rank,
        "count": len(tflops_list),
        "t_mean": t_mean,
        "t_p95": t_p95,
        "t_max": t_max,
        "t_topk": t_topk,
        "b_mean": b_mean,
        "b_p95": b_p95,
        "b_max": b_max,
        "b_topk": b_topk,
    }


def benchmark_model_dense(report_dir_path: str, model_config: dict, mbs_list: list[int]):
    """
    Benchmark dense GEMM operations for a given model configuration.

    Each rank:
      1. Generates GEMM shapes for QKV, attention out, MLP, and LM head.
      2. Profiles each GEMM for TFLOPS performance.
      3. Calculates local summary statistics (mean, p95, max, top-k mean).
      4. Sends summary to rank 0 via gather_all_results().

    Rank 0:
      - Aggregates results from all ranks.
      - Prints a clean summary table.

    Args:
        report_dir_path (str): Directory to save optional benchmark reports.
        model_config (dict): Model specification dictionary.
    """
    model_name = model_config["model"]
    hostname = socket.gethostname()

    # Generate GEMM shapes for key dense ops
    seq = model_config["seqlen"]
    hidden = model_config["hidden_size"]
    interm = model_config["intermediate_size"]
    nh = model_config["num_attention_heads"]
    nkv = model_config["num_key_value_heads"]
    hd = model_config["head_dim"]
    vocab = model_config["vocab_size"]

    # Define GEMM shapes for major dense layers
    gemm_shapes = [
        [seq, (nh + 2 * nkv) * hd, hidden],  # QKV projection
        [seq, hidden, hidden],  # Attention output projection
        [seq, 2 * interm, hidden],  # MLP gate + up projection
        [seq, hidden, interm],  # MLP down projection
        [seq, vocab, hidden],  # LM head projection
    ]

    combos = itertools.product(
        [torch.bfloat16],
        mbs_list,
        gemm_shapes,
        [profile_gemm_fwd, profile_gemm_wgrad, profile_gemm_dgrad],
    )

    tflops_list, bandwidth_list = [], []
    for dtype, mbs, shape, func in tqdm(combos, desc=f"[{model_name}] GEMM Benchmark"):
        m, n, k = mbs * shape[0], shape[1], shape[2]
        _, _, _, _, _, _, _, _, tflops, bandwidth = func(m, n, k, dtype)
        tflops_list.append(tflops)
        bandwidth_list.append(bandwidth)

    # gather and write
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_summary = calc_summary_stats(hostname, rank, tflops_list, bandwidth_list)
    all_summaries = gather_all_results(local_summary)

    if is_rank_0():
        Path(report_dir_path).mkdir(parents=True, exist_ok=True)
        markdown_path = os.path.join(report_dir_path, f"benchmark_gemm_{model_name}.md")
        write_markdown_summary(all_summaries, markdown_path, model_name)
        print(f"[Rank 0] Markdown summary saved to: {markdown_path}")


def benchmark_gemm(model: str, report_dir: str, mbs_list: list[int]):
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    matched = False
    for config in MODEL_CONFIGS:
        name = config["model"]

        if model.upper() != "ALL" and model != name:
            continue

        matched = True
        if name in DENSE_MODELS:
            benchmark_model_dense(report_dir, config, mbs_list)
        elif name in DEEPSEEK_MODELS:
            pass
            # benchmark_model_deepseek(report_dir_path, config)
        else:
            raise ValueError(f"[Benchmark] Unsupported model: {name}")

    if not matched:
        raise ValueError(f"[Benchmark] No matching model found for: {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="If run all model, set --model=all. If only run specific model, set --model=xxx, for example --model=Llama2_7B.",
    )
    parser.add_argument(
        "--mbs-list",
        type=int,
        nargs="+",
        default=[1],
        help="List of micro-batch sizes to benchmark, e.g., --mbs-list 1 2 4",
    )
    parser.add_argument("--report-dir-path", type=str)
    args = parser.parse_args()

    benchmark_gemm(args.model, args.report_dir_path)
