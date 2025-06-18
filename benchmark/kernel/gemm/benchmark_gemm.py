import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

CACHE_ROTATING_BUFFER_BYTES = 512 * (1024**2)  # 512 MB


DEVICE = "cuda:0"


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


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def profile_gemm(m, n, k, dtype, transA, transB):
    assert dtype in [torch.float16, torch.bfloat16]
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = math.ceil(CACHE_ROTATING_BUFFER_BYTES / mem_size_bytes) + 1
    num_run = 100

    # Shape and Tensor
    a_shape = (k, m) if transA else (m, k)
    # In PyTorch, weights are typically stored as [n, k] rather than [k, n].
    b_shape = (n, k) if transB else (k, n)
    a_list = [torch.randn(a_shape, device=DEVICE, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=DEVICE, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.randn((m, n), device=DEVICE, dtype=dtype) for _ in range(num_rotations)]

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


def benchmark_model_dense(report_dir_path, model_config):
    model_name = model_config["model"]
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]
    vocab_size = model_config["vocab_size"]

    # Generate shapes
    gemm_shape_list = []  # [[m, n, k]...]
    # attn qkv
    gemm_shape_list.append(
        [
            seq,
            int((num_attention_heads + 2 * num_key_value_heads) * head_dim),
            hidden_size,
        ]
    )
    # attn out
    gemm_shape_list.append([seq, hidden_size, hidden_size])
    # mlp gate+up
    gemm_shape_list.append([seq, int(2 * intermediate_size), hidden_size])
    # mlp down
    gemm_shape_list.append([seq, hidden_size, intermediate_size])
    # vocab
    gemm_shape_list.append([seq, vocab_size, hidden_size])

    perf_results = []

    param_combos = list(
        itertools.product(
            [torch.bfloat16],
            MBS_LIST,
            gemm_shape_list,
            [profile_gemm_fwd, profile_gemm_wgrad, profile_gemm_dgrad],
        )
    )
    for dtype, mbs, shape, func in tqdm(param_combos, desc=f"{model_name} Benchmarking"):
        (
            m,
            n,
            k,
            transA,
            transB,
            dtype,
            avg_time_s,
            tflop,
            tflops,
            bandwidth,
        ) = func(mbs * shape[0], shape[1], shape[2], dtype=dtype)
        result = {
            "model": model_name,
            "m": m,
            "n": n,
            "k": k,
            "transA": "T" if transA else "N",
            "transB": "T" if transB else "N",
            "dtype": str(dtype),
            "Time(s)": avg_time_s,
            "TFLOPS": tflops,
            "Bandwidth(GB/s)": bandwidth,
        }
        perf_results.append(result)

    csv_filename = f"benchmark_gemm_{model_name}.csv"
    csv_path = os.path.join(report_dir_path, csv_filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
        writer.writeheader()
        for result in perf_results:
            writer.writerow(result)

    json_filename = f"benchmark_gemm_{model_name}.json"
    json_path = os.path.join(report_dir_path, json_filename)
    with open(json_path, mode="w", encoding="utf-8") as jsonfile:
        json.dump(perf_results, jsonfile, indent=4)


def benchmark_model_deepseek(report_dir_path, model_config):
    model_name = model_config["model"]
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    kv_lora_rank = model_config["kv_lora_rank"]
    moe_intermediate_size = model_config["moe_intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    n_routed_experts = model_config["n_routed_experts"]
    n_shared_experts = model_config["n_shared_experts"]
    num_experts_per_tok = model_config["num_experts_per_tok"]
    q_lora_rank = model_config["q_lora_rank"]
    qk_nope_head_dim = model_config["qk_nope_head_dim"]
    qk_rope_head_dim = model_config["qk_rope_head_dim"]
    v_head_dim = model_config["v_head_dim"]
    vocab_size = model_config["vocab_size"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Generate shapes
    gemm_shape_list = []  # [[m, n, k]...]
    # q down
    if q_lora_rank is None:
        gemm_shape_list.append(
            [
                seq,
                int(num_attention_heads * q_head_dim),
                hidden_size,
            ]
        )
    else:
        gemm_shape_list.append([seq, q_lora_rank, hidden_size])
        gemm_shape_list.append([seq, int(num_attention_heads * q_head_dim), q_lora_rank])

    # kv_down
    gemm_shape_list.append([seq, kv_lora_rank + qk_rope_head_dim, hidden_size])
    # kv_up
    gemm_shape_list.append([seq, int(num_attention_heads * (qk_nope_head_dim + v_head_dim)), kv_lora_rank])
    # attn out
    gemm_shape_list.append([seq, hidden_size, int(v_head_dim * num_attention_heads)])

    # Router
    gemm_shape_list.append([seq, n_routed_experts, hidden_size])

    # MoE
    # ShareExpert
    if n_shared_experts > 0:
        gemm_shape_list.append([seq, intermediate_size * 2, hidden_size])  # GateUp
        gemm_shape_list.append([seq, hidden_size, intermediate_size])  # Down

    # Force balance
    balance_seq = int(seq * num_experts_per_tok // n_routed_experts)
    gemm_shape_list.append([balance_seq, moe_intermediate_size * 2, hidden_size])  # GateUp
    gemm_shape_list.append([balance_seq, hidden_size, moe_intermediate_size])  # Down

    # vocab
    gemm_shape_list.append([seq, vocab_size, hidden_size])

    #
    perf_results = []
    param_combos = list(
        itertools.product(
            [torch.bfloat16],
            MBS_LIST,
            gemm_shape_list,
            [profile_gemm_fwd, profile_gemm_wgrad, profile_gemm_dgrad],
        )
    )
    for dtype, mbs, shape, func in tqdm(param_combos, desc=f"{model_name} Benchmarking"):
        (
            m,
            n,
            k,
            transA,
            transB,
            dtype,
            avg_time_s,
            tflop,
            tflops,
            bandwidth,
        ) = func(mbs * shape[0], shape[1], shape[2], dtype=dtype)
        result = {
            "model": model_name,
            "m": m,
            "n": n,
            "k": k,
            "transA": "T" if transA else "N",
            "transB": "T" if transB else "N",
            "dtype": str(dtype),
            "Time(s)": avg_time_s,
            "TFLOPS": tflops,
            "Bandwidth(GB/s)": bandwidth,
        }
        perf_results.append(result)

    csv_filename = f"benchmark_gemm_{model_name}.csv"
    csv_path = os.path.join(report_dir_path, csv_filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
        writer.writeheader()
        for result in perf_results:
            writer.writerow(result)

    json_filename = f"benchmark_gemm_{model_name}.json"
    json_path = os.path.join(report_dir_path, json_filename)
    with open(json_path, mode="w", encoding="utf-8") as jsonfile:
        json.dump(perf_results, jsonfile, indent=4)


def benchmark_deepseek_moe(report_dir_path, model_config):
    model_name = model_config["model"]
    hidden_size = model_config["hidden_size"]
    moe_intermediate_size = model_config["moe_intermediate_size"]

    # Generate shapes
    perf_results = []
    for func in [profile_gemm_fwd, profile_gemm_wgrad, profile_gemm_dgrad]:
        gate_up_shape = [1, moe_intermediate_size * 2, hidden_size]
        down_shape = [1, hidden_size, moe_intermediate_size]
        for shape in [gate_up_shape, down_shape]:
            for dtype in [torch.bfloat16]:
                for m in tqdm(range(1, 4096 + 1)):
                    shape[0] = m
                    (
                        m,
                        n,
                        k,
                        transA,
                        transB,
                        dtype,
                        avg_time_s,
                        tflop,
                        tflops,
                        bandwidth,
                    ) = func(shape[0], shape[1], shape[2], dtype=dtype)
                    result = {
                        "model": model_name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "transA": "T" if transA else "N",
                        "transB": "T" if transB else "N",
                        "dtype": str(dtype),
                        "Time(s)": avg_time_s,
                        "TFLOPS": tflops,
                        "Bandwidth(GB/s)": bandwidth,
                    }
                    perf_results.append(result)
    csv_filename = f"benchmark_moe_{model_name}.csv"
    csv_path = os.path.join(report_dir_path, csv_filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
        writer.writeheader()
        for result in perf_results:
            writer.writerow(result)

    json_filename = f"benchmark_moe_{model_name}.json"
    json_path = os.path.join(report_dir_path, json_filename)
    with open(json_path, mode="w", encoding="utf-8") as jsonfile:
        json.dump(perf_results, jsonfile, indent=4)


def benchmark(model, model_config_path, report_dir_path):
    benchmark_dir = Path(report_dir_path)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config_list: list[dict] = json.load(f)

    for model_config in model_config_list:
        model_name = model_config["model"]
        if model.upper() != "ALL" and model != model_name:
            continue

        if model_name in DENSE_MODELS:
            benchmark_model_dense(report_dir_path, model_config)
        elif model_name in DEEPSEEK_MODELS:
            benchmark_model_deepseek(report_dir_path, model_config)
        else:
            assert False, f"model({model_name}) don't have "
        # benchmark moe
        # if model_name in DEEPSEEK_MODELS:
        #     benchmark_deepseek_moe(report_dir_path, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="If run all model, set --model=all. If only run specific model, set --model=xxx, for example --model=Llama2_7B.",
    )
    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--report-dir-path", type=str)
    # parser.add_argument(
    #     "--stage", type=int, choices=[0, 1], help="benchmark: 0, tune + benchmark: 1"
    # )
    args = parser.parse_args()

    benchmark(args.model, args.model_config_path, args.report_dir_path)
