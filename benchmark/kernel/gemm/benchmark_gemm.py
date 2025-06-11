import argparse
import csv
import json
import math
import multiprocessing
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
    "Mistral_8x7B",
    "Mistral_8x22B",
]
DEEPSEEK_MODELS = ["Deepseek_V2_Lite", "Deepseek_V2", "Deepseek_V3"]
MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def profile_gemm(m, n, k, dtype, transA, transB, device):
    assert dtype in [torch.float16, torch.bfloat16]
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = math.ceil(CACHE_ROTATING_BUFFER_BYTES / mem_size_bytes) + 1
    num_run = 100

    a_shape = (k, m) if transA else (m, k)
    b_shape = (n, k) if transB else (k, n)
    a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.randn((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    for i in range(num_rotations):
        a = maybe_transpose(a_list[i], transA)
        b = maybe_transpose(b_list[i], transB)
        c_list[i] = torch.matmul(a, b)
    torch.cuda.synchronize()

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

    avg_time_s = start_event.elapsed_time(end_event) / 1000 / (num_rotations * num_run)
    tflop = 2 * m * n * k / 1e12
    tflops = tflop / avg_time_s
    bandwidth = mem_size_bytes / 1e9 / avg_time_s
    return (m, n, k, transA, transB, dtype, avg_time_s, tflop, tflops, bandwidth)


def profile_gemm_fwd(m, n, k, dtype, device): return profile_gemm(m, n, k, dtype, False, True, device)
def profile_gemm_wgrad(m, n, k, dtype, device): return profile_gemm(n, k, m, dtype, True, False, device)
def profile_gemm_dgrad(m, n, k, dtype, device): return profile_gemm(m, k, n, dtype, False, False, device)


def generate_gemm_shapes_dense(config):
    seq = config["seqlen"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab_size = config["vocab_size"]
    return [
        [seq, int((num_attention_heads + 2 * num_key_value_heads) * head_dim), hidden_size],
        [seq, hidden_size, hidden_size],
        [seq, int(2 * intermediate_size), hidden_size],
        [seq, hidden_size, intermediate_size],
        [seq, vocab_size, hidden_size],
    ]


def generate_gemm_shapes_deepseek(config):
    seq = config["seqlen"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    kv_lora_rank = config["kv_lora_rank"]
    moe_intermediate_size = config["moe_intermediate_size"]
    num_attention_heads = config["num_attention_heads"]
    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    num_experts_per_tok = config["num_experts_per_tok"]
    q_lora_rank = config["q_lora_rank"]
    qk_nope_head_dim = config["qk_nope_head_dim"]
    qk_rope_head_dim = config["qk_rope_head_dim"]
    v_head_dim = config["v_head_dim"]
    vocab_size = config["vocab_size"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    shapes = []
    if q_lora_rank is None:
        shapes.append([seq, int(num_attention_heads * q_head_dim), hidden_size])
    else:
        shapes.append([seq, q_lora_rank, hidden_size])
        shapes.append([seq, int(num_attention_heads * q_head_dim), q_lora_rank])

    shapes.append([seq, kv_lora_rank + qk_rope_head_dim, hidden_size])
    shapes.append([seq, int(num_attention_heads * (qk_nope_head_dim + v_head_dim)), kv_lora_rank])
    shapes.append([seq, hidden_size, int(v_head_dim * num_attention_heads)])
    shapes.append([seq, n_routed_experts, hidden_size])

    if n_shared_experts > 0:
        shapes.append([seq, intermediate_size * 2, hidden_size])
        shapes.append([seq, hidden_size, intermediate_size])

    balance_seq = int(seq * num_experts_per_tok // n_routed_experts)
    shapes.append([balance_seq, moe_intermediate_size * 2, hidden_size])
    shapes.append([balance_seq, hidden_size, moe_intermediate_size])
    shapes.append([seq, vocab_size, hidden_size])
    return shapes


def benchmark_model(report_dir_path, model_config, device, json_output, model_idx, total_models):
    model_name = model_config["model"]
    if model_name in DENSE_MODELS:
        gemm_shape_list = generate_gemm_shapes_dense(model_config)
    elif model_name in DEEPSEEK_MODELS:
        gemm_shape_list = generate_gemm_shapes_deepseek(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    perf_results = []
    total = len(MBS_LIST) * len(gemm_shape_list) * 3
    count = 0

    for dtype in [torch.bfloat16]:
        for mbs in MBS_LIST:
            for shape in gemm_shape_list:
                for func in [profile_gemm_fwd, profile_gemm_wgrad, profile_gemm_dgrad]:
                    (
                        m, n, k, transA, transB, dtype, avg_time_s, tflop, tflops, bandwidth
                    ) = func(mbs * shape[0], shape[1], shape[2], dtype=dtype, device=device)
                    result = {
                        "model": model_name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "transA": "T" if transA else "N",
                        "transB": "T" if transB else "N",
                        "dtype": dtype,
                        "Time(s)": avg_time_s,
                        "TFLOPS": tflops,
                        "Bandwidth(GB/s)": bandwidth,
                    }
                    if json_output:
                        print(json.dumps({
                            "type": "gemm_progress",
                            "model_index": model_idx,
                            "model_count": total_models,
                            "model_name": model_name,
                            "current": count,
                            "total": total,
                            "device": device,
                        }))
                    count += 1
                    perf_results.append(result)

    filename = f"benchmark_gemm_{model_name}_{device}.csv"
    csv_path = os.path.join(report_dir_path, filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
        writer.writeheader()
        for result in perf_results:
            writer.writerow(result)


def benchmark(model, model_config_path, report_dir_path, gpu_idx=0, json_output=False):
    benchmark_dir = Path(report_dir_path)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_idx}"
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config_list = json.load(f)

    total_models = len(model_config_list)
    if json_output:
        for model_idx, model_config in enumerate(model_config_list):
            model_name = model_config["model"]
            if model.upper() != "ALL" and model != model_name:
                continue

            print(json.dumps({
                "type": "model_start",
                "model_index": model_idx,
                "model_count": total_models,
                "model_name": model_name,
                "device": device,
            }))

            benchmark_model(report_dir_path, model_config, device, json_output, model_idx, total_models)
    else:
        model_idx = 0
        for model_config in tqdm(model_config_list):
            model_name = model_config["model"]
            if model.upper() != "ALL" and model != model_name:
                continue
            print(f"{model_name}")
            benchmark_model(report_dir_path, model_config, device, json_output, model_idx, total_models)
            model_idx += 1

def run_on_gpu(gpu_idx, model, model_config_path, report_dir_path, json_output):
    benchmark(model, model_config_path, report_dir_path, gpu_idx=gpu_idx, json_output=json_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--multi-gpu-eval", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    if args.multi_gpu_eval:
        num_gpus = torch.cuda.device_count()
        processes = []
        for i in range(num_gpus):
            if 0 < args.gpus < i:
                continue
            p = multiprocessing.Process(
                target=run_on_gpu,
                args=(i, args.model, args.model_config_path, args.report_dir_path, args.json_output),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        benchmark(args.model, args.model_config_path, args.report_dir_path, json_output=args.json_output)
