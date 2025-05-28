import argparse
import csv
import json
import multiprocessing
import os
from pathlib import Path

import torch

from attn_profiler import ck_attention_profile, flash_attention_profile
from tqdm import tqdm

MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


def profile_case(test_shape_dict, model_count, model_index,device_id=-1,json_output=False):
    assert test_shape_dict["seqlen_q"] == test_shape_dict["seqlen_kv"]

    results = []
    backends = {
        "flash": flash_attention_profile,
        #"ck": ck_attention_profile,
    }
    total = len(backends) * len(MBS_LIST)
    current = 0
    for backend_name, profile_func in backends.items():
        for mbs in MBS_LIST:
            fwd_tflops, fwd_time, bwd_tflops, bwd_time = profile_func(
                batch_size=mbs,
                seq_len=test_shape_dict["seqlen_q"],
                num_head_q=test_shape_dict["num_head_q"],
                num_head_kv=test_shape_dict["num_head_kv"],
                head_dim_qk=test_shape_dict["head_dim_qk"],
                head_dim_v=test_shape_dict["head_dim_v"],
                causal=test_shape_dict["causal"],
                device_id = device_id,
            )
            if json_output:
                print(json.dumps({
                    'type':'progressing',
                    'device_id':device_id,
                    'total': total,
                    'model': test_shape_dict['model'],
                    'model_count': model_count,
                    'model_index':model_index,
                }))
            current += 1
            result = test_shape_dict.copy()
            result["batch_size"] = mbs
            result["fwd_tflops"] = fwd_tflops
            result["fwd_time"] = fwd_time
            result["bwd_tflops"] = bwd_tflops
            result["bwd_time"] = bwd_time
            result["backend"] = backend_name
            results.append(result)
    return results


def benchmark(attn_shapes_json_path, output_csv_path,device_id=-1,json_output=False):
    benchmark_dir = Path(output_csv_path)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if device_id > 0 and not json_output:
        print(f"benchmark for {device_id}")
    with open(attn_shapes_json_path, "r", encoding="utf-8") as f:
        shape_data_dict_list: list[dict] = json.load(f)

    benchmark_results = []
    model_count = len(shape_data_dict_list)
    model_index = 0
    if not json_output:
        for shape_data_dict in tqdm(shape_data_dict_list):
            if shape_data_dict["model"] in ["deepseek_v2_lite", "deepseek_v2/v3"]:
                continue
            results = profile_case(shape_data_dict,device_id=device_id,model_count=model_count, model_index=model_index,json_output=json_output)
            benchmark_results.extend(results)
            model_index += 1
    else:
        for shape_data_dict in shape_data_dict_list:
            if shape_data_dict["model"] in ["deepseek_v2_lite", "deepseek_v2/v3"]:
                continue
            print(json.dumps({
                    'type': 'shape_start',
                    'device_id': device_id,
                    'model': shape_data_dict['model'],
                    'model_count': model_count,
                    'model_index': model_index,
            }))
            results = profile_case(shape_data_dict, device_id=device_id, model_count=model_count,
                                   model_index=model_index,json_output=json_output)
            benchmark_results.extend(results)
            model_index += 1
    if json_output:
        print(json.dumps({
            'type': 'device_done',
            'device_id': device_id,
        }))
    fieldnames = [
        "model",
        "batch_size",
        "seqlen_q",
        "seqlen_kv",
        "num_head_q",
        "num_head_kv",
        "head_dim_qk",
        "head_dim_v",
        "causal",
        "fwd_tflops",
        "fwd_time",
        "bwd_tflops",
        "bwd_time",
        "backend",
    ]
    filename = "benchmark_attention_result.csv"
    if device_id >=0:
        filename = f"benchmark_attention_result_{device_id}.csv"
    output_file_path = os.path.join(output_csv_path,filename)
    with open(output_file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in benchmark_results:
            writer.writerow(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-json-path", type=str)
    parser.add_argument("--report-csv-path", type=str)
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
                target=benchmark,
                args=(args.shapes_json_path,args.report_csv_path,i,args.json_output),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        if args.json_output:
            print(json.dumps({
                'type':'done'
            }))
    else:
        benchmark(
            attn_shapes_json_path=args.shapes_json_path,
            output_csv_path=args.report_csv_path,
            json_output=args.json_output,
        )

