###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import json

from attn_profiler import flash_attention_profile
from tqdm import tqdm

MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]


def profile_case(test_shape_dict):
    assert test_shape_dict["seqlen_q"] == test_shape_dict["seqlen_kv"]

    results = []
    backends = {
        "flash": flash_attention_profile,
        # "ck": ck_attention_profile,
    }
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
            )
            result = test_shape_dict.copy()
            result["batch_size"] = mbs
            result["fwd_tflops"] = fwd_tflops
            result["fwd_time"] = fwd_time
            result["bwd_tflops"] = bwd_tflops
            result["bwd_time"] = bwd_time
            result["backend"] = backend_name
            results.append(result)
    return results


def benchmark(attn_shapes_json_path, output_csv_path):
    with open(attn_shapes_json_path, "r", encoding="utf-8") as f:
        shape_data_dict_list: list[dict] = json.load(f)

    benchmark_results = []
    for shape_data_dict in tqdm(shape_data_dict_list):
        if shape_data_dict["model"] in ["deepseek_v2_lite", "deepseek_v2/v3"]:
            continue
        results = profile_case(shape_data_dict)
        benchmark_results.extend(results)

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

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in benchmark_results:
            writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-json-path", type=str)
    parser.add_argument("--report-csv-path", type=str)
    args = parser.parse_args()

    benchmark(
        attn_shapes_json_path=args.shapes_json_path,
        output_csv_path=args.report_csv_path,
    )
