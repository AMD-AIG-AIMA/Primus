#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import argparse
import csv
import re
from pathlib import Path


def parse_file_name(file_path: Path):
    match = re.search(
        r"nodes(\d+)_dp(\d+)_tp(\d+)_pp(\d+)_ep(\d+)_mbs(\d+)_gbs(\d+)_seqlen(\d+)_iters(\d+)\.log",
        file_path.name,
    )
    if match:
        return {
            "NODES": int(match.group(1)),
            "DP": int(match.group(2)),
            "TP": int(match.group(3)),
            "PP": int(match.group(4)),
            "EP": int(match.group(5)),
            "MBS": int(match.group(6)),
            "GBS": int(match.group(7)),
            "SEQ_LENGTH": int(match.group(8)),
            "ITERS": int(match.group(9)),
        }
    return None


def find_match(file_path, search_pattern):
    with open(file_path, "r") as file:
        content = file.read()
    matches = re.findall(search_pattern, content)
    match = matches[-1]
    return match


def process_one(model, input_file):
    data_dict = {
        "model": model,
        "NODES": None,
        "DP": None,
        "TP": None,
        "PP": None,
        "EP": None,
        "MBS": None,
        "GBS": None,
        "SEQ_LENGTH": None,
        "ITERS": None,
        "TFLOP/s/GPU": 0,
        "Step Time(s)": 0,
        "Memory Usage(%)": 0,
    }
    params = parse_file_name(Path(input_file))
    data_dict.update(params)

    data_dict["TFLOP/s/GPU"] = float(find_match(input_file, r"TFLOP/s/GPU\):\s*([\d.]+)/([\d.]+)")[-1])
    data_dict["Step Time(s)"] = "{:.2f}".format(
        float(find_match(input_file, r"elapsed time per iteration \(ms\): ([\d.]+)/([\d.]+)")[-1]) / 1000
    )
    data_dict["Memory Usage(%)"] = "{:.2f}%".format(
        float(find_match(input_file, r"mem usages:\s*([\d.]+)")) * 100
    )
    return data_dict


def process_all(model, input_folder):
    data_list = []
    input_folder_path = Path(input_folder)

    for log_file in input_folder_path.glob("*.log"):
        data_dict = process_one(model, log_file)
        data_list.append(data_dict)
    return data_list


def main(model, benchmark_log_dir, report_csv_path):
    all_data = process_all(model, benchmark_log_dir)

    fieldnames = [
        "model",
        "NODES",
        "DP",
        "TP",
        "PP",
        "EP",
        "MBS",
        "GBS",
        "SEQ_LENGTH",
        "ITERS",
        "TFLOP/s/GPU",
        "Step Time(s)",
        "Memory Usage(%)",
    ]

    with open(report_csv_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--benchmark-log-dir", type=str)
    parser.add_argument("--report-csv-path", type=str)
    args = parser.parse_args()

    main(args.model, args.benchmark_log_dir, args.report_csv_path)
