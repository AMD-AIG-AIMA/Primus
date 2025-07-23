###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import glob
import json
import os


def merge_traces_in_folder(path, output_file):
    merged_trace = {"traceEvents": [], "displayTimeUnit": "ns"}

    json_files = sorted(glob.glob(os.path.join(path, "*.json")))

    if not json_files:
        print("No JSON files found in the folder.", flush=True)
        return

    for filepath in json_files:
        with open(filepath, "r") as f:
            trace = json.load(f)
            merged_trace["traceEvents"].extend(trace.get("traceEvents", []))

    with open(output_file, "w") as f:
        json.dump(merged_trace, f)

    print(f"Merged trace saved to: {output_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-trace-path", type=str)
    parser.add_argument("--merged-trace-file", type=str)
    args = parser.parse_args()

    merge_traces_in_folder(args.profile_trace_path, args.merged_trace_file)
