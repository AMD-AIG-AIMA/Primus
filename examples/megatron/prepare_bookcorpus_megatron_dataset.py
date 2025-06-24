###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from pathlib import Path

import nltk
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=False, default="tmp/data", help="Path to output JSON")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    nltk.download("punkt_tab")

    dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    dataset.to_json(out_dir / "bookcorpus_megatron.json")
