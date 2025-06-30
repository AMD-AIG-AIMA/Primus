###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from time import sleep

import nltk
from datasets import load_dataset

from primus.core.launcher.parser import PrimusParser


# ---------- Logging ----------
def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK", "0"))


def get_hostname():
    return socket.gethostname()


def log_info(msg):
    if get_node_rank() == 0:
        print(f"[NODE-{get_node_rank()}({get_hostname()})] [INFO] {msg}", file=sys.stderr)


def log_error_and_exit(msg):
    if get_node_rank() == 0:
        print(f"[NODE-{get_node_rank()}({get_hostname()})] [ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------- Helpers ----------
def check_dir_nonempty(path: Path, name: str):
    if not path.is_dir() or not any(path.iterdir()):
        log_error_and_exit(
            f"{name} ({path}) does not exist or is empty.\n"
            "Please ensure Primus is properly initialized.\n"
            "If not yet cloned, run:\n"
            "    git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git\n"
            "Or if already cloned, initialize submodules with:\n"
            "    git submodule update --init --recursive"
        )


def load_yaml_field(file_path: Path, field: str):
    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith(f"{field}:"):
                return line.split(":", 1)[1].strip().strip('"')
    return None


def prepare_dataset(
    primus_path: Path, data_path: Path, tokenizer_type: str, tokenizer_model: str, tokenized_data_path: Path
):
    dataset = "bookcorpus"
    dataset_path = data_path / dataset
    output_path = dataset_path / tokenizer_type
    hf_home = Path(os.environ.get("HF_HOME", data_path / "huggingface"))
    os.environ["HF_HOME"] = str(hf_home)

    tokenized_bin = tokenized_data_path.with_suffix(".bin")
    tokenized_idx = tokenized_data_path.with_suffix(".idx")

    if tokenized_bin.exists() and tokenized_idx.exists():
        log_info(f"Tokenized files {tokenized_bin} and {tokenized_idx} exist, skipping preprocessing.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_json = dataset_path / "bookcorpus_megatron.json"

    if dataset_json.exists():
        log_info(f"Found dataset file: {dataset_json}, skipping download.")
    else:
        log_info(f"Downloading and saving BookCorpus dataset to {dataset_json} ...")
        nltk.download("punkt")
        dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
        dataset.to_json(str(dataset_json))
        log_info("Download and save completed.")

    log_info(f"Preprocessing dataset with tokenizer {tokenizer_type} / {tokenizer_model}")
    start = time.time()
    subprocess.run(
        [
            "python3",
            str(primus_path / "examples/megatron/preprocess_data.py"),
            "--input",
            str(dataset_json),
            "--tokenizer-type",
            tokenizer_type,
            "--tokenizer-model",
            tokenizer_model,
            "--output-prefix",
            str(output_path / "bookcorpus"),
            "--workers",
            str(os.cpu_count()),
            "--split-sentences",
            "--partitions",
            "2",
        ],
        check=True,
    )
    log_info(f"Preprocessing completed in {int(time.time() - start)} s")


def prepare_dataset_if_needed(primus_config, primus_path: Path, data_path: Path):
    tokenizer_type = primus_config.get_module_config("pre_trainer").tokenizer_type
    tokenized_data_path = Path(
        os.environ.get(
            "TOKENIZED_DATA_PATH", data_path / f"bookcorpus/{tokenizer_type}/bookcorpus_text_sentence"
        )
    )

    done_flag = tokenized_data_path.with_suffix(".done")
    node_rank = get_node_rank()

    if node_rank == 0:
        if not done_flag.exists():
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                log_error_and_exit("Environment variable HF_TOKEN must be set.")

            tokenizer_type = primus_config.get_module_config("pre_trainer").tokenizer_type
            tokenizer_model = primus_config.get_module_config("pre_trainer").tokenizer_model

            log_info(f"TOKENIZER_DATA_PATH is {tokenized_data_path}")

            prepare_dataset(
                primus_path=primus_path,
                data_path=data_path,
                tokenizer_type=tokenizer_type,
                tokenizer_model=tokenizer_model,
                tokenized_data_path=tokenized_data_path,
            )
            done_flag.touch()
            log_info("Dataset preparation completed.")
    else:
        while not done_flag.exists():
            log_info("Waiting for dataset...")
            sleep(30)


def get_env_case_insensitive(var_name: str) -> str | None:
    """Get environment variable by name, ignoring case."""
    for key, value in os.environ.items():
        if key.lower() == var_name.lower():
            return value
    return None


def build_megatron_helper(primus_path: Path):
    """Build Megatron's helper C++ dataset library."""
    megatron_env = get_env_case_insensitive("MEGATRON_PATH")
    if megatron_env:
        megatron_path = Path(megatron_env).resolve()
        log_info(f"MEGATRON_PATH found in environment: {megatron_path}")
    else:
        megatron_path = primus_path / "third_party/Megatron-LM"
        log_info(f"MEGATRON_PATH not found, falling back to: {megatron_path}")

    # pip install -e .
    log_info(f"Installing Megatron in editable mode via pip (path: {megatron_path})")
    ret = subprocess.run(["pip", "install", "-e", ".", "-q"], cwd=megatron_path)
    if ret.returncode != 0:
        log_error_and_exit("Failed to install Megatron via pip.")

    # build C++ helper
    dataset_cpp_dir = megatron_path / "megatron/core/datasets"
    log_info(f"Building Megatron dataset helper in {dataset_cpp_dir}")

    ret = subprocess.run(["make"], cwd=dataset_cpp_dir)
    if ret.returncode != 0:
        log_error_and_exit("Building Megatron C++ helper failed.")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Prepare Primus environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    primus_config = PrimusParser().parse(args)

    primus_path = Path(args.primus_path).resolve()
    log_info(f"PRIMUS_PATH is set to: {primus_path}")

    data_path = Path(args.data_path).resolve()
    log_info(f"DATA_PATH is set to: {data_path}")

    exp_path = Path(args.exp).resolve()
    if not exp_path.is_file():
        log_error_and_exit(f"The specified EXP file does not exist: {exp_path}")
    log_info(f"EXP is set to: {exp_path}")

    mock_data = primus_config.get_module_config("pre_trainer").mock_data
    if mock_data:
        log_info(f"'mock_data: true' is set in {exp_path}, skipping dataset preparation.")
        # os.environ.pop("TOKENIZED_DATA_PATH", None)
    else:
        prepare_dataset_if_needed(primus_config=primus_config, primus_path=primus_path, data_path=data_path)

    build_megatron_helper(primus_path=primus_path)


if __name__ == "__main__":
    log_info("========== Prepare Megatron dataset ==========")
    main()
