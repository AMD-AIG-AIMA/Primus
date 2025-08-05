###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

from requests.exceptions import HTTPError

from examples.scripts.utils import (
    get_env_case_insensitive,
    get_node_rank,
    log_error_and_exit,
    log_info,
    write_patch_args,
)
from primus.core.launcher.parser import PrimusParser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Primus environment (Torchtitan/DeepSeek)")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write patch_args file (used during training)",
    )
    return parser.parse_args()


def pip_install_editable(path: Path, name: str):
    """Install a package in editable mode via pip."""
    log_info(f"Installing {name} in editable mode (path: {path})")
    ret = subprocess.run(["pip", "install", "-e", ".", "-q"], cwd=path)
    if ret.returncode != 0:
        log_error_and_exit(f"Failed to install {name}.")


def resolve_backend_path(env_var: str, default_subdir: str, primus_path: Path, name: str) -> Path:
    """Resolve backend path from environment variable or fallback to default path."""
    env_value = get_env_case_insensitive(env_var)
    if env_value:
        path = Path(env_value).resolve()
        log_info(f"{env_var.upper()} found in environment: {path}")
    else:
        path = primus_path / default_subdir
        log_info(f"{env_var.upper()} not set, falling back to: {path}")
    return path


# def prepare_tokenizer(tokenizer_path: str, full_path: Path, hf_token: str) -> Path:
#     """
#     Unified tokenizer download logic:
#       1. DeepSeek models → use AutoTokenizer to cache files
#       2. Other HF models → download tokenizer.model via huggingface_hub
#     """
#     from huggingface_hub import hf_hub_download
#     from transformers import AutoTokenizer

#     full_path.mkdir(parents=True, exist_ok=True)

#     # Case 1: DeepSeek models
#     if tokenizer_path.startswith("deepseek-ai/"):
#         log_info(f"Detected DeepSeek model: {tokenizer_path}")
#         log_info(f"Downloading via AutoTokenizer to cache_dir: {full_path}")

#         # AutoTokenizer automatically downloads model files to cache_dir
#         AutoTokenizer.from_pretrained(
#             tokenizer_path,
#             cache_dir=str(full_path),
#             trust_remote_code=True,
#             token=hf_token,
#         )

#         # Locate tokenizer file
#         tokenizer_files = list(full_path.rglob("tokenizer.model")) or list(full_path.rglob("tokenizer.json"))
#         if not tokenizer_files:
#             log_error_and_exit(f"No tokenizer file found for DeepSeek model: {tokenizer_path}")
#         tokenizer_file = tokenizer_files[0]
#         log_info(f"Tokenizer ready: {tokenizer_file}")
#         return tokenizer_file

#     # Case 2: Non-DeepSeek models
#     log_info(f"Using hf_hub_download to download tokenizer: {tokenizer_path}")
#     original_dir = full_path / "original"
#     original_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         hf_hub_download(
#             repo_id=tokenizer_path,
#             filename="tokenizer.model",
#             local_dir=str(original_dir),
#             local_dir_use_symlinks=False,
#             token=hf_token,
#         )
#     except HTTPError as e:
#         if e.response.status_code == 401:
#             log_error_and_exit("Download failed: HF_TOKEN required for private checkpoints.")
#         else:
#             raise e

#     tokenizer_file = original_dir / "tokenizer.model"
#     log_info(f"Tokenizer ready: {tokenizer_file}")
#     return tokenizer_file

def prepare_tokenizer(tokenizer_path: str, full_path: Path, hf_token: str) -> Path:
    """
    Unified tokenizer download logic:
      1. DeepSeek models → AutoTokenizer
      2. Other HF models → huggingface_hub
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    full_path.mkdir(parents=True, exist_ok=True)

    # DeepSeek
    if tokenizer_path.startswith("deepseek-ai/"):
        log_info(f"Detected DeepSeek model: {tokenizer_path}")
        log_info(f"Downloading via AutoTokenizer to cache_dir: {full_path}")

        # Only rank0 downloads
        if get_node_rank() == 0:
            AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=str(full_path),
                trust_remote_code=True,
                token=hf_token,
            )

        # All ranks wait for tokenizer file
        tokenizer_file = None
        while tokenizer_file is None:
            tokenizer_files = list(full_path.rglob("tokenizer.model")) or list(full_path.rglob("tokenizer.json"))
            if tokenizer_files:
                tokenizer_file = tokenizer_files[0]
            else:
                time.sleep(5)

        log_info(f"Tokenizer ready: {tokenizer_file}")
        return tokenizer_file

    # Non-DeepSeek
    log_info(f"Using hf_hub_download to download tokenizer: {tokenizer_path}")
    original_dir = full_path / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    if get_node_rank() == 0:
        try:
            hf_hub_download(
                repo_id=tokenizer_path,
                filename="tokenizer.model",
                local_dir=str(original_dir),
                local_dir_use_symlinks=False,
                token=hf_token,
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                log_error_and_exit("Download failed: HF_TOKEN required for private checkpoints.")
            else:
                raise e

    tokenizer_file = original_dir / "tokenizer.model"
    while not tokenizer_file.exists():
        time.sleep(5)

    log_info(f"Tokenizer ready: {tokenizer_file}")
    return tokenizer_file


def main():
    args = parse_args()

    primus_path = Path(args.primus_path).resolve()
    data_path = Path(args.data_path).resolve()
    exp_path = Path(args.exp).resolve()
    patch_args_file = Path(args.patch_args).resolve()

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH  : {data_path}")
    log_info(f"EXP        : {exp_path}")
    log_info(f"PATCH-ARGS : {patch_args_file}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    # Parse Primus config
    primus_cfg = PrimusParser().parse(args)

    # Step 1: Install TorchTitan in editable mode
    torchtitan_path = resolve_backend_path(
        "TORCHTITAN_PATH", "third_party/torchtitan", primus_path, "torchtitan"
    )
    pip_install_editable(torchtitan_path, "TorchTitan")

    # Step 2: Retrieve pre_trainer module config
    try:
        pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not hasattr(pre_trainer_cfg, "model") or pre_trainer_cfg.model is None:
        log_error_and_exit("Missing field: pre_trainer.model")

    if not hasattr(pre_trainer_cfg.model, "tokenizer_path") or not pre_trainer_cfg.model.tokenizer_path:
        log_error_and_exit("Missing field: pre_trainer.model.tokenizer_path")

    tokenizer_path = pre_trainer_cfg.model.tokenizer_path
    # Replace '/' with '_' to avoid nested directories
    full_path = data_path / "torchtitan" / tokenizer_path.replace("/", "_")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log_error_and_exit("HF_TOKEN is not set. Please export HF_TOKEN=your_token")

    tokenizer_file = full_path / "tokenizer.model"

    tokenizer_file = prepare_tokenizer(tokenizer_path, full_path, hf_token)
    # Step 3: Distributed download logic
    # if get_node_rank() == 0:
    #     tokenizer_file = prepare_tokenizer(tokenizer_path, full_path, hf_token)
    # else:
    #     log_info(f"Rank {get_node_rank()} waiting for tokenizer file ...")
    #     while not tokenizer_file.exists():
    #         time.sleep(5)

    # Step 4: Write patch_args
    log_info(f"Tokenizer prepared: {tokenizer_file}")
    write_patch_args(patch_args_file, "train_args", {"model.tokenizer_path": str(tokenizer_file)})
    write_patch_args(patch_args_file, "torchrun_args", {"local-ranks-filter": "0"})


if __name__ == "__main__":
    log_info("========== Prepare Torchtitan / DeepSeek dataset ==========")
    main()
