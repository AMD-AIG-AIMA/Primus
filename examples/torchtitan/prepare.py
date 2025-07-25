###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import subprocess
import time
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


def hf_download(repo_id: str, tokenizer_path: str, local_dir: str, hf_token: Optional[str] = None) -> None:

    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{tokenizer_path}/tokenizer.model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            log_error_and_exit("You need to pass a valid `HF_TOKEN` to download private checkpoints.")
        else:
            raise e


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    return parser.parse_args()


def pip_install_editable(path: Path, name: str):
    log_info(f"Installing {name} in editable mode via pip (path: {path})")
    ret = subprocess.run(["pip", "install", "-e", ".", "-q"], cwd=path)
    if ret.returncode != 0:
        log_error_and_exit(f"Failed to install {name} via pip.")


def resolve_backend_path(env_var: str, default_subdir: str, primus_path: Path, name: str) -> Path:
    env_value = get_env_case_insensitive(env_var)
    if env_value:
        path = Path(env_value).resolve()
        log_info(f"{env_var.upper()} found in environment: {path}")
    else:
        path = primus_path / default_subdir
        log_info(f"{env_var.upper()} not found, falling back to: {path}")
    return path


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

    primus_cfg = PrimusParser().parse(args)

    torchtitan_path = resolve_backend_path(
        "TORCHTITAN_PATH", "third_party/torchtitan", primus_path, "torchtitan"
    )
    pip_install_editable(torchtitan_path, "TorchTitan")

    try:
        pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not hasattr(pre_trainer_cfg, "model") or pre_trainer_cfg.model is None:
        log_error_and_exit("Missing required field: pre_trainer.model")

    if not hasattr(pre_trainer_cfg.model, "tokenizer_path") or not pre_trainer_cfg.model.tokenizer_path:
        log_error_and_exit("Missing required field: pre_trainer.model.tokenizer_path")

    tokenizer_path = pre_trainer_cfg.model.tokenizer_path

    full_path = data_path / "torchtitan" / tokenizer_path.lstrip("/")
    tokenizer_file = full_path / "original/tokenizer.model"

    if not tokenizer_file.is_file():
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            log_error_and_exit("HF_TOKEN not set. Please export HF_TOKEN.")

        if get_node_rank() == 0:
            log_info(f"Downloading tokenizer to {full_path} ...")
            (full_path / "original").mkdir(parents=True, exist_ok=True)
            hf_download(
                repo_id=tokenizer_path, tokenizer_path="original", local_dir=str(full_path), hf_token=hf_token
            )
        else:
            log_info(f"Rank {get_node_rank()} waiting for tokenizer file ...")
            while not tokenizer_file.exists():
                time.sleep(5)
    else:
        log_info(f"Tokenizer file exists: {tokenizer_file}")
    write_patch_args(patch_args_file, "train_args", {"model.tokenizer_path": str(tokenizer_file)})
    write_patch_args(patch_args_file, "torchrun_args", {"local-ranks-filter": "0"})


if __name__ == "__main__":
    log_info("========== Prepare Torchtitan dataset ==========")
    main()
