import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from requests.exceptions import HTTPError


def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK", "0"))


def log_info(msg):
    if get_node_rank() == 0:
        print(f"[INFO] {msg}", file=sys.stderr)


def log_error_and_exit(msg):
    if get_node_rank() == 0:
        print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


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
    return parser.parse_args()


def main():
    args = parse_args()

    primus_path = Path(args.primus_path).resolve()
    data_path = Path(args.data_path).resolve()
    exp_path = Path(args.exp).resolve()
    # hf_home = data_path / "huggingface"

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH: {data_path}")
    log_info(f"EXP: {exp_path}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    with open(exp_path, "r") as f:
        cfg = yaml.safe_load(f)
    try:
        model_file = cfg["modules"]["pre_trainer"]["model"]
    except KeyError:
        log_error_and_exit(f"Invalid EXP file: missing model info: {exp_path}")

    model_config_path = primus_path / "primus/configs/models/torchtitan" / model_file
    if not model_config_path.is_file():
        log_error_and_exit(f"Model config not found: {model_config_path}")

    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    try:
        tokenizer_path = model_cfg["model"]["tokenizer_path"]
    except KeyError:
        log_error_and_exit(f"Missing tokenizer_path in model config: {model_config_path}")

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

    # Final output: key=value
    print(f'TOKENIZER_PATH="{tokenizer_file}"')
    print(f'LOCAL_RANKS_FILTER="0"')


if __name__ == "__main__":
    main()
