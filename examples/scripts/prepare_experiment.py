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
from pathlib import Path

import yaml

# from examples.scripts.utils import log_error_and_exit, log_info
# from .utils import log_error_and_exit, log_info
from primus.core.launcher.parser import PrimusParser

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


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


def format_cli_args(args: dict) -> str:
    """Format a dictionary into CLI-style arguments: --key val --key2 val2 ..."""
    parts = []
    for k, v in args.items():
        parts.extend([f"--{k}", str(v)])
    return " ".join(parts)


def parse_cli_args_string(arg_string: str) -> dict:
    """Parse a CLI-style string into a dict: --key val --key2 val2 → {"key": "val", ...}"""
    parts = arg_string.strip().split()
    result = {}
    i = 0
    while i < len(parts):
        if parts[i].startswith("--") and i + 1 < len(parts):
            key = parts[i][2:]
            val = parts[i + 1]
            result[key] = val
            i += 2
        else:
            i += 1
    return result


def write_patch_args(path: Path, section: str, args_dict: dict):
    """Write or merge args_dict into the given section in YAML patch file"""
    if path.exists():
        with open(path, "r") as f:
            patch = yaml.safe_load(f) or {}
    else:
        patch = {}

    existing_section = patch.get(section, {})

    if isinstance(existing_section, str):
        existing_args = parse_cli_args_string(existing_section)
    elif isinstance(existing_section, dict):
        existing_args = existing_section
    else:
        existing_args = {}

    # Merge the new args into existing
    existing_args.update(args_dict)

    # Save the merged args
    patch[section] = format_cli_args(existing_args)

    with open(path, "w") as f:
        yaml.safe_dump(patch, f)


def get_env_case_insensitive(var_name: str) -> str | None:
    """Get environment variable by name, ignoring case."""
    for key, value in os.environ.items():
        if key.lower() == var_name.lower():
            return value
    return None


def log(msg, level="INFO"):
    if int(os.environ.get("NODE_RANK", "0")) == 0:
        print(f"[NODE-0({socket.gethostname()})] [{level}] {msg}", file=sys.stderr)
        if level == "ERROR":
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Primus Backend Preparation Entry")
    parser.add_argument("--exp", required=True, help="Path to experiment YAML config file")
    parser.add_argument("--data_path", required=True, help="Root directory for datasets and tokenizer")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    args = parser.parse_args()

    primus_path = Path.cwd()
    patch_args_path = Path(args.patch_args).resolve()
    patch_args_path.parent.mkdir(parents=True, exist_ok=True)

    config = PrimusParser().parse(args)
    framework = config.get_module_config("pre_trainer").framework
    script = primus_path / "examples" / framework / "prepare.py"

    if not script.exists():
        log_info(f"Backend prepare script not found: {script}")

    log_info(f"Running backend prepare: {script}")
    try:
        subprocess.run(
            [
                "python",
                str(script),
                "--exp",
                args.exp,
                "--data_path",
                args.data_path,
                "--primus_path",
                primus_path,
                "--patch_args",
                str(patch_args_path),
            ],
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        log_error_and_exit(f"Backend script({script}) failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
