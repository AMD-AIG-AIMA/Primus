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

from primus.core.launcher.parser import PrimusParser


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
        log(f"Backend prepare script not found: {script}", level="ERROR")

    log(f"Running backend prepare: {script}")
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
        log(f"Backend script({script}) failed with exit code {e.returncode}", level="ERROR")


if __name__ == "__main__":
    main()
