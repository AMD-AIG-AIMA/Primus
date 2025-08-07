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

from examples.scripts.utils import log_error_and_exit, log_info
from primus.core.launcher.parser import PrimusParser


def log_rank0(msg: str, level: str = "INFO") -> None:
    """Print log only on NODE_RANK=0 for cleaner multi-node output."""
    if int(os.environ.get("NODE_RANK", "0")) == 0:
        print(f"[NODE-0({socket.gethostname()})] [{level}] {msg}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Primus Backend Preparation Entry")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config file")
    parser.add_argument("--data_path", required=True, help="Root directory for datasets and tokenizer")
    parser.add_argument(
        "--patch-args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    parser.add_argument(
        "--backend-path",
        type=str,
        default=None,
        help="Optional path to backend (e.g., Megatron), will be added to PYTHONPATH",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")

    args = parser.parse_args()

    primus_path = Path.cwd()
    patch_args_path = Path(args.patch_args).resolve()
    patch_args_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Always create an empty patch_args file first ---
    # This ensures run_pretrain.sh can safely source it even if the backend step fails.
    patch_args_path.write_text("# Auto-generated patch args\n")

    # --- Parse experiment config to identify backend framework ---
    config = PrimusParser().parse(args)
    framework = config.get_module_config("pre_trainer").framework

    # --- Map aliases to canonical framework folder names ---
    framework_map = {
        "megatron": "megatron",
        "light-megatron": "megatron",
        "torchtitan": "torchtitan",
    }
    framework_dir = framework_map.get(framework, framework)

    # --- Determine backend prepare script path ---
    script = primus_path / "examples" / framework_dir / "prepare.py"
    if not script.exists():
        log_error_and_exit(f"Backend prepare script not found: {script}")

    log_info(f"Running backend prepare: {script}")

    # --- Construct command for subprocess ---
    cmd = [
        sys.executable,
        str(script),
        "--exp",
        args.exp,
        "--data_path",
        args.data_path,
        "--primus_path",
        str(primus_path),
        "--patch_args",
        str(patch_args_path),
    ]
    if args.backend_path:
        cmd += ["--backend_path", args.backend_path]

    if args.dry_run:
        # Print the command without executing (useful for debugging)
        log_rank0(f"[DRY RUN] {' '.join(cmd)}")
        return

    # --- Execute backend prepare.py ---
    try:
        subprocess.run(cmd, check=True)
        log_rank0(f"Backend prepare completed. Patch args written to {patch_args_path}")
    except subprocess.CalledProcessError as e:
        # Fail fast if backend script returns non-zero exit code
        log_error_and_exit(f"Backend script({script}) failed with exit code {e.returncode}")

        # try:
        #     subprocess.run(
        #         [
        #             "python",
        #             str(script),
        #             "--config",
        #             args.config,
        #             "--data_path",
        #             args.data_path,
        #             "--primus_path",
        #             primus_path,
        #             "--patch_args",
        #             str(patch_args_path),
        #         ],
        #         check=True,
        #         text=True,
        #         stdout=sys.stdout,
        #         stderr=sys.stderr,
        #     )
        # except subprocess.CalledProcessError as e:
        log_error_and_exit(f"Backend script({script}) failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
