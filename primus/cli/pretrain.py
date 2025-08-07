###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import sys
from pathlib import Path

from primus.core.launcher.parser import load_primus_config
from primus.core.utils.yaml_utils import (
    dump_namespace_to_yaml,
    nested_namespace_to_dict,
)


def register_subcommand(subparsers):
    """
    Register the 'train' subcommand to the main CLI parser.

    Example:
        primus-cli train --config exp.yaml --backend-path /path/to/megatron

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """
    parser = subparsers.add_parser("pretrain", help="Launch Primus pretrain with Megatron or TorchTitan")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config file.")
    parser.add_argument(
        "--backend-path",
        nargs="?",
        default=None,
        help=(
            "Optional backend import path for Megatron or TorchTitan. "
            "If provided, it will be appended to PYTHONPATH dynamically."
        ),
    )
    parser.add_argument(
        "--export-config", default=None, help="Optional path to export the merged Primus config as YAML."
    )
    return parser


def setup_backend_path(framework: str, backend_path=None, verbose: bool = True):
    """
    Setup Python path for backend modules.

    Priority order:
    1. --backend-path from CLI
    2. BACKEND_PATH from environment
    3. Source tree fallback: <primus>/../../third_party/{framework}

    Returns:
        str: The first valid backend path inserted into sys.path.
    """
    candidate_paths = []

    # 1) From CLI
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        candidate_paths.extend(backend_path)

    # 2) From environment variable
    env_path = os.getenv("BACKEND_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3) Fallback to source tree under third_party
    default_path = Path(__file__).resolve().parent.parent.parent / "third_party" / framework
    candidate_paths.append(default_path)

    # Normalize & deduplicate
    candidate_paths = list(dict.fromkeys(os.path.normpath(os.path.abspath(p)) for p in candidate_paths))

    # Insert the first existing path into sys.path
    for path in candidate_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
                if verbose:
                    print(f"[Primus] sys.path.insert: {path}")
            return path  # Return the first valid path

    # None of the candidate paths exist
    raise FileNotFoundError(
        f"[Primus] backend_path not found for framework '{framework}'. " f"Tried paths: {candidate_paths}"
    )


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.

     Steps:
        1. Load and parse the experiment YAML config
        2. Merge CLI overrides into the config
        3. Optionally export the merged config
        4. Setup backend path
        5. Launch the training
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Train] Config file '{cfg_path}' not found.")

    primus_cfg = load_primus_config(args, overrides)

    # Export merged config if requested
    if args.export_config:
        export_path = Path(args.export_config).resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)

        dump_namespace_to_yaml(nested_namespace_to_dict(primus_cfg._exp), str(export_path))
        print(f"[Primus:Train] Exported merged config to {export_path}")

    # Setup backend path for dynamic import
    framework = primus_cfg.get_module_config("pre_trainer").framework
    setup_backend_path(framework=framework, backend_path=args.backend_path, verbose=True)

    # Run the training
    import primus.train as train

    train.run_train(primus_cfg)
