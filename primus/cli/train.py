###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import click

import click
import sys

def setup_pythonpath(framework: str, backend_path: str = None):
    python_paths = []

    if backend_path:
        # Use custom backend path provided via CLI
        python_paths.append(os.path.realpath(backend_path[0]))
        print(f"backend_path{os.path.realpath(backend_path[0])}")
    else:
        # Fallback to default third_party path if backend path is not provided
        primus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_backend_path = os.path.join(primus_root, "third_party", framework)
        print(f"backend_path{default_backend_path}")
        if os.path.isdir(default_backend_path):
            python_paths.append(default_backend_path)

    # Add Primus root path to PYTHONPATH
    primus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_paths.append(primus_root)

    # Prepend new paths to sys.path
    sys.path = python_paths + sys.path

@click.command()
@click.option(
    "--config", 
    required=True,
    help="Path to experiment YAML config."
)
@click.option(
    "--backend-path", 
    type=str,
    help="Additional import paths, e.g., --backend-path /path/to/megatron"
)
def train(config, backend_path):
    setup_pythonpath("megatron", backend_path)
    print(f"test {config}")
    # train_main(exp_path=exp)