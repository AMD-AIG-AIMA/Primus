###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import argparse
import os
import sys

from primus.modules.trainer.torchtitan.pre_trainer import TorchtitanPretrainTrainer


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    parser = argparse.ArgumentParser(description="Primus Arguments", allow_abbrev=False)

    group = parser.add_argument_group(title="Primus exp arguments")
    group.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Primus experiment yaml config file.",
    )

    if extra_args_provider is not None:
        extra_args_provider(parser)

    if ignore_unknown_args:
        args, unknown_args = parser.parse_known_args()
    else:
        args = parser.parse_args()
        unknown_args = []

    # Merge known args back into CLI-style list
    known_args_list = []
    for key, value in vars(args).items():
        if isinstance(value, bool):
            if value:
                known_args_list.append(f"--{key.replace('_', '-')}")
        elif value is not None:
            known_args_list.extend([f"--{key.replace('_', '-')}", str(value)])

    merged_args = known_args_list + unknown_args
    return merged_args


def get_distributed_env():
    try:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        master_addr = os.getenv("MASTER_ADDR", "localhost")
        master_port = int(os.getenv("MASTER_PORT", "29500"))
        return rank, world_size, master_addr, master_port
    except Exception as e:
        raise RuntimeError(f"Failed to read torchrun env vars: {e}")


def main():
    print("--------------------- torchtitan")
    print("Arguments passed to pretrain.py:", sys.argv)

    # rank, world_size, master_addr, master_port = get_distributed_env()

    trainer = TorchtitanPretrainTrainer()
    trainer.init()
    trainer.run()


if __name__ == "__main__":
    main()
