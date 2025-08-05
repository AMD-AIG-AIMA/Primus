###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse

from primus.cli import train


def main():
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch Megatron / TorchTitan training.

    Reserved for future expansion:
    - preflight: Environment and configuration checks.
      ...
    """
    parser = argparse.ArgumentParser(
        prog="primus-cli", description="Primus Unified CLI for Training & Utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register train subcommand (only implemented one for now)
    train.register_subcommand(subparsers)

    args, unknown_args = parser.parse_known_args()

    # Dispatch to the implemented subcommands
    if args.command == "train":
        train.run(args, unknown_args)
    else:
        # Future subcommands can be dispatched here
        parser.print_help()


if __name__ == "__main__":
    main()
