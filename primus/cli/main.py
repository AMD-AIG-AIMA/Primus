###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import click
from primus.cli.train import train

@click.group()
def primus():
    """Primus unified CLI."""
    pass

primus.add_command(train)

if __name__ == "__main__":
    primus()
