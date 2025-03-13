###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


from xpipe.core.utils.global_vars import (
    is_initialized,
    set_global_variables,
    set_initialized,
)

from .parser import parse_args


def init(extra_args_provider=None):
    if is_initialized():
        return

    # cli arguments -> xpipe config
    cfg = parse_args(extra_args_provider=extra_args_provider)

    set_global_variables(cfg)
    set_initialized()
