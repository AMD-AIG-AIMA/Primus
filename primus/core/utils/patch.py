# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import inspect
import os
import warnings

warnings.filterwarnings(action="once")


def is_rank0():
    True


try:
    import torch

    def is_rank0():
        is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        env_rank = os.getenv("RANK", None)

        if is_distributed:
            if torch.distributed.get_rank() == 0:
                return True
        elif env_rank:
            if env_rank == "0":
                return True
        else:
            return True

except ImportError:
    pass


def monkey_patch_fn(src_var, dst_var, rank0=True):
    assert type(src_var) == type(dst_var), "The type of src var and dst var are not matched."

    if rank0 and is_rank0():
        msg = f"""<'{src_var.__module__}.{src_var.__name__}' from '{inspect.getfile(src_var)}'> was replaced by \
<'{dst_var.__module__}.{dst_var.__name__}' from '{inspect.getfile(dst_var)}'>"""
        warnings.warn(msg, ImportWarning)

    src_var = dst_var
