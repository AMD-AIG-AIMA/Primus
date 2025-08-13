###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os

import torch
import torch.distributed as dist


def init_distributed():
    """Init only when launched with >1 processes via torchrun."""
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def finalize_distributed():
    """Destroy process group if initialized."""
    if dist.is_initialized():
        try:
            dist.barrier()  # optional: ensure all ranks reach this point
        except Exception:
            pass  # ignore barrier errors on exit
        dist.destroy_process_group()


def gather_all_results(obj):
    """
    Gather arbitrary Python object(s) from all ranks.
    - If `obj` is a list, returns a single flattened list (concatenate).
    - If `obj` is a dict (your per-rank summary), returns a list[dict].
    - If `obj` is None, it's skipped.
    """
    # Single-process fallback
    if not dist.is_initialized():
        if obj is None:
            return []
        # keep list as-is; wrap non-list as list
        return obj if isinstance(obj, list) else [obj]

    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)

    # Filter out None from any rank that returned nothing
    gathered = [g for g in gathered if g is not None]

    # If ranks each returned a list, concatenate
    if len(gathered) > 0 and isinstance(gathered[0], list):
        out = []
        for g in gathered:
            out.extend(g)
        return out

    # Otherwise, assume ranks returned single objects (e.g., dict summaries)
    return gathered


def is_rank_0() -> bool:
    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized, cannot determine rank.")
    return dist.get_rank() == 0


def get_local_rank() -> int:
    """Return local rank for this node (torchrun/Slurm/fallback)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() % max(1, torch.cuda.device_count())
    if "LOCAL_RANK" in os.environ:  # torchrun
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_current_device() -> torch.device:
    """Set and return the current CUDA device for this rank."""
    lr = get_local_rank()
    torch.cuda.set_device(lr)
    return torch.device(f"cuda:{lr}")


# @torch.no_grad()
# def allreduce_once(tensor):
#     dist.all_reduce(tensor)
