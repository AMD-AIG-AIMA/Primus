import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)

from primus.modules.module_utils import log_rank_0, warning_rank_0


class PrimusFileSystemWriterAsync(FileSystemWriterAsync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def preload_tensors(*args, **kwargs):
        # (limou)
        # change argument non_blocking to False on HIP platform
        # the tensors will be stored in pinned memory if non_blocking=True
        # currently on the ROCm platform
        # forking a subprocess afterward with pinned_memory=True will trigger segmentation fault
        if torch.version.hip:
            log_rank_0("HIP env detected, change argument non_blocking in FileSystemWriterAsync to False")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 0 and type(args[-1]) == type(True):
                # TODO (limou)
                # non_blocking may NOT always be the last argument in the future
                args = args[:-1] + (False,)
            else:
                warning_rank_0("found argument non_blocking failed")

        return super(PrimusFileSystemWriterAsync, PrimusFileSystemWriterAsync).preload_tensors(
            *args, **kwargs
        )
