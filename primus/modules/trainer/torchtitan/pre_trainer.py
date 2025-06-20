###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import primus_turbo.pytorch as pt
import torch
import torch.distributed._symmetric_memory as symm_module
import torch.distributed.distributed_c10d as c10d
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from primus.backends.transformer_engine.transformer_engine_torch.comm_overlap import (
    get_backend_stream,
)
from primus.modules.trainer.torchtitan.parse_utils import get_torchtitan_config_args


def flatten_config(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat_dict = {}

    if is_dataclass(obj):
        obj = asdict(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if is_dataclass(value) or isinstance(value, dict):
                flat_dict.update(flatten_config(value, full_key))
            else:
                flat_dict[full_key] = value
    else:
        flat_dict[prefix] = obj

    return flat_dict


def log_config(logger, obj: Any, header: str = "TorchTitan Config"):
    logger.info("========== %s ==========" % header)
    flat = flatten_config(obj)
    for key in sorted(flat):  # optional: sorted for readability
        logger.info(f"arguments {key}: {flat[key]}")


class TorchtitanPretrainTrainer:
    def __init__(self):
        titan_args = get_torchtitan_config_args()
        init_logger()
        self.titan_config: JobConfig = ConfigManager().parse_args(titan_args)

        tokenizer_path = os.getenv("TOKENIZER_PATH")
        if tokenizer_path is not None:
            self.titan_config.model.tokenizer_path = tokenizer_path
        self.trainer = None

        logger.warning(f"TorchtitanPretrainTrainer: Patch Async TP")
        self.patch_torch_async_tp()

    def init(self, *init_args, **kwargs):
        log_config(logger, self.titan_config)
        self.trainer = Trainer(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()

    @staticmethod
    def patch_torch_async_tp(cls):
        def _fused_all_gather_matmul_impl(
            mm_out_op: torch._ops.OpOverload,
            A_shard: torch.Tensor,
            Bs: list[torch.Tensor],
            A_scale: Optional[torch.Tensor],
            kwargs_list: list[dict[str, Any]],
            out_dtypes: list[Optional[torch.dtype]],
            gather_dim: int,
            group_name: str,
            return_A: bool,
        ) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
            assert A_scale is None, "fused_all_gather_matmul not support for fp8"

            layouts = ["NN" for _ in range(len(Bs))]
            group = c10d._resolve_process_group(group_name)
            gemm_streams = [torch.cuda.current_stream()]
            comm_streams = get_backend_stream(size=group.size() - 1, priority=0, prefix="comm")

            copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")
            A, outputs = pt.ops.fused_all_gather_matmul(
                A_shard,
                Bs,
                layouts,
                gather_dim=gather_dim,
                group_name=group_name,
                gemm_streams=gemm_streams,
                comm_streams=comm_streams,
                copy_streams=copy_streams,
                comm_method="pipeline",
                num_splits=4,
                return_A=return_A,
                out_dtypes=out_dtypes,
            )

            return A, outputs

        symm_module._fused_all_gather_matmul_impl = _fused_all_gather_matmul_impl
