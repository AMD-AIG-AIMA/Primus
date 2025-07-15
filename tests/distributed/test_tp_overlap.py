###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from contextlib import contextmanager

import torch
import torch.distributed as dist
import transformer_engine as te
import transformer_engine_torch as tex
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from transformer_engine.pytorch import LayerNormLinear, Linear

from primus.backends.transformer_engine import transformer_engine_torch as ptex
from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import gemm
from primus.backends.transformer_engine.pytorch.module.base import (
    get_workspace,
    initialize_ub,
)
from primus.core.utils import logger
from primus.modules.module_utils import set_logging_rank


@contextmanager
def custom_te_patch():
    prev_CommOverlap = tex.CommOverlap
    prev_CommOverlapP2P = tex.CommOverlapP2P
    prev_CommOverlapAlgo = tex.CommOverlapAlgo
    prev_gemm = te.pytorch.cpp_extensions.gemm
    prev_initialize_ub = te.pytorch.module.base.initialize_ub
    prev_get_workspace = te.pytorch.module.base.get_workspace
    try:
        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType
        tex.CommOverlapAlgo = ptex.CommOverlapAlgo
        te.pytorch.cpp_extensions.gemm = gemm
        te.pytorch.module.linear.gemm = gemm
        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace
        te.pytorch.cpp_extensions.CommOverlapAlgo = ptex.CommOverlapAlgo
        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

        yield
    finally:
        tex.CommOverlap = prev_CommOverlap
        tex.CommOverlapP2P = prev_CommOverlapP2P
        tex.CommOverlapAlgo = prev_CommOverlapAlgo
        te.pytorch.cpp_extensions.gemm = prev_gemm
        te.pytorch.module.linear.gemm = prev_gemm
        te.pytorch.module.base.initialize_ub = prev_initialize_ub
        te.pytorch.module.base.get_workspace = prev_get_workspace
        te.pytorch.cpp_extensions.CommOverlapAlgo = prev_CommOverlapAlgo


def te_linear(seed, batch_size, seqlen, in_features, out_features, ub_overlap_ag, ub_overlap_rs, **cfg):
    torch.manual_seed(seed)
    tp_size = cfg["tp_size"]
    dtype = cfg.get("params_dtype", torch.bfloat16)
    parallel_mode = cfg.get("parallel_mode", "row")

    if ub_overlap_ag or ub_overlap_rs:
        te.pytorch.module.base._ub_communicators = None
        input_shape = [seqlen * batch_size, in_features]
        te.pytorch.module.base.initialize_ub(shape=input_shape, tp_size=tp_size, use_fp8=False)

    if parallel_mode == "column":
        inp_shape = (batch_size * seqlen // tp_size, in_features)
        grad_out_shape = (batch_size * seqlen, out_features // tp_size)
        model = LayerNormLinear(in_features, out_features, ub_overlap_ag=ub_overlap_ag, **cfg)
    else:
        inp_shape = (batch_size * seqlen, in_features // tp_size)
        grad_out_shape = (batch_size * seqlen // tp_size, in_features)
        model = Linear(in_features, out_features, ub_overlap_rs=ub_overlap_rs, **cfg)

    inp = torch.rand(inp_shape, dtype=dtype, device="cuda", requires_grad=True)
    grad_output = torch.rand(grad_out_shape, dtype=dtype, device="cuda")

    out = model(inp)
    out.backward(grad_output)

    return (out, inp.grad, model.weight.grad)


@instantiate_parametrized_tests
class TPOverlapTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        logger_cfg = logger.LoggerConfig(
            exp_root_path="/tmp",
            work_group="",
            user_name="",
            exp_name="test",
            module_name="linear",
            rank=self.rank,
            world_size=self.world_size,
        )

        logger.setup_logger(logger_cfg)
        set_logging_rank(rank=self.rank, world_size=self.world_size)

    @skip_if_lt_x_gpu(2)
    @parametrize("out_features", [1024])
    @parametrize("in_features", [1024])
    @parametrize("seqlen", [4096])
    @parametrize("batch_size", [1])
    @parametrize("ub_name", ["qkv", "proj"])
    @parametrize("parallel_mode", ["column", "row"])
    @parametrize("ub_overlap_ag", [True])
    @parametrize("ub_overlap_rs", [False])
    def test_te_linear(
        self,
        batch_size,
        seqlen,
        in_features,
        out_features,
        ub_name,
        parallel_mode,
        ub_overlap_ag,
        ub_overlap_rs,
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        seed = 42 + rank

        cfg = {
            "tp_group": group,
            "tp_size": self.world_size,
            "parallel_mode": parallel_mode,
            "sequence_parallel": True,
            "bias": False,
            "ub_name": ub_name,
            "params_dtype": torch.bfloat16,
        }

        base_outputs = te_linear(
            seed,
            batch_size,
            seqlen,
            in_features,
            out_features,
            ub_overlap_ag=False,
            ub_overlap_rs=False,
            **cfg
        )

        with custom_te_patch():
            patch_outputs = te_linear(
                seed,
                batch_size,
                seqlen,
                in_features,
                out_features,
                ub_overlap_ag=ub_overlap_ag,
                ub_overlap_rs=ub_overlap_rs,
                **cfg
            )

        for base_out, patch_out in zip(base_outputs, patch_outputs):
            torch.testing.assert_close(base_out, patch_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
