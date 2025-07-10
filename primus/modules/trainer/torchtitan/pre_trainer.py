###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule

class TorchTitanPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # important: make sure patch torchtitan logger first
        self.patch_torchtitan_logger()

        from torchtitan.config_manager import JobConfig
        from torchtitan.train import Trainer

        self.TrainerClass = Trainer
        self.JobConfigClass = JobConfig

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        cfg_dict = nested_namespace_to_dict(pre_trainer_cfg)

        self.titan_config = self.build_job_config(cfg_dict, self.JobConfigClass)
        self.trainer = None

        self.patch_torch_async_tp()

    def setup(self):
        pass

    def init(self, *init_args, **kwargs):
        self.log_config(self.titan_config)
        self.trainer = self.TrainerClass(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()

    def patch_torchtitan_logger(self):
        from primus.core.utils.logger import _logger as primus_logger

        primus_logger.info("Mokey patch torchtitan logger...")

        import torchtitan.tools.logging as titan_logging

        titan_logging.logger = primus_logger
        titan_logging.init_logger = lambda: None

    def patch_torch_async_tp(self):
        import torch
        import torch.distributed._symmetric_memory as symm_module
        import torch.distributed.distributed_c10d as c10d
        from torchtitan.tools.logging import logger

        if not self.titan_config.parallelism.enable_async_tensor_parallel:
            return

        try:
            import primus_turbo.pytorch as pt

            from primus.backends.transformer_engine.transformer_engine_torch.comm_overlap import (
                get_backend_stream,
            )

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

            logger.warning(f"TorchtitanPretrainTrainer: Patch Async TP")

        except ImportError as e:
            logger.warning(f"TorchtitanPretrainTrainer: Patch Async TP failed - {e}")

    def flatten_config(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        flat_dict = {}
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if is_dataclass(value) or isinstance(value, dict):
                    flat_dict.update(self.flatten_config(value, full_key))
                else:
                    flat_dict[full_key] = value
        else:
            flat_dict[prefix] = obj

        return flat_dict

    def log_config(self, obj: Any, header: str = "TorchTitan Config"):
        from torchtitan.tools.logging import logger

        logger.info("========== %s ==========" % header)
        flat = self.flatten_config(obj)
        max_key_len = max(len(k) for k in flat.keys())
        for key in sorted(flat):
            val = flat[key]
            formatted_line = f"arguments {key.ljust(max_key_len, '.')} {val}"
            logger.info(formatted_line)

    def build_job_config(self, cfg_dict: dict, JobConfigType) -> Any:
        from third_party.torchtitan.torchtitan.config_manager import (
            MX,
            ActivationCheckpoint,
            Checkpoint,
            Comm,
            Experimental,
            FaultTolerance,
            Float8,
            Job,
            LRScheduler,
            MemoryEstimation,
            Metrics,
            Model,
            Optimizer,
            Parallelism,
            Profiling,
            Training,
        )
        from dataclasses import is_dataclass
        import importlib
        from torchtitan.tools.logging import logger
            

        # Step 1: Parse the experimental section to check for a custom JobConfig extension
        experimental_cfg = cfg_dict.get("experimental", {})
        experimental = Experimental(**experimental_cfg)

        custom_job_config_cls = JobConfigType
        if getattr(experimental, "custom_args_module", None):
            try:
                module = importlib.import_module(experimental.custom_args_module)
                ExtendedJobConfig = getattr(module, "JobConfig")
                # Dynamically merge the base and custom JobConfig classes
                custom_job_config_cls = self.merge_configs(JobConfigType, ExtendedJobConfig)
            except Exception as e:
                logger.warning(f"Failed to load custom_args_module: {e}")

        # Step 2: Construct config sections using the merged or base JobConfig class
        flat_config = {
            "job": Job(**cfg_dict.get("job", {})),
            "profiling": Profiling(**cfg_dict.get("profiling", {})),
            "metrics": Metrics(**cfg_dict.get("metrics", {})),
            "model": Model(**cfg_dict.get("model", {})),
            "optimizer": Optimizer(**cfg_dict.get("optimizer", {})),
            "lr_scheduler": LRScheduler(**cfg_dict.get("lr_scheduler", {})),
            "training": Training(**cfg_dict.get("training", {})),
            "parallelism": Parallelism(**cfg_dict.get("parallelism", {})),
            "checkpoint": Checkpoint(**cfg_dict.get("checkpoint", {})),
            "activation_checkpoint": ActivationCheckpoint(**cfg_dict.get("activation_checkpoint", {})),
            "float8": Float8(**cfg_dict.get("float8", {})),
            "mx": MX(**cfg_dict.get("mx", {})),
            "comm": Comm(**cfg_dict.get("comm", {})),
            "memory_estimation": MemoryEstimation(**cfg_dict.get("memory_estimation", {})),
            "fault_tolerance": FaultTolerance(**cfg_dict.get("fault_tolerance", {})),
            "experimental": experimental,
        }

        # Step 3: Return a dataclass instance constructed from config dictionary
        logger.info(f"load custom_args_module")
        return custom_job_config_cls(**flat_config)

    @staticmethod
    def merge_configs(base_cls, custom_cls):
        """
        Merges two dataclass types into one unified dataclass.

        Merge logic:
        - If a field exists in both:
            - If both fields are dataclasses, recursively merge them.
            - Otherwise, the custom field overrides the base.
        - Fields only in base or only in custom are included as-is.
        """
        from dataclasses import field, fields, make_dataclass

        base_fields = {f.name: f for f in fields(base_cls)}
        custom_fields = {f.name: f for f in fields(custom_cls)}

        merged = []

        # Merge overlapping and base-only fields
        for name, base_f in base_fields.items():
            if name in custom_fields:
                custom_f = custom_fields[name]
                if is_dataclass(base_f.type) and is_dataclass(custom_f.type):
                    merged_type = TorchTitanPretrainTrainer.merge_configs(base_f.type, custom_f.type)
                    merged.append((name, merged_type, field(default_factory=merged_type)))
                else:
                    merged.append((name, custom_f.type, custom_f))
            else:
                merged.append((name, base_f.type, base_f))

        # Add custom-only fields
        for name, custom_f in custom_fields.items():
            if name not in base_fields:
                merged.append((name, custom_f.type, custom_f))

        return make_dataclass(f"Merged{base_cls.__name__}", merged, bases=(base_cls,))


