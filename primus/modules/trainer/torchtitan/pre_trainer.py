###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule


def monkey_patch_torchtitan_logger():
    from primus.core.utils.logger import _logger as primus_logger

    primus_logger.info("Mokey patch torchtitan logger...")

    import torchtitan.tools.logging as titan_logging

    titan_logging.logger = primus_logger
    titan_logging.init_logger = lambda: None

    import logging

    logging.getLogger("transformer_engine").setLevel(logging.ERROR)
    for name in logging.root.manager.loggerDict:
        print("Logger detected:", name)


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


def log_config(obj: Any, header: str = "TorchTitan Config"):
    from torchtitan.tools.logging import logger

    logger.info("========== %s ==========" % header)
    flat = flatten_config(obj)
    max_key_len = max(len(k) for k in flat.keys())
    for key in sorted(flat):
        val = flat[key]
        formatted_line = f"arguments {key.ljust(max_key_len, '.')} {val}"
        logger.info(formatted_line)


def build_job_config(cfg_dict: dict, JobConfigType) -> Any:
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

    return JobConfigType(
        job=Job(**cfg_dict.get("job", {})),
        profiling=Profiling(**cfg_dict.get("profiling", {})),
        metrics=Metrics(**cfg_dict.get("metrics", {})),
        model=Model(**cfg_dict.get("model", {})),
        optimizer=Optimizer(**cfg_dict.get("optimizer", {})),
        lr_scheduler=LRScheduler(**cfg_dict.get("lr_scheduler", {})),
        training=Training(**cfg_dict.get("training", {})),
        parallelism=Parallelism(**cfg_dict.get("parallelism", {})),
        checkpoint=Checkpoint(**cfg_dict.get("checkpoint", {})),
        activation_checkpoint=ActivationCheckpoint(**cfg_dict.get("activation_checkpoint", {})),
        float8=Float8(**cfg_dict.get("float8", {})),
        mx=MX(**cfg_dict.get("mx", {})),
        comm=Comm(**cfg_dict.get("comm", {})),
        memory_estimation=MemoryEstimation(**cfg_dict.get("memory_estimation", {})),
        fault_tolerance=FaultTolerance(**cfg_dict.get("fault_tolerance", {})),
        experimental=Experimental(**cfg_dict.get("experimental", {})),
    )


class TorchTitanPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        monkey_patch_torchtitan_logger()

        from torchtitan.config_manager import JobConfig
        from torchtitan.train import Trainer

        self.TrainerClass = Trainer
        self.JobConfigClass = JobConfig

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        cfg_dict = nested_namespace_to_dict(pre_trainer_cfg)

        self.titan_config = build_job_config(cfg_dict, self.JobConfigClass)
        self.trainer = None

    def setup(self):
        pass

    def init(self, *init_args, **kwargs):
        log_config(self.titan_config)
        self.trainer = self.TrainerClass(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()
