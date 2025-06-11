###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

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

    def init(self, *init_args, **kwargs):
        log_config(logger, self.titan_config)
        self.trainer = Trainer(self.titan_config)

    def run(self, *args, **kwargs):
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call init() first.")
        self.trainer.train()
