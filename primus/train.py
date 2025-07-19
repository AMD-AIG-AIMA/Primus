###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import sys

from primus.core.launcher.parser import parse_args


# Lazy backend loader
def load_backend_trainer(framework: str):
    if framework == "megatron":
        from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

        return MegatronPretrainTrainer
    elif framework == "torchtitan":
        from primus.modules.trainer.torchtitan.pre_trainer import (
            TorchTitanPretrainTrainer,
        )

        return TorchTitanPretrainTrainer
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def setup_backend_path(backend_path):
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        for path in backend_path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[Primus] backend_path does not exist: {path}")
            if path not in sys.path:
                sys.path.insert(0, path)


if __name__ == "__main__":
    primus_cfg = parse_args()

    # envs set by torchrun
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")

    # Setup backend path before lazy import
    setup_backend_path(pre_trainer_cfg.backend_path)

    TrainerClass = load_backend_trainer(pre_trainer_cfg.framework)

    trainer = TrainerClass(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    trainer.init()
    trainer.run()
