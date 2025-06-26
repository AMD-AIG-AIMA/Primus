###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

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


if __name__ == "__main__":
    primus_cfg = parse_args()

    # envs set by torchrun
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    if rank == 0:
        print(f"{primus_cfg}")
        # log_init(primus_cfg, primus_cfg.platform_config)

    pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")

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
