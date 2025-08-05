###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import parse_args


# Lazy backend loader
def load_backend_trainer(framework: str):
    """
    Dynamically load the backend trainer class based on the selected framework.

    Notes:
        - This function should be called after sys.path has been updated by the CLI,
          or after the required backend path has been set via external environment
          variables (e.g., PYTHONPATH / MEGATRON_PATH / TORCHTITAN_PATH).
        - External PYTHONPATH setup is also supported and may be preferred in
          cluster/Slurm environments for consistency across multiple ranks.
    """
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


def run_train(primus_cfg: PrimusConfig):
    """
    Launch the training using the Primus trainer.

    Args:
        primus_cfg (PrimusConfig): Parsed Primus configuration object.
    """
    # envs set by torchrun
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    # Get pre_trainer module configuration
    pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    framework = pre_trainer_cfg.framework

    # Lazy import backend trainer
    TrainerClass = load_backend_trainer(framework)

    # Initialize trainer
    trainer = TrainerClass(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    # Launch training
    trainer.init()
    trainer.run()


if __name__ == "__main__":
    # Fallback mode: allow `python primus/train.py --config exp.yaml`
    from primus.core.launcher.parser import parse_args

    primus_cfg = parse_args()

    cfg = parse_args()
    if not isinstance(cfg, PrimusConfig):
        raise TypeError("primus_cfg must be an instance of PrimusConfig")
    run_train(cfg)
