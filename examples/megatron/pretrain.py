###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.launcher.initialize import log_init
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

if __name__ == "__main__":
    primus_cfg = parse_args()

    # envs set by torchrun
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    trainer = MegatronPretrainTrainer(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    if rank == 0:
        log_init(primus_cfg, trainer.platform)

    trainer.init()
    trainer.run()
