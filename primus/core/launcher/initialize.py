###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


from primus.core.utils import constant_vars, logger
from primus.core.utils.global_vars import (
    get_target_platform,
    is_initialized,
    set_global_variables,
    set_initialized,
)

from .parser import parse_args


def setup_master_logger(cfg, platform):
    # setup logger of primus master
    logger_cfg = logger.LoggerConfig(
        exp_root_path=cfg.exp_root_path,
        work_group=cfg.exp_meta_info["work_group"],
        user_name=cfg.exp_meta_info["user_name"],
        exp_name=cfg.exp_meta_info["exp_name"],
        module_name=constant_vars.PRIMUS_MASTER,
        file_sink_level=platform.sink_level,
        stderr_sink_level=platform.sink_level,
        node_ip=platform.get_addr(),
        rank=platform.get_node_rank(),  # node rank for master
        world_size=platform.get_num_nodes(),  # node number for master
    )
    logger.setup_logger(logger_cfg, is_head=True)


def log_init(cfg, platform):
    logger.info(f"Start Primus initialize...")
    logger.log_kv(f"-WorkGroup", f"{cfg.exp_meta_info['work_group']}")
    logger.log_kv(f"-UserName", f"{cfg.exp_meta_info['user_name']}")
    logger.log_kv(f"-Experiment", f"{cfg.exp_meta_info['exp_name']}")
    logger.log_kv(f"  -Modules", f"{cfg.module_keys}")
    logger.log_kv(f"  -RootPath", f"{cfg.exp_root_path}")
    logger.log_kv(f"-Platform", f"{cfg.platform_config.name}")
    logger.log_kv(f"  -MASTER_ADDR", f"{platform.get_master_addr()}")
    logger.log_kv(f"  -MASTER_PORT", f"{platform.get_master_port()}")
    logger.log_kv(f"  -NUM_NODES", f"{platform.get_num_nodes()}")
    logger.log_kv(f"  -NODE_RANK", f"{platform.get_node_rank()}")
    logger.log_kv(f"  -GPUS_PER_NODE", f"{platform.get_gpus_per_node()}")


def init(extra_args_provider=None):
    # Note: primus.init is for ray training master node init
    if is_initialized():
        return

    # cli arguments -> primus config
    cfg = parse_args(extra_args_provider=extra_args_provider)
    set_global_variables(cfg)

    # get platform object, then do the following:
    platform = get_target_platform()
    setup_master_logger(cfg, platform)

    log_init(cfg, platform)

    set_initialized()
