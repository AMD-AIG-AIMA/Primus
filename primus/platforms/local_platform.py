###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.utils import constant_vars, logger

from .base_platform import BasePlatform


class LocalPlatform(BasePlatform):
    def __init__(self, name="local"):
        super().__init__(name)

    def get_num_nodes(self):
        return constant_vars.LOCAL_NUM_NODES

    def get_node_rank(self):
        return constant_vars.LOCAL_NODE_RANK

    def get_master_addr(self):
        return constant_vars.LOCAL_MASTER_ADDR

    def get_master_port(self):
        return constant_vars.LOCAL_MASTER_PORT

    def get_platform_supported_submit_methods(self):
        return []

    def setup(self):
        logger.info(f"setup platform {self.name}... node_rank={self.get_node_rank()}")

    def teardown(self):
        if self.get_node_rank() == 0:
            logger.info(f"teardown platform {self.name}... node_rank={self.get_node_rank()}")
