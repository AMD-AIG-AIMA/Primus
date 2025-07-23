###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.utils import logger

from .base_platform import BasePlatform


class RemotePlatform(BasePlatform):
    def __init__(self, name: str):
        super().__init__(name)
        self.num_nodes = int(self.get_platform_env(self.args.num_nodes_env_key))
        self.node_rank = int(self.get_platform_env(self.args.node_rank_env_key))
        self.master_addr = self.get_platform_env(self.args.master_addr_env_key)
        self.master_port = int(self.get_platform_env(self.args.master_port_env_key))

    def get_platform_supported_submit_methods(self):
        if self._name == "polaris":
            return ["polaris-direct"]
        return []

    def get_num_nodes(self):
        return self.num_nodes

    def get_node_rank(self):
        return self.node_rank

    def get_master_addr(self):
        return self.master_addr

    def get_master_port(self):
        return self.master_port

    def setup(self):
        logger.info(f"setup platform {self.name}... node_rank={self.get_node_rank()}")

    def teardown(self):
        if self.get_node_rank() == 0:
            logger.info(f"teardown platform {self.name}... node_rank={self.get_node_rank()}")
