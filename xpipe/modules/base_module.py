###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

from abc import ABC, abstractmethod
from xpipe.core.launcher.config import XPipeConfig


class BaseModule(ABC):
    def __init__(self, module_name: str, xpipe_config: XPipeConfig):
        self.module_name = module_name
        self.xpipe_config = xpipe_config

    @abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
