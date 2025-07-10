###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field

from torchtitan.config_manager import JobConfig as TTJobConfig

@dataclass
class PrimusTurboConfig:
    enable_primus_turbo: bool = False
    enable_attention_float8: bool = False


@dataclass
class JobConfig(TTJobConfig):
    primus_turbo: PrimusTurboConfig = field(default_factory=PrimusTurboConfig)