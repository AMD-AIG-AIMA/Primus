from dataclasses import dataclass, field

from torchtitan.config_manager import JobConfig as TTJobConfig

# TODO: float8 quant config
# Tensorwise / Rowwise / Blockwise  etc.
# @dataclass
# class PrimusTurboFloat8Config:
#     pass


@dataclass
class PrimusTurboConfig:
    enable_primus_turbo: bool = False
    enable_attention_float8: bool = False
    # float8_config: PrimusTurboFloat8Config = field(default_factory=PrimusTurboFloat8Config)


@dataclass
class JobConfig(TTJobConfig):
    primus_turbo: PrimusTurboConfig = field(default_factory=PrimusTurboConfig)
