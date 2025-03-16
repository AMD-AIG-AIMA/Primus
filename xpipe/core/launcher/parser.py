import argparse
import os
from pathlib import Path

from xpipe.core.utils import constant_vars, yaml_utils

from .config import XPipeConfig


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    # cli arguments
    parser = argparse.ArgumentParser(description="XPipe Arguments", allow_abbrev=False)
    parser = _add_exp_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        cli_args, _ = parser.parse_known_args()
    else:
        cli_args = parser.parse_args()

    config_parser = XPipeParser()
    xpipe_config = config_parser.parse(cli_args)

    return xpipe_config


def _add_exp_args(parser):
    """Add exp arguments for XPipe."""
    group = parser.add_argument_group(title="XPipe exp arguments")
    group.add_argument(
        "--exp",
        type=str,
        required=True,
        help="XPipe experiment yaml config file.",
    )
    return parser


class XPipeParser(object):
    def __init__(self):
        pass

    def parse(self, cli_args: argparse.Namespace) -> XPipeConfig:
        exp_yaml_cfg = cli_args.exp
        self.xpipe_home = Path(os.path.dirname(__file__)).parent.parent.absolute()
        self.parse_exp(exp_yaml_cfg)
        self.parse_meta_info()
        self.parse_platform()
        self.parse_modules()
        return XPipeConfig(cli_args, self.exp)

    def parse_exp(self, config_file: str):
        self.exp = yaml_utils.parse_yaml_to_namespace(config_file)
        self.exp.name = constant_vars.XPIPE_CONFIG_NAME
        self.exp.config_file = config_file

    def parse_meta_info(self):
        yaml_utils.check_key_in_namespace(self.exp, "work_group")
        yaml_utils.check_key_in_namespace(self.exp, "user_name")
        yaml_utils.check_key_in_namespace(self.exp, "exp_name")
        yaml_utils.check_key_in_namespace(self.exp, "workspace")

    def parse_platform(self):
        yaml_utils.check_key_in_namespace(self.exp, "platform")
        self.exp.platform.name = "exp.platform"

        # parse platform config
        yaml_utils.check_key_in_namespace(self.exp.platform, "config")
        platform_config_file = os.path.join(self.xpipe_home, "configs/platforms", self.exp.platform.config)
        platform_config = yaml_utils.parse_yaml_to_namespace(platform_config_file)

        # override args
        if yaml_utils.has_key_in_namespace(self.exp.platform, "overrides"):
            yaml_utils.override_namespace(platform_config, self.exp.platform.overrides)

        # final check
        yaml_utils.check_key_in_namespace(platform_config, "name")
        yaml_utils.check_key_in_namespace(platform_config, "num_nodes_env_key")
        yaml_utils.check_key_in_namespace(platform_config, "node_rank_env_key")
        yaml_utils.check_key_in_namespace(platform_config, "master_addr_env_key")
        yaml_utils.check_key_in_namespace(platform_config, "master_port_env_key")
        yaml_utils.check_key_in_namespace(platform_config, "gpus_per_node_env_key")
        yaml_utils.check_key_in_namespace(platform_config, "master_sink_level")
        yaml_utils.check_key_in_namespace(platform_config, "workspace")

        # update exp.platform
        yaml_utils.set_value_by_key(self.exp, "platform", platform_config, allow_override=True)

    def get_model_format(self, module_framework: str):
        map = {
            "jax": "jax",
            "megatron": "megatron",
            "sglang": "huggingface",
        }
        assert module_framework in map, f"Invalid module framework: {module_framework}."
        return map[module_framework]

    def parse_trainer_module(self, module_name):
        module = yaml_utils.get_value_by_key(self.exp.modules, module_name)
        module.name = f"exp.modules.{module_name}"
        yaml_utils.check_key_in_namespace(module, "framework")
        yaml_utils.check_key_in_namespace(module, "config")
        yaml_utils.check_key_in_namespace(module, "model")
        framework = module.framework

        # config
        module_config_file = os.path.join(self.xpipe_home, "configs/modules", framework, module.config)
        module_config = yaml_utils.parse_yaml_to_namespace(module_config_file)
        module_config.name = f"exp.modules.{module_name}.config"

        # framework
        module_config.framework = framework

        # model
        model_format = self.get_model_format(framework)
        model_config_file = os.path.join(self.xpipe_home, "configs/models", model_format, module.model)
        model_config = yaml_utils.parse_yaml_to_namespace(model_config_file)
        model_config.name = f"exp.modules.{module_name}.model"

        # module config = config + model + overrides
        yaml_utils.merge_namespace(module_config, model_config, allow_override=False, excepts=["name"])
        if yaml_utils.has_key_in_namespace(module, "overrides"):
            yaml_utils.override_namespace(module_config, module.overrides)

        # flatten args of exp.modules.module_name
        module_config.name = module_name
        yaml_utils.set_value_by_key(self.exp.modules, module_name, module_config, allow_override=True)

    def parse_modules(self):
        yaml_utils.check_key_in_namespace(self.exp, "modules")
        for module_name in vars(self.exp.modules):
            if "trainer" in module_name:
                self.parse_trainer_module(module_name)
            else:
                raise ValueError(f"Not supported module: {module_name}")
