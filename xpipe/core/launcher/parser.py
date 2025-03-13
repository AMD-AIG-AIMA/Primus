import argparse
import os
from pathlib import Path

from xpipe.core.utils import yaml_utils

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
        self.align_orch_home = Path(os.path.dirname(__file__)).parent.parent.absolute()
        self.parse_exp(exp_yaml_cfg)
        self.parse_meta_info()
        self.parse_platform()
        self.parse_dataset()
        self.parse_modules()
        self.parse_pipeline()
        return XPipeConfig(cli_args, self.exp)

    def parse_exp(self, config_file: str):
        self.exp = yaml_utils.parse_yaml_to_namespace(config_file)
        self.exp.name = "XPipe"
        self.exp.config_file = config_file

    def parse_meta_info(self):
        yaml_utils.check_key_in_namespace(self.exp, "work_group")
        yaml_utils.check_key_in_namespace(self.exp, "user_name")
        yaml_utils.check_key_in_namespace(self.exp, "exp_name")

    def parse_platform(self):
        yaml_utils.check_key_in_namespace(self.exp, "platform")
        self.exp.platform.name = "exp.platform"

        # parse platform config
        yaml_utils.check_key_in_namespace(self.exp.platform, "config")
        platform_config_file = os.path.join(
            self.align_orch_home, "configs/platforms", self.exp.platform.config
        )
        platform_config = yaml_utils.parse_yaml_to_namespace(platform_config_file)

        # parse autotask manager config
        if hasattr(self.exp.platform, "autotask_manager_config"):
            autotask_manager_config_file = os.path.join(
                self.align_orch_home, "configs/autotask_managers", self.exp.platform.autotask_manager_config
            )
            autotask_manager_config = yaml_utils.parse_yaml_to_namespace(autotask_manager_config_file)
        else:
            autotask_manager_config = None
        platform_config.autotask_manager_config = autotask_manager_config

        # override args
        if yaml_utils.has_key_in_namespace(self.exp.platform, "overrides"):
            yaml_utils.override_namespace(platform_config, self.exp.platform.overrides)

        # auto task paths
        if platform_config.autotask_manager_config is not None:
            autotask_manager_config.autotask_config_paths = [
                os.path.join(self.align_orch_home, "configs/autotasks", _)
                for _ in autotask_manager_config.autotask_config_paths
            ]

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
            "huggingface": "huggingface",
            "megatron": "megatron",
            "vllm": "huggingface",
            "sglang": "huggingface",
        }
        # TODO: logger
        assert module_framework in map, f"Invalid module framework: {module_framework}."
        return map[module_framework]

    def parse_dataset(self):
        if hasattr(self.exp, "dataset"):
            assert isinstance(self.exp.dataset, str), f"The dataset path config should be a string"
            dataset_config_file = os.path.join(self.align_orch_home, "configs/datasets", self.exp.dataset)
            dataset_config = yaml_utils.parse_yaml_to_namespace(dataset_config_file)
            yaml_utils.check_key_in_namespace(dataset_config, "readable_dataset_type")
            yaml_utils.check_key_in_namespace(dataset_config, "readable_dataset_source")
            yaml_utils.set_value_by_key(self.exp, "dataset", dataset_config, allow_override=True)

    def parse_trainer_module(self, module_name):
        module = yaml_utils.get_value_by_key(self.exp.modules, module_name)
        module.name = f"exp.modules.{module_name}"
        # TODO(jingjie): some module may have no model and tokenizer
        yaml_utils.check_key_in_namespace(module, "framework")
        yaml_utils.check_key_in_namespace(module, "config")
        yaml_utils.check_key_in_namespace(module, "model")
        # yaml_utils.check_key_in_namespace(module, "dataset")
        # this is not essestial for ondpo like modules
        yaml_utils.check_key_in_namespace(module, "tokenizer")
        framework = module.framework

        # config
        module_config_file = os.path.join(self.align_orch_home, "configs/modules", framework, module.config)
        module_config = yaml_utils.parse_yaml_to_namespace(module_config_file)
        module_config.name = f"exp.modules.{module_name}.config"

        # framework
        module_config.framework = framework

        # model
        model_format = self.get_model_format(framework)
        model_config_file = os.path.join(self.align_orch_home, "configs/models", model_format, module.model)
        model_config = yaml_utils.parse_yaml_to_namespace(model_config_file)
        model_config.name = f"exp.modules.{module_name}.model"

        # datasets
        if hasattr(module, "dataset"):
            dataset_config_file = os.path.join(self.align_orch_home, "configs/datasets", module.dataset)
            dataset_config = yaml_utils.parse_yaml_to_namespace(dataset_config_file)
            dataset_config.name = f"exp.modules.{module_name}.dataset"
            yaml_utils.set_value_by_key(module_config, "dataset", dataset_config, allow_override=False)

        # tokenizer
        tokenizer_config_file = os.path.join(self.align_orch_home, "configs/tokenizers", module.tokenizer)
        tokenizer_config = yaml_utils.parse_yaml_to_namespace(tokenizer_config_file)
        tokenizer_config.name = f"exp.modules.{module_name}.tokenizer"
        yaml_utils.set_value_by_key(module_config, "tokenizer", tokenizer_config, allow_override=False)

        # module config = config + model + datasets + tokenizer + overrides
        yaml_utils.merge_namespace(module_config, model_config, allow_override=False, excepts=["name"])
        if yaml_utils.has_key_in_namespace(module, "overrides"):
            yaml_utils.override_namespace(module_config, module.overrides)

        # flatten args of exp.modules.module_name
        module_config.name = module_name
        yaml_utils.set_value_by_key(self.exp.modules, module_name, module_config, allow_override=True)

    def parse_inferer_module(self, module_name):
        module = yaml_utils.get_value_by_key(self.exp.modules, module_name)
        module.name = f"exp.modules.{module_name}"
        # TODO(jingjie): some module may have no model and tokenizer
        yaml_utils.check_key_in_namespace(module, "framework")
        yaml_utils.check_key_in_namespace(module, "config")
        framework = module.framework

        # config
        module_config_file = os.path.join(self.align_orch_home, "configs/modules", framework, module.config)
        module_config = yaml_utils.parse_yaml_to_namespace(module_config_file)
        module_config.name = f"exp.modules.{module_name}.config"

        # tokenizer
        if hasattr(module, "tokenizer"):
            tokenizer_config_file = os.path.join(self.align_orch_home, "configs/tokenizers", module.tokenizer)
            tokenizer_config = yaml_utils.parse_yaml_to_namespace(tokenizer_config_file)
            tokenizer_config.name = f"exp.modules.{module_name}.tokenizer"
            yaml_utils.set_value_by_key(module_config, "tokenizer", tokenizer_config, allow_override=False)

        # framework
        module_config.framework = framework

        # module config = config + overrides
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
            elif module_name in ["policy_generator", "reference", "reward"]:
                self.parse_inferer_module(module_name)
            else:
                raise ValueError(f"Not supported module: {module_name}")

    def parse_pipeline(self):
        # Currently the 'algo' argument is used to determine which algorithm to run, and
        # in the future, a general DAG configuration will be used to define the algorithm calculation flow.
        yaml_utils.check_key_in_namespace(self.exp, "algo")
