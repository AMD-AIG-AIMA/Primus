import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

from primus.core.launcher.config import PrimusConfig
from primus.core.utils import constant_vars, yaml_utils


def _parse_args(extra_args_provider=None, ignore_unknown_args=False) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Primus Arguments", allow_abbrev=False)

    parser.add_argument(
        "--config",
        "--exp",
        dest="exp",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )

    parser.add_argument(
        "--export-config",
        type=str,
        help="Optional path to export the final merged config to a file.",
    )

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    return parser.parse_known_args() if ignore_unknown_args else (parser.parse_args(), [])


def _parse_kv_overrides(args: list[str]) -> dict:
    """
    Parse CLI arguments of the form:
      --key=value
      --key value
      --flag (boolean True)
    into a nested dictionary structure.

    Supports nested keys using dot notation, e.g., --a.b.c=1.
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        # Ignore non-option arguments (not starting with "--")
        if not arg.startswith("--"):
            i += 1
            continue

        # Strip the "--" prefix
        key = arg[2:]

        if "=" in key:
            # Format: --key=value
            key, val = key.split("=", 1)
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            # Format: --key value
            val = args[i + 1]
            i += 1
        else:
            # Format: --flag (boolean True)
            val = True

        # Try to evaluate the value to correct type (int, float, bool, etc.)
        try:
            val = eval(val, {}, {})
        except Exception:
            pass  # Leave as string if evaluation fails

        # Handle nested keys, e.g., modules.pre_trainer.lr
        d = overrides
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

        i += 1

    return overrides


def _deep_merge_namespace(ns, override_dict):
    for k, v in override_dict.items():
        if hasattr(ns, k) and isinstance(getattr(ns, k), SimpleNamespace) and isinstance(v, dict):
            _deep_merge_namespace(getattr(ns, k), v)
        else:
            setattr(ns, k, v)


def _check_keys_exist(ns: SimpleNamespace, overrides: dict, prefix=""):
    for k, v in overrides.items():
        full_key = f"{prefix}.{k}" if prefix else k
        assert hasattr(ns, k), f"Override key '{full_key}' does not exist in pre_trainer config."
        attr_val = getattr(ns, k)
        if isinstance(v, dict):
            assert isinstance(
                attr_val, SimpleNamespace
            ), f"Override key '{full_key}' expects a namespace/dict but got {type(attr_val)}"
            _check_keys_exist(attr_val, v, prefix=full_key)


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args, unknown_args = _parse_args(extra_args_provider, ignore_unknown_args=True)

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)

    overrides = _parse_kv_overrides(unknown_args)
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    _check_keys_exist(pre_trainer_cfg, overrides)
    _deep_merge_namespace(pre_trainer_cfg, overrides)

    if args.export_config:
        from primus.core.utils.yaml_utils import dump_namespace_to_yaml

        dump_namespace_to_yaml(primus_config.exp, args.export_config)

    return primus_config


class PrimusParser(object):
    def __init__(self):
        pass

    def parse(self, cli_args: argparse.Namespace) -> PrimusConfig:
        exp_yaml_cfg = cli_args.exp
        self.primus_home = Path(os.path.dirname(__file__)).parent.parent.absolute()
        self.parse_exp(exp_yaml_cfg)
        self.parse_meta_info()
        self.parse_platform()
        self.parse_modules()
        return PrimusConfig(cli_args, self.exp)

    def parse_exp(self, config_file: str):
        self.exp = yaml_utils.parse_yaml_to_namespace(config_file)
        self.exp.name = constant_vars.PRIMUS_CONFIG_NAME
        self.exp.config_file = config_file

    def parse_meta_info(self):
        yaml_utils.check_key_in_namespace(self.exp, "work_group")
        yaml_utils.check_key_in_namespace(self.exp, "user_name")
        yaml_utils.check_key_in_namespace(self.exp, "exp_name")
        yaml_utils.check_key_in_namespace(self.exp, "workspace")

    def parse_platform(self):
        # If platform is set in exp config
        if not hasattr(self.exp, "platform"):
            self.exp.platform = SimpleNamespace(
                config="platform_azure.yaml", overrides=SimpleNamespace(master_sink_level="INFO")
            )

        # Load platform config
        config_path = os.path.join(self.primus_home, "configs/platforms", self.exp.platform.config)
        platform_config = yaml_utils.parse_yaml_to_namespace(config_path)

        # Optional overrides
        if yaml_utils.has_key_in_namespace(self.exp.platform, "overrides"):
            yaml_utils.override_namespace(platform_config, self.exp.platform.overrides)

        # Final required key checks
        for key in [
            "name",
            "num_nodes_env_key",
            "node_rank_env_key",
            "master_addr_env_key",
            "master_port_env_key",
            "gpus_per_node_env_key",
            "master_sink_level",
            "workspace",
        ]:
            yaml_utils.check_key_in_namespace(platform_config, key)

        yaml_utils.set_value_by_key(self.exp, "platform", platform_config, allow_override=True)

    def get_model_format(self, framework: str):
        map = {"megatron": "megatron", "light-megatron": "megatron", "torchtitan": "torchtitan"}
        assert framework in map, f"Invalid module framework: {framework}."
        return map[framework]

    def parse_trainer_module(self, module_name):
        module = yaml_utils.get_value_by_key(self.exp.modules, module_name)
        module.name = f"exp.modules.{module_name}"
        yaml_utils.check_key_in_namespace(module, "framework")
        yaml_utils.check_key_in_namespace(module, "config")
        yaml_utils.check_key_in_namespace(module, "model")
        framework = module.framework

        model_format = self.get_model_format(framework)

        # config
        module_config_file = os.path.join(self.primus_home, "configs/modules", model_format, module.config)
        module_config = yaml_utils.parse_yaml_to_namespace(module_config_file)
        module_config.name = f"exp.modules.{module_name}.config"

        # framework
        module_config.framework = framework

        # backend_path
        if yaml_utils.has_key_in_namespace(module, "backend_path"):
            module_config.backend_path = module.backend_path
        else:
            module_config.backend_path = os.path.join(
                self.primus_home.parent.absolute(), "third_party", model_format
            )

        # model
        model_config_file = os.path.join(self.primus_home, "configs/models", model_format, module.model)
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
                raise ValueError(f"Unsupported module: {module_name}")
