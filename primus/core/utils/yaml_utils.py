import json
import os
import re
from types import SimpleNamespace

import yaml


def parse_yaml(yaml_file: str):
    def replace_env_variables(config):
        """Recursively replace environment variable placeholders in the config."""
        if isinstance(config, dict):
            return {replace_env_variables(key): replace_env_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [replace_env_variables(item) for item in config]
        elif isinstance(config, str):
            return re.sub(
                r"\${(.*?)}",
                lambda m: os.environ.get(m.group(1).split(":")[0], m.group(1).split(":")[1]),
                config,
            )
        return config

    with open(yaml_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = replace_env_variables(config)

        if config is None:
            return {}

        if "bases" in config:
            for base_file in config["bases"]:
                full_base_file = os.path.join(os.path.dirname(yaml_file), base_file)
                base_config = parse_yaml(full_base_file)
                # key must in base config
                for key in config:
                    if key != "bases":
                        assert key in base_config, (
                            f"The argument '{key}' in the a config file '{yaml_file}' "
                            f"cannot be found in the base config file '{base_file}'."
                        )
                for key, value in base_config.items():
                    if key != "bases" and key not in config:
                        config[key] = value
                # remove bases config
                del config["bases"]

        if "includes" in config:
            for include_file in config["includes"]:
                full_include_file = os.path.join(os.path.dirname(yaml_file), include_file)
                include_config = parse_yaml(full_include_file)
                # overrides if exist
                for key, value in include_config.items():
                    if key == "includes":
                        continue
                    if key not in config:
                        config[key] = value
                # remove includes config
                del config["includes"]

        return config


def dict_to_nested_namespace(d: dict):
    """Recursively convert dictionary to a nested SimpleNamespace."""
    return SimpleNamespace(
        **{k: dict_to_nested_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def nested_namespace_to_dict(obj):
    """Recursively convert nested SimpleNamespace to a dictionary."""
    if isinstance(obj, SimpleNamespace):
        return {key: nested_namespace_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [nested_namespace_to_dict(item) for item in obj]
    return obj


def parse_yaml_to_namespace(yaml_file: str):
    return dict_to_nested_namespace(parse_yaml(yaml_file))


def parse_nested_namespace_to_str(namespace: SimpleNamespace, indent=4):
    return json.dumps(nested_namespace_to_dict(namespace), indent=indent)


def delete_namespace_key(namespace: SimpleNamespace, key: str):
    if hasattr(namespace, key):
        delattr(namespace, key)


def has_key_in_namespace(namespace: SimpleNamespace, key: str):
    return hasattr(namespace, key)


def check_key_in_namespace(namespace: SimpleNamespace, key: str):
    # WARN: namespace should have name attr
    assert has_key_in_namespace(namespace, key), f"Failed to find key({key}) in namespace({namespace.name})"


def get_value_by_key(namespace: SimpleNamespace, key: str):
    check_key_in_namespace(namespace, key)
    return getattr(namespace, key)


def set_value_by_key(namespace: SimpleNamespace, key: str, value, allow_override=False):
    if not allow_override:
        assert not hasattr(namespace, key), f"Not allowed to override key({key}) in namespace({namespace})"
    return setattr(namespace, key, value)


def override_namespace(original_ns: SimpleNamespace, overrides_ns: SimpleNamespace):
    if overrides_ns is None:
        return

    for key in vars(overrides_ns):
        if not has_key_in_namespace(original_ns, key):
            raise Exception(
                f"Override namespace failed: can't find key({key}) in namespace({original_ns.name})"
            )
        new_value = get_value_by_key(overrides_ns, key)
        if isinstance(new_value, SimpleNamespace):
            override_namespace(get_value_by_key(original_ns, key), new_value)
        else:
            set_value_by_key(original_ns, key, new_value, allow_override=True)


def merge_namespace(dst: SimpleNamespace, src: SimpleNamespace, allow_override=False, excepts: list = None):
    src_dict = vars(src)
    dst_dict = vars(dst)
    for key, value in src_dict.items():
        if key in excepts:
            continue
        if key in dst_dict and not allow_override:
            raise ValueError(f"Key '{key}' from {src.name} already exists in {dst.name}.")
        else:
            setattr(dst, key, value)
