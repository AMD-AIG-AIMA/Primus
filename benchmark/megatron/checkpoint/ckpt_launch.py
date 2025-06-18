import os
import re
import logging
import argparse
import tempfile
import subprocess

import yaml

logger = logging.getLogger(__name__)

CURDIR = os.getcwd()

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description = "Launch traning task and save checkpoint")
    parser.add_argument(
        "--yaml-config-path",
        type = str,
        required = True,
        help = "Primus pretrainer config yaml path",
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="Number of nodes to run training on (default: 1)"
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type = int,
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type = int,
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type = int,
    )
    parser.add_argument(
        "--train-iters",
        type = int,
        default = 300,
    )
    parser.add_argument(
        "--save-interval",
        type = int,
        default = 100,
    )
    parser.add_argument(
        "--ckpt-format",
        type = str,
    )
    parser.add_argument(
        "--async-save",
        action = "store_true",
    )
    parser.add_argument(
        "--ckpt-fully-parallel-save",
        action = "store_true",
    )
    args = parser.parse_args()
    return args

def load_yaml_config(yaml_config_path):
    def substitute_env_vars(obj):
        PATTERN = re.compile(r"\$\{([^:}]+):([^}]+)\}")
        if isinstance(obj, dict):
            return {k: substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_env_vars(i) for i in obj]
        elif isinstance(obj, str):
            def repl(m): return os.environ.get(m.group(1), m.group(2))
            replaced = PATTERN.sub(repl, obj)
            try:
                return yaml.safe_load(replaced)
            except Exception:
                return replaced
        else:
            return obj
    with open(args.yaml_config_path, "r") as f:
        raw = yaml.safe_load(f)
    return substitute_env_vars(raw)
    
def overwrite_yaml_config(args, yaml_config):
    config = yaml_config["modules"]["pre_trainer"]["overrides"]

    if args.tensor_model_parallel_size:
        config["tensor_model_parallel_size"] = args.tensor_model_parallel_size
    if args.pipeline_model_parallel_size:
        config["pipeline_model_parallel_size"] = args.pipeline_model_parallel_size
    if args.expert_model_parallel_size:
        config["expert_model_parallel_size"] = args.expert_model_parallel_size
    if args.train_iters:
        config["train_iters"] = args.train_iters
    if args.save_interval:
        config["save_interval"] = args.save_interval
    if args.ckpt_format:
        config["ckpt_format"] = args.ckpt_format
    if args.async_save:
        config["async_save"] = args.async_save
    if args.ckpt_fully_parallel_save:
        config["ckpt_fully_parallel_save"] = args.ckpt_fully_parallel_save
    
    config["no_save_rng"] = None
    config["no_save_optim"] = None
    config["disable_last_saving"] = True
    config["auto_continue_train"] = False
    config["finetune"] = False

def train_with_overwritten_config(args, yaml_config):
    NEW_YAML_FILE = "training_config.yaml"
    exp_root_path = os.path.join(
        CURDIR,
        yaml_config["workspace"],
        yaml_config["work_group"],
        yaml_config["user_name"],
        yaml_config["exp_name"])
    logger.debug(f"exp_root_path={exp_root_path}")
    new_yaml_config_path = os.path.join(CURDIR, NEW_YAML_FILE)
    logger.debug(f"new_yaml_config_path={new_yaml_config_path}")
    with open(new_yaml_config_path, "w") as f:
        yaml.safe_dump(yaml_config, f)
    
    # launch training task
    env = os.environ.copy()
    env["EXP"] = new_yaml_config_path
    env["NUM_NODES"] = str(args.nnodes)
    env["BACKEND"] = "megatron"
    command = "bash examples/run_slurm_pretrain.sh"
    result = subprocess.run(command,
        shell = True,
        # capture_output = True,
        text = True,
        env = env)
    # logger.debug(f"training subprocess stdout : {result.stdout}")
    # logger.debug(f"training subprocess stderr : {result.stderr}")
    logger.info(f"training subprocess exit code : {result.returncode}")

def main(args):
    logger.debug(args)
    yaml_config = load_yaml_config(args.yaml_config_path)
    overwrite_yaml_config(args, yaml_config)
    logger.debug(yaml_config)
    train_with_overwritten_config(args, yaml_config)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.DEBUG,
        format = "[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()])
    args = parse_cli_args()
    main(args)