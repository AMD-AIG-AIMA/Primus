import os
import re
import logging
import argparse

logger = logging.getLogger(__name__)

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description = "Parse Primus training logs and generate checkpoint report",
        formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--primus-log-dir",
        type=str,
        required=True,
        help=(
            "Directory containing Primus training log folders. "
            "The general structure is as follows: "
            """
            .
            ├── rank-0
            │ ├── debug.log
            │ ├── error.log
            │ ├── info.log
            │ └── warning.log
            ├── rank-1
            │ ├── debug.log
            │ ├── error.log
            │ ├── info.log
            │ └── warning.log
            ...
            """
            )
        )
    args = parser.parse_args()
    return args

def get_full_log(log_dir):
    PRIMUS_LOG_PATTERN = r"\[(.*?)\].*?\.py:\d+\]:\s*(.*)"
    max_rank = max([int(m.group(1)) for m in map(lambda name: re.match(r"rank-(\d+)$", name),
        os.listdir(log_dir)) if m])
    logger.info(f"max_rank for training log dir : {max_rank}")
    full_log = []
    # get debug && info message type from first && last rank
    for rank in set([0, max_rank]):
        for level in ["debug", "info"]:
            with open(os.path.join(log_dir, f"rank-{rank}", f"{level}.log"), "r") as f:
                logger.debug(f"get log from file {f.name} ...")
                for line in f:
                    match = re.search(PRIMUS_LOG_PATTERN, line)
                    if match:
                        full_log.append((match.group(1).strip(), match.group(2).strip()))

    return sorted(list(set(full_log)), key=lambda x: x[0])

def get_arguments_from_log(full_log):
    arguments = {}
    for _, line in full_log:
        match = re.match(r'(\S+)\s\.{3,}\s(.+)', line)
        if match:
            arguments[match.group(1)] = match.group(2)
    return arguments

def get_metrics_from_log(full_log):
    pass

def get_arguments_and_metrics(log_dir):
    full_log = get_full_log(log_dir)
    arguments = get_arguments_from_log(full_log)
    metrics = get_metrics_from_log(full_log)
    return arguments, metrics

def main(args):
    logger.info(args)
    arguments, metrics = get_arguments_and_metrics(args.primus_log_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.DEBUG,
        format = "[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()])
    args = parse_cli_args()
    main(args)
