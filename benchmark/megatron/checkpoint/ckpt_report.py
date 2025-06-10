import os
import re
import logging
import argparse

from datetime import datetime

logger = logging.getLogger(__name__)

def log_and_exit(message):
    logger.error(message)
    raise Exception("ABORT")

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

def remove_ansi_escape(text: str) -> str:
    ANSI_ESCAPE_PATTERN = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ANSI_ESCAPE_PATTERN.sub("", text)

def get_time_elapsed_in_sec(start_time, end_time):
    FMT = "%Y%m%d %H:%M:%S"
    start = datetime.strptime(start_time, FMT)
    end = datetime.strptime(end_time, FMT)
    return int((end - start).total_seconds())

def get_full_log(log_dir):
    PRIMUS_LOG_PATTERN = re.compile(r"\[(.*?)\].*?\.py:\d+\]:\s*(.*)")
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
                        full_log.append((
                            match.group(1).strip(),
                            remove_ansi_escape(match.group(2).strip())
                        ))

    return sorted(list(set(full_log)), key=lambda x: x[0])

def get_arguments_from_log(full_log):
    ARGUMENTS_PATTERN = re.compile(r"(\S+)\s\.{3,}\s(.+)")
    arguments = {}
    for _, line in full_log:
        match = ARGUMENTS_PATTERN.search(line)
        if match:
            arguments[match.group(1)] = match.group(2)
    return arguments

def get_metrics_from_log(full_log, arguments):
    async_save = True if arguments["async_save"] == "True" else False
    save_start_indice = [idx for (idx, log) in enumerate(full_log) if "saving checkpoint at iteration" in log[1]]
    save_end_indice = [idx for (idx, log) in enumerate(full_log) if "successfully saved checkpoint from iteration" in log[1]]
    for idx in save_start_indice:
        print(full_log[idx])
    for idx in save_end_indice:
        print(full_log[idx])
    if len(save_start_indice)==0 or len(save_start_indice) != len(save_end_indice):
        log_and_exit("check save indice failed")
    total_time = get_time_elapsed_in_sec(full_log[save_start_indice[-1]][0], full_log[save_end_indice[-1]][0])
    metrics = {
        "block_time" : 0,
        "total_time" : total_time,
    }
    if not async_save:
        metrics["block_time"] = total_time
    else:
        pass
    return metrics

def get_arguments_and_metrics(log_dir):
    full_log = get_full_log(log_dir)
    #
    # for log in full_log:
    #     print(log)
    #
    arguments = get_arguments_from_log(full_log)
    metrics = get_metrics_from_log(full_log, arguments)
    return arguments, metrics

def main(args):
    logger.info(args)
    arguments, metrics = get_arguments_and_metrics(args.primus_log_dir)
    #
    # for k, v in arguments.items():
    #     print(f"{k} -- {v}")
    #
    print(metrics)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.DEBUG,
        format = "[%(asctime)s][%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()])
    args = parse_cli_args()
    main(args)
