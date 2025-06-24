###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any

from . import checker
from .decorator import call_once
from .file_utils import create_path_if_not_exists

_logger = None

LOGGING_BANNER = ">>>>>>>>>>"

# "[<white>{process}</>]"
# "[<magenta>{extra[module_name]: <11}</>]"
# "[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow>]: <level>{message}</level>"
# "[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow>]: <level>{message}</level>"
master_stderr_sink_format = (
    "<blue>(PrimusMaster  pid={process}) </>"
    "[<green>{time:YYYYMMDD HH:mm:ss}</>]"
    "[<cyan>node-{extra[rank]}/{extra[world_size]}</>]"
    "[<level>{level: <5}</level>]"
    "<level>{message}</level>"
)
stderr_sink_format = (
    "[<green>{time:YYYYMMDD HH:mm:ss}</>]"
    "[<cyan>rank-{extra[rank]}/{extra[world_size]}</>]"
    "[<level>{level: <5}</level>]"
    "<level>{message}</level>"
)
master_file_sink_format = (
    "<blue>(PrimusMaster  pid={process}, ip={extra[node_ip]}) </>"
    "[<green>{time:YYYYMMDD HH:mm:ss}</>]"
    "[<blue>{extra[user]}/{extra[team]}</>]"
    "[<magenta>{extra[module_name]: <11}</>]"
    "[<cyan>node-{extra[rank]}/{extra[world_size]}</>]"
    "[<level>{level: <5}</level>]"
    "<level>{message}</level>"
)
file_sink_format = (
    "[<green>{time:YYYYMMDD HH:mm:ss}</>]"
    "[<blue>{extra[user]}/{extra[team]}</>]"
    "[<magenta>{extra[module_name]: <11}</>]"
    "[<cyan>ip-{extra[node_ip]}</>]"
    "[<cyan>rank-{extra[rank]}/{extra[world_size]}</>]"
    "[<level>{level: <5}</level>]"
    "<level>{message}</level>"
)


@dataclass(frozen=True)
class LoggerConfig:
    exp_root_path: str
    work_group: str
    user_name: str
    exp_name: str

    module_name: str
    file_sink_level: str = "INFO"
    stderr_sink_level: str = "INFO"

    node_ip: str = "localhost"
    rank: int = 0
    world_size: int = 1


def add_file_sink(
    logger,
    log_path: str,
    file_sink_level: str,
    prefix: str,
    is_head: bool,
    level: str,
    rotation: str = "10 MB",
    retention: str = None,
    encoding: str = "utf-8",
    backtrace: bool = True,
    diagnose: bool = True,
):
    assert level in [
        "trace",
        "debug",
        "info",
        "success",
        "warning",
        "error",
        "critical",
    ]
    sink_format = master_file_sink_format if is_head else file_sink_format
    if logger.level(level.upper()) >= logger.level(file_sink_level.upper()):
        logger.add(
            os.path.join(log_path, f"{prefix}{level}.log"),
            level=level.upper(),
            backtrace=backtrace,
            diagnose=diagnose,
            format=sink_format,
            colorize=False,
            rotation=rotation,
            retention=retention,
            encoding=encoding,
            filter=lambda record: record["level"].no >= logger.level(level.upper()).no,
        )


@call_once
def setup_logger(
    cfg: LoggerConfig,
    is_head: bool = False,
):
    create_path_if_not_exists(cfg.exp_root_path)
    if is_head:
        log_path = os.path.join(cfg.exp_root_path, f"logs/master")
    else:
        log_path = os.path.join(cfg.exp_root_path, f"logs/{cfg.module_name}/rank-{cfg.rank}")

    from loguru import logger as loguru_logger

    # remove default stderr sink
    loguru_logger.remove(0)

    # bind extra attributes to the loguru_logger
    loguru_logger = loguru_logger.bind(team=cfg.work_group)
    loguru_logger = loguru_logger.bind(user=cfg.user_name)
    loguru_logger = loguru_logger.bind(exp=cfg.exp_name)
    loguru_logger = loguru_logger.bind(module_name=cfg.module_name)
    loguru_logger = loguru_logger.bind(node_ip=cfg.node_ip)
    loguru_logger = loguru_logger.bind(rank=cfg.rank)
    loguru_logger = loguru_logger.bind(world_size=cfg.world_size)

    sink_file_prefix = f"master-" if is_head else ""
    sinked_levels = ["debug", "info", "warning", "error"]
    for sinked_level in sinked_levels:
        add_file_sink(
            loguru_logger,
            log_path,
            cfg.file_sink_level,
            sink_file_prefix,
            is_head,
            sinked_level,
        )

    sink_format = master_stderr_sink_format if is_head else stderr_sink_format
    loguru_logger.add(
        sys.stderr,
        level=cfg.stderr_sink_level.upper(),
        backtrace=True,
        diagnose=True,
        format=sink_format,
        colorize=True,
        filter=lambda record: record["level"].no >= loguru_logger.level(cfg.stderr_sink_level.upper()).no,
    )

    global _logger
    checker.check_true(_logger is None, "logger Must be None at first logger setup.")
    _logger = loguru_logger


def module_format(module_name: str, line: int):
    return "[" + f"{module_name}.py:{line}".rjust(23, "-") + "]"


def debug(__message: str, *args: Any, **kwargs: Any) -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.debug(__message, *args, **kwargs)


def debug_with_caller(__message: str, module_name: str, function_name: str, line: int) -> None:
    global _logger
    __message = f"{module_format(module_name, line)}: {__message}"
    _logger.debug(__message)


def log(__message: str, *args: Any, **kwargs: Any) -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.info(__message, *args, **kwargs)


def log_kv_with_caller(
    key: str,
    value: str,
    module_name: str,
    function_name: str,
    line: int,
    width=20,
    fillchar=" ",
) -> None:
    global _logger

    __message = f"{key}:".ljust(width, fillchar) + f"{value}"
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.info(__message)


def log_kv(key: str, value: str, width=18, fillchar=" ") -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{key}:".ljust(width, fillchar) + f"{value}"
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.info(__message)


def info(__message: str, *args: Any, **kwargs: Any) -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.info(__message, *args, **kwargs)


def info_with_caller(__message: str, module_name: str, function_name: str, line: int) -> None:
    global _logger
    __message = f"{module_format(module_name, line)}: {__message}"
    _logger.info(__message)


def warning(__message: str, *args: Any, **kwargs: Any) -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.warning(__message, *args, **kwargs)


def warning_with_caller(__message: str, module_name: str, function_name: str, line: int) -> None:
    global _logger
    __message = f"[{module_name}.py:{line}]: {__message}"
    _logger.warning(__message)


def error(__message: str, *args: Any, **kwargs: Any) -> None:
    global _logger

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno
    __message = f"{module_format(module_name, line)}: {__message}"

    _logger.error(__message, *args, **kwargs)


def error_with_caller(__message: str, module_name: str, function_name: str, line: int) -> None:
    global _logger
    __message = f"{module_format(module_name, line)}: {__message}"
    _logger.warning(__message)
