"""megatron utils"""

import inspect
import os

import megatron
import torch

from xpipe.core.utils import logger


######################################################log after torch distributed initialized
def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(msg):
    """If distributed is initialized, print only on last rank."""
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if torch.distributed.is_initialized():
        if is_last_rank():
            log_func(msg, module_name, function_name, line)
    else:
        log_func(msg, module_name, function_name, line)


def set_wandb_writer_patch(args):  # monkey patch
    """
    This function is adapted from the original Megatron implementation, with an additional
    wandb argument `entity` be added.
    Monkey-patch note:
    - The original function will be replaced at runtime by this implementation.

    """

    megatron.training.global_vars._ensure_var_is_not_initialized(
        megatron.training.global_vars._GLOBAL_WANDB_WRITER, "wandb writer"
    )

    if getattr(args, "wandb_project", "") and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == "":
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, "wandb")
        wandb_kwargs = {
            "dir": save_dir,
            "name": args.wandb_exp_name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "config": vars(args),
        }
        os.makedirs(wandb_kwargs["dir"], exist_ok=True)
        wandb.init(**wandb_kwargs)
        megatron.training.global_vars._GLOBAL_WANDB_WRITER = wandb
