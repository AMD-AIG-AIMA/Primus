###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""Extra Megatron tokenizers."""

import math

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.arguments import _add_tokenizer_args as megatron_add_tokenizer_args
from megatron.training.tokenizer import build_tokenizer as megatron_build_tokenizer
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer 

from primus.modules.module_utils import log_rank_0


def _add_tokenizer_args(parser):
    return megatron_add_tokenizer_args(parser)

def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""

    log_rank_0(f"-building {args.tokenizer_type} tokenizer...")

    # Select and instantiate the tokenizer.
    if args.tokenizer_type in {
        "DeepSeekV2Tokenizer",
        "DeepSeekV3Tokenizer",
        "Llama2Tokenizer",
        "Llama3Tokenizer",
    }:
        tokenizer = _HuggingFaceTokenizer(args.tokenizer_model)
    else:
        return megatron_build_tokenizer(args, **kwargs)

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after
