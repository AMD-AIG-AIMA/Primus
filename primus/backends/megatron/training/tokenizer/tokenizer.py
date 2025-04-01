# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 Alibaba PAI Team.
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extra Megatron tokenizers."""

import math

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.training.tokenizer import build_tokenizer as megatron_build_tokenizer

from primus.modules.module_utils import log_rank_0


def _add_extra_tokenizer_args(parser):
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title="extra tokenizer")
    group.add_argument(
        "--extra-tokenizer-type",
        type=str,
        default=None,
        choices=["DeepSeekV2Tokenizer", "DeepSeekV3Tokenizer"],
        help="What extra type of tokenizer to use.",
    )
    return parser


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""

    # Select and instantiate the tokenizer.
    if args.extra_tokenizer_type is not None:
        log_rank_0(f"-building extra {args.extra_tokenizer_type} tokenizer...")
        if args.tokenizer_type is not None:
            log_rank_0(f"  -skip args.tokenizer_type={args.tokenizer_type}")

        if args.extra_tokenizer_type == "DeepSeekV2Tokenizer":
            tokenizer = _DeepSeekV2Tokenizer(args.tokenizer_model)
        elif args.extra_tokenizer_type == "DeepSeekV3Tokenizer":
            tokenizer = _DeepSeekV3Tokenizer(args.tokenizer_model)
        else:
            raise NotImplementedError("{} tokenizer is not " "implemented.".format(args.extra_tokenizer_type))
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


class _DeepSeekV2Tokenizer(MegatronTokenizer):
    def __init__(self, tokenizer_path, extra_vocab_size=0):
        super().__init__(tokenizer_path, extra_vocab_size)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface_tokenizer_provider"
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.extra_vocab_size = extra_vocab_size

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=None,
        max_length=None,
        truncation=None,
        add_special_tokens=None,
    ):

        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id


class _DeepSeekV3Tokenizer(MegatronTokenizer):
    def __init__(self, tokenizer_path, extra_vocab_size=0):
        super().__init__(tokenizer_path, extra_vocab_size)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface_tokenizer_provider"
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.extra_vocab_size = extra_vocab_size

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=None,
        max_length=None,
        truncation=None,
        add_special_tokens=None,
    ):

        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
