###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extra Megatron tokenizers."""

import math
from typing import List

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.arguments import (
    _add_tokenizer_args as megatron_add_tokenizer_args,
)
from megatron.training.tokenizer import build_tokenizer as megatron_build_tokenizer
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

from primus.modules.module_utils import log_rank_0

CUSTOM_TOKENIZER_TYPES = {
    "DeepSeekV2Tokenizer",
    "DeepSeekV3Tokenizer",
    "Llama2Tokenizer",
    "Llama3Tokenizer",
    "MixtralTokenizer",
    "OriginalTikTokenizer",
}


def _add_tokenizer_args(parser):
    parser = megatron_add_tokenizer_args(parser)
    tokenizer_arg = next(action for action in parser._actions if action.dest == "tokenizer_type")
    custom_choices = [t for t in CUSTOM_TOKENIZER_TYPES]
    tokenizer_arg.choices = list(set(tokenizer_arg.choices).union(custom_choices))
    return parser


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""

    log_rank_0(f"-building {args.tokenizer_type} tokenizer...")

    # Select and instantiate the tokenizer.
    if args.tokenizer_type in CUSTOM_TOKENIZER_TYPES:
        if args.tokenizer_type == "OriginalTikTokenizer":
            tokenizer = OriginalTikTokenizer(args.tokenizer_model)
        else:
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


class OriginalTikTokenizer(MegatronTokenizer):
    """Original tiktoken tokenizer."""

    def __init__(self, tiktoken_name: str):
        super().__init__(tiktoken_name)

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for OriginalTikTokenizer. Install it with: pip install tiktoken"
            )

        from megatron.training.utils import print_rank_0  # To prevent circular import.

        # Load the  encoding directly from tiktoken
        # If path is a model name, load it directly
        self._encoding = tiktoken.get_encoding(tiktoken_name)

        # Set up vocabulary mappings
        self._vocab_size = self._encoding.n_vocab

        # Create token-to-ID and ID-to-token mappings
        self._token_to_id = {}
        self._id_to_token = {}

        # Build vocabulary mappings by iterating through all token IDs
        for token_id in range(self._vocab_size):
            try:
                token_bytes = self._encoding.decode_single_token_bytes(token_id)
                token_str = token_bytes.decode("utf-8", errors="replace")
                self._token_to_id[token_str] = token_id
                self._id_to_token[token_id] = token_str
            except Exception:
                # For tokens that can't be decoded properly, use a placeholder
                placeholder = f"<token_{token_id}>"
                self._token_to_id[placeholder] = token_id
                self._id_to_token[token_id] = placeholder

        # Set special token IDs based on o200k_base
        self._bos_id = (
            self._encoding.encode("<|begin_of_text|>")[0]
            if "<|begin_of_text|>" in str(self._encoding.special_tokens_set)
            else None
        )
        self._eos_id = self._encoding.eot_token  # End of text token
        self._eod_id = self._eos_id

        print_rank_0(f"Tokenizer initialized with vocab size: {self._vocab_size}")
        print_rank_0(f"EOT token ID: {self._eos_id}")

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self._token_to_id

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self._id_to_token

    @property
    def decoder(self):
        return self._id_to_token

    @property
    def encoder(self):
        return self._token_to_id

    def tokenize(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Tokenize text into token IDs."""
        # Use tiktoken's encode method
        tokens = self._encoding.encode(text)

        # Add BOS/EOS tokens if requested
        if bos and self._bos_id is not None:
            tokens = [self._bos_id] + tokens
        if eos and self._eos_id is not None:
            tokens = tokens + [self._eos_id]

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self._encoding.decode(token_ids)

    def offsets(self, ids: List[int], text: str) -> List[int]:
        """Get character offsets for tokens."""
        # For o200k_base, we'll use a simple approach
        # This is a simplified implementation
        decoded_tokens = []
        offsets = []
        current_offset = 0

        for token_id in ids:
            try:
                token_text = self._encoding.decode([token_id])
                decoded_tokens.append(token_text)
                offsets.append(current_offset)
                current_offset += len(token_text)
            except Exception:
                offsets.append(current_offset)

        return offsets

    @property
    def bos(self):
        """Beginning of sequence token ID."""
        return self._bos_id if self._bos_id is not None else -1

    @property
    def eos(self):
        """End of sequence token ID."""
        return self._eos_id

    @property
    def eod(self):
        """End of document token ID."""
        return self._eod_id

    @property
    def cls(self):
        """Classification token ID (not used in o200k_base)."""
        return -1

    @property
    def sep(self):
        """Separator token ID (not used in o200k_base)."""
        return -1

    @property
    def pad(self):
        """Padding token ID (not used in o200k_base)."""
        return -1

    @property
    def mask(self):
        """Mask token ID (not used in o200k_base)."""
        return -1

    @property
    def additional_special_tokens_ids(self):
        """Additional special tokens (not used in o200k_base)."""
        return None
