# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine

from primus.backends.transformer_engine.pytorch.permutation import _moe_chunk_sort
from primus.core.utils.patch import monkey_patch_fn as _monkey_patch_fn

PRIMUS_PATCHED_TE = False

try:
    _monkey_patch_fn(transformer_engine.pytorch.permutation._moe_chunk_sort, _moe_chunk_sort)

    PRIMUS_PATCHED_TE = True
except Exception as e:
    raise ImportError(e, "Primus TransformerEngine patch failed !")
