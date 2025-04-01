# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

from megatron.core.transformer.moe import moe_layer

from primus.backends.megatron.core.transformer.moe.router import BalancedTopKRouter
from primus.core.utils.patch import monkey_patch_fn as _monkey_patch_fn

PRIMUS_PATCHED_MEGATRON = False

try:
    _monkey_patch_fn(moe_layer.TopKRouter, BalancedTopKRouter)

    PRIMUS_PATCHED_MEGATRON = True
except Exception as e:
    raise ImportError(e, "Primus megatron patch failed !")
