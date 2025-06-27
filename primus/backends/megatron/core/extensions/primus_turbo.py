###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Optional

import primus_turbo.pytorch as pt
import torch
import transformer_engine as te
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


class PrimusTurboAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        self.config = config
        self.qkv_format: str = "sbhd"
        self.softmax_scale = softmax_scale

        if model_comm_pgs is None:
            # For backward compatibility, remove in v0.14 and raise error
            # raise ValueError("TEDotProductAttention was called without ModelCommProcessGroups")
            model_comm_pgs = ModelCommProcessGroups(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(model_comm_pgs, "tp"), "TEDotProductAttention model_comm_pgs must have tp pg"
            assert hasattr(model_comm_pgs, "cp"), "TEDotProductAttention model_comm_pgs must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    model_comm_pgs, "hcp"
                ), "TEDotProductAttention model_comm_pgs must have hierarchical cp pg"

        # todo: cp
        if self.config.context_parallel_size > 1:
            pass

        assert config.window_size is None, "primus_turbo does not support sliding window attention"
        # Check version

        kv_channels = (
            (k_channels, v_channels)
            if k_channels is not None and v_channels is not None
            else self.config.kv_channels
        )

        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            num_gqa_groups=self.config.num_query_groups,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            qkv_format="sbhd",
            attn_mask_type=attn_mask_type.name,
            window_size=None,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=None,
            tp_group=model_comm_pgs.tp,
            layer_number=layer_number,
            attention_type=attention_type,
            # cp is not support
            softmax_scale=softmax_scale,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )

        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        assert qkv_format == "sbhd", "qkv_format only support bshd, but got {qkv_format}"
        if qkv_format == "sbhd":
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
        mask_type = attn_mask_type.name
        if mask_type == AttnMaskType.causal.name:
            causal = True
        elif mask_type == AttnMaskType.no_mask.name:
            causal = False
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")
        if self.softmax_scale is None:
            self.softmax_scale = query.shape[-1] ** (-0.5)
        o = pt.ops.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type="ck",
        )
        o = o.reshape(o.shape[0], o.shape[1], -1)
        o = o.transpose(0, 1)
        return o
