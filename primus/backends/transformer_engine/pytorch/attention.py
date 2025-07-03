###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from contextlib import nullcontext
from typing import Callable, Optional, Tuple, Union

import torch
from transformer_engine.pytorch.constants import QKVLayouts
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.softmax import _get_default_causal_mask
from transformer_engine.pytorch.utils import attention_mask_func


class ScaleMaskSoftmaxWithSinkToken(torch.nn.Module):
    """
    operation: scaling + mask + softmax

    Arguments:
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
    """

    def __init__(
        self,
        mask_func: Callable,
        softmax_in_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32

    def forward(
        self,
        inp: torch.Tensor,
        mask: torch.Tensor,
        attn_mask_type: str,
        scale: Optional[float] = None,
        sink_param: torch.Tensor = None,
    ) -> torch.Tensor:
        """ScaleMaskSoftmaxWithSinkToken fprop"""
        # [b, np, sq, sk]
        assert inp.dim() == 4
        self.input_in_fp16 = inp.dtype == torch.float16
        self.input_in_bf16 = inp.dtype == torch.bfloat16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type

        assert scale is None or self.softmax_in_fp32, "softmax should be in fp32 when scaled"

        return self.forward_torch_softmax(inp, mask, scale, sink_param)

    def forward_torch_softmax(
        self,
        inp: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[float] = None,
        sink_param: torch.Tensor = None,
    ) -> torch.Tensor:
        """Framework softmax"""
        if self.input_in_float16 and self.softmax_in_fp32:
            inp = inp.float()

        if scale is not None:
            inp = inp * scale

        batch, n_heads, seq_len_q, seq_len_k = inp.shape
        if self.attn_mask_type in ["causal", "causal_bottom_right"]:
            # seq_len_q, seq_len_k = inp.size(2), inp.size(3)
            causal_mask = _get_default_causal_mask(self.attn_mask_type, seq_len_q, seq_len_k)

            if mask is None:
                mask = causal_mask
            else:
                mask = torch.logical_or(mask, causal_mask)

        mask_output = inp
        if mask is not None and self.attn_mask_type != "no_mask":
            mask_output = self.mask_func(inp, mask)

        # [batch, n_heads, seq_len_q, 1]
        sink_param = sink_param.reshape(batch, n_heads, 1, 1).expand(-1, -1, seq_len_q, -1)
        # Softmax Stabilization via Sink Token
        # [batch, n_heads, seq_len_q, seq_len_k] -> [batch, n_heads, seq_len_q, seq_len_k + 1]
        mask_output = torch.cat([mask_output, sink_param], dim=-1)

        probs = torch.nn.Softmax(dim=-1)(mask_output)

        # [batch, n_heads, seq_len_q, seq_len_k]
        probs = probs[..., :-1]

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


class UnfusedDotProductAttentionWithSinkToken(torch.nn.Module):
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        softmax_scale: float,
        attention_type: str = "self",
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_type = attention_type
        self.attention_dropout_ctx = attention_dropout_ctx
        self.layer_number = layer_number

        self.scale_mask_softmax = ScaleMaskSoftmaxWithSinkToken(attention_mask_func)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # An FP16 training trick required for certain GPT-like models.
        self.apply_qk_layer_scaling = (
            bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and layer_number is not None
        )

        # attention
        self.num_attention_heads = None
        self.sink_params = torch.nn.ParameterList()

    def set_num_attention_heads(self, num_attention_heads):
        assert self.num_attention_heads is None
        self.num_attention_heads = num_attention_heads
        init_method = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.023)
        sink_param = torch.nn.Parameter(
            torch.empty(self.num_attention_heads, device=torch.cuda.current_device(), dtype=torch.bfloat16)
        )
        init_method(sink_param)
        self.sink_params.append(sink_param)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        cu_seqlens_kv: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        attn_mask_type: str = "causal",
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unfused attention fprop"""
        assert (
            qkv_layout in QKVLayouts
        ), f"UnfusedDotProductAttention does not support qkv_layout = {qkv_layout}!"
        qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
        if qkv_format == "bshd":
            # convert to sbhd and use sbhd implementation for now
            query_layer, key_layer, value_layer = [
                x.transpose(0, 1) for x in [query_layer, key_layer, value_layer]
            ]
        batch_size, max_seqlen_q, max_seqlen_kv = (
            query_layer.shape[1],
            query_layer.shape[0],
            key_layer.shape[0],
        )
        if "padding" in attn_mask_type:
            if self.attention_type == "self":
                assert attention_mask.shape == (
                    batch_size,
                    1,
                    1,
                    max_seqlen_q,
                ), "attention_mask should be a single tensor with [b, 1, 1, sq] shape!"
                attention_mask = torch.logical_or(attention_mask.squeeze(1).unsqueeze(3), attention_mask)
            else:
                assert (
                    len(attention_mask) == 2
                    and attention_mask[0].shape == (batch_size, 1, 1, max_seqlen_q)
                    and attention_mask[1].shape == (batch_size, 1, 1, max_seqlen_kv)
                ), (
                    "attention_mask should be a tuple of two tensors with shapes "
                    "[b, 1, 1, sq] and [b, 1, 1, skv]!"
                )
                attention_mask = torch.logical_or(
                    attention_mask[0].squeeze(1).unsqueeze(3), attention_mask[1]
                )
            mask = attention_mask.squeeze(1).logical_not()
            actual_seqlens_q = mask[:, :, 0].sum(dim=1)
            actual_seqlens_kv = mask[:, 0, :].sum(dim=1)
            mask = torch.arange(max_seqlen_q, dtype=torch.int32, device="cuda").view(
                1, 1, max_seqlen_q, 1
            ) - torch.arange(max_seqlen_kv, dtype=torch.int32, device="cuda").view(1, 1, 1, max_seqlen_kv)
            if attn_mask_type == "padding_causal":
                attention_mask = torch.logical_or(
                    torch.where(mask.view(1, 1, max_seqlen_q, max_seqlen_kv) < 0, 1, 0),
                    attention_mask,
                )
            if attn_mask_type == "padding_causal_bottom_right":
                attention_mask = torch.logical_or(
                    torch.where(
                        mask.expand(batch_size, 1, max_seqlen_q, max_seqlen_kv)
                        + (actual_seqlens_kv - actual_seqlens_q).view(batch_size, 1, 1, 1)
                        < 0,
                        1,
                        0,
                    ),
                    attention_mask,
                )

        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]
        apply_qk_layer_scaling = self.apply_qk_layer_scaling and key_layer.dtype == torch.float16

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        if key_layer.shape[2] != query_layer.shape[2]:
            assert (
                query_layer.shape[2] % key_layer.shape[2] == 0
            ), "The number of attention heads must be divisible by the number of GQA groups!"
            key_layer = key_layer.repeat_interleave(int(query_layer.shape[2] / key_layer.shape[2]), dim=2)
            value_layer = value_layer.repeat_interleave(
                int(query_layer.shape[2] / value_layer.shape[2]), dim=2
            )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        # WAR to set dtype to FP32 as ONNX lacks BF16 support for ConstantOfShape operator
        is_bf16 = query_layer.dtype == torch.bfloat16
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=torch.float32 if is_in_onnx_export_mode() and is_bf16 else query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        if is_in_onnx_export_mode() and is_bf16:
            matmul_result = matmul_result.bfloat16()

        scale = self.softmax_scale
        if apply_qk_layer_scaling:
            scale /= self.layer_number

        # Raw attention scores. [b * np, sq, sk]
        if core_attention_bias_type == "no_bias":
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            ).view(*output_size)

        elif core_attention_bias_type == "pre_scale_bias":
            assert core_attention_bias is not None, "core_attention_bias should not be None!"
            matmul_result = torch.bmm(
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            )
            matmul_result = matmul_result.view(*output_size) + core_attention_bias
            matmul_result *= scale

        elif core_attention_bias_type in ["post_scale_bias", "alibi"]:
            if core_attention_bias_type == "post_scale_bias":
                assert core_attention_bias is not None, "core_attention_bias should not be None!"
            if core_attention_bias_type == "alibi":
                _, core_attention_bias = get_alibi(
                    output_size[1],
                    output_size[2],
                    output_size[3],
                    actual_seqlens_q=actual_seqlens_q if "padding" in attn_mask_type else None,
                    actual_seqlens_kv=actual_seqlens_kv if "padding" in attn_mask_type else None,
                    alibi_slopes=alibi_slopes,
                    bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                )
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=scale,
            )
            matmul_result = (matmul_result.view(*output_size) + core_attention_bias).to(
                dtype=query_layer.dtype
            )

        # attention scores and attention mask [b, np, sq, sk]
        softmax_scale = self.layer_number if apply_qk_layer_scaling else None
        assert self.num_attention_heads is not None
        attention_probs = self.scale_mask_softmax(
            matmul_result,
            attention_mask,
            attn_mask_type,
            softmax_scale,
            self.sink_params[0],
        )

        # mask out the pad positions in softmax results, mostly for the rows (pad tokens from q)
        # the columns (pad tokens from k) are already zeroed out during softmax
        if "padding" in attn_mask_type:
            attention_probs = attention_probs.masked_fill(attention_mask, 0)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        if qkv_format == "sbhd":
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.view(seqlen, batch_size, -1)

        if qkv_format == "bshd":
            # [b, np, sq, hn] --> [b, sq, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            # [b, sq, np, hn] --> [b, sq, hp]
            context_layer = context_layer.view(batch_size, seqlen, -1)

        return context_layer
