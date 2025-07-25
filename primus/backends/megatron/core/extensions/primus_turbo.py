###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Callable, List, Optional

import primus_turbo.pytorch as pt
import torch
import transformer_engine as te
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, condition_init_method
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from megatron.training.global_vars import get_args
from torch import Tensor


class PrimusTurboAttention(te.pytorch.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.

    Note that if Megatron's parallel_state has not been initialized yet, the
    tp_group and cp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group() and set_context_parallel_group().
    """

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

        args = get_args()
        if args.enable_turbo_attention_float8:
            self.attn = pt.ops.attention_fp8_blockwise
            self.attention_backend = "triton"
        else:
            self.attn = pt.ops.attention
            self.attention_backend = "ck"
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
        self.cp_param_bundle = None
        if self.config.context_parallel_size > 1:
            self.cp_param_bundle = {"cp_group": model_comm_pgs.cp, "cp_comm_type": cp_comm_type}

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
        assert qkv_format in ("sbhd", "bhsd"), "qkv_format only support bshd, but got {qkv_format}"
        if qkv_format == "sbhd":
            query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
        mask_type = attn_mask_type.name
        if mask_type == AttnMaskType.causal.name:
            causal = True
        elif mask_type == AttnMaskType.no_mask.name:
            causal = False
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")

        o = self.attn(
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
            backend_type=self.attention_backend,
            cp_param_bundle=self.cp_param_bundle,
        )

        o = o.reshape(o.shape[0], o.shape[1], -1).transpose(0, 1)
        if not o.is_contiguous():
            o = o.contiguous()
        return o


class PrimusTurboRowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not input_is_parallel:
            raise ValueError("Transformer Engine linear layers do not support input_is_parallel = False")

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        args = get_args()
        if args.enable_turbo_gemm_float8:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None, config=None: pt.ops.gemm_fp8_blockwise(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype, config=config
            )
        else:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None: pt.ops.gemm(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {"weight": 1}, sharded_offsets)

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(
        self,
        inp: torch.Tensor,
    ):
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)  # or set weights = self._parameters['weight']
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = inp.size()
        if not inp.is_contiguous():
            inp = inp.contiguous()
        inp = inp.view(-1, original_shape[-1])
        out = self.gemm(inp, weights)
        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
        return out, None


class PrimusTurboColumnParallelLinear(TELinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        args = get_args()
        if args.enable_turbo_gemm_float8:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None, config=None: pt.ops.gemm_fp8_blockwise(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype, config=config
            )
        else:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None: pt.ops.gemm(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(
        self,
        inp: torch.Tensor,
    ):
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)  # or set weights = self._parameters['weight']
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = inp.size()
        if not inp.is_contiguous():
            inp = inp.contiguous()
        inp = inp.view(-1, original_shape[-1])
        out = self.gemm(inp, weights)
        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
        return out, None


class PrimusTurboColumnParallelLinearTorch(ColumnParallelLinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):

        args = get_args()
        if args.enable_turbo_gemm_float8:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None, config=None: pt.ops.gemm_fp8_blockwise(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype, config=config
            )
        else:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None: pt.ops.gemm(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype
            )

        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
            tp_group=tp_group,
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        if weight is None:
            weight = self.weight
        bias_tensor = self.bias if not self.skip_bias_add else None

        original_shape = input_.size()
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        input_ = input_.view(-1, original_shape[-1])
        out = self.gemm(input_, weight)
        out = out.view(original_shape[0], original_shape[1], -1)

        return out, bias_tensor


class PrimusTurboLayerNormColumnParallelLinear(te.pytorch.LayerNormLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.config = config

        if gather_output:
            raise ValueError("Primus Turbo linear layers do not support gather_output = True")

        if is_expert:
            raise ValueError("Primus Turbo linear layers do not yet support MoE")

        if skip_weight_param_allocation:
            raise ValueError("Primus Turbo linear layers do not support skip_weight_param_allocation")

        # TODO: For backward compatibility, remove in v0.15.
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias

        args = get_args()
        if args.enable_turbo_gemm_float8:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None, config=None: pt.ops.gemm_fp8_blockwise(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype, config=config
            )
        else:
            self.gemm = lambda a, b, transA=False, transB=True, out_dtype=None: pt.ops.gemm(
                a, b, transA=transA, transB=transB, out_dtype=out_dtype
            )

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=None,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            normalization=self.config.normalization,
            return_bias=self.te_return_bias,
            parallel_mode="column",
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )

    def forward(self, x):
        """Forward."""

        if self.config.normalization == "LayerNorm":
            norm_out = torch.nn.functional.layer_norm(
                x, [x.size(-1)], self.layer_norm_weight, self.layer_norm_bias, self.eps
            )
        elif self.config.normalization == "RMSNorm":
            norm_out = torch.nn.functional.rms_norm(x, [x.size(-1)], self.layer_norm_weight, self.eps)
        # weights = [getattr(self, name) for name in self.weight_names]
        # weights = torch.cat(weights, dim=0)
        weights = self._parameters["weight"]
        if self.use_bias:
            bias_tensor = torch.cat([getattr(self, name) for name in self.bias_names])
        original_shape = x.size()
        if not norm_out.is_contiguous():
            norm_out = norm_out.contiguous()
        inp = norm_out.view(-1, original_shape[-1])
        out = self.gemm(inp, weights)
        out = out.view(original_shape[0], original_shape[1], -1)
        if self.te_return_bias:
            return out, bias_tensor
        if self.use_bias:
            return out + bias_tensor, None
        return out, None


class PrimusTurboGroupedMLP(GroupedMLP):
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super().__init__(
            num_local_experts,
            config,
            model_comm_pgs,
        )

        self.weight1 = torch.nn.Parameter(
            self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            .transpose(1, 2)
            .contiguous()
        )
        self.weight2 = torch.nn.Parameter(
            self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
            .transpose(1, 2)
            .contiguous()
        )

        # primus turbo only support grouped gemm fp8 for now
        self.group_gemm = pt.ops.grouped_gemm_fp8_blockwise

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1
            w2 = self.weight2
            tokens_per_expert = tokens_per_expert.cuda()
            assert w1.is_contiguous(), "w1 must be contiguous"
            assert w2.is_contiguous(), "w2 must be contiguous"
            fc1_output = self.group_gemm(permuted_local_hidden_states, w1, tokens_per_expert)
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = self.group_gemm(intermediate_parallel, w2, tokens_per_expert)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = self.group_gemm(intermediate_parallel, w2, tokens_per_expert)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.transpose(1, 2).contiguous().view(self.config.hidden_size, -1)
            w2 = self.weight2.transpose(1, 2).contiguous().view(self.config.hidden_size, -1)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None
