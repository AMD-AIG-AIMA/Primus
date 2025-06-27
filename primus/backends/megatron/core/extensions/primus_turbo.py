###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Callable, Optional

import primus_turbo.pytorch as pt
import torch
import transformer_engine as te
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
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


class PrimusRowParallelTurboLinear(TELinear):
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

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=None,
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
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ):  # we use fp8_kernel by default. input & output is bf16
        weights = [getattr(self, name) for name in self.weight_names]
        weights = torch.cat(weights, dim=0)
        out = pt.ops.gemm_fp8_blockwise(inp, weights)
        return out


if __name__ == "__main__":
    from megatron.core.utils import init_method_normal

    # test the accuracy of Turbo and TE
    transformer_config = TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_comm_backend=None,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
        context_parallel_size=1,
        hierarchical_context_parallel_sizes=None,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        moe_extended_tp=False,
        perform_initialization=True,
        use_cpu_initialization=None,
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        timers=None,
        finalize_model_grads_func=None,
        grad_scale_func=None,
        no_sync_func=None,
        grad_sync_func=None,
        param_sync_func=None,
        deterministic_mode=False,
        enable_autocast=False,
        autocast_dtype=torch.bfloat16,
        num_microbatches_with_partial_activation_checkpoints=None,
        gradient_accumulation_fusion=True,
        async_tensor_model_parallel_allreduce=True,
        use_te_rng_tracker=False,
        tp_comm_overlap=False,
        tp_comm_bulk_wgrad=True,
        tp_comm_bulk_dgrad=True,
        tp_comm_overlap_ag=True,
        tp_comm_overlap_rs=True,
        tp_comm_overlap_rs_dgrad=False,
        tp_comm_split_ag=True,
        tp_comm_atomic_ag=False,
        tp_comm_split_rs=True,
        tp_comm_atomic_rs=False,
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        tp_comm_overlap_disable_qkv=False,
        tp_comm_overlap_disable_fc1=False,
        tp_comm_bootstrap_backend="nccl",
        pipeline_dtype=torch.bfloat16,
        variable_seq_lengths=False,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
        batch_p2p_sync=True,
        use_ring_exchange_p2p=False,
        deallocate_pipeline_outputs=True,
        defer_embedding_wgrad_compute=False,
        wgrad_deferral_limit=0,
        pipeline_model_parallel_split_rank=None,
        overlap_p2p_comm_warmup_flush=False,
        microbatch_group_size_per_vp_stage=1,
        cpu_offloading=False,
        cpu_offloading_num_layers=0,
        _cpu_offloading_context=None,
        cpu_offloading_activations=True,
        cpu_offloading_weights=True,
        barrier_with_L1_time=True,
        num_layers=32,
        mtp_num_layers=None,
        mtp_loss_scaling_factor=0.1,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        hidden_size=4096,
        num_attention_heads=32,
        attention_backend="auto",
        softmax_scale=None,
        num_query_groups=8,
        ffn_hidden_size=14336,
        kv_channels=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        fp32_residual_connection=False,
        apply_residual_connection_post_layernorm=False,
        layernorm_epsilon=1e-06,
        layernorm_zero_centered_gamma=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        activation_func_fp8_input_store=False,
        num_moe_experts=None,
        rotary_interleaved=False,
        window_size=None,
        normalization="RMSNorm",
        qk_layernorm=False,
        test_mode=False,
        calculate_per_token_loss=False,
        multi_latent_attention=False,
        no_rope_freq=None,
        init_method=init_method_normal(0.008),
        output_layer_init_method=init_method_normal(0.001),
        init_method_std=0.008,
        init_model_with_meta_device=False,
        apply_query_key_layer_scaling=False,
        attention_softmax_in_fp32=True,
        disable_bf16_reduced_precision_matmul=False,
        bias_activation_fusion=True,
        masked_softmax_fusion=False,
        persist_layer_norm=True,
        memory_efficient_layer_norm=False,
        bias_dropout_fusion=True,
        apply_rope_fusion=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        distribute_saved_activations=False,
        recompute_modules=["core_attn"],
        fp8=None,
        fp8_recipe="delayed",
        fp8_param=False,
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_wgrad=True,
        fp8_dot_product_attention=False,
        fp8_multi_head_attention=False,
        tp_only_amax_red=False,
        first_last_layers_bf16=False,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_overlap=False,
        moe_layer_freq=1,
        moe_ffn_hidden_size=None,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=2,
        moe_router_topk_limited_devices=None,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=None,
        moe_router_score_function="softmax",
        moe_router_dtype=None,
        moe_router_enable_expert_bias=False,
        moe_router_bias_update_rate=0.001,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=None,
        moe_input_jitter_eps=None,
        moe_token_dropping=False,
        moe_token_dispatcher_type="allgather",
        moe_enable_deepep=False,
        moe_per_layer_logging=False,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_token_drop_policy="probs",
        moe_layer_recompute=False,
        moe_permute_fusion=False,
        moe_apply_probs_on_input=False,
        cp_comm_type="p2p",
        enable_cuda_graph=False,
        cuda_graph_use_single_mempool=False,
        cuda_graph_retain_backward_graph=False,
        cuda_graph_warmup_steps=3,
        external_cuda_graph=False,
        cuda_graph_scope="full",
        clone_scatter_output_in_embedding=True,
        disable_parameter_transpose_cache=False,
        config_logger_dir="",
        flash_decode=False,
        inference_rng_tracker=False,
        symmetric_ar_type=None,
        mrope_section=None,
        is_hybrid_model=False,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=None,
        use_mamba_mem_eff_path=True,
        mlp_chunks_for_prefill=1,
        heterogeneous_block_specs=False,
    )
