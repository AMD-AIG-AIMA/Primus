import dataclasses

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed

# from tests.unit_tests.test_utilities import Utils


class PrimusMoEModelTestContainer:
    """The difference between original MoEModelTestContainer and this one is whether to init the process group
    PrimusMoEModelTestContainer use MultiProcessTestCase so that it can be launched without torchrun
    """

    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        if moe_tp_size is None:
            moe_tp_size = tp_size

        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        local_expert_indices_offset = parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_enable_deepep=kwargs.get("moe_enable_deepep", False),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=self.config.moe_grouped_gemm
        )
        new_config = dataclasses.replace(self.config, **kargs)
        moe_layer = MoELayer(new_config, transformer_layer_spec.submodules.mlp.submodules).cuda()
        moe_layer.set_layer_number(0)
        return moe_layer, self.config
