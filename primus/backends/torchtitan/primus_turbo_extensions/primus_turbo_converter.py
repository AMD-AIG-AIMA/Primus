import torch
from primus_turbo.pytorch.modules import CoreAttention  # TODO: import Check
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)


def replace_turbo_attention_modules(model: torch.nn.Module):
    for name, module in model.named_children():
        if isinstance(module, (FlexAttention, ScaledDotProductAttention)):
            setattr(model, name, CoreAttention(causal=True))
        else:
            replace_turbo_attention_modules(module)


class PrimusTubroConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = True

    def convert(self, model: torch.nn.Module):
        if self.enabled == False:
            return

        replace_turbo_attention_modules(model)
        return model

    def post_optimizer_hook(self, model: torch.nn.Module | list[torch.nn.Module]):
        return


register_model_converter(PrimusTubroConverter, "primus_turbo")
