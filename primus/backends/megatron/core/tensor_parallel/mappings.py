import transformer_engine  # pylint: disable=unused-import
from primus_turbo.pytorch.core.float8 import Float8QuantConfig, Format
from primus_turbo.pytorch.dist import FP8AllToAll
from transformer_engine.common.recipe import Format as TEFormat
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager


def fp8_all_to_all(
    group, input_, output_split_sizes_=None, input_split_sizes=None, fwd_quant=False, bwd_quant=False, fwd_quant_scale=None, bwd_quant_scale=None
):
    """Wrapper for autograd function"""

    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
    if fp8_recipe.fp8_format == TEFormat.HYBRID:
        fp8_format = Format.HYBRID
    if fp8_recipe.fp8_format == TEFormat.E4M3:
        fp8_format = Format.E4M3

    config = Float8QuantConfig(format=fp8_format)
    args = (group, input_, output_split_sizes_, input_split_sizes, fwd_quant, bwd_quant, fwd_quant_scale, bwd_quant_scale, config)

    return FP8AllToAll.apply(*args)
