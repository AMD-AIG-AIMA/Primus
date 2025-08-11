from transformer_engine.common.recipe import Format as TEFormat
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

try:
    from primus_turbo.pytorch.core.float8 import Float8QuantConfig, Format
    from primus_turbo.pytorch.dist import FP8AllToAll

    HAVE_PRIMUS_TURBO = True
except ImportError:
    HAVE_PRIMUS_TURBO = False


def fp8_all_to_all(
    group, input_, output_split_sizes_=None, input_split_sizes=None, fwd_quant=False, bwd_quant=False
):
    """Wrapper for autograd function"""
    if not HAVE_PRIMUS_TURBO:
        raise ValueError("Failed to import 'primus_turbo'. Please make sure it is installed.")

    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
    if fp8_recipe.fp8_format == TEFormat.HYBRID:
        fp8_format = Format.HYBRID
    if fp8_recipe.fp8_format == TEFormat.E4M3:
        fp8_format = Format.E4M3

    config = Float8QuantConfig(format=fp8_format)
    args = (group, input_, output_split_sizes_, input_split_sizes, fwd_quant, bwd_quant, config)

    return FP8AllToAll.apply(*args)
