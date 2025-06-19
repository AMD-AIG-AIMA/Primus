import transformer_engine  # pylint: disable=unused-import
from primus_turbo.pytorch.core.fp8 import Format
from primus_turbo.pytorch.dist import FP8AllToAll
from transformer_engine.common.recipe import DelayedScaling as TEDelayedScaling
from transformer_engine.common.recipe import Format as TEFormat
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager


def fp8_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""

    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
    if isinstance(fp8_recipe, TEDelayedScaling):
        if fp8_recipe.fp8_format == TEFormat.HYBRID:
            fp8_format = Format.HYBRID
        if fp8_recipe.fp8_format == TEFormat.E4M3:
            fp8_format = Format.E4M3
        if fp8_recipe.fp8_format == TEFormat.E5M2:
            fp8_format = Format.E5M2
    else:
        raise f"Not support recipe. Current support recipe: [`DelayedScaling`]."

    args = (group, input_, output_split_sizes_, input_split_sizes, fp8_format, None)

    return FP8AllToAll.apply(*args)
