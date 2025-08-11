from megatron.core.fusions.fused_indices_converter import IndicesToMultihot


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert moe topk indices to multihot representation.

    This function is an experimental feature and may change in future versions.
    """
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)
