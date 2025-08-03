from megatron.core.transformer.moe.fused_a2a import FusedCombine, FusedDispatch


def fused_dispatch(x, token_indices, token_probs, num_experts, group, previous_event=None):
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event

    Returns:
        Result of FusedDispatch
    """
    return FusedDispatch.apply(x.contiguous(), token_indices, token_probs, num_experts, group, previous_event)


def fused_combine(x, group, handle, previous_event=None):
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event

    Returns:
        Result of FusedCombine
    """
    return FusedCombine.apply(x, group, handle, previous_event)
