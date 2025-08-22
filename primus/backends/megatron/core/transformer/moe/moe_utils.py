import torch


class RandomSTE(torch.autograd.Function):
    """
    Straight-Through Estimator(STE) function that returns random values
    with different seed for each rank.

    This is used to generate random logits of router for load-balanced benchmark.
    """

    generator = None

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass returns random logits with rank-specific seed.
        """
        if RandomSTE.generator is None:
            global_rank = torch.distributed.get_rank()
            base_seed = 42
            seed = base_seed + global_rank
            RandomSTE.generator = torch.Generator(device=logits.device)
            RandomSTE.generator.manual_seed(seed)

        random_logits = logits.clone().normal_(generator=RandomSTE.generator)
        return random_logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass propagates the gradient for logits.
        """
        return grad_output


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)
