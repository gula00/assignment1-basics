"""Neural network utility functions."""

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy loss.

    Args:
        inputs: (batch_size, vocab_size) unnormalized logits
        targets: (batch_size,) target class indices

    Returns:
        Scalar average loss
    """
    log_softmax = inputs - inputs.logsumexp(dim=-1, keepdim=True)
    loss = -log_softmax[torch.arange(inputs.size(0), device=inputs.device), targets]
    return loss.mean()


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: Iterable of parameters with gradients
        max_l2_norm: Maximum l2 norm for gradients
        eps: Small value for numerical stability (default: 1e-6)
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
