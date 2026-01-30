"""Optimizer implementations."""

import math
import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Parameter update
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        it: Current iteration
        max_learning_rate: Maximum learning rate (after warmup)
        min_learning_rate: Minimum learning rate (after cosine decay)
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Number of iterations for one cosine cycle

    Returns:
        Learning rate for the current iteration
    """
    if it < warmup_iters:
        # Linear warmup
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        # Cosine decay
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
    else:
        # After cycle, return min
        return min_learning_rate
