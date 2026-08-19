"""
termux_train.optim.adamw
========================
AdamW (Adam with Decoupled Weight Decay) Optimizer.
"""

from typing import Iterable, Tuple
from .optimizer import Optimizer
from ..nn.parameter import Parameter

class AdamW(Optimizer):
    """
    Implements AdamW (Adam with decoupled weight decay).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (must be > 0).
        betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps: Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 1e-2).
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be > 0)")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} (must be > 0)")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple/list of two floats, got {betas}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]} (must be in [0.0, 1.0))")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]} (must be in [0.0, 1.0))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} (must be >= 0)")

        defaults = dict(
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single optimization step with decoupled weight decay."""
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        for idx, p in enumerate(self.params):
            if p.grad is None or not p.requires_grad:
                continue

            if p.grad.shape != p.shape:
                raise RuntimeError(
                    f"Gradient shape {p.grad.shape} does not match parameter shape {p.shape}"
                )
            if p.grad.backend.name != p.backend.name:
                raise RuntimeError(
                    f"Gradient backend ({p.grad.backend.name}) does not match parameter backend ({p.backend.name})"
                )

            # 1. Decoupled Weight Decay: theta_t = theta_(t-1) * (1 - lr * lambda)
            if weight_decay != 0.0:
                p._data = p.backend.mul(
                    p._data,
                    1.0 - lr * weight_decay
                )

            # 2. Pure gradient (unpolluted by weight decay)
            grad_data = p.grad._data

            # 3. State Initialization
            if idx not in self.state:
                self.state[idx] = {
                    "step": 0,
                    "exp_avg": p.backend.zeros(p.shape),
                    "exp_avg_sq": p.backend.zeros(p.shape),
                }

            state = self.state[idx]
            state["step"] += 1
            step_count = state["step"]

            # 4. Update biased 1st and 2nd moment estimates
            # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            state["exp_avg"] = p.backend.add(
                p.backend.mul(state["exp_avg"], beta1),
                p.backend.mul(grad_data, 1.0 - beta1)
            )

            # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)
            grad_sq = p.backend.mul(grad_data, grad_data)
            state["exp_avg_sq"] = p.backend.add(
                p.backend.mul(state["exp_avg_sq"], beta2),
                p.backend.mul(grad_sq, 1.0 - beta2)
            )

            # 5. Bias Corrections
            bias_correction1 = 1.0 - (beta1 ** step_count)
            bias_correction2 = 1.0 - (beta2 ** step_count)

            # Corrected moments: m_hat, v_hat
            m_hat = p.backend.div(state["exp_avg"], bias_correction1)
            v_hat = p.backend.div(state["exp_avg_sq"], bias_correction2)

            # 6. Compute denominator: sqrt(v_hat) + eps
            sqrt_v = p.backend.pow(v_hat, 0.5)
            denom = p.backend.add(sqrt_v, eps)

            # 7. Parameter update: theta_t = theta_t - lr * (m_hat / denom)
            step_update = p.backend.mul(p.backend.div(m_hat, denom), lr)
            p._data = p.backend.sub(p._data, step_update)
