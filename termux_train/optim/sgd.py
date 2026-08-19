"""
termux_train.optim.sgd
======================
Stochastic Gradient Descent (SGD) with optional Momentum, Dampening, Weight Decay, and Nesterov acceleration.
"""

import copy
from typing import Iterable
from .optimizer import Optimizer
from ..nn.parameter import Parameter

class SGD(Optimizer):
    """
    Implements Stochastic Gradient Descent (with momentum and weight decay).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (must be > 0).
        momentum: Momentum factor (must be >= 0).
        dampening: Dampening for momentum (must be >= 0).
        weight_decay: Weight decay (L2 penalty) factor (must be >= 0).
        nesterov: Enables Nesterov momentum (requires momentum > 0 and dampening == 0).
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be > 0)")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum} (must be >= 0)")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening} (must be >= 0)")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} (must be >= 0)")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires a momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(self) -> None:
        """Performs a single optimization step."""
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        weight_decay = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        for idx, p in enumerate(self.params):
            if p.grad is None or not p.requires_grad:
                continue

            # Verify gradient compatibility
            if p.grad.shape != p.shape:
                raise RuntimeError(
                    f"Gradient shape {p.grad.shape} does not match parameter shape {p.shape}"
                )
            if p.grad.backend.name != p.backend.name:
                raise RuntimeError(
                    f"Gradient backend ({p.grad.backend.name}) does not match parameter backend ({p.backend.name})"
                )

            grad_data = p.grad._data

            # 1. L2 Weight Decay (applied to gradient)
            if weight_decay != 0.0:
                grad_data = p.backend.add(
                    grad_data,
                    p.backend.mul(p._data, weight_decay)
                )

            # 2. Momentum
            if momentum != 0.0:
                if idx not in self.state:
                    # First step: v_1 = g_1 (no dampening on step 1)
                    self.state[idx] = {"momentum_buffer": copy.deepcopy(grad_data)}
                    buf = self.state[idx]["momentum_buffer"]
                else:
                    buf = self.state[idx]["momentum_buffer"]
                    # v_t = mu * v_(t-1) + (1 - dampening) * g_t
                    dampened_g = p.backend.mul(grad_data, 1.0 - dampening)
                    buf = p.backend.add(p.backend.mul(buf, momentum), dampened_g)
                    self.state[idx]["momentum_buffer"] = buf

                if nesterov:
                    # update_t = g_t + mu * v_t
                    update_data = p.backend.add(grad_data, p.backend.mul(buf, momentum))
                else:
                    update_data = buf
            else:
                update_data = grad_data

            # 3. Parameter Update: theta_(t+1) = theta_t - lr * update_t
            step_delta = p.backend.mul(update_data, lr)
            p._data = p.backend.sub(p._data, step_delta)
