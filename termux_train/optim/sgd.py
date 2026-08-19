"""
termux_train.optim.sgd
======================
Stochastic Gradient Descent (SGD) with optional Momentum, Dampening, Weight Decay, and Nesterov acceleration.
Fully hardened with real-number type validation, fail-fast finite checks, and atomic state schemas.
"""

import copy
from typing import Iterable, Dict, Any
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
        raw_defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, raw_defaults)

    def _validate_and_normalize_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes SGD hyperparameters."""
        if "lr" not in defaults:
            raise ValueError("SGD defaults missing 'lr'")
        lr = self._validate_real("lr", defaults["lr"], strictly_positive=True)

        momentum = self._validate_real("momentum", defaults.get("momentum", 0.0), allow_negative=False)
        dampening = self._validate_real("dampening", defaults.get("dampening", 0.0), allow_negative=False)
        weight_decay = self._validate_real("weight_decay", defaults.get("weight_decay", 0.0), allow_negative=False)

        nesterov = defaults.get("nesterov", False)
        if not isinstance(nesterov, bool):
            raise TypeError(f"nesterov must be a bool, got {type(nesterov).__name__}")

        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")

        return dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def _validate_state_entry(self, index: int, param: Parameter, state_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and restores SGD parameter state entry."""
        if self.defaults.get("momentum", 0.0) > 0.0:
            if "momentum_buffer" not in state_entry:
                raise ValueError(f"State entry for parameter {index} missing 'momentum_buffer'")

        restored = {}
        if "momentum_buffer" in state_entry:
            buf_raw = state_entry["momentum_buffer"]
            if not isinstance(buf_raw, list):
                raise TypeError(f"momentum_buffer for parameter {index} must be a list structure")
            buf_data = param.backend.from_data(copy.deepcopy(buf_raw))
            if tuple(param.backend.get_shape(buf_data)) != tuple(param.shape):
                raise RuntimeError(
                    f"Shape mismatch for momentum_buffer parameter {index}: expected {param.shape}, got {param.backend.get_shape(buf_data)}"
                )
            self._assert_all_finite(param, buf_data, f"momentum_buffer for parameter {index}")
            restored["momentum_buffer"] = buf_data
        return restored

    def step(self) -> None:
        """Performs a single optimization step using a two-phase transactional commit (Policy B)."""
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        weight_decay = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        # Phase 1: Validation and candidate calculation
        candidates = []
        for idx, p in enumerate(self.params):
            if not self._validate_param_and_grad(idx, p):
                continue

            # Fail-fast check on initial parameter & gradient
            self._assert_all_finite(p, p._data, f"parameter {idx}")
            self._assert_all_finite(p, p.grad._data, f"gradient for parameter {idx}")

            grad_data = p.grad._data

            # 1. L2 Weight Decay (applied to gradient)
            if weight_decay != 0.0:
                grad_data = p.backend.add(
                    grad_data,
                    p.backend.mul(p._data, weight_decay)
                )
                self._assert_all_finite(p, grad_data, f"weight decayed gradient for parameter {idx}")

            # 2. Momentum
            new_state = None
            if momentum != 0.0:
                if idx not in self.state:
                    # First step: v_1 = g_1 (no dampening on step 1)
                    buf = self._clone_backend_data(p, grad_data)
                    new_state = {"momentum_buffer": buf}
                else:
                    buf = self.state[idx]["momentum_buffer"]
                    self._assert_all_finite(p, buf, f"previous momentum_buffer for parameter {idx}")

                    # v_t = mu * v_(t-1) + (1 - dampening) * g_t
                    dampened_g = p.backend.mul(grad_data, 1.0 - dampening)
                    buf = p.backend.add(p.backend.mul(buf, momentum), dampened_g)
                    self._assert_all_finite(p, buf, f"new momentum_buffer for parameter {idx}")
                    new_state = {"momentum_buffer": buf}

                if nesterov:
                    # update_t = g_t + mu * v_t
                    update_data = p.backend.add(grad_data, p.backend.mul(buf, momentum))
                else:
                    update_data = buf
            else:
                update_data = grad_data

            self._assert_all_finite(p, update_data, f"update delta for parameter {idx}")

            # 3. Parameter Update: theta_(t+1) = theta_t - lr * update_t
            step_delta = p.backend.mul(update_data, lr)
            new_param_data = p.backend.sub(p._data, step_delta)
            self._assert_all_finite(p, new_param_data, f"new parameter data for parameter {idx}")

            candidates.append((idx, p, new_param_data, new_state))

        # Phase 2: Atomic commit across all parameters and states
        for idx, p, new_data, new_state in candidates:
            p._data = new_data
            if new_state is not None:
                self.state[idx] = new_state
