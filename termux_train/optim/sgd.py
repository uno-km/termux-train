"""
termux_train.optim.sgd
======================
Stochastic Gradient Descent (SGD) with optional Momentum, Dampening, Weight Decay, and Nesterov acceleration.
Fully hardened with real-number type validation, fail-fast finite checks, strict state schema enforcement,
and two-phase full-step transactional commits (Policy B).
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

    def _validate_state_entry(
        self,
        index: int,
        param: Parameter,
        state_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validates and restores SGD parameter state schema strictly."""
        allowed_keys = {"momentum_buffer"}
        unexpected_keys = set(state_entry) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected SGD state fields for parameter {index}: {sorted(unexpected_keys)}"
            )

        momentum = self.defaults["momentum"]

        if momentum > 0.0:
            if "momentum_buffer" not in state_entry:
                raise ValueError(
                    f"State entry for parameter {index} missing 'momentum_buffer'"
                )
        elif "momentum_buffer" in state_entry:
            raise ValueError(
                f"State entry for parameter {index} contains 'momentum_buffer' while momentum is disabled"
            )

        if "momentum_buffer" not in state_entry:
            return {}

        buffer_raw = state_entry["momentum_buffer"]

        if not isinstance(buffer_raw, list):
            raise TypeError(
                f"momentum_buffer for parameter {index} must be a list structure"
            )

        buffer_data = param.backend.from_data(
            copy.deepcopy(buffer_raw)
        )
        buffer_shape = tuple(
            param.backend.get_shape(buffer_data)
        )

        if buffer_shape != tuple(param.shape):
            raise RuntimeError(
                f"Shape mismatch for momentum_buffer parameter {index}: expected {param.shape}, got {buffer_shape}"
            )

        self._assert_all_finite(
            param,
            buffer_data,
            f"momentum_buffer for parameter {index}",
        )

        return {
            "momentum_buffer": buffer_data,
        }

    def step(self) -> None:
        """Performs a single optimization step using a two-phase transactional commit (Policy B)."""
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        weight_decay = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        update_mask = self._validate_all_params_and_grads()
        pending_state = self._clone_state()
        pending_param_data = {}

        for index, param in enumerate(self.params):
            if not update_mask[index]:
                continue

            grad_data = param.grad._data

            if weight_decay != 0.0:
                grad_data = param.backend.add(
                    grad_data,
                    param.backend.mul(
                        param._data,
                        weight_decay,
                    ),
                )

                self._assert_all_finite(
                    param,
                    grad_data,
                    f"weight decayed gradient for parameter {index}",
                )

            if momentum != 0.0:
                if index not in pending_state:
                    buffer_data = self._clone_backend_data(
                        param,
                        grad_data,
                    )
                    pending_state[index] = {
                        "momentum_buffer": buffer_data,
                    }
                else:
                    buffer_data = pending_state[index][
                        "momentum_buffer"
                    ]

                    self._assert_all_finite(
                        param,
                        buffer_data,
                        f"previous momentum_buffer for parameter {index}",
                    )

                    dampened_gradient = param.backend.mul(
                        grad_data,
                        1.0 - dampening,
                    )
                    buffer_data = param.backend.add(
                        param.backend.mul(
                            buffer_data,
                            momentum,
                        ),
                        dampened_gradient,
                    )

                    self._assert_all_finite(
                        param,
                        buffer_data,
                        f"new momentum_buffer for parameter {index}",
                    )

                    pending_state[index][
                        "momentum_buffer"
                    ] = buffer_data

                if nesterov:
                    update_data = param.backend.add(
                        grad_data,
                        param.backend.mul(
                            buffer_data,
                            momentum,
                        ),
                    )
                else:
                    update_data = buffer_data
            else:
                update_data = grad_data

            self._assert_all_finite(
                param,
                update_data,
                f"update for parameter {index}",
            )

            step_delta = param.backend.mul(
                update_data,
                lr,
            )
            new_param_data = param.backend.sub(
                param._data,
                step_delta,
            )

            self._assert_all_finite(
                param,
                new_param_data,
                f"new parameter data for parameter {index}",
            )

            pending_param_data[index] = new_param_data

        self._commit_step(
            pending_param_data,
            pending_state,
        )

        return None
