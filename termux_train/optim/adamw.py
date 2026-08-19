"""
termux_train.optim.adamw
========================
AdamW (Adam with Decoupled Weight Decay) Optimizer.
Fully hardened with real-number type validation, fail-fast finite checks, atomic state schemas,
and non-destructive decoupled decay calculations.
"""

import copy
from typing import Iterable, Tuple, Dict, Any
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
        raw_defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, raw_defaults)

    def _validate_and_normalize_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and normalizes AdamW hyperparameters."""
        if "lr" not in defaults:
            raise ValueError("AdamW defaults missing 'lr'")
        lr = self._validate_real("lr", defaults["lr"], strictly_positive=True)

        if "eps" not in defaults:
            raise ValueError("AdamW defaults missing 'eps'")
        eps = self._validate_real("eps", defaults["eps"], strictly_positive=True)

        weight_decay = self._validate_real("weight_decay", defaults.get("weight_decay", 1e-2), allow_negative=False)

        betas = defaults.get("betas", (0.9, 0.999))
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"betas must be a tuple/list of two real numbers, got {betas}")

        beta1 = self._validate_real("betas[0]", betas[0], allow_negative=False)
        beta2 = self._validate_real("betas[1]", betas[1], allow_negative=False)

        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1 parameter: {beta1} (must be in [0.0, 1.0))")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2 parameter: {beta2} (must be in [0.0, 1.0))")

        return dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    def _validate_state_entry(self, index: int, param: Parameter, state_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and restores AdamW parameter state entry."""
        if "step" not in state_entry:
            raise ValueError(f"State entry for parameter {index} missing 'step'")
        step_val = state_entry["step"]
        if isinstance(step_val, bool) or not isinstance(step_val, int) or step_val < 0:
            raise ValueError(f"optimizer step must be a non-negative integer, got {step_val}")

        if "exp_avg" not in state_entry:
            raise ValueError(f"State entry for parameter {index} missing 'exp_avg'")
        exp_avg_raw = state_entry["exp_avg"]
        if not isinstance(exp_avg_raw, list):
            raise TypeError(f"exp_avg for parameter {index} must be a list structure")
        exp_avg_data = param.backend.from_data(copy.deepcopy(exp_avg_raw))
        if tuple(param.backend.get_shape(exp_avg_data)) != tuple(param.shape):
            raise RuntimeError(
                f"Shape mismatch for exp_avg parameter {index}: expected {param.shape}, got {param.backend.get_shape(exp_avg_data)}"
            )
        self._assert_all_finite(param, exp_avg_data, f"exp_avg for parameter {index}")

        if "exp_avg_sq" not in state_entry:
            raise ValueError(f"State entry for parameter {index} missing 'exp_avg_sq'")
        exp_avg_sq_raw = state_entry["exp_avg_sq"]
        if not isinstance(exp_avg_sq_raw, list):
            raise TypeError(f"exp_avg_sq for parameter {index} must be a list structure")
        exp_avg_sq_data = param.backend.from_data(copy.deepcopy(exp_avg_sq_raw))
        if tuple(param.backend.get_shape(exp_avg_sq_data)) != tuple(param.shape):
            raise RuntimeError(
                f"Shape mismatch for exp_avg_sq parameter {index}: expected {param.shape}, got {param.backend.get_shape(exp_avg_sq_data)}"
            )
        self._assert_all_finite(param, exp_avg_sq_data, f"exp_avg_sq for parameter {index}")

        return {
            "step": step_val,
            "exp_avg": exp_avg_data,
            "exp_avg_sq": exp_avg_sq_data,
        }

    def step(self) -> None:
        """Performs a single optimization step using a two-phase transactional commit (Policy B)."""
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        # Phase 1: Validation and candidate calculation
        candidates = []
        for idx, p in enumerate(self.params):
            if not self._validate_param_and_grad(idx, p):
                continue

            self._assert_all_finite(p, p._data, f"parameter {idx}")
            self._assert_all_finite(p, p.grad._data, f"gradient for parameter {idx}")

            # 1. Non-destructive Decoupled Weight Decay: theta_decayed = theta_(t-1) * (1 - lr * lambda)
            decayed_param_data = p._data
            if weight_decay != 0.0:
                decayed_param_data = p.backend.mul(
                    p._data,
                    1.0 - lr * weight_decay
                )
                self._assert_all_finite(p, decayed_param_data, f"decoupled decayed parameter {idx}")

            # 2. Pure gradient (unpolluted by weight decay)
            grad_data = p.grad._data

            # 3. State Initialization
            if idx not in self.state:
                prev_step = 0
                prev_exp_avg = p.backend.zeros(p.shape)
                prev_exp_avg_sq = p.backend.zeros(p.shape)
            else:
                prev_state = self.state[idx]
                prev_step = prev_state["step"]
                prev_exp_avg = prev_state["exp_avg"]
                prev_exp_avg_sq = prev_state["exp_avg_sq"]
                self._assert_all_finite(p, prev_exp_avg, f"previous exp_avg for parameter {idx}")
                self._assert_all_finite(p, prev_exp_avg_sq, f"previous exp_avg_sq for parameter {idx}")

            new_step_count = prev_step + 1

            # 4. Update biased 1st and 2nd moment estimates
            new_exp_avg = p.backend.add(
                p.backend.mul(prev_exp_avg, beta1),
                p.backend.mul(grad_data, 1.0 - beta1)
            )
            self._assert_all_finite(p, new_exp_avg, f"new exp_avg for parameter {idx}")

            grad_sq = p.backend.mul(grad_data, grad_data)
            new_exp_avg_sq = p.backend.add(
                p.backend.mul(prev_exp_avg_sq, beta2),
                p.backend.mul(grad_sq, 1.0 - beta2)
            )
            self._assert_all_finite(p, new_exp_avg_sq, f"new exp_avg_sq for parameter {idx}")

            # 5. Bias Corrections
            bias_correction1 = 1.0 - (beta1 ** new_step_count)
            bias_correction2 = 1.0 - (beta2 ** new_step_count)

            m_hat = p.backend.div(new_exp_avg, bias_correction1)
            v_hat = p.backend.div(new_exp_avg_sq, bias_correction2)

            # 6. Compute denominator: sqrt(v_hat) + eps
            sqrt_v = p.backend.pow(v_hat, 0.5)
            denom = p.backend.add(sqrt_v, eps)

            # 7. Step update from pure gradient
            step_update = p.backend.mul(p.backend.div(m_hat, denom), lr)
            self._assert_all_finite(p, step_update, f"step update for parameter {idx}")

            # 8. Final parameter: theta_t = theta_decayed - step_update
            new_param_data = p.backend.sub(decayed_param_data, step_update)
            self._assert_all_finite(p, new_param_data, f"new parameter data for parameter {idx}")

            new_state = {
                "step": new_step_count,
                "exp_avg": new_exp_avg,
                "exp_avg_sq": new_exp_avg_sq,
            }
            candidates.append((idx, p, new_param_data, new_state))

        # Phase 2: Atomic commit across all parameters and states
        for idx, p, new_data, new_state in candidates:
            p._data = new_data
            self.state[idx] = new_state
