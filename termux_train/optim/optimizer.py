"""
termux_train.optim.optimizer
============================
Abstract Base Class for all termux-train first-order optimizers.
Manages parameter references, validation, lifecycle, DAG-safety, fail-fast finite checks,
and atomic state_dict serialization.
"""

import copy
import math
from typing import List, Dict, Any, Iterable, Optional, Tuple
from ..tensor import Tensor
from ..nn.parameter import Parameter

class Optimizer:
    """
    Base class for all termux-train optimizers.

    Args:
        params: Iterable of Parameter or Tensor instances to optimize.
        defaults: Dict of default hyperparameters (e.g. lr, weight_decay).
    """

    def __init__(self, params: Iterable[Parameter], defaults: Dict[str, Any]):
        if not isinstance(defaults, dict):
            raise TypeError(f"defaults must be a dict, got {type(defaults).__name__}")

        param_list = list(params)

        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")

        seen_ids = set()
        self._param_shapes: List[Tuple[int, ...]] = []
        for p in param_list:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer params must be Tensor or Parameter instances, got {type(p).__name__}")
            if id(p) in seen_ids:
                raise ValueError("optimizer got duplicate parameter")
            if any(dim == 0 for dim in p.shape):
                raise ValueError(f"optimizer does not support zero-size parameter: shape {p.shape}")

            seen_ids.add(id(p))
            self._param_shapes.append(tuple(p.shape))

        self.params: List[Parameter] = param_list
        self.defaults = self._validate_and_normalize_defaults(defaults)
        self.state: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _validate_real(
        name: str,
        value: Any,
        allow_negative: bool = False,
        strictly_positive: bool = False,
    ) -> float:
        """Validates that a hyperparameter is a finite real number (disallows bool)."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        if strictly_positive and val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")
        if not allow_negative and val < 0.0:
            raise ValueError(f"{name} must be >= 0, got {val}")
        return val

    def _validate_and_normalize_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Subclasses override this to validate and normalize their hyperparameters."""
        return copy.deepcopy(defaults)

    def _validate_param_and_grad(self, index: int, param: Parameter) -> bool:
        """
        Validates parameter shape consistency, requirements, and gradient compatibility.
        Returns True if step update should proceed, False if skipped.
        """
        expected_shape = self._param_shapes[index]
        if tuple(param.shape) != expected_shape:
            raise RuntimeError(
                f"parameter shape changed after optimizer construction: expected {expected_shape}, got {tuple(param.shape)}"
            )

        if not param.requires_grad or param.grad is None:
            return False

        if tuple(param.grad.shape) != tuple(param.shape):
            raise RuntimeError(
                f"gradient shape mismatch: parameter shape {param.shape}, gradient shape {param.grad.shape}"
            )

        if param.grad.backend.name != param.backend.name:
            raise RuntimeError(
                f"gradient backend mismatch: parameter backend {param.backend.name}, gradient backend {param.grad.backend.name}"
            )

        return True

    def _assert_all_finite(self, param: Parameter, data: Any, label: str) -> None:
        """Fail-fast check ensuring tensor data contains only finite numbers (no NaN or Inf)."""
        flat_vals = param.backend.to_flat_list(data)
        if not all(math.isfinite(float(v)) for v in flat_vals):
            raise FloatingPointError(f"non-finite value detected in {label}")

    def _clone_backend_data(self, param: Parameter, data: Any) -> Any:
        """Creates a portable deep copy of backend native data."""
        nested = param.backend.to_nested_list(data)
        return param.backend.from_data(copy.deepcopy(nested))

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clears the gradients of all optimized parameters.

        Args:
            set_to_none: If True (default), sets param.grad to None to release memory.
                         If False, creates a zeros Tensor matching param.shape.
        """
        for param in self.params:
            param.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Performs a single optimization step (parameter update)."""
        raise NotImplementedError("Optimizer subclasses must implement step()")

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a deepcopy dictionary.
        Keys are stable parameter indices rather than object references.
        """
        serializable_state = {}
        for idx, s in self.state.items():
            param = self.params[idx]
            param_s = {}
            for k, v in s.items():
                if isinstance(v, (int, float, bool, str)):
                    param_s[k] = v
                else:
                    # Backend native tensor data
                    param_s[k] = param.backend.to_nested_list(v)
            serializable_state[idx] = param_s

        return {
            "class": self.__class__.__name__,
            "defaults": copy.deepcopy(self.defaults),
            "state": serializable_state,
            "param_count": len(self.params),
        }

    def _validate_state_entry(self, index: int, param: Parameter, state_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Subclasses validate and restore their parameter state schema."""
        return copy.deepcopy(state_entry)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Atomically loads the optimizer state from a state_dict dictionary.
        Validates all fields, schemas, shapes, and finiteness before mutating any internal state.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")
        if state_dict.get("class") != self.__class__.__name__:
            raise ValueError(
                f"Optimizer class mismatch: expected {self.__class__.__name__}, got {state_dict.get('class')}"
            )

        param_count = state_dict.get("param_count")
        if isinstance(param_count, bool) or not isinstance(param_count, int):
            raise TypeError(f"param_count must be an integer, got {type(param_count).__name__}")
        if param_count != len(self.params):
            raise ValueError(
                f"Parameter count mismatch: expected {len(self.params)}, got {param_count}"
            )

        if "defaults" not in state_dict or not isinstance(state_dict["defaults"], dict):
            raise ValueError("state_dict missing valid 'defaults' dictionary")

        # 1. Validate defaults atomically
        validated_defaults = self._validate_and_normalize_defaults(state_dict["defaults"])

        # 2. Validate and build state entries atomically
        saved_state = state_dict.get("state", {})
        if not isinstance(saved_state, dict):
            raise TypeError(f"state must be a dict, got {type(saved_state).__name__}")

        restored_state = {}
        for raw_idx, s in saved_state.items():
            if isinstance(raw_idx, bool):
                raise TypeError(f"state index cannot be bool: {raw_idx}")
            if isinstance(raw_idx, int):
                idx = raw_idx
            elif isinstance(raw_idx, str) and raw_idx.isdigit():
                idx = int(raw_idx)
            else:
                raise TypeError(f"Invalid state index type: {raw_idx}")

            if idx < 0 or idx >= len(self.params):
                raise ValueError(f"State index {idx} out of range [0, {len(self.params) - 1}]")

            if not isinstance(s, dict):
                raise TypeError(f"State entry for parameter {idx} must be a dict, got {type(s).__name__}")

            param = self.params[idx]
            restored_entry = self._validate_state_entry(idx, param, s)
            restored_state[idx] = restored_entry

        # 3. Apply changes atomically only after 100% verification
        self.defaults = validated_defaults
        self.state = restored_state

    def __repr__(self) -> str:
        options = ", ".join(f"{key}={value!r}" for key, value in self.defaults.items())
        return f"{type(self).__name__}({options}, params={len(self.params)})"
