"""
termux_train.optim.optimizer
============================
Abstract Base Class for all termux-train first-order optimizers.
Manages parameter references, states, zero_grad lifecycle, and state_dict serialization.
"""

import copy
from typing import List, Dict, Any, Iterable, Optional
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

        self.defaults = defaults
        param_list = list(params)

        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")

        seen_ids = set()
        for p in param_list:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer params must be Tensor or Parameter instances, got {type(p).__name__}")
            if id(p) in seen_ids:
                raise ValueError("optimizer got duplicate parameter")
            seen_ids.add(id(p))

        self.params: List[Parameter] = param_list
        self.state: Dict[int, Dict[str, Any]] = {}

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
            param_s = {}
            for k, v in s.items():
                if isinstance(v, (int, float, bool, str)):
                    param_s[k] = v
                elif hasattr(v, "tolist"):
                    param_s[k] = v.tolist()
                elif isinstance(v, list):
                    param_s[k] = copy.deepcopy(v)
                else:
                    param_s[k] = copy.deepcopy(v)
            serializable_state[idx] = param_s

        return {
            "class": self.__class__.__name__,
            "defaults": copy.deepcopy(self.defaults),
            "state": serializable_state,
            "param_count": len(self.params)
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state from a state_dict dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")
        if state_dict.get("class") != self.__class__.__name__:
            raise ValueError(
                f"Optimizer class mismatch: expected {self.__class__.__name__}, got {state_dict.get('class')}"
            )
        if state_dict.get("param_count") != len(self.params):
            raise ValueError(
                f"Parameter count mismatch: expected {len(self.params)}, got {state_dict.get('param_count')}"
            )

        self.defaults = copy.deepcopy(state_dict["defaults"])

        saved_state = state_dict.get("state", {})
        self.state = {}
        for idx_str, s in saved_state.items():
            idx = int(idx_str)
            if idx < 0 or idx >= len(self.params):
                raise ValueError(f"Invalid state index {idx}")
            p = self.params[idx]
            restored_s = {}
            for k, v in s.items():
                if isinstance(v, list):
                    restored_data = p.backend.from_data(copy.deepcopy(v))
                    if p.backend.get_shape(restored_data) != p.shape:
                        raise RuntimeError(
                            f"Shape mismatch for optimizer state {k}: expected {p.shape}, got {p.backend.get_shape(restored_data)}"
                        )
                    restored_s[k] = restored_data
                else:
                    restored_s[k] = copy.deepcopy(v)
            self.state[idx] = restored_s
