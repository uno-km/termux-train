"""
termux_train.nn.module
======================
Base Module class for all Neural Network layers and models.
"""

from typing import Dict, List, Tuple, Any, Optional, Iterator
from .parameter import Parameter
from ..tensor import Tensor

class Module:
    """
    Base class for all neural network modules in termux-train.
    Provides PyTorch-compatible interface for parameter tracking, state serialization,
    and train/eval mode toggling with deduplicated, cycle-guarded traversal.
    """
    
    def __init__(self):
        # Use object.__setattr__ to avoid triggering custom __setattr__ during init
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def forward(self, *args, **kwargs) -> Any:
        """Defines the computation performed at every call. Must be overridden by subclasses."""
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing forward implementation")

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            self._parameters[name] = value
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("cannot assign modules before Module.__init__() call")
            self._modules[name] = value
        else:
            # If reassigning over an existing parameter or module
            if params is not None and name in params:
                del self._parameters[name]
            if modules is not None and name in modules:
                del self._modules[name]
                
        super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Adds a parameter to the module."""
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameters before Module.__init__() call")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param)}' object to parameter '{name}'")
        else:
            self._parameters[name] = param

    def _walk_modules(self) -> Iterator[Tuple[str, 'Module']]:
        """Unified, cycle-guarded and shared-submodule safe module graph walker."""
        visited = set()
        stack = [("", self)]
        while stack:
            prefix, module = stack.pop()
            if id(module) in visited:
                continue
            visited.add(id(module))
            yield prefix, module
            for name, child in reversed(list(module._modules.items())):
                if child is not None:
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    stack.append((child_prefix, child))

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> List[Tuple[str, Parameter]]:
        """Returns a list of (name, parameter) tuples in the module with deduplication."""
        seen_params = set()
        named_params: List[Tuple[str, Parameter]] = []

        if not recurse:
            for name, p in self._parameters.items():
                if p is not None and id(p) not in seen_params:
                    seen_params.add(id(p))
                    full_name = f"{prefix}.{name}" if prefix else name
                    named_params.append((full_name, p))
            return named_params

        for mod_prefix, mod in self._walk_modules():
            for name, p in mod._parameters.items():
                if p is not None and id(p) not in seen_params:
                    seen_params.add(id(p))
                    eff_prefix = f"{prefix}.{mod_prefix}" if (prefix and mod_prefix) else (mod_prefix or prefix)
                    full_name = f"{eff_prefix}.{name}" if eff_prefix else name
                    named_params.append((full_name, p))
        return named_params

    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """Returns an iterator or list over deduplicated module parameters."""
        return [p for _, p in self.named_parameters(recurse=recurse)]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Sets gradients of all model parameters to zero or None.
        
        Args:
            set_to_none: if True (default), sets param.grad to None for optimal mobile RAM usage.
                         if False, sets param.grad to a tensor of zeros.
        """
        for p in self.parameters():
            p.zero_grad(set_to_none=set_to_none)

    def train(self, mode: bool = True) -> 'Module':
        """Sets the module and all submodules in training mode."""
        for _, mod in self._walk_modules():
            object.__setattr__(mod, "training", mode)
        return self

    def eval(self) -> 'Module':
        """Sets the module in evaluation (inference) mode."""
        return self.train(False)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a whole state of the module.
        Ensures a detached deep copy so modifying model parameters does not corrupt saved checkpoints.
        """
        import copy
        state = {}
        for name, p in self.named_parameters():
            state[name] = copy.deepcopy(p.tolist())
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Copies parameters from state_dict into this module and its descendants with atomic 2-phase commit.
        If any validation error or shape mismatch occurs, NO parameters are modified and rollback occurs.
        """
        named_params = dict(self.named_parameters())
        
        if strict:
            unexpected = set(state_dict.keys()) - set(named_params.keys())
            if unexpected:
                raise KeyError(f"Unexpected key(s) in state_dict: {unexpected}")
            missing = set(named_params.keys()) - set(state_dict.keys())
            if missing:
                raise KeyError(f"Missing key(s) in state_dict: {missing}")

        # Phase 1: Validation and staging
        staged_updates: Dict[str, Any] = {}
        for name, data in state_dict.items():
            if name in named_params:
                param = named_params[name]
                new_data = param.backend.from_data(data, dtype=param.dtype)
                new_shape = param.backend.get_shape(new_data)
                if new_shape != param.shape:
                    raise RuntimeError(f"Shape mismatch for parameter '{name}': expected {param.shape}, got {new_shape}")
                staged_updates[name] = new_data
            elif strict:
                raise KeyError(f"Unexpected key '{name}' in state_dict")

        # Phase 2: Atomic commit with rollback safety and monotonic version bump
        old_data: Dict[str, Any] = {}
        try:
            for name, new_data in staged_updates.items():
                param = named_params[name]
                old_data[name] = param._data
                param._replace_data(new_data, bump_version=True)
        except Exception as commit_err:
            for name, prev in old_data.items():
                named_params[name]._replace_data(prev, bump_version=True)
            raise RuntimeError(f"Failed to commit state_dict atomically: {commit_err}") from commit_err

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, m in self._modules.items():
            mod_str = repr(m)
            mod_str = "  " + "\n  ".join(mod_str.split("\n"))
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines) if len(self._modules) > 0 else f"{type(self).__name__}()"
