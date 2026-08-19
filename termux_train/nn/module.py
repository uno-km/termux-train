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
    and train/eval mode toggling.
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

    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """Returns an iterator or list over module parameters."""
        params: List[Parameter] = []
        for p in self._parameters.values():
            if p is not None:
                params.append(p)
        if recurse:
            for m in self._modules.values():
                params.extend(m.parameters(recurse=True))
        return params

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> List[Tuple[str, Parameter]]:
        """Returns a list of (name, parameter) tuples in the module."""
        named_params = []
        for name, p in self._parameters.items():
            if p is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                named_params.append((full_name, p))
        if recurse:
            for m_name, m in self._modules.items():
                sub_prefix = f"{prefix}.{m_name}" if prefix else m_name
                named_params.extend(m.named_parameters(prefix=sub_prefix, recurse=True))
        return named_params

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
        """Sets the module in training mode."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
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
        """Copies parameters from state_dict into this module and its descendants."""
        named_params = dict(self.named_parameters())
        
        for name, data in state_dict.items():
            if name in named_params:
                param = named_params[name]
                new_data = param.backend.from_data(data)
                new_shape = param.backend.get_shape(new_data)
                if new_shape != param.shape:
                    raise RuntimeError(f"Shape mismatch for parameter '{name}': expected {param.shape}, got {new_shape}")
                param._data = new_data
            elif strict:
                raise KeyError(f"Unexpected key '{name}' in state_dict")
                
        if strict:
            missing = set(named_params.keys()) - set(state_dict.keys())
            if missing:
                raise KeyError(f"Missing keys in state_dict: {missing}")

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, m in self._modules.items():
            mod_str = repr(m)
            mod_str = "  " + "\n  ".join(mod_str.split("\n"))
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines) if len(self._modules) > 0 else f"{type(self).__name__}()"
