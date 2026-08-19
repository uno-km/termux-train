"""
termux_train.nn.sequential
==========================
Sequential container module chaining layers in pipeline order.
"""

from typing import Union, List, Tuple, Iterator
from .module import Module

class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    The forward() method of Sequential accepts any input and forwards it to the first module it contains.
    It then 'chains' outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
    """
    
    def __init__(self, *args: Union[Module, List[Module]]):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            modules = args[0]
        else:
            modules = args
            
        for idx, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(f"Sequential argument {idx} is not a Module subclass: {type(module)}")
            self._modules[str(idx)] = module

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise TypeError(f"Module to append is not a Module subclass: {type(module)}")
        self._modules[str(len(self._modules))] = module
        return self

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for name, m in self._modules.items():
            mod_str = repr(m)
            mod_str = "  " + "\n  ".join(mod_str.split("\n"))
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines)
