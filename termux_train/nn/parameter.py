"""
termux_train.nn.parameter
=========================
Parameter class representing trainable weights and biases in a Neural Network Module.
"""

from typing import Any, Optional
from ..tensor import Tensor
from ..backend import BaseBackend

class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    Parameters are Tensor subclasses that have requires_grad=True by default.
    """
    
    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        backend: Optional[BaseBackend] = None
    ):
        super().__init__(data, requires_grad=requires_grad, backend=backend)

    def __repr__(self) -> str:
        data_str = str(self.tolist())
        return f"Parameter({data_str}, shape={self.shape}, requires_grad={self.requires_grad})"
