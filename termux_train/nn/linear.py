"""
termux_train.nn.linear
======================
Fully Connected Linear (Dense) Layer with learnable weight and bias parameters.
"""

import math
import random
from typing import Optional
from .module import Module
from .parameter import Parameter
from ..backend import get_backend, BaseBackend

class Linear(Module):
    """
    Applies an affine linear transformation to the incoming data: y = x @ weight + bias.
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: Optional[BaseBackend] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        b = backend or get_backend()
        
        # Initialize weights: He/Kaiming uniform distribution U(-1/sqrt(in), 1/sqrt(in))
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        
        # Generate random weights of shape (in_features, out_features)
        weight_data = [
            [random.uniform(-bound, bound) for _ in range(out_features)]
            for _ in range(in_features)
        ]
        self.weight = Parameter(weight_data, requires_grad=True, backend=b)
        
        if bias:
            bias_data = [[random.uniform(-bound, bound) for _ in range(out_features)]]
            self.bias: Optional[Parameter] = Parameter(bias_data, requires_grad=True, backend=b)
        else:
            self.bias: Optional[Parameter] = None

    def forward(self, x):
        """
        Forward computation: y = x @ weight + bias.
        Supported inputs:
          1D: (in_features,) -> (out_features,)
          2D: (batch_size, in_features) -> (batch_size, out_features)
          3D: (batch_size, sequence_length, in_features) -> (batch_size, sequence_length, out_features)
        """
        if x.ndim not in (1, 2, 3):
            raise ValueError(
                "Linear expects a 1D, 2D, or 3D input, "
                f"but received shape {x.shape}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected input.shape[-1] == {self.in_features}, "
                f"but received shape {x.shape}"
            )
        out = x @ self.weight
        if self.bias is not None:
            if x.ndim == 1:
                out = out + self.bias.reshape((self.out_features,))
            else:
                out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
