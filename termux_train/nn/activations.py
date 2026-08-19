"""
termux_train.nn.activations
===========================
Non-linear activation modules (ReLU, Sigmoid, Tanh).
"""

from .module import Module

class ReLU(Module):
    """Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)."""
    
    def forward(self, x):
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Applies the Sigmoid function element-wise: Sigmoid(x) = 1 / (1 + exp(-x))."""
    
    def forward(self, x):
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Applies the Hyperbolic Tangent function element-wise: Tanh(x)."""
    
    def forward(self, x):
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"
