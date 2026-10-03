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


class SiLU(Module):
    """Applies the Sigmoid Linear Unit (SiLU/Swish) function element-wise: SiLU(x) = x * sigmoid(x)."""

    def forward(self, x):
        return x * x.sigmoid()

    def __repr__(self) -> str:
        return "SiLU()"


class GELU(Module):
    """Applies Gaussian Error Linear Units function element-wise: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))."""

    def forward(self, x):
        return 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * (x ** 3))).tanh())

    def __repr__(self) -> str:
        return "GELU()"
