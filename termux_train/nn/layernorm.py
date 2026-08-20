"""
termux_train.nn.layernorm
=========================
Layer Normalization over Last Dimension for Deep Transformers.
"""

from typing import Union, Tuple
from .module import Module
from .parameter import Parameter
from ..tensor import Tensor, ones, zeros


class LayerNorm(Module):
    """
    Layer Normalization over specified normalized_shape.
      y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(ones(self.normalized_shape))
            self.bias = Parameter(zeros(self.normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        norm_ndim = len(self.normalized_shape)
        if input.shape[-norm_ndim:] != self.normalized_shape:
            raise ValueError(
                f"Input shape {input.shape} does not match LayerNorm normalized_shape {self.normalized_shape}"
            )

        # Normalize over last len(normalized_shape) dimensions
        axis = tuple(range(-norm_ndim, 0))
        mean = input.mean(axis=axis, keepdims=True)
        diff = input - mean
        var = (diff ** 2).mean(axis=axis, keepdims=True)
        std = (var + self.eps).sqrt()
        x_norm = diff / std

        if self.elementwise_affine:
            return x_norm * self.weight + self.bias
        return x_norm

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"
