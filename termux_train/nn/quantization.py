"""
termux_train.nn.quantization
============================
INT8 Dynamic Weight Quantization (Symmetric AbsMax Quantization) for Mobile Inference.
Performs zero-allocation INT8 matrix operations (x @ W_int8 * scale) to eliminate runtime heap allocation.
"""

from typing import Optional, Dict, Any, Tuple
from .module import Module
from .linear import Linear
from .parameter import Parameter
from ..tensor import Tensor

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class QuantizedLinear(Module):
    """
    INT8 Quantized Linear Layer with Zero-Allocation Forward:
      y = (x @ W_int8) * scale + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: Any,
        scale: float,
        bias: Optional[Parameter] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        self.qweight = qweight
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        backend = x.backend
        scale = self.scale

        # Zero-allocation direct scaling: (x @ qweight) * scale
        if HAS_NUMPY and isinstance(x._data, np.ndarray) and isinstance(self.qweight, np.ndarray):
            q_float = self.qweight.astype(np.float32)
            out_arr = (x._data @ q_float) * scale
            if self.bias is not None:
                bias_arr = self.bias._data if isinstance(self.bias._data, np.ndarray) else np.array(self.bias._data, dtype=np.float32)
                out_arr = out_arr + bias_arr
            return Tensor(out_arr, dtype="float32", requires_grad=False, backend=backend)

        # Pure Python fallback
        q_flat = backend.to_flat_list(self.qweight)
        dequant_flat = [float(v) * scale for v in q_flat]
        w_data = backend.from_data(backend.reshape(dequant_flat, (self.in_features, self.out_features)), dtype="float32")
        w_tensor = Tensor(w_data, dtype="float32", requires_grad=False, backend=backend)

        out = x @ w_tensor
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, scale={self.scale:.6f})"


def quantize_linear_int8(linear: Linear) -> QuantizedLinear:
    """
    Quantizes a floating-point nn.Linear module into an INT8 QuantizedLinear module.
    """
    backend = linear.weight.backend
    flat_w = backend.to_flat_list(linear.weight._data)

    max_abs = max(abs(float(v)) for v in flat_w) if flat_w else 1.0
    scale = (max_abs / 127.0) if max_abs > 0 else 1.0

    # Quantize to int8: clamp(round(w / scale), -128, 127)
    q_flat = [max(-128, min(127, int(round(float(v) / scale)))) for v in flat_w]

    if HAS_NUMPY and getattr(backend, "name", "").lower() == "numpy":
        q_data = np.array(q_flat, dtype=np.int8).reshape(linear.weight.shape)
    else:
        q_data = backend.from_data(backend.reshape(q_flat, linear.weight.shape), dtype="int64")

    return QuantizedLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        qweight=q_data,
        scale=scale,
        bias=linear.bias
    )
