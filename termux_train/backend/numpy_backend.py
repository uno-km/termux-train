"""
termux_train.backend.numpy_backend
==================================
High-Performance C-Accelerated Backend using NumPy.
Enabled when numpy is available on Android Termux (pkg install python-numpy).
"""

from typing import Any, Tuple, List, Union, Optional
from .base import BaseBackend, Shape

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

class NumPyBackend(BaseBackend):
    """C-accelerated NumPy compute backend."""
    
    def __init__(self):
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not installed. Install via 'pkg install python-numpy' or 'pip install numpy'.")

    @property
    def name(self) -> str:
        return "numpy"

    def from_data(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        try:
            arr = np.array(data, dtype=np.float32)
            if arr.dtype == object:
                raise ValueError("Ragged nested list is not supported")
            return arr
        except Exception as e:
            raise ValueError(f"Invalid tensor data or ragged list: {e}")

    def get_shape(self, data: Any) -> Shape:
        if isinstance(data, np.ndarray):
            return tuple(data.shape)
        return ()

    def to_flat_list(self, data: Any) -> List[float]:
        if isinstance(data, np.ndarray):
            return [float(x) for x in data.flatten()]
        return [float(data)]

    def to_nested_list(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def zeros(self, shape: Shape) -> Any:
        return np.zeros(shape, dtype=np.float32)

    def ones(self, shape: Shape) -> Any:
        return np.ones(shape, dtype=np.float32)

    def randn(self, shape: Shape, mean: float = 0.0, std: float = 1.0) -> Any:
        arr = np.random.randn(*shape).astype(np.float32)
        return arr * std + mean

    def reshape(self, data: Any, new_shape: Shape) -> Any:
        return np.reshape(data, new_shape)

    def transpose(self, data: Any, axes: Tuple[int, ...] = None) -> Any:
        return np.transpose(data, axes)

    def add(self, a: Any, b: Any) -> Any:
        return np.add(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        return np.subtract(a, b)

    def mul(self, a: Any, b: Any) -> Any:
        return np.multiply(a, b)

    def div(self, a: Any, b: Any) -> Any:
        return np.divide(a, np.where(b == 0, 1e-12, b))

    def pow(self, a: Any, exp: float) -> Any:
        return np.power(a, exp)

    def neg(self, a: Any) -> Any:
        return np.negative(a)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def sum(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        return np.sum(data, axis=axis, keepdims=keepdims)

    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        return np.mean(data, axis=axis, keepdims=keepdims)

    def relu(self, data: Any) -> Any:
        return np.maximum(data, 0.0)

    def sigmoid(self, data: Any) -> Any:
        clipped = np.clip(data, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def tanh(self, data: Any) -> Any:
        return np.tanh(data)

    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        if isinstance(grad, (int, float)):
            grad = np.array(grad, dtype=np.float32)
            
        grad_shape = tuple(grad.shape)
        if grad_shape == target_shape:
            return grad
            
        # Sum over leading added dimensions
        num_added_dims = len(grad_shape) - len(target_shape)
        for _ in range(num_added_dims):
            grad = np.sum(grad, axis=0)
            
        # Sum over broadcasted dimensions (where target_shape was 1)
        for i, (gd, td) in enumerate(zip(grad.shape, target_shape)):
            if td == 1 and gd != 1:
                grad = np.sum(grad, axis=i, keepdims=True)
                
        return grad.astype(np.float32)

    def clamp(self, data: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        return np.clip(data, min_val, max_val).astype(np.float32)

    def log(self, data: Any) -> Any:
        return np.log(np.maximum(data, 1e-12)).astype(np.float32)

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        return np.take(data, index, axis=axis)
