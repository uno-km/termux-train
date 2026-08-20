"""
termux_train.backend.numpy_backend
==================================
High-Performance C-Accelerated Backend using NumPy.
Enabled when numpy is available on Android Termux (pkg install python-numpy).
Supports multi-dtype representation (float32, int64, bool) and general N-D batched matmul.
"""

from typing import Any, Tuple, List, Union, Optional
from .base import BaseBackend, Shape

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

DTYPE_MAP = {
    "float32": np.float32 if np else None,
    "int64": np.int64 if np else None,
    "bool": np.bool_ if np else None,
}


class NumPyBackend(BaseBackend):
    """C-accelerated NumPy compute backend."""

    def __init__(self):
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not installed. Install via 'pkg install python-numpy' or 'pip install numpy'.")

    @property
    def name(self) -> str:
        return "numpy"

    def from_data(self, data: Any, dtype: Optional[str] = "float32") -> Any:
        dtype = dtype or "float32"
        target_np_dtype = DTYPE_MAP.get(dtype, np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(target_np_dtype)
        try:
            arr = np.array(data, dtype=target_np_dtype)
            if arr.dtype == object:
                raise ValueError("Ragged nested list is not supported")
            return arr
        except Exception as e:
            raise ValueError(f"Invalid tensor data or ragged list: {e}")

    def get_shape(self, data: Any) -> Shape:
        if isinstance(data, np.ndarray):
            return tuple(data.shape)
        return ()

    def to_flat_list(self, data: Any) -> List[Any]:
        if isinstance(data, np.ndarray):
            if data.dtype == np.int64:
                return [int(x) for x in data.flatten()]
            elif data.dtype == np.bool_:
                return [bool(x) for x in data.flatten()]
            return [float(x) for x in data.flatten()]
        return [data]

    def to_nested_list(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def zeros(self, shape: Shape, dtype: str = "float32") -> Any:
        target_np_dtype = DTYPE_MAP.get(dtype, np.float32)
        return np.zeros(shape, dtype=target_np_dtype)

    def ones(self, shape: Shape, dtype: str = "float32") -> Any:
        target_np_dtype = DTYPE_MAP.get(dtype, np.float32)
        return np.ones(shape, dtype=target_np_dtype)

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
        return np.divide(a, b)

    def pow(self, a: Any, exp: float) -> Any:
        return np.power(a, exp)

    def exp(self, a: Any) -> Any:
        return np.exp(a)

    def sqrt(self, a: Any) -> Any:
        return np.sqrt(a)

    def neg(self, a: Any) -> Any:
        return np.negative(a)

    def matmul(self, a: Any, b: Any) -> Any:
        if (isinstance(a, np.ndarray) and a.ndim == 0) or (isinstance(b, np.ndarray) and b.ndim == 0):
            raise ValueError(f"Cannot perform matmul with scalar operand (shapes {getattr(a, 'shape', ())} and {getattr(b, 'shape', ())})")
        return np.matmul(a, b)

    def sum(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        return np.sum(data, axis=axis, keepdims=keepdims)

    def max(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        return np.max(data, axis=axis, keepdims=keepdims)

    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        return np.mean(data, axis=axis, keepdims=keepdims)

    def relu(self, data: Any) -> Any:
        return np.maximum(data, 0.0)

    def sigmoid(self, data: Any) -> Any:
        clamped = np.clip(data, -88.0, 88.0)
        return 1.0 / (1.0 + np.exp(-clamped))

    def tanh(self, data: Any) -> Any:
        return np.tanh(data)

    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        if isinstance(grad, (int, float)):
            grad = np.array(grad, dtype=np.float32)

        cur_shape = tuple(grad.shape)
        if cur_shape == target_shape:
            return grad

        cur_ndim = len(cur_shape)
        tgt_ndim = len(target_shape)
        pad = cur_ndim - tgt_ndim

        out = grad
        for _ in range(pad):
            out = np.sum(out, axis=0, keepdims=False)

        for i in range(tgt_ndim):
            if target_shape[i] == 1 and cur_shape[i + pad] > 1:
                out = np.sum(out, axis=i, keepdims=True)

        return out

    def clamp(self, data: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        a_min = min_val if min_val is not None else -np.inf
        a_max = max_val if max_val is not None else np.inf
        return np.clip(data, a_min, a_max)

    def log(self, data: Any) -> Any:
        return np.log(data)

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        return np.take(data, index, axis=axis)

    def gather_rows(self, weight_data: Any, row_indices: List[int], out_shape: Tuple[int, ...]) -> Any:
        idx_arr = np.array(row_indices, dtype=np.int64)
        gathered = weight_data[idx_arr]
        return gathered.reshape(out_shape)

    def scatter_add_rows(self, target_data: Any, row_indices: List[int], grad_data: Any, padding_idx: Optional[int] = None) -> Any:
        idx_arr = np.array(row_indices, dtype=np.int64)
        e_dim = target_data.shape[-1]
        if isinstance(grad_data, np.ndarray):
            flat_grad = grad_data.reshape(-1, e_dim)
        else:
            flat_grad = np.array(grad_data, dtype=np.float32).reshape(-1, e_dim)

        if padding_idx is not None:
            mask = (idx_arr != padding_idx)
            idx_arr = idx_arr[mask]
            flat_grad = flat_grad[mask]

        if len(idx_arr) > 0:
            np.add.at(target_data, idx_arr, flat_grad)
        return target_data
