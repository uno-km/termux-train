"""
termux_train.backend.amuda_backend
===================================
AMUDA GPU-Accelerated Backend for termux-train.
Routes compute-intensive operations (matmul, elementwise math) to
AMUDA Vulkan GPU kernels while delegating shape/indexing ops to NumPy.

Requires: ameva-compute (AMUDA) package with ac_runtime and pre-compiled
SPIR-V shaders.
"""

from typing import Any, Tuple, List, Union, Optional
from .base import BaseBackend, Shape

import numpy as np

# Lazy-loaded AMUDA context and ops
_amuda_ctx = None
_amuda_ops = None
_amuda_available = None


def _ensure_amuda():
    """Lazily initialize AMUDA GPU context and ops module."""
    global _amuda_ctx, _amuda_ops, _amuda_available
    if _amuda_available is not None:
        return _amuda_available

    try:
        import ameva_compute as ac
        _amuda_ops = ac.ops
        _amuda_ctx = ac.init()
        _amuda_available = True
        return True
    except Exception as e:
        print(f"[AmudaBackend] GPU init failed: {e}. Falling back to NumPy.")
        _amuda_available = False
        return False


def _get_ctx():
    """Get the AMUDA GPU context."""
    _ensure_amuda()
    return _amuda_ctx


def _get_ops():
    """Get the AMUDA ops module."""
    _ensure_amuda()
    return _amuda_ops


# Minimum tensor size to offload to GPU.
# Benchmark data (Mali-G68): GPU wins at ~700x700+ (490K+ elements).
# Below this, buffer creation overhead exceeds compute savings.
GPU_OFFLOAD_THRESHOLD = 262144  # ~512x512


def _should_gpu(a, b=None) -> bool:
    """Determine whether to use GPU based on tensor sizes."""
    if not _ensure_amuda():
        return False
    size_a = a.size if isinstance(a, np.ndarray) else 0
    size_b = b.size if isinstance(b, np.ndarray) else 0
    return max(size_a, size_b) >= GPU_OFFLOAD_THRESHOLD


DTYPE_MAP = {
    "float32": np.float32,
    "int64": np.int64,
    "bool": np.bool_,
}


class AmudaBackend(BaseBackend):
    """
    Hybrid GPU+CPU backend: AMUDA Vulkan for heavy compute,
    NumPy for everything else.
    """

    def __init__(self):
        if not _ensure_amuda():
            raise RuntimeError(
                "AMUDA GPU backend unavailable. "
                "Ensure ameva-compute is installed and Vulkan drivers are accessible."
            )
        self._gpu_matmul_count = 0
        self._cpu_fallback_count = 0

    @property
    def name(self) -> str:
        return "amuda"

    # ================================================================
    # Data conversion (NumPy-based, no GPU needed)
    # ================================================================

    def from_data(self, data: Any, dtype: Optional[str] = "float32") -> Any:
        dtype = dtype or "float32"
        target = DTYPE_MAP.get(dtype, np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(target)
        arr = np.array(data, dtype=target)
        if arr.dtype == object:
            raise ValueError("Ragged nested list is not supported")
        return arr

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

    # ================================================================
    # Creation ops (NumPy, then potentially GPU-resident in future)
    # ================================================================

    def zeros(self, shape: Shape, dtype: str = "float32") -> Any:
        return np.zeros(shape, dtype=DTYPE_MAP.get(dtype, np.float32))

    def ones(self, shape: Shape, dtype: str = "float32") -> Any:
        return np.ones(shape, dtype=DTYPE_MAP.get(dtype, np.float32))

    def randn(self, shape: Shape, mean: float = 0.0, std: float = 1.0) -> Any:
        return (np.random.randn(*shape).astype(np.float32) * std + mean)

    # ================================================================
    # Shape ops (NumPy, zero-cost)
    # ================================================================

    def reshape(self, data: Any, new_shape: Shape) -> Any:
        return np.reshape(data, new_shape)

    def transpose(self, data: Any, axes: Tuple[int, ...] = None) -> Any:
        return np.transpose(data, axes)

    # ================================================================
    # CORE COMPUTE: GPU-accelerated when beneficial
    # ================================================================

    def matmul(self, a: Any, b: Any) -> Any:
        """GPU-accelerated matrix multiplication for large tensors."""
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return np.matmul(a, b)

        # Only GPU-accelerate 2D matmuls above threshold
        if a.ndim == 2 and b.ndim == 2 and _should_gpu(a, b):
            try:
                ops = _get_ops()
                ctx = _get_ctx()
                # AMUDA matmul returns float16, convert back to float32
                result = ops.matmul(ctx, a, b, dtype='float32')
                self._gpu_matmul_count += 1
                return result
            except Exception:
                self._cpu_fallback_count += 1
                return np.matmul(a, b)

        # Batched matmul or small tensors: NumPy
        return np.matmul(a, b)

    def add(self, a: Any, b: Any) -> Any:
        """GPU-accelerated element-wise addition for large tensors."""
        if (_should_gpu(a, b)
                and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.ndim <= 2):
            try:
                ops = _get_ops()
                ctx = _get_ctx()
                return ops.elementwise(ctx, a, b, op='add').astype(np.float32)
            except Exception:
                pass
        return np.add(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        if (_should_gpu(a, b)
                and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.ndim <= 2):
            try:
                ops = _get_ops()
                ctx = _get_ctx()
                return ops.elementwise(ctx, a, b, op='sub').astype(np.float32)
            except Exception:
                pass
        return np.subtract(a, b)

    def mul(self, a: Any, b: Any) -> Any:
        if (_should_gpu(a, b)
                and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.ndim <= 2):
            try:
                ops = _get_ops()
                ctx = _get_ctx()
                return ops.elementwise(ctx, a, b, op='mul').astype(np.float32)
            except Exception:
                pass
        return np.multiply(a, b)

    def div(self, a: Any, b: Any) -> Any:
        return np.divide(a, b)

    # ================================================================
    # Unary math ops (NumPy for now, GPU candidates for future)
    # ================================================================

    def pow(self, a: Any, exp: float) -> Any:
        return np.power(a, exp)

    def exp(self, a: Any) -> Any:
        return np.exp(a)

    def sqrt(self, a: Any) -> Any:
        return np.sqrt(a)

    def neg(self, a: Any) -> Any:
        return np.negative(a)

    def log(self, data: Any) -> Any:
        return np.log(data)

    # ================================================================
    # Activation functions
    # ================================================================

    def relu(self, data: Any) -> Any:
        return np.maximum(data, 0.0)

    def sigmoid(self, data: Any) -> Any:
        clamped = np.clip(data, -88.0, 88.0)
        return 1.0 / (1.0 + np.exp(-clamped))

    def tanh(self, data: Any) -> Any:
        return np.tanh(data)

    # ================================================================
    # Reduction ops (NumPy — GPU reduction less beneficial for small dims)
    # ================================================================

    def sum(self, data: Any, axis=None, keepdims: bool = False) -> Any:
        return np.sum(data, axis=axis, keepdims=keepdims)

    def max(self, data: Any, axis=None, keepdims: bool = False) -> Any:
        return np.max(data, axis=axis, keepdims=keepdims)

    def mean(self, data: Any, axis=None, keepdims: bool = False) -> Any:
        return np.mean(data, axis=axis, keepdims=keepdims)

    # ================================================================
    # Broadcast / indexing ops (NumPy)
    # ================================================================

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

    def clamp(self, data: Any, min_val=None, max_val=None) -> Any:
        a_min = min_val if min_val is not None else -np.inf
        a_max = max_val if max_val is not None else np.inf
        return np.clip(data, a_min, a_max)

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        return np.take(data, index, axis=axis)

    def gather_rows(self, weight_data: Any, row_indices: List[int],
                    out_shape: Tuple[int, ...]) -> Any:
        idx_arr = np.array(row_indices, dtype=np.int64)
        return weight_data[idx_arr].reshape(out_shape)

    def scatter_add_rows(self, target_data: Any, row_indices: List[int],
                         grad_data: Any, padding_idx=None) -> Any:
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

    # ================================================================
    # Diagnostics
    # ================================================================

    def get_stats(self) -> dict:
        """Return GPU offload statistics."""
        return {
            "gpu_matmul_count": self._gpu_matmul_count,
            "cpu_fallback_count": self._cpu_fallback_count,
            "gpu_available": _amuda_available,
        }
