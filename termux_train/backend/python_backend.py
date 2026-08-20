"""
termux_train.backend.python_backend
===================================
0-Dependency Pure-Python Compute Backend.
Uses nested Python lists and pure scalar math without any external binary dependencies.
Supports multi-dtype representation (float32, int64, bool) and general N-D batched matmul.
"""

import math
import random
from typing import Any, Tuple, List, Union, Callable, Sequence, Optional
from .base import BaseBackend, Shape

def _infer_shape(data: Any) -> Shape:
    """Recursively infer dimension tuple from nested lists."""
    if not isinstance(data, (list, tuple)):
        return ()
    if len(data) == 0:
        return (0,)
    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        s = _infer_shape(item)
        if s != first_shape:
            raise ValueError(f"Ragged nested list is not supported: shape mismatch between {first_shape} and {s}")
    return (len(data),) + first_shape

def _wrap_int64(val: Any) -> int:
    i_val = int(val)
    return ((i_val + (1 << 63)) % (1 << 64)) - (1 << 63)

def _normalize_data_dtype(data: Any, dtype: Optional[str] = "float32") -> Any:
    """Recursively cast elements to the target dtype with 64-bit overflow wrap."""
    if dtype == "int64":
        if isinstance(data, (bool, int, float)):
            return _wrap_int64(data)
        if isinstance(data, (list, tuple)):
            return [_normalize_data_dtype(x, dtype) for x in data]
    elif dtype == "bool":
        if isinstance(data, (bool, int, float)):
            return bool(data)
        if isinstance(data, (list, tuple)):
            return [_normalize_data_dtype(x, dtype) for x in data]
    else:  # float32 / default
        if isinstance(data, (int, float)):
            return float(data)
        if isinstance(data, (list, tuple)):
            return [_normalize_data_dtype(x, "float32") for x in data]
    raise TypeError(f"Unsupported data element type: {type(data)}")

def _flatten(data: Any) -> List[Any]:
    """Recursively flatten nested lists to 1D flat list."""
    if not isinstance(data, (list, tuple)):
        return [data]
    res = []
    for item in data:
        res.extend(_flatten(item))
    return res

def _unflatten(flat: Sequence[Any], shape: Shape) -> Any:
    """Rebuild nested lists from flat list and shape tuple."""
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat[:shape[0]])

    stride = 1
    for dim in shape[1:]:
        stride *= dim

    res = []
    for i in range(shape[0]):
        chunk = flat[i * stride : (i + 1) * stride]
        res.append(_unflatten(chunk, shape[1:]))
    return res

def _strides_for_shape(shape: Shape) -> List[int]:
    """Compute row-major strides for a given shape."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides

def _broadcast_shapes(s1: Shape, s2: Shape) -> Shape:
    """Compute standard numpy-style broadcasted output shape."""
    l1, l2 = len(s1), len(s2)
    max_len = max(l1, l2)
    pad_s1 = (1,) * (max_len - l1) + s1
    pad_s2 = (1,) * (max_len - l2) + s2
    out_shape = []
    for d1, d2 in zip(pad_s1, pad_s2):
        if d1 == d2:
            out_shape.append(d1)
        elif d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        else:
            raise ValueError(f"Operands could not be broadcast together with shapes {s1} and {s2}")
    return tuple(out_shape)

def _elementwise_binary(a_data: Any, b_data: Any, op: Callable[[Any, Any], Any]) -> Any:
    """Perform elementwise binary operation with full ND broadcasting."""
    s1 = _infer_shape(a_data)
    s2 = _infer_shape(b_data)
    out_shape = _broadcast_shapes(s1, s2)

    flat_a = _flatten(a_data)
    flat_b = _flatten(b_data)

    if len(out_shape) == 0:
        return op(flat_a[0], flat_b[0])

    num_elements = 1
    for d in out_shape:
        num_elements *= d

    pad_s1 = (1,) * (len(out_shape) - len(s1)) + s1
    pad_s2 = (1,) * (len(out_shape) - len(s2)) + s2

    strides_out = _strides_for_shape(out_shape)
    strides_a = _strides_for_shape(pad_s1)
    strides_b = _strides_for_shape(pad_s2)

    out_flat = [0.0] * num_elements
    for idx in range(num_elements):
        temp = idx
        coords = []
        for d, s in zip(out_shape, strides_out):
            coord = (temp // s) % d
            coords.append(coord)
            temp %= s

        idx_a = 0
        for c, d, s in zip(coords, pad_s1, strides_a):
            idx_a += (c % d) * s

        idx_b = 0
        for c, d, s in zip(coords, pad_s2, strides_b):
            idx_b += (c % d) * s

        out_flat[idx] = op(flat_a[idx_a], flat_b[idx_b])

    return _unflatten(out_flat, out_shape)

def _elementwise_unary(data: Any, op: Callable[[Any], Any]) -> Any:
    """Perform elementwise unary operation."""
    shape = _infer_shape(data)
    flat = _flatten(data)
    out_flat = [op(x) for x in flat]
    return _unflatten(out_flat, shape)


class PythonBackend(BaseBackend):
    """Zero-dependency pure Python list compute backend."""

    @property
    def name(self) -> str:
        return "python"

    def from_data(self, data: Any, dtype: Optional[str] = "float32") -> Any:
        dtype = dtype or "float32"
        if isinstance(data, (int, float, bool)):
            return _normalize_data_dtype(data, dtype)
        if isinstance(data, (list, tuple)):
            _infer_shape(data)
            return _normalize_data_dtype(data, dtype)
        if hasattr(data, "tolist"):
            res = data.tolist()
            _infer_shape(res)
            return _normalize_data_dtype(res, dtype)
        raise TypeError(f"Cannot convert {type(data)} to PythonBackend structure")

    def get_shape(self, data: Any) -> Shape:
        return _infer_shape(data)

    def to_flat_list(self, data: Any) -> List[Any]:
        return _flatten(data)

    def to_nested_list(self, data: Any) -> Any:
        return data

    def zeros(self, shape: Shape, dtype: str = "float32") -> Any:
        val = 0 if dtype == "int64" else (False if dtype == "bool" else 0.0)
        if len(shape) == 0:
            return val
        num = 1
        for d in shape:
            num *= d
        return _unflatten([val] * num, shape)

    def ones(self, shape: Shape, dtype: str = "float32") -> Any:
        val = 1 if dtype == "int64" else (True if dtype == "bool" else 1.0)
        if len(shape) == 0:
            return val
        num = 1
        for d in shape:
            num *= d
        return _unflatten([val] * num, shape)

    def randn(self, shape: Shape, mean: float = 0.0, std: float = 1.0) -> Any:
        if len(shape) == 0:
            return random.gauss(mean, std)
        num = 1
        for d in shape:
            num *= d
        flat = [random.gauss(mean, std) for _ in range(num)]
        return _unflatten(flat, shape)

    def reshape(self, data: Any, new_shape: Shape) -> Any:
        flat = _flatten(data)
        target_num = 1
        for d in new_shape:
            target_num *= d
        if len(flat) != target_num:
            raise ValueError(f"Cannot reshape tensor of size {len(flat)} into shape {new_shape}")
        return _unflatten(flat, new_shape)

    def transpose(self, data: Any, axes: Tuple[int, ...] = None) -> Any:
        shape = _infer_shape(data)
        ndim = len(shape)
        if ndim < 2:
            return data

        if axes is None:
            axes = tuple(range(ndim - 1, -1, -1))
        elif len(axes) != ndim:
            raise ValueError(f"axes must match dimension count {ndim}, got {len(axes)}")

        new_shape = tuple(shape[i] for i in axes)
        flat = _flatten(data)
        strides_old = _strides_for_shape(shape)
        strides_new = _strides_for_shape(new_shape)

        num_elements = len(flat)
        out_flat = [0.0] * num_elements

        for idx in range(num_elements):
            temp = idx
            new_coords = []
            for d, s in zip(new_shape, strides_new):
                new_coords.append((temp // s) % d)
                temp %= s

            old_idx = 0
            for axis_idx, coord in enumerate(new_coords):
                orig_dim = axes[axis_idx]
                old_idx += coord * strides_old[orig_dim]

            out_flat[idx] = flat[old_idx]

        return _unflatten(out_flat, new_shape)

    def add(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x + y)

    def sub(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x - y)

    def mul(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x * y)

    def div(self, a: Any, b: Any) -> Any:
        def _div(x: Any, y: Any) -> float:
            fx, fy = float(x), float(y)
            if fy == 0.0:
                if fx > 0.0:
                    return float('inf')
                elif fx < 0.0:
                    return float('-inf')
                return float('nan')
            return fx / fy
        return _elementwise_binary(a, b, _div)

    def pow(self, a: Any, exp: float) -> Any:
        return _elementwise_unary(a, lambda x: x ** exp)

    def exp(self, a: Any) -> Any:
        return _elementwise_unary(a, lambda x: math.exp(x))

    def sqrt(self, a: Any) -> Any:
        def _sqrt(x: Any) -> float:
            fx = float(x)
            if fx < 0.0:
                return float("nan")
            return math.sqrt(fx)
        return _elementwise_unary(a, _sqrt)

    def neg(self, a: Any) -> Any:
        return _elementwise_unary(a, lambda x: -x)

    def matmul(self, a: Any, b: Any) -> Any:
        """
        Generalized N-D Batched Matrix Multiplication with right-aligned batch broadcasting.
        """
        s1 = _infer_shape(a)
        s2 = _infer_shape(b)
        r1, r2 = len(s1), len(s2)

        if r1 == 0 or r2 == 0:
            raise ValueError(f"Cannot perform matmul with scalar operand (shapes {s1} and {s2})")

        # 1D @ 1D -> scalar dot product
        if r1 == 1 and r2 == 1:
            if s1[0] != s2[0]:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for vector dot product: {s1[0]} != {s2[0]}")
            flat_a, flat_b = _flatten(a), _flatten(b)
            return sum(x * y for x, y in zip(flat_a, flat_b))

        # Promote 1D operands
        s1_prom = (1, s1[0]) if r1 == 1 else s1
        s2_prom = (s2[0], 1) if r2 == 1 else s2

        batch1, batch2 = s1_prom[:-2], s2_prom[:-2]
        M, K1 = s1_prom[-2], s1_prom[-1]
        K2, N = s2_prom[-2], s2_prom[-1]

        if K1 != K2:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for matmul: contracting dim {K1} != {K2}")

        out_batch = _broadcast_shapes(batch1, batch2)
        out_prom_shape = out_batch + (M, N)

        # Pad batch shapes for indexing
        pad_batch1 = (1,) * (len(out_batch) - len(batch1)) + batch1
        pad_batch2 = (1,) * (len(out_batch) - len(batch2)) + batch2

        strides_out_batch = _strides_for_shape(out_batch) if out_batch else []
        strides_b1 = _strides_for_shape(pad_batch1) if pad_batch1 else []
        strides_b2 = _strides_for_shape(pad_batch2) if pad_batch2 else []

        flat_a = _flatten(a)
        flat_b = _flatten(b)

        matrix_size_a = M * K1
        matrix_size_b = K2 * N
        matrix_size_out = M * N

        num_batches = 1
        for d in out_batch:
            num_batches *= d

        total_out_elements = num_batches * matrix_size_out
        out_flat = [0.0] * total_out_elements

        for b_idx in range(num_batches):
            # Compute batch coordinate
            temp = b_idx
            coords = []
            for d, s in zip(out_batch, strides_out_batch):
                coords.append((temp // s) % d)
                temp %= s

            # Compute source batch index for A and B
            src_b1 = 0
            for c, d, s in zip(coords, pad_batch1, strides_b1):
                src_b1 += (c % d) * s

            src_b2 = 0
            for c, d, s in zip(coords, pad_batch2, strides_b2):
                src_b2 += (c % d) * s

            offset_a = src_b1 * matrix_size_a
            offset_b = src_b2 * matrix_size_b
            offset_out = b_idx * matrix_size_out

            # 2D Matrix multiply for this batch slice
            for i in range(M):
                row_a_start = offset_a + i * K1
                row_out_start = offset_out + i * N
                for k in range(K1):
                    a_ik = flat_a[row_a_start + k]
                    if a_ik == 0.0:
                        continue
                    row_b_start = offset_b + k * N
                    for j in range(N):
                        out_flat[row_out_start + j] += a_ik * flat_b[row_b_start + j]

        # Squeeze promoted 1D dimensions
        final_out = _unflatten(out_flat, out_prom_shape)
        if r1 == 1 and r2 > 1:
            # Squeeze dim -2 (was shape (..., 1, N) -> (..., N))
            s_final = out_batch + (N,)
            return _unflatten(out_flat, s_final)
        elif r1 > 1 and r2 == 1:
            # Squeeze dim -1 (was shape (..., M, 1) -> (..., M))
            s_final = out_batch + (M,)
            return _unflatten(out_flat, s_final)

        return final_out

    def sum(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        shape = _infer_shape(data)
        ndim = len(shape)
        flat = _flatten(data)
        if axis is None or ndim == 0:
            total = sum(flat)
            if keepdims:
                target_shape = (1,) * ndim
                return _unflatten([total], target_shape)
            return total

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a + ndim if a < 0 else a for a in axes)

        out_shape = tuple(shape[i] for i in range(ndim) if i not in axes)
        pad_out_shape = tuple(1 if i in axes else shape[i] for i in range(ndim))

        num_out = 1
        for d in pad_out_shape:
            num_out *= d

        strides_in = _strides_for_shape(shape)
        strides_pad = _strides_for_shape(pad_out_shape)

        out_flat = [0.0] * num_out
        for idx_in, val in enumerate(flat):
            temp = idx_in
            coords = []
            for d, s in zip(shape, strides_in):
                coords.append((temp // s) % d)
                temp %= s

            idx_out = 0
            for axis_idx, (c, s) in enumerate(zip(coords, strides_pad)):
                if axis_idx not in axes:
                    idx_out += c * s
            out_flat[idx_out] += val

        target_shape = pad_out_shape if keepdims else out_shape
        return _unflatten(out_flat, target_shape)

    def max(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        shape = _infer_shape(data)
        ndim = len(shape)
        flat = _flatten(data)
        if axis is None or ndim == 0:
            max_val = max(flat)
            if keepdims:
                target_shape = (1,) * ndim
                return _unflatten([max_val], target_shape)
            return max_val

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a + ndim if a < 0 else a for a in axes)

        out_shape = tuple(shape[i] for i in range(ndim) if i not in axes)
        pad_out_shape = tuple(1 if i in axes else shape[i] for i in range(ndim))

        num_out = 1
        for d in pad_out_shape:
            num_out *= d

        strides_in = _strides_for_shape(shape)
        strides_pad = _strides_for_shape(pad_out_shape)

        out_flat = [-float('inf')] * num_out
        for idx_in, val in enumerate(flat):
            temp = idx_in
            coords = []
            for d, s in zip(shape, strides_in):
                coords.append((temp // s) % d)
                temp %= s

            idx_out = 0
            for axis_idx, (c, s) in enumerate(zip(coords, strides_pad)):
                if axis_idx not in axes:
                    idx_out += c * s
            if val > out_flat[idx_out]:
                out_flat[idx_out] = val

        target_shape = pad_out_shape if keepdims else out_shape
        return _unflatten(out_flat, target_shape)

    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        shape = _infer_shape(data)
        ndim = len(shape)
        summed = self.sum(data, axis=axis, keepdims=keepdims)

        if axis is None or ndim == 0:
            num_elements = len(_flatten(data))
            return self.div(summed, float(max(1, num_elements)))

        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = 1
        for a in axes:
            norm_a = a + ndim if a < 0 else a
            count *= shape[norm_a]

        return self.div(summed, float(max(1, count)))

    def relu(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: max(0.0, x))

    def sigmoid(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: 1.0 / (1.0 + math.exp(-max(-88.0, min(88.0, x)))))

    def tanh(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: math.tanh(x))

    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        current_shape = _infer_shape(grad)
        if current_shape == target_shape:
            return grad

        cur_ndim = len(current_shape)
        tgt_ndim = len(target_shape)
        pad = cur_ndim - tgt_ndim

        out = grad
        for _ in range(pad):
            out = self.sum(out, axis=0, keepdims=False)

        for i in range(tgt_ndim):
            if target_shape[i] == 1 and current_shape[i + pad] > 1:
                out = self.sum(out, axis=i, keepdims=True)

        return out

    def clamp(self, data: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        def _c(x):
            if min_val is not None and x < min_val:
                return min_val
            if max_val is not None and x > max_val:
                return max_val
            return x
        return _elementwise_unary(data, _c)

    def log(self, data: Any) -> Any:
        def _log(x: Any) -> float:
            fx = float(x)
            if fx < 0.0:
                return float('nan')
            elif fx == 0.0:
                return float('-inf')
            return math.log(fx)
        return _elementwise_unary(data, _log)

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        shape = _infer_shape(data)
        if axis != 0:
            raise NotImplementedError("take is currently only supported along axis 0")
        if not isinstance(data, list):
            raise TypeError(f"Cannot index non-list data of shape {shape}")
        return data[index]

    def gather_rows(self, weight_data: Any, row_indices: List[int], out_shape: Tuple[int, ...]) -> Any:
        w_flat = self.to_flat_list(weight_data)
        e_dim = out_shape[-1]
        out_flat = []
        for idx in row_indices:
            start = idx * e_dim
            out_flat.extend(w_flat[start:start + e_dim])
        return self.from_data(self.reshape(out_flat, out_shape), dtype="float32")

    def scatter_add_rows(self, target_data: Any, row_indices: List[int], grad_data: Any, padding_idx: Optional[int] = None) -> Any:
        target_flat = self.to_flat_list(target_data)
        grad_flat = self.to_flat_list(grad_data)
        e_dim = len(target_flat) // self.get_shape(target_data)[0]
        for sample_i, token_idx in enumerate(row_indices):
            if padding_idx is not None and token_idx == padding_idx:
                continue
            g_start = sample_i * e_dim
            w_start = token_idx * e_dim
            for dim_j in range(e_dim):
                target_flat[w_start + dim_j] += grad_flat[g_start + dim_j]
        return self.from_data(self.reshape(target_flat, self.get_shape(target_data)), dtype="float32")
