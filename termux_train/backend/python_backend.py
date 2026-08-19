"""
termux_train.backend.python_backend
===================================
0-Dependency Pure-Python Compute Backend.
Uses nested Python lists and pure scalar math without any external binary dependencies.
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

def _to_float_nested(data: Any) -> Any:
    """Recursively cast all elements to float."""
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, (list, tuple)):
        return [_to_float_nested(x) for x in data]
    raise TypeError(f"Unsupported data element type: {type(data)}")

def _flatten(data: Any) -> List[float]:
    """Recursively flatten nested lists to 1D flat list."""
    if isinstance(data, (int, float)):
        return [float(data)]
    res = []
    for item in data:
        res.extend(_flatten(item))
    return res

def _unflatten(flat: Sequence[float], shape: Shape) -> Any:
    """Rebuild nested lists from flat list and shape tuple."""
    if len(shape) == 0:
        return float(flat[0])
    if len(shape) == 1:
        return [float(x) for x in flat[:shape[0]]]
    
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

def _elementwise_binary(a_data: Any, b_data: Any, op: Callable[[float, float], float]) -> Any:
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
        # Decode multi-dimensional index
        temp = idx
        coords = []
        for d, s in zip(out_shape, strides_out):
            coord = (temp // s) % d
            coords.append(coord)
            temp %= s
            
        # Map to a index
        idx_a = 0
        for c, d, s in zip(coords, pad_s1, strides_a):
            idx_a += (c % d) * s
            
        # Map to b index
        idx_b = 0
        for c, d, s in zip(coords, pad_s2, strides_b):
            idx_b += (c % d) * s
            
        out_flat[idx] = op(flat_a[idx_a], flat_b[idx_b])
        
    return _unflatten(out_flat, out_shape)

def _elementwise_unary(data: Any, op: Callable[[float], float]) -> Any:
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

    def from_data(self, data: Any) -> Any:
        if isinstance(data, (int, float)):
            return float(data)
        if isinstance(data, (list, tuple)):
            _infer_shape(data) # Validates shape and catches ragged lists
            return _to_float_nested(data)
        # Handle objects with tolist() (like numpy arrays)
        if hasattr(data, "tolist"):
            res = data.tolist()
            _infer_shape(res)
            return _to_float_nested(res)
        raise TypeError(f"Cannot convert {type(data)} to PythonBackend structure")

    def get_shape(self, data: Any) -> Shape:
        return _infer_shape(data)

    def to_flat_list(self, data: Any) -> List[float]:
        return _flatten(data)

    def to_nested_list(self, data: Any) -> Any:
        return data

    def zeros(self, shape: Shape) -> Any:
        if len(shape) == 0:
            return 0.0
        num = 1
        for d in shape:
            num *= d
        return _unflatten([0.0] * num, shape)

    def ones(self, shape: Shape) -> Any:
        if len(shape) == 0:
            return 1.0
        num = 1
        for d in shape:
            num *= d
        return _unflatten([1.0] * num, shape)

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

    def transpose(self, data: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        shape = _infer_shape(data)
        ndim = len(shape)
        if ndim <= 1:
            return data
        if axes is None:
            axes = tuple(reversed(range(ndim)))
        else:
            axes = tuple(a + ndim if a < 0 else a for a in axes)
            if len(axes) != ndim or set(axes) != set(range(ndim)):
                raise ValueError(f"Invalid axes {axes} for {ndim}D tensor")

        out_shape = tuple(shape[a] for a in axes)
        flat = _flatten(data)
        out_flat = [0.0] * len(flat)
        
        in_strides = _strides_for_shape(shape)
        out_strides = _strides_for_shape(out_shape)

        for in_idx in range(len(flat)):
            temp = in_idx
            coords = [0] * ndim
            for i in range(ndim):
                coords[i] = temp // in_strides[i]
                temp %= in_strides[i]
                
            out_coords = [coords[a] for a in axes]
            out_idx = sum(c * s for c, s in zip(out_coords, out_strides))
            out_flat[out_idx] = flat[in_idx]

        return _unflatten(out_flat, out_shape)

    def add(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x + y)

    def sub(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x - y)

    def mul(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x * y)

    def div(self, a: Any, b: Any) -> Any:
        return _elementwise_binary(a, b, lambda x, y: x / (y if y != 0 else 1e-12))

    def pow(self, a: Any, exp: float) -> Any:
        return _elementwise_unary(a, lambda x: x ** exp)

    def neg(self, a: Any) -> Any:
        return _elementwise_unary(a, lambda x: -x)

    def matmul(self, a: Any, b: Any) -> Any:
        s1 = _infer_shape(a)
        s2 = _infer_shape(b)
        r1, r2 = len(s1), len(s2)

        if r1 not in (1, 2, 3) or r2 not in (1, 2, 3):
            raise NotImplementedError(
                "matmul supports every rank combination where both operands "
                f"are between 1D and 3D. Received shapes {s1} and {s2}. "
                "Scalar operands and 4D+ ND matmul are not supported."
            )

        # 1. 1D @ 1D -> scalar dot product ()
        if r1 == 1 and r2 == 1:
            if s1[0] != s2[0]:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for vector dot product: {s1[0]} != {s2[0]}")
            flat_a, flat_b = _flatten(a), _flatten(b)
            return sum(x * y for x, y in zip(flat_a, flat_b))

        # 2. 1D @ 2D -> (N,)
        if r1 == 1 and r2 == 2:
            K1 = s1[0]
            K2, N = s2
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 1D@2D matmul: {K1} != {K2}")
            flat_a = _flatten(a)
            flat_b = _flatten(b)
            out = [0.0] * N
            for j in range(N):
                out[j] = sum(flat_a[k] * flat_b[k * N + j] for k in range(K1))
            return out

        # 3. 1D @ 3D -> (B, N)
        if r1 == 1 and r2 == 3:
            K1 = s1[0]
            B, K2, N = s2
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 1D@3D matmul: {K1} != {K2}")
            return [self.matmul(a, b[b_idx]) for b_idx in range(B)]

        # 4. 2D @ 1D -> (M,)
        if r1 == 2 and r2 == 1:
            M, K1 = s1
            K2 = s2[0]
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 2D@1D matmul: {K1} != {K2}")
            flat_a = _flatten(a)
            flat_b = _flatten(b)
            out = [0.0] * M
            for i in range(M):
                out[i] = sum(flat_a[i * K1 + k] * flat_b[k] for k in range(K1))
            return out

        # 5. 2D @ 2D -> (M, N)
        if r1 == 2 and r2 == 2:
            M, K1 = s1
            K2, N = s2
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 2D@2D matmul: {K1} != {K2}")
            flat_a = _flatten(a)
            flat_b = _flatten(b)
            out_flat = [0.0] * (M * N)
            for i in range(M):
                for k in range(K1):
                    a_ik = flat_a[i * K1 + k]
                    if a_ik == 0:
                        continue
                    for j in range(N):
                        out_flat[i * N + j] += a_ik * flat_b[k * N + j]
            return _unflatten(out_flat, (M, N))

        # 6. 2D @ 3D -> (B, M, N)
        if r1 == 2 and r2 == 3:
            M, K1 = s1
            B, K2, N = s2
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 2D@3D matmul: {K1} != {K2}")
            return [self.matmul(a, b[b_idx]) for b_idx in range(B)]

        # 7. 3D @ 1D -> (B, M)
        if r1 == 3 and r2 == 1:
            B, M, K1 = s1
            K2 = s2[0]
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@1D matmul: {K1} != {K2}")
            return [self.matmul(a[b_idx], b) for b_idx in range(B)]

        # 8. 3D @ 2D -> (B, M, N)
        if r1 == 3 and r2 == 2:
            B, M, K1 = s1
            K2, N = s2
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@2D matmul: {K1} != {K2}")
            return [self.matmul(a[b_idx], b) for b_idx in range(B)]

        # 9. 3D @ 3D -> (B, M, N)
        if r1 == 3 and r2 == 3:
            B1, M, K1 = s1
            B2, K2, N = s2
            if B1 != B2:
                raise ValueError(f"Batch dimensions must match: {B1} != {B2}")
            if K1 != K2:
                raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@3D matmul: {K1} != {K2}")
            return [self.matmul(a[b_idx], b[b_idx]) for b_idx in range(B1)]

        raise NotImplementedError(f"Matmul for shapes {s1} and {s2} not supported in PythonBackend")

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
            
        out_flat = [0.0] * num_out
        in_strides = _strides_for_shape(shape)
        out_strides = _strides_for_shape(pad_out_shape)

        for in_idx in range(len(flat)):
            temp = in_idx
            coords = [0] * ndim
            for i in range(ndim):
                coords[i] = temp // in_strides[i]
                temp %= in_strides[i]
                
            out_coords = [0 if i in axes else coords[i] for i in range(ndim)]
            out_idx = sum(c * s for c, s in zip(out_coords, out_strides))
            out_flat[out_idx] += flat[in_idx]

        final_shape = pad_out_shape if keepdims else out_shape
        return _unflatten(out_flat, final_shape)

    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        sum_res = self.sum(data, axis=axis, keepdims=keepdims)
        shape = _infer_shape(data)
        ndim = len(shape)
        if axis is None or ndim == 0:
            count = len(_flatten(data))
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(a + ndim if a < 0 else a for a in axes)
            count = 1
            for a in axes:
                count *= shape[a]
        scale = 1.0 / max(count, 1)
        return self.mul(sum_res, scale)

    def relu(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: max(0.0, x))

    def sigmoid(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x)))))

    def tanh(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: math.tanh(x))

    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        """Collapse broadcasted gradient back to target shape."""
        grad_shape = _infer_shape(grad)
        if grad_shape == target_shape:
            return grad
            
        flat_grad = _flatten(grad)
        if len(target_shape) == 0:
            return sum(flat_grad)
            
        # If target shape has fewer dimensions, sum over leading dimensions
        dim_diff = len(grad_shape) - len(target_shape)
        pad_target = (1,) * dim_diff + target_shape
        
        # Collapse dimensions where target has 1
        num_target = 1
        for d in target_shape:
            num_target *= d
            
        target_flat = [0.0] * num_target
        
        strides_grad = _strides_for_shape(grad_shape)
        strides_target = _strides_for_shape(pad_target)
        
        for idx in range(len(flat_grad)):
            temp = idx
            coords = []
            for d, s in zip(grad_shape, strides_grad):
                coords.append((temp // s) % d)
                temp %= s
                
            idx_t = 0
            for c, d, s in zip(coords, pad_target, strides_target):
                idx_t += (c % d) * s
                
            target_flat[idx_t] += flat_grad[idx]
            
        return _unflatten(target_flat, target_shape)

    def clamp(self, data: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        def _c(x):
            if min_val is not None and x < min_val:
                return float(min_val)
            if max_val is not None and x > max_val:
                return float(max_val)
            return float(x)
        return _elementwise_unary(data, _c)

    def log(self, data: Any) -> Any:
        return _elementwise_unary(data, lambda x: math.log(max(x, 1e-12)))

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        if axis != 0:
            raise NotImplementedError("take currently supports axis=0 only in PythonBackend")
        return data[index]
