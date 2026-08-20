"""
termux_train.tensor
===================
Core Multi-Dimensional Tensor Class with Dynamic DAG Autograd Engine.
Supports multi-dtype representation (float32, int64, bool), explicit Type Promotion,
in-place version tracking, in-flight DAG dissection (instant memory release on backward),
selective value saving, iterative DAG traversal, generalized N-D batched matmul,
no_grad context manager, and Transformer mathematical primitives.
"""

import functools
import math
from contextvars import ContextVar
from typing import Any, Tuple, Set, List, Optional, Union, Callable
from .backend import get_backend, BaseBackend

VALID_DTYPES = {"float32", "int64", "bool"}

_GRAD_ENABLED_VAR: ContextVar[bool] = ContextVar("termux_train_grad_enabled", default=True)

def _promote_dtype(dt1: str, dt2: str) -> str:
    """Explicit type promotion rule matrix."""
    if dt1 == "float32" or dt2 == "float32":
        return "float32"
    if dt1 == "int64" or dt2 == "int64":
        return "int64"
    return "bool"

def _invert_permutation(axes: Tuple[int, ...]) -> Tuple[int, ...]:
    inv = [0] * len(axes)
    for i, a in enumerate(axes):
        inv[a] = i
    return tuple(inv)

def _normalize_axes(axis: Union[int, Tuple[int, ...], List[int], None], ndim: int) -> Tuple[int, ...]:
    if axis is None or ndim == 0:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    res = []
    for a in axes:
        norm = a + ndim if a < 0 else a
        if not (0 <= norm < ndim):
            raise ValueError(f"axis {a} is out of bounds for tensor of dimension {ndim}")
        res.append(norm)
    return tuple(sorted(set(res)))

def _reduced_count(shape: Tuple[int, ...], axes: Tuple[int, ...]) -> int:
    count = 1
    for a in axes:
        if a < len(shape):
            count *= shape[a]
    return max(1, count)

def _flatten_data_types(data: Any) -> List[Any]:
    if isinstance(data, (list, tuple)):
        res = []
        for x in data:
            res.extend(_flatten_data_types(x))
        return res
    return [data]

def _infer_dtype_from_data(data: Any) -> str:
    flat = _flatten_data_types(data)
    if len(flat) == 0:
        return "float32"
    if all(isinstance(x, bool) for x in flat):
        return "bool"
    if all(isinstance(x, int) and not isinstance(x, bool) for x in flat):
        return "int64"
    return "float32"

def _unbroadcast_to(grad_tensor: 'Tensor', target_shape: Tuple[int, ...]) -> 'Tensor':
    """Sum out broadcast dimensions to match target_shape."""
    current_shape = grad_tensor.shape
    if current_shape == target_shape:
        return grad_tensor

    cur_ndim = len(current_shape)
    tgt_ndim = len(target_shape)
    pad = cur_ndim - tgt_ndim

    out = grad_tensor
    for _ in range(pad):
        out = out.sum(axis=0, keepdims=False)

    for i in range(tgt_ndim):
        if target_shape[i] == 1 and out.shape[i] > 1:
            out = out.sum(axis=i, keepdims=True)

    return out

def _attach_grad_fn(out: 'Tensor', parents: Tuple['Tensor', ...], backward_fn: Optional[Callable[[], None]]) -> None:
    """Attach autograd closure only when out.requires_grad is True, preventing closure leaks in no_grad."""
    if out.requires_grad and backward_fn is not None:
        out._prev = set(parents)
        out._backward = backward_fn
        out._grad_fn_state = "live"
    else:
        out._prev = set()
        out._backward = None
        out._grad_fn_state = "leaf"


class no_grad:
    """
    Context-manager and decorator that disables gradient calculation.
    Thread-safe and async-safe via contextvars.
    """
    def __init__(self):
        self._token = None

    def __enter__(self):
        self._token = _GRAD_ENABLED_VAR.set(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _GRAD_ENABLED_VAR.reset(self._token)
            self._token = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class Tensor:
    """
    Core Tensor class supporting multi-dimensional arrays, pluggable backends,
    multi-dtype representation (float32, int64, bool), in-place version tracking,
    in-flight memory release on backward, and dynamic reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: Any,
        dtype: Optional[str] = None,
        requires_grad: bool = False,
        _prev: Tuple['Tensor', ...] = (),
        _op: str = "",
        backend: Optional[BaseBackend] = None
    ):
        if dtype is not None and dtype not in VALID_DTYPES:
            raise ValueError(f"Unsupported dtype: '{dtype}'. Valid dtypes are {sorted(VALID_DTYPES)}")

        if isinstance(data, Tensor):
            self.dtype = dtype or data.dtype
            self.backend = backend or data.backend
            self._data = self.backend.from_data(data.tolist(), dtype=self.dtype)
        else:
            self.backend = backend or get_backend()
            inferred = _infer_dtype_from_data(data) if dtype is None else dtype
            self.dtype = inferred
            self._data = self.backend.from_data(data, dtype=self.dtype)

        effective_requires_grad = requires_grad and _GRAD_ENABLED_VAR.get()

        if self.dtype in ("int64", "bool") and effective_requires_grad:
            raise ValueError(f"Only Tensors with floating point dtype can require gradients (got dtype='{self.dtype}')")

        self.requires_grad: bool = effective_requires_grad
        self.grad: Optional['Tensor'] = None
        self._backward: Optional[Callable[[], None]] = None
        self._prev: Set['Tensor'] = set(_prev) if effective_requires_grad else set()
        self._op: str = _op
        self._version: int = 0
        self._grad_fn_state: str = "live" if (effective_requires_grad and _prev) else "leaf"

    @classmethod
    def is_grad_enabled(cls) -> bool:
        return _GRAD_ENABLED_VAR.get()

    @classmethod
    def set_grad_enabled(cls, mode: bool) -> None:
        _GRAD_ENABLED_VAR.set(bool(mode))

    def _replace_data(self, value: Any, *, bump_version: bool = True) -> None:
        """Atomic internal data replacement with monotonic version increment."""
        if isinstance(value, Tensor):
            self._data = self.backend.from_data(value.tolist(), dtype=self.dtype)
        else:
            self._data = self.backend.from_data(value, dtype=self.dtype)
        if bump_version:
            self._version += 1

    def _accumulate_grad_data(self, grad_data: Any) -> None:
        """Accumulate incoming raw gradient data into self.grad safely."""
        if not self.requires_grad:
            return
        g_data = self.backend.from_data(grad_data, dtype="float32")
        if self.grad is None:
            self.grad = Tensor(g_data, dtype="float32", requires_grad=False, backend=self.backend)
        else:
            self.grad._data = self.backend.add(self.grad._data, g_data)

    def _ensure_tensor_on_self_backend(self, other: Any) -> 'Tensor':
        """Normalize other operand to Tensor on self.backend."""
        if not isinstance(other, Tensor):
            return Tensor(other, backend=self.backend)

        if other.backend is self.backend or other.backend.name == self.backend.name:
            return other

        if other.requires_grad:
            raise RuntimeError(
                f"Cross-backend autograd operation is not supported between {self.backend.name} "
                f"and {other.backend.name} when requires_grad=True."
            )

        return Tensor(other.tolist(), dtype=other.dtype, requires_grad=False, backend=self.backend)

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._replace_data(value, bump_version=True)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.backend.get_shape(self._data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    def item(self) -> Union[float, int, bool]:
        flat = self.backend.to_flat_list(self._data)
        if len(flat) != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars (got size {len(flat)})")
        val = flat[0]
        if self.dtype == "int64":
            return int(val)
        if self.dtype == "bool":
            return bool(val)
        return float(val)

    def tolist(self) -> Any:
        return self.backend.to_nested_list(self._data)

    def to(self, backend: Union[str, BaseBackend]) -> 'Tensor':
        if isinstance(backend, str):
            from .backend import get_backend
            target_backend = get_backend(backend)
        else:
            target_backend = backend

        if target_backend.name == self.backend.name:
            return self

        return Tensor(
            self.tolist(),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            backend=target_backend
        )

    def detach(self) -> 'Tensor':
        """Returns a new Tensor detached from the current autograd computation graph."""
        return Tensor(
            self.tolist(),
            dtype=self.dtype,
            requires_grad=False,
            backend=self.backend
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        if not self.requires_grad:
            self.grad = None
            return
        if set_to_none:
            self.grad = None
        else:
            self.grad = Tensor(self.backend.zeros(self.shape), dtype="float32", requires_grad=False, backend=self.backend)

    # =========================================================================
    # Reshaping & Permutation
    # =========================================================================

    def reshape(self, *shape: Union[int, Tuple[int, ...], List[int]]) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)

        num_neg_ones = new_shape.count(-1)
        if num_neg_ones > 1:
            raise ValueError("can only specify one unknown dimension (-1)")

        if num_neg_ones == 1:
            cur_elements = 1
            for d in self.shape:
                cur_elements *= d

            known_elements = 1
            for d in new_shape:
                if d != -1:
                    known_elements *= d

            if cur_elements % known_elements != 0:
                raise ValueError(f"Cannot reshape tensor of size {cur_elements} into shape {new_shape}")

            inferred_dim = cur_elements // known_elements
            new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)

        reshaped_data = self.backend.reshape(self._data, new_shape)
        out = Tensor(
            reshaped_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="reshape",
            backend=self.backend
        )

        orig_shape = self.shape
        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                d_self = backend.reshape(out.grad._data, orig_shape)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def flatten(self) -> 'Tensor':
        """Flatten tensor to a 1D 1-dimensional tensor."""
        return self.reshape(-1)

    def transpose(self, *axes: int) -> 'Tensor':
        ndim = self.ndim
        if len(axes) == 0:
            if ndim < 2:
                return self
            axes_tuple = tuple(range(ndim - 1, -1, -1))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_tuple = tuple(axes[0])
        else:
            axes_tuple = tuple(axes)

        if len(axes_tuple) != ndim:
            raise ValueError(f"axes must match dimension count {ndim}, got {len(axes_tuple)}")

        norm_axes = []
        for a in axes_tuple:
            na = a + ndim if a < 0 else a
            if not (0 <= na < ndim):
                raise ValueError(f"axis {a} is out of bounds for tensor of dimension {ndim}")
            norm_axes.append(na)

        if len(set(norm_axes)) != ndim:
            raise ValueError(f"repeated axis in transpose: {axes_tuple}")

        axes_tuple = tuple(norm_axes)
        trans_data = self.backend.transpose(self._data, axes_tuple)
        out = Tensor(
            trans_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="transpose",
            backend=self.backend
        )

        inv_axes = _invert_permutation(axes_tuple)
        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                d_self = backend.transpose(out.grad._data, inv_axes)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def swapaxes(self, dim0: int, dim1: int) -> 'Tensor':
        """Swap two dimensions of a tensor."""
        ndim = self.ndim
        d0 = dim0 + ndim if dim0 < 0 else dim0
        d1 = dim1 + ndim if dim1 < 0 else dim1
        if not (0 <= d0 < ndim) or not (0 <= d1 < ndim):
            raise ValueError(f"swapaxes dimensions ({dim0}, {dim1}) out of bounds for ndim={ndim}")
        axes = list(range(ndim))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        return self.transpose(*axes)

    # =========================================================================
    # Arithmetic & Autograd Operators
    # =========================================================================

    def __add__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        out_dtype = _promote_dtype(self.dtype, other.dtype)
        out_data = self.backend.add(self._data, other._data)
        out = Tensor(
            out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="+",
            backend=self.backend
        )

        saved_v_self = self._version
        saved_v_other = other._version
        s_self = self.shape
        s_other = other.shape
        req_self = self.requires_grad
        req_other = other.requires_grad
        backend = self.backend

        def _backward():
            if self._version != saved_v_self or other._version != saved_v_other:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if out.grad is not None:
                if req_self:
                    d_self = backend.unbroadcast(out.grad._data, s_self)
                    self._accumulate_grad_data(d_self)
                if req_other:
                    d_other = backend.unbroadcast(out.grad._data, s_other)
                    other._accumulate_grad_data(d_other)

        _attach_grad_fn(out, (self, other), _backward)
        return out

    def __radd__(self, other: Any) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        out_dtype = _promote_dtype(self.dtype, other.dtype)
        out_data = self.backend.sub(self._data, other._data)
        out = Tensor(
            out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="-",
            backend=self.backend
        )

        saved_v_self = self._version
        saved_v_other = other._version
        s_self = self.shape
        s_other = other.shape
        req_self = self.requires_grad
        req_other = other.requires_grad
        backend = self.backend

        def _backward():
            if self._version != saved_v_self or other._version != saved_v_other:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if out.grad is not None:
                if req_self:
                    d_self = backend.unbroadcast(out.grad._data, s_self)
                    self._accumulate_grad_data(d_self)
                if req_other:
                    neg_grad = backend.neg(out.grad._data)
                    d_other = backend.unbroadcast(neg_grad, s_other)
                    other._accumulate_grad_data(d_other)

        _attach_grad_fn(out, (self, other), _backward)
        return out

    def __rsub__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        return other.__sub__(self)

    def __mul__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        out_dtype = _promote_dtype(self.dtype, other.dtype)
        out_data = self.backend.mul(self._data, other._data)
        out = Tensor(
            out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="*",
            backend=self.backend
        )

        saved_v_self = self._version
        saved_v_other = other._version
        s_self = self.shape
        s_other = other.shape
        req_self = self.requires_grad
        req_other = other.requires_grad
        backend = self.backend

        def _backward():
            if self._version != saved_v_self or other._version != saved_v_other:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if out.grad is not None:
                if req_self:
                    g_self = backend.mul(out.grad._data, other._data)
                    d_self = backend.unbroadcast(g_self, s_self)
                    self._accumulate_grad_data(d_self)
                if req_other:
                    g_other = backend.mul(out.grad._data, self._data)
                    d_other = backend.unbroadcast(g_other, s_other)
                    other._accumulate_grad_data(d_other)

        _attach_grad_fn(out, (self, other), _backward)
        return out

    def __rmul__(self, other: Any) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        out_data = self.backend.div(self._data, other._data)
        out = Tensor(
            out_data,
            dtype="float32",
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="/",
            backend=self.backend
        )

        saved_v_self = self._version
        saved_v_other = other._version
        s_self = self.shape
        s_other = other.shape
        req_self = self.requires_grad
        req_other = other.requires_grad
        backend = self.backend

        def _backward():
            if self._version != saved_v_self or other._version != saved_v_other:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if out.grad is not None:
                if req_self:
                    g_self = backend.div(out.grad._data, other._data)
                    d_self = backend.unbroadcast(g_self, s_self)
                    self._accumulate_grad_data(d_self)
                if req_other:
                    b_sq = backend.mul(other._data, other._data)
                    neg_a = backend.neg(self._data)
                    deriv_b = backend.div(neg_a, b_sq)
                    g_other = backend.mul(out.grad._data, deriv_b)
                    d_other = backend.unbroadcast(g_other, s_other)
                    other._accumulate_grad_data(d_other)

        _attach_grad_fn(out, (self, other), _backward)
        return out

    def __rtruediv__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        return other.__truediv__(self)

    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be a Python int or float")

        out_data = self.backend.pow(self._data, float(exponent))
        out = Tensor(
            out_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op=f"**{exponent}",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                deriv = backend.mul(
                    backend.pow(self._data, exponent - 1.0),
                    backend.from_data(float(exponent), dtype="float32")
                )
                d_self = backend.mul(deriv, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def __neg__(self) -> 'Tensor':
        out_data = self.backend.neg(self._data)
        out = Tensor(
            out_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="neg",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                d_self = backend.neg(out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Generalized N-D Batched Matrix Multiplication with right-aligned batch broadcasting.
        """
        other = self._ensure_tensor_on_self_backend(other)
        out_dtype = _promote_dtype(self.dtype, other.dtype)
        out_data = self.backend.matmul(self._data, other._data)
        out = Tensor(
            out_data,
            dtype=out_dtype,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="@",
            backend=self.backend
        )

        s1, s2 = self.shape, other.shape
        r1, r2 = len(s1), len(s2)
        saved_v_self = self._version
        saved_v_other = other._version

        def _backward():
            if self._version != saved_v_self or other._version != saved_v_other:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if out.grad is None:
                return

            if r1 == 1 and r2 == 1:
                grad_data = out.grad._data
                if self.requires_grad:
                    d_self = self.backend.mul(grad_data, other._data)
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    d_other = self.backend.mul(grad_data, self._data)
                    other._accumulate_grad_data(d_other)
                return

            G = out.grad
            A = self
            B = other

            if r1 == 1:
                A_prom = A.reshape(1, s1[0])
                G_prom = G.reshape(*G.shape[:-1], 1, G.shape[-1])
            else:
                A_prom = A
                G_prom = G

            if r2 == 1:
                B_prom = B.reshape(s2[0], 1)
                if r1 > 1:
                    G_prom = G.reshape(*G.shape, 1)
            else:
                B_prom = B

            b_ndim = B_prom.ndim
            b_axes = list(range(b_ndim))
            b_axes[-2], b_axes[-1] = b_axes[-1], b_axes[-2]
            B_prom_T = B_prom.transpose(*b_axes)

            a_ndim = A_prom.ndim
            a_axes = list(range(a_ndim))
            a_axes[-2], a_axes[-1] = a_axes[-1], a_axes[-2]
            A_prom_T = A_prom.transpose(*a_axes)

            if self.requires_grad:
                dA_prom = G_prom @ B_prom_T
                dA_unbroadcast = _unbroadcast_to(dA_prom, A_prom.shape)
                if r1 == 1:
                    dA_final = dA_unbroadcast.reshape(s1[0])
                else:
                    dA_final = dA_unbroadcast
                self._accumulate_grad_data(dA_final._data)

            if other.requires_grad:
                dB_prom = A_prom_T @ G_prom
                dB_unbroadcast = _unbroadcast_to(dB_prom, B_prom.shape)
                if r2 == 1:
                    dB_final = dB_unbroadcast.reshape(s2[0])
                else:
                    dB_final = dB_unbroadcast
                other._accumulate_grad_data(dB_final._data)

        _attach_grad_fn(out, (self, other), _backward)
        return out

    def sum(self, axis: Union[int, Tuple[int, ...], List[int], None] = None, keepdims: bool = False) -> 'Tensor':
        ndim = self.ndim
        norm_axes = _normalize_axes(axis, ndim)
        sum_data = self.backend.sum(self._data, axis=norm_axes, keepdims=keepdims)
        out = Tensor(
            sum_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sum",
            backend=self.backend
        )

        orig_shape = self.shape
        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                if not keepdims:
                    pad_shape = tuple(1 if i in norm_axes else orig_shape[i] for i in range(ndim))
                    grad_reshaped = backend.reshape(out.grad._data, pad_shape)
                else:
                    grad_reshaped = out.grad._data

                ones_data = backend.ones(orig_shape)
                d_self = backend.mul(ones_data, grad_reshaped)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def max(self, axis: Union[int, Tuple[int, ...], List[int], None] = None, keepdims: bool = False) -> 'Tensor':
        """
        Maximum reduction along specified axes with subgradient mass conservation across tied maximums.
        """
        ndim = self.ndim
        norm_axes = _normalize_axes(axis, ndim)
        max_data = self.backend.max(self._data, axis=norm_axes, keepdims=keepdims)
        out = Tensor(
            max_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="max",
            backend=self.backend
        )

        orig_shape = self.shape
        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                if not keepdims:
                    pad_shape = tuple(1 if i in norm_axes else orig_shape[i] for i in range(ndim))
                    grad_reshaped = backend.reshape(out.grad._data, pad_shape)
                    max_reshaped = backend.reshape(out._data, pad_shape)
                else:
                    grad_reshaped = out.grad._data
                    max_reshaped = out._data

                ones_data = backend.ones(orig_shape)
                b_grad = backend.mul(ones_data, grad_reshaped)
                b_max = backend.mul(ones_data, max_reshaped)

                flat_x = backend.to_flat_list(self._data)
                flat_max = backend.to_flat_list(b_max)
                flat_g = backend.to_flat_list(b_grad)

                # 1. Indicator mask for tied maximums
                mask_flat = [1.0 if x == m else 0.0 for x, m in zip(flat_x, flat_max)]
                mask_data = backend.from_data(backend.reshape(mask_flat, orig_shape), dtype="float32")

                # 2. Count tied maximums along reduction axes
                k_data = backend.sum(mask_data, axis=norm_axes, keepdims=True)
                k_broadcast = backend.mul(ones_data, k_data)
                flat_k = backend.to_flat_list(k_broadcast)

                # 3. Equidistribute subgradient mass: g / k
                d_flat = [(g / k_val) if m_val > 0.0 else 0.0 for g, m_val, k_val in zip(flat_g, mask_flat, flat_k)]
                d_self = backend.from_data(backend.reshape(d_flat, orig_shape), dtype="float32")
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def mean(self, axis: Union[int, Tuple[int, ...], List[int], None] = None, keepdims: bool = False) -> 'Tensor':
        ndim = self.ndim
        norm_axes = _normalize_axes(axis, ndim)
        count = _reduced_count(self.shape, norm_axes)
        return self.sum(axis=norm_axes, keepdims=keepdims) / float(count)

    def exp(self) -> 'Tensor':
        """Elementwise natural exponential exp(x)."""
        exp_data = self.backend.exp(self._data)
        out = Tensor(
            exp_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="exp",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                d_self = backend.mul(out._data, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def sqrt(self) -> 'Tensor':
        """Elementwise square root sqrt(x)."""
        sqrt_data = self.backend.sqrt(self._data)
        out = Tensor(
            sqrt_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sqrt",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                inv_two_sqrt = backend.div(backend.from_data(0.5, dtype="float32"), out._data)
                d_self = backend.mul(inv_two_sqrt, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def log(self) -> 'Tensor':
        """Natural logarithm ln(x)."""
        log_data = self.backend.log(self._data)
        out = Tensor(
            log_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="log",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                inv_x = backend.div(backend.ones(self.shape), self._data)
                d_self = backend.mul(inv_x, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def relu(self) -> 'Tensor':
        relu_data = self.backend.relu(self._data)
        out = Tensor(
            relu_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="relu",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                flat_data = backend.to_flat_list(self._data)
                flat_grad = backend.to_flat_list(out.grad._data)
                d_flat = [g if x > 0 else 0.0 for x, g in zip(flat_data, flat_grad)]
                d_self = backend.from_data(backend.reshape(d_flat, self.shape), dtype="float32")
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def sigmoid(self) -> 'Tensor':
        sig_data = self.backend.sigmoid(self._data)
        out = Tensor(
            sig_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sigmoid",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                one = backend.ones(self.shape)
                one_minus_sig = backend.sub(one, out._data)
                deriv = backend.mul(out._data, one_minus_sig)
                d_self = backend.mul(deriv, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def tanh(self) -> 'Tensor':
        tanh_data = self.backend.tanh(self._data)
        out = Tensor(
            tanh_data,
            dtype="float32",
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="tanh",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                one = backend.ones(self.shape)
                tanh_sq = backend.mul(out._data, out._data)
                deriv = backend.sub(one, tanh_sq)
                d_self = backend.mul(deriv, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def clamp(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'Tensor':
        clamped_data = self.backend.clamp(self._data, min_val, max_val)
        out = Tensor(
            clamped_data,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="clamp",
            backend=self.backend
        )

        saved_v = self._version
        backend = self.backend

        def _backward():
            if self._version != saved_v:
                raise RuntimeError("one of the variables needed for gradient computation has been modified by an inplace operation")
            if self.requires_grad and out.grad is not None:
                flat_x = backend.to_flat_list(self._data)
                mask = [1.0 if ((min_val is None or v >= min_val) and (max_val is None or v <= max_val)) else 0.0 for v in flat_x]
                mask_data = backend.from_data(backend.reshape(mask, self.shape), dtype="float32")
                d_self = backend.mul(mask_data, out.grad._data)
                self._accumulate_grad_data(d_self)

        _attach_grad_fn(out, (self,), _backward)
        return out

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'Tensor':
        return self.clamp(min_val=min_val, max_val=max_val)

    # =========================================================================
    # Transformer Math Primitives
    # =========================================================================

    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> 'Tensor':
        """
        Numerically stable Log-Sum-Exp: logsumexp(x) = m + log(sum(exp(x - m)))
        """
        m = self.max(axis=axis, keepdims=True).detach()
        shifted = self - m
        exp_shifted = shifted.exp()
        sum_exp = exp_shifted.sum(axis=axis, keepdims=True)
        log_sum = sum_exp.log()
        out = m + log_sum
        if not keepdims:
            ndim = self.ndim
            norm_axis = axis + ndim if axis < 0 else axis
            new_shape = tuple(self.shape[i] for i in range(ndim) if i != norm_axis)
            return out.reshape(new_shape)
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """Numerically stable Log-Softmax: log_softmax(x) = x - logsumexp(x)."""
        lse = self.logsumexp(axis=axis, keepdims=True)
        return self - lse

    def softmax(self, axis: int = -1) -> 'Tensor':
        """Numerically stable Softmax: softmax(x) = exp(log_softmax(x))."""
        return self.log_softmax(axis=axis).exp()

    # =========================================================================
    # Reverse Mode Autograd Entrypoint (In-Flight Memory Release on Backward)
    # =========================================================================

    def backward(
        self,
        gradient: Optional[Union['Tensor', Any]] = None,
        retain_graph: bool = False,
        allow_implicit_grad: bool = False
    ) -> None:
        """
        Computes the gradient of current tensor w.r.t. graph leaves using iterative DFS.
        Releases intermediate graph references and closures in-flight when retain_graph=False.
        """
        if not self.requires_grad and self._backward is None:
            if self._grad_fn_state == "freed":
                raise RuntimeError(
                    "Trying to backward through the graph a second time, but the saved intermediate variables "
                    "have already been freed. Specify retain_graph=True when calling .backward() if you need "
                    "to backward through the graph more than once."
                )
            return

        if self._grad_fn_state == "freed":
            raise RuntimeError(
                "Trying to backward through the graph a second time, but the saved intermediate variables "
                "have already been freed. Specify retain_graph=True when calling .backward() if you need "
                "to backward through the graph more than once."
            )
        if self._grad_fn_state == "invalid":
            raise RuntimeError("Cannot backward through an invalid/poisoned computation graph.")

        topo: List['Tensor'] = []
        visited: Set[int] = set()
        stack: List[Tuple['Tensor', bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            if processed:
                visited.add(node_id)
                topo.append(node)
            else:
                stack.append((node, True))
                for child in node._prev:
                    if id(child) not in visited:
                        stack.append((child, False))

        # Reset intermediate non-leaf gradients for clean multi-backward accumulation
        for node in topo:
            if len(node._prev) > 0 and node is not self:
                node.grad = None

        if gradient is None:
            if self.shape == ():
                self.grad = Tensor(1.0, dtype="float32", backend=self.backend)
            elif allow_implicit_grad:
                self.grad = Tensor(self.backend.ones(self.shape), dtype="float32", backend=self.backend)
            else:
                raise RuntimeError("gradient must be specified for non-scalar Tensor.")
        else:
            grad_tensor = gradient if isinstance(gradient, Tensor) else Tensor(gradient, dtype="float32", backend=self.backend)
            if grad_tensor.shape != self.shape:
                raise RuntimeError(
                    f"Mismatch in shape: grad_output shape {grad_tensor.shape} != output shape {self.shape}"
                )
            self.grad = grad_tensor

        all_nodes = list(topo)
        try:
            for i in range(len(topo) - 1, -1, -1):
                node = topo[i]
                if node._backward is not None:
                    node._backward()
                if not retain_graph and node is not self:
                    node._prev.clear()
                    node._backward = None
                    node._grad_fn_state = "freed"
                topo[i] = None
        except Exception as e:
            for node in all_nodes:
                node._grad_fn_state = "invalid"
            raise e

        if not retain_graph:
            self._prev.clear()
            self._backward = None
            self._grad_fn_state = "freed"

    # =========================================================================
    # Representation & Printing
    # =========================================================================

    def __repr__(self) -> str:
        data_str = str(self.tolist())
        dtype_str = f", dtype='{self.dtype}'" if self.dtype != "float32" else ""
        req_str = f", requires_grad=True" if self.requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({data_str}, shape={self.shape}{dtype_str}{req_str}{op_str})"

    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# Factory Functions
# =============================================================================

def tensor(data: Any, dtype: Optional[str] = None, requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a new Tensor."""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, backend=backend)

def zeros(shape: Tuple[int, ...], dtype: str = "float32", requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor filled with zeros."""
    b = backend or get_backend()
    return Tensor(b.zeros(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad, backend=b)

def ones(shape: Tuple[int, ...], dtype: str = "float32", requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor filled with ones."""
    b = backend or get_backend()
    return Tensor(b.ones(shape, dtype=dtype), dtype=dtype, requires_grad=requires_grad, backend=b)

def zeros_like(t: Tensor, dtype: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    """Create a zero Tensor with the same shape as input tensor."""
    dt = dtype or t.dtype
    return zeros(t.shape, dtype=dt, requires_grad=requires_grad, backend=t.backend)

def ones_like(t: Tensor, dtype: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    """Create a one Tensor with the same shape as input tensor."""
    dt = dtype or t.dtype
    return ones(t.shape, dtype=dt, requires_grad=requires_grad, backend=t.backend)

def randn(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor with Gaussian normal distributed random values."""
    b = backend or get_backend()
    return Tensor(b.randn(shape, mean=mean, std=std), dtype="float32", requires_grad=requires_grad, backend=b)
