"""
termux_train.tensor
===================
Core Multi-Dimensional Tensor Class with Dynamic DAG Autograd Engine.
"""

from typing import Any, Tuple, Set, List, Optional, Union, Callable
from .backend import get_backend, BaseBackend

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

def _classify_matmul(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> str:
    """Classify 1D~3D matmul shape contract or raise dimension mismatch / NotImplementedError."""
    r1, r2 = len(s1), len(s2)
    if r1 not in (1, 2, 3) or r2 not in (1, 2, 3):
        raise NotImplementedError(
            "matmul supports every rank combination where both operands "
            f"are between 1D and 3D. Received shapes {s1} and {s2}. "
            "Scalar operands and 4D+ ND matmul are not supported."
        )

    # 1D @ 1D -> ()
    if r1 == 1 and r2 == 1:
        if s1[0] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for vector dot product: {s1[0]} != {s2[0]}")
        return "1d_1d"

    # 1D @ 2D -> (N,)
    if r1 == 1 and r2 == 2:
        if s1[0] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 1D@2D matmul: {s1[0]} != {s2[0]}")
        return "1d_2d"

    # 1D @ 3D -> (B, N)
    if r1 == 1 and r2 == 3:
        if s1[0] != s2[1]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 1D@3D matmul: {s1[0]} != {s2[1]}")
        return "1d_3d"

    # 2D @ 1D -> (M,)
    if r1 == 2 and r2 == 1:
        if s1[1] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for matrix-vector multiplication: {s1[1]} != {s2[0]}")
        return "2d_1d"

    # 2D @ 2D -> (M, N)
    if r1 == 2 and r2 == 2:
        if s1[1] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for matrix multiplication: {s1[1]} != {s2[0]}")
        return "2d_2d"

    # 2D @ 3D -> (B, M, N)
    if r1 == 2 and r2 == 3:
        if s1[1] != s2[1]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 2D@3D matmul: {s1[1]} != {s2[1]}")
        return "2d_3d"

    # 3D @ 1D -> (B, M)
    if r1 == 3 and r2 == 1:
        if s1[2] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@1D matmul: {s1[2]} != {s2[0]}")
        return "3d_1d"

    # 3D @ 2D -> (B, M, N)
    if r1 == 3 and r2 == 2:
        if s1[2] != s2[0]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@2D matmul: {s1[2]} != {s2[0]}")
        return "3d_2d"

    # 3D @ 3D -> (B, M, N)
    if r1 == 3 and r2 == 3:
        if s1[0] != s2[0]:
            raise ValueError(f"Batch dimensions must match: {s1[0]} != {s2[0]}")
        if s1[2] != s2[1]:
            raise ValueError(f"Shapes {s1} and {s2} not aligned for 3D@3D matmul: {s1[2]} != {s2[1]}")
        return "3d_3d"

    raise NotImplementedError(
        f"matmul unsupported for shapes {s1} and {s2}."
    )

class Tensor:
    """
    Core Tensor class supporting multi-dimensional arrays, pluggable backends,
    and dynamic reverse-mode automatic differentiation (Autograd).
    """
    
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _prev: Tuple['Tensor', ...] = (),
        _op: str = "",
        backend: Optional[BaseBackend] = None
    ):
        if isinstance(data, Tensor):
            if backend is None:
                self.backend = data.backend
                self._data = data._data
            else:
                self.backend = backend
                self._data = self.backend.from_data(data.tolist())
        else:
            self.backend = backend or get_backend()
            self._data = self.backend.from_data(data)
            
        self.requires_grad: bool = requires_grad
        self.grad: Optional['Tensor'] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = set(_prev)
        self._op: str = _op

    def _accumulate_grad_data(self, grad_data: Any) -> None:
        """Accumulate incoming raw gradient data into self.grad safely."""
        if not self.requires_grad:
            return
        g_data = self.backend.from_data(grad_data)
        if self.grad is None:
            self.grad = Tensor(g_data, requires_grad=False, backend=self.backend)
        else:
            self.grad._data = self.backend.add(self.grad._data, g_data)

    def _ensure_tensor_on_self_backend(self, other: Any) -> 'Tensor':
        """
        Normalize other operand to Tensor on self.backend.
        If other is a Tensor on a different backend:
          - If requires_grad=False: automatically converted to self.backend.
          - If requires_grad=True: raises RuntimeError to prevent broken autograd parentage.
        """
        if not isinstance(other, Tensor):
            return Tensor(other, backend=self.backend)

        if other.backend is self.backend or other.backend.name == self.backend.name:
            return other

        if other.requires_grad:
            raise RuntimeError(
                f"Cross-backend autograd operation is not supported between {self.backend.name} "
                f"and {other.backend.name} when requires_grad=True. "
                "Ensure all trainable tensors reside on the same backend before forward/backward."
            )

        return Tensor(other.tolist(), requires_grad=False, backend=self.backend)

    @property
    def data(self) -> Any:
        """Return native underlying data."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """
        Safely update underlying data while strictly preserving self.backend.
        If a Tensor is assigned (even from another backend), its data is converted to self.backend.
        """
        if isinstance(value, Tensor):
            self._data = self.backend.from_data(value.tolist())
        else:
            self._data = self.backend.from_data(value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape tuple."""
        return self.backend.get_shape(self._data)

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.shape)

    @property
    def T(self) -> 'Tensor':
        """2D Transpose shortcut."""
        return self.transpose()

    def item(self) -> float:
        """Extract a single scalar value from a 0D/1D 1-element tensor."""
        flat = self.backend.to_flat_list(self._data)
        if len(flat) != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars (got size {len(flat)})")
        return flat[0]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Sets gradient of this tensor to zero or None.
        
        Args:
            set_to_none: if True, sets self.grad to None to release memory (default).
                         if False, initializes self.grad as a zero tensor.
        """
        if not self.requires_grad:
            self.grad = None
            return
        if set_to_none:
            self.grad = None
        else:
            if self.grad is None:
                self.grad = Tensor(self.backend.zeros(self.shape), backend=self.backend)
            else:
                self.grad._data = self.backend.zeros(self.shape)

    def tolist(self) -> Any:
        """Return nested Python list representation."""
        return self.backend.to_nested_list(self._data)

    def numpy(self) -> Any:
        """Convert tensor to a NumPy array if available."""
        import numpy as np
        if isinstance(self._data, np.ndarray):
            return self._data
        flat = self.backend.to_flat_list(self._data)
        return np.array(flat, dtype=np.float32).reshape(self.shape)

    def reshape(self, *new_shape: Union[int, Tuple[int, ...]]) -> 'Tensor':
        """Reshape tensor to new dimensions."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            target_shape = tuple(new_shape[0])
        else:
            target_shape = tuple(new_shape)
            
        old_shape = self.shape
        out = Tensor(
            self.backend.reshape(self._data, target_shape),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="reshape",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_reshaped = self.backend.reshape(out.grad._data, old_shape)
                if self.grad is None:
                    self.grad = Tensor(grad_reshaped, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, grad_reshaped)

        out._backward = _backward
        return out

    def flatten(self) -> 'Tensor':
        """Flatten tensor to 1D."""
        num_elements = 1
        for d in self.shape:
            num_elements *= d
        return self.reshape(num_elements)

    def transpose(self, *axes: Union[int, Tuple[int, ...], List[int]]) -> 'Tensor':
        """
        Transpose tensor dimensions according to given axes permutation.
        Supports 2D .T as well as arbitrary ND permutations with inverse permutation backward.
        """
        if len(axes) == 0:
            axes_norm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_norm = tuple(axes[0])
        elif len(axes) == 1 and axes[0] is None:
            axes_norm = tuple(reversed(range(self.ndim)))
        else:
            axes_norm = tuple(axes)

        axes_norm = tuple(a + self.ndim if a < 0 else a for a in axes_norm)
        if len(axes_norm) != self.ndim or set(axes_norm) != set(range(self.ndim)):
            raise ValueError(f"Invalid axes {axes_norm} for transpose of {self.ndim}D tensor")

        out = Tensor(
            self.backend.transpose(self._data, axes=axes_norm),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="transpose",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                inv_axes = _invert_permutation(axes_norm)
                grad_t = self.backend.transpose(out.grad._data, axes=inv_axes)
                if self.grad is None:
                    self.grad = Tensor(grad_t, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, grad_t)

        out._backward = _backward
        return out

    # =========================================================================
    # Elementwise Arithmetic & Autograd
    # =========================================================================

    def __add__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.backend.add(self._data, other._data),
            requires_grad=req_grad,
            _prev=(self, other),
            _op="+",
            backend=self.backend
        )

        def _backward():
            if out.grad is not None:
                if self.requires_grad:
                    d_self = self.backend.unbroadcast(out.grad._data, self.shape)
                    if self.grad is None:
                        self.grad = Tensor(d_self, backend=self.backend)
                    else:
                        self.grad._data = self.backend.add(self.grad._data, d_self)
                        
                if other.requires_grad:
                    d_other = self.backend.unbroadcast(out.grad._data, other.shape)
                    if other.grad is None:
                        other.grad = Tensor(d_other, backend=other.backend)
                    else:
                        other.grad._data = self.backend.add(other.grad._data, d_other)

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> 'Tensor':
        return self + other

    def __sub__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.backend.sub(self._data, other._data),
            requires_grad=req_grad,
            _prev=(self, other),
            _op="-",
            backend=self.backend
        )

        def _backward():
            if out.grad is not None:
                if self.requires_grad:
                    d_self = self.backend.unbroadcast(out.grad._data, self.shape)
                    if self.grad is None:
                        self.grad = Tensor(d_self, backend=self.backend)
                    else:
                        self.grad._data = self.backend.add(self.grad._data, d_self)
                        
                if other.requires_grad:
                    neg_grad = self.backend.neg(out.grad._data)
                    d_other = self.backend.unbroadcast(neg_grad, other.shape)
                    if other.grad is None:
                        other.grad = Tensor(d_other, backend=other.backend)
                    else:
                        other.grad._data = self.backend.add(other.grad._data, d_other)

        out._backward = _backward
        return out

    def __rsub__(self, other: Any) -> 'Tensor':
        other_t = self._ensure_tensor_on_self_backend(other)
        return other_t - self

    def __mul__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.backend.mul(self._data, other._data),
            requires_grad=req_grad,
            _prev=(self, other),
            _op="*",
            backend=self.backend
        )

        def _backward():
            if out.grad is not None:
                if self.requires_grad:
                    d_self = self.backend.mul(other._data, out.grad._data)
                    d_self = self.backend.unbroadcast(d_self, self.shape)
                    if self.grad is None:
                        self.grad = Tensor(d_self, backend=self.backend)
                    else:
                        self.grad._data = self.backend.add(self.grad._data, d_self)
                        
                if other.requires_grad:
                    d_other = self.backend.mul(self._data, out.grad._data)
                    d_other = self.backend.unbroadcast(d_other, other.shape)
                    if other.grad is None:
                        other.grad = Tensor(d_other, backend=other.backend)
                    else:
                        other.grad._data = self.backend.add(other.grad._data, d_other)

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> 'Tensor':
        return self * other

    def __truediv__(self, other: Any) -> 'Tensor':
        other = self._ensure_tensor_on_self_backend(other)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.backend.div(self._data, other._data),
            requires_grad=req_grad,
            _prev=(self, other),
            _op="/",
            backend=self.backend
        )

        def _backward():
            if out.grad is not None:
                if self.requires_grad:
                    # d(a/b)/da = 1/b
                    inv_b = self.backend.div(self.backend.ones(()), other._data)
                    d_self = self.backend.mul(inv_b, out.grad._data)
                    d_self = self.backend.unbroadcast(d_self, self.shape)
                    if self.grad is None:
                        self.grad = Tensor(d_self, backend=self.backend)
                    else:
                        self.grad._data = self.backend.add(self.grad._data, d_self)
                        
                if other.requires_grad:
                    # d(a/b)/db = -a / (b^2)
                    b_sq = self.backend.mul(other._data, other._data)
                    neg_a = self.backend.neg(self._data)
                    grad_b_raw = self.backend.mul(self.backend.div(neg_a, b_sq), out.grad._data)
                    d_other = self.backend.unbroadcast(grad_b_raw, other.shape)
                    if other.grad is None:
                        other.grad = Tensor(d_other, backend=other.backend)
                    else:
                        other.grad._data = self.backend.add(other.grad._data, d_other)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Any) -> 'Tensor':
        other_t = self._ensure_tensor_on_self_backend(other)
        return other_t / self

    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        exp_val = float(exponent)
        out = Tensor(
            self.backend.pow(self._data, exp_val),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op=f"**{exp_val}",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # d(x^p)/dx = p * x^(p-1)
                term = self.backend.mul(exp_val, self.backend.pow(self._data, exp_val - 1.0))
                d_self = self.backend.mul(term, out.grad._data)
                d_self = self.backend.unbroadcast(d_self, self.shape)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        out = Tensor(
            self.backend.neg(self._data),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="neg",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                d_self = self.backend.neg(out.grad._data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    # =========================================================================
    # Matrix Multiplication (@ / matmul)
    # =========================================================================

    def matmul(self, other: Any) -> 'Tensor':
        """
        Matrix multiplication supporting every rank combination
        where both operands are between 1D and 3D.

        Supported:
          - 1D @ 1D -> () (Scalar Dot Product)
          - 1D @ 2D -> (N,) (1D Vector)
          - 1D @ 3D -> (B, N) (2D Batch Matrix)
          - 2D @ 1D -> (M,) (1D Vector)
          - 2D @ 2D -> (M, N) (2D Matrix Multiplication)
          - 2D @ 3D -> (B, M, N) (3D Batched Matrix)
          - 3D @ 1D -> (B, M) (2D Matrix)
          - 3D @ 2D -> (B, M, N) (3D Sequence / LoRA Projection)
          - 3D @ 3D -> (B, M, N) (3D Transformer Attention Product)

        Scalar operands (0D) and 4D+ operands are not supported.
        For 3D @ 3D, batch dimensions must currently match.
        """
        other = self._ensure_tensor_on_self_backend(other)
        kind = _classify_matmul(self.shape, other.shape)
        
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(
            self.backend.matmul(self._data, other._data),
            requires_grad=req_grad,
            _prev=(self, other),
            _op="@",
            backend=self.backend
        )

        def _backward():
            if out.grad is None:
                return

            if kind == "1d_1d":
                # A: (K,), B: (K,) -> Y: ()
                if self.requires_grad:
                    self._accumulate_grad_data(self.backend.mul(other._data, out.grad._data))
                if other.requires_grad:
                    other._accumulate_grad_data(other.backend.mul(self._data, out.grad._data))

            elif kind == "1d_2d":
                # A: (K,), B: (K, N) -> Y: (N,), dY: (N,)
                # dA = dY @ B^T: (N,) @ (N, K) -> (K,)
                # dB = outer(A, dY) = (K, 1) @ (1, N) -> (K, N)
                if self.requires_grad:
                    b_t = self.backend.transpose(other._data, axes=(1, 0))
                    d_self = self.backend.matmul(out.grad._data, b_t)
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    a_col = self.backend.reshape(self._data, (self.shape[0], 1))
                    dy_row = self.backend.reshape(out.grad._data, (1, other.shape[1]))
                    d_other = self.backend.matmul(a_col, dy_row)
                    other._accumulate_grad_data(d_other)

            elif kind == "1d_3d":
                # A: (K,), B: (B, K, N) -> Y: (B, N), dY: (B, N)
                # dA = sum_b dY[b] @ B[b]^T: sum over B ((N,) @ (N, K) -> (K,))
                # dB[b] = outer(A, dY[b]) = (K, 1) @ (1, N) -> (K, N)
                batch_size = other.shape[0]
                if self.requires_grad:
                    d_self = self.backend.zeros(self.shape)
                    for b in range(batch_size):
                        b_mat = self.backend.take(other._data, b, axis=0)
                        b_t = self.backend.transpose(b_mat, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_self = self.backend.add(d_self, self.backend.matmul(dy_b, b_t))
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    a_col = self.backend.reshape(self._data, (self.shape[0], 1))
                    d_other_batches = []
                    for b in range(batch_size):
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        dy_row = self.backend.reshape(dy_b, (1, other.shape[2]))
                        d_other_batches.append(self.backend.matmul(a_col, dy_row))
                    other._accumulate_grad_data(d_other_batches)

            elif kind == "2d_1d":
                # A: (M, K), B: (K,) -> Y: (M,), dY: (M,)
                # dA = outer(dY, B): (M, 1) @ (1, K) -> (M, K)
                # dB = A^T @ dY: (K, M) @ (M,) -> (K,)
                if self.requires_grad:
                    dy_col = self.backend.reshape(out.grad._data, (self.shape[0], 1))
                    b_row = self.backend.reshape(other._data, (1, other.shape[0]))
                    d_self = self.backend.matmul(dy_col, b_row)
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    a_t = self.backend.transpose(self._data, axes=(1, 0))
                    d_other = self.backend.matmul(a_t, out.grad._data)
                    other._accumulate_grad_data(d_other)

            elif kind == "2d_2d":
                # A: (M, K), B: (K, N) -> Y: (M, N), dY: (M, N)
                # dA = dY @ B^T: (M, N) @ (N, K) -> (M, K)
                # dB = A^T @ dY: (K, M) @ (M, N) -> (K, N)
                if self.requires_grad:
                    b_t = self.backend.transpose(other._data, axes=(1, 0))
                    d_self = self.backend.matmul(out.grad._data, b_t)
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    a_t = self.backend.transpose(self._data, axes=(1, 0))
                    d_other = self.backend.matmul(a_t, out.grad._data)
                    other._accumulate_grad_data(d_other)

            elif kind == "2d_3d":
                # A: (M, K), B: (B, K, N) -> Y: (B, M, N), dY: (B, M, N)
                # dA = sum_b dY[b] @ B[b]^T: sum over B ((M, N) @ (N, K) -> (M, K))
                # dB[b] = A^T @ dY[b]: (K, M) @ (M, N) -> (K, N)
                batch_size = other.shape[0]
                if self.requires_grad:
                    d_self = self.backend.zeros(self.shape)
                    for b in range(batch_size):
                        b_mat = self.backend.take(other._data, b, axis=0)
                        b_t = self.backend.transpose(b_mat, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_self = self.backend.add(d_self, self.backend.matmul(dy_b, b_t))
                    self._accumulate_grad_data(d_self)
                if other.requires_grad:
                    a_t = self.backend.transpose(self._data, axes=(1, 0))
                    d_other_batches = []
                    for b in range(batch_size):
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_other_batches.append(self.backend.matmul(a_t, dy_b))
                    other._accumulate_grad_data(d_other_batches)

            elif kind == "3d_1d":
                # A: (B, M, K), V: (K,) -> Y: (B, M), dY: (B, M)
                # dA[b] = outer(dY[b], V): (M, 1) @ (1, K) -> (M, K)
                # dV = sum_b A[b]^T @ dY[b]: sum over B ((K, M) @ (M,) -> (K,))
                batch_size = self.shape[0]
                if self.requires_grad:
                    v_row = self.backend.reshape(other._data, (1, other.shape[0]))
                    d_self_batches = []
                    for b in range(batch_size):
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        dy_col = self.backend.reshape(dy_b, (self.shape[1], 1))
                        d_self_batches.append(self.backend.matmul(dy_col, v_row))
                    self._accumulate_grad_data(d_self_batches)
                if other.requires_grad:
                    d_other = self.backend.zeros(other.shape)
                    for b in range(batch_size):
                        a_mat = self.backend.take(self._data, b, axis=0)
                        a_t = self.backend.transpose(a_mat, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_other = self.backend.add(d_other, self.backend.matmul(a_t, dy_b))
                    other._accumulate_grad_data(d_other)

            elif kind == "3d_2d":
                # A: (B, M, K), W: (K, N) -> Y: (B, M, N), dY: (B, M, N)
                # dA[b] = dY[b] @ W^T: (M, N) @ (N, K) -> (M, K)
                # dW = sum_b A[b]^T @ dY[b]: (K, M) @ (M, N) -> (K, N)
                batch_size = self.shape[0]
                if self.requires_grad:
                    w_t = self.backend.transpose(other._data, axes=(1, 0))
                    d_self_batches = [
                        self.backend.matmul(self.backend.take(out.grad._data, b, axis=0), w_t)
                        for b in range(batch_size)
                    ]
                    self._accumulate_grad_data(d_self_batches)
                if other.requires_grad:
                    d_other = self.backend.zeros(other.shape)
                    for b in range(batch_size):
                        a_b = self.backend.take(self._data, b, axis=0)
                        a_t = self.backend.transpose(a_b, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        batch_grad = self.backend.matmul(a_t, dy_b)
                        d_other = self.backend.add(d_other, batch_grad)
                    other._accumulate_grad_data(d_other)

            elif kind == "3d_3d":
                # A: (B, M, K), B: (B, K, N) -> Y: (B, M, N), dY: (B, M, N)
                # dA[b] = dY[b] @ B[b]^T: (M, N) @ (N, K) -> (M, K)
                # dB[b] = A[b]^T @ dY[b]: (K, M) @ (M, N) -> (K, N)
                batch_size = self.shape[0]
                if self.requires_grad:
                    d_self_batches = []
                    for b in range(batch_size):
                        b_mat = self.backend.take(other._data, b, axis=0)
                        b_t = self.backend.transpose(b_mat, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_self_batches.append(self.backend.matmul(dy_b, b_t))
                    self._accumulate_grad_data(d_self_batches)
                if other.requires_grad:
                    d_other_batches = []
                    for b in range(batch_size):
                        a_mat = self.backend.take(self._data, b, axis=0)
                        a_t = self.backend.transpose(a_mat, axes=(1, 0))
                        dy_b = self.backend.take(out.grad._data, b, axis=0)
                        d_other_batches.append(self.backend.matmul(a_t, dy_b))
                    other._accumulate_grad_data(d_other_batches)

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> 'Tensor':
        return self.matmul(other)

    def __rmatmul__(self, other: Any) -> 'Tensor':
        other_t = self._ensure_tensor_on_self_backend(other)
        return other_t.matmul(self)

    # =========================================================================
    # Reductions: sum & mean
    # =========================================================================

    def sum(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        """Sum reduction with axis-aware gradient propagation."""
        norm_axes = _normalize_axes(axis, self.ndim)
        out_data = self.backend.sum(self._data, axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sum",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_data = out.grad._data
                if not keepdims:
                    if axis is None or self.ndim == 0:
                        grad_data = self.backend.reshape(grad_data, (1,) * self.ndim)
                    else:
                        pad_shape = list(self.shape)
                        for a in norm_axes:
                            pad_shape[a] = 1
                        grad_data = self.backend.reshape(grad_data, tuple(pad_shape))

                ones_arr = self.backend.ones(self.shape)
                d_self = self.backend.mul(ones_arr, grad_data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def mean(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        """Mean reduction with axis-aware gradient propagation."""
        norm_axes = _normalize_axes(axis, self.ndim)
        count = _reduced_count(self.shape, norm_axes)
        out_data = self.backend.mean(self._data, axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="mean",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                grad_data = out.grad._data
                if not keepdims:
                    if axis is None or self.ndim == 0:
                        grad_data = self.backend.reshape(grad_data, (1,) * self.ndim)
                    else:
                        pad_shape = list(self.shape)
                        for a in norm_axes:
                            pad_shape[a] = 1
                        grad_data = self.backend.reshape(grad_data, tuple(pad_shape))

                scale = 1.0 / float(count)
                scale_arr = self.backend.mul(self.backend.ones(self.shape), scale)
                d_self = self.backend.mul(scale_arr, grad_data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    # =========================================================================
    # Non-linear Activations: relu, sigmoid, tanh
    # =========================================================================

    def relu(self) -> 'Tensor':
        """Elementwise ReLU activation."""
        out = Tensor(
            self.backend.relu(self._data),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="relu",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # Mask: 1.0 if x > 0 else 0.0
                flat_data = self.backend.to_flat_list(self._data)
                flat_grad = self.backend.to_flat_list(out.grad._data)
                d_flat = [g if x > 0 else 0.0 for x, g in zip(flat_data, flat_grad)]
                d_self = self.backend.from_data(self.backend.reshape(d_flat, self.shape))
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        """Elementwise Sigmoid activation."""
        sig_data = self.backend.sigmoid(self._data)
        out = Tensor(
            sig_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sigmoid",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                one = self.backend.ones(self.shape)
                one_minus_sig = self.backend.sub(one, out._data)
                deriv = self.backend.mul(out._data, one_minus_sig)
                d_self = self.backend.mul(deriv, out.grad._data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        """Elementwise Tanh activation."""
        tanh_data = self.backend.tanh(self._data)
        out = Tensor(
            tanh_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="tanh",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # d(tanh(x))/dx = 1 - tanh(x)^2
                one = self.backend.ones(self.shape)
                tanh_sq = self.backend.mul(out._data, out._data)
                deriv = self.backend.sub(one, tanh_sq)
                d_self = self.backend.mul(deriv, out.grad._data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def clamp(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'Tensor':
        """
        Clamp elements in tensor between [min_val, max_val].
        
        Note on Subgradient:
            At clamp boundaries (x == min_val or x == max_val), termux-train uses a pass-through subgradient of 1.0.
        """
        clamped_data = self.backend.clamp(self._data, min_val, max_val)
        out = Tensor(
            clamped_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="clamp",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # Gradient flows only through elements that are strictly within clamp bounds
                flat_x = self.backend.to_flat_list(self._data)
                mask = [1.0 if ((min_val is None or v >= min_val) and (max_val is None or v <= max_val)) else 0.0 for v in flat_x]
                mask_data = self.backend.from_data(self.backend.reshape(mask, self.shape))
                d_self = self.backend.mul(mask_data, out.grad._data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'Tensor':
        """Alias for clamp."""
        return self.clamp(min_val=min_val, max_val=max_val)

    def log(self) -> 'Tensor':
        """Natural logarithm ln(x)."""
        log_data = self.backend.log(self._data)
        out = Tensor(
            log_data,
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="log",
            backend=self.backend
        )

        def _backward():
            if self.requires_grad and out.grad is not None:
                # d(ln(x))/dx = 1 / x
                inv_x = self.backend.div(self.backend.ones(self.shape), self._data)
                d_self = self.backend.mul(inv_x, out.grad._data)
                if self.grad is None:
                    self.grad = Tensor(d_self, backend=self.backend)
                else:
                    self.grad._data = self.backend.add(self.grad._data, d_self)

        out._backward = _backward
        return out

    # =========================================================================
    # Reverse-Mode Autograd Engine (Topological Sort)
    # =========================================================================

    def backward(
        self,
        gradient: Optional['Tensor'] = None,
        allow_implicit_grad: bool = False
    ) -> None:
        """
        Execute Reverse-Mode Automatic Differentiation (Autograd).
        Constructs DAG topological ordering via DFS and propagates gradients.

        Gradient Policy:
            - Scalar Tensor (shape=()): backward() is allowed without gradient (seed = 1.0).
            - Non-Scalar Tensor: requires explicit `gradient=...` OR `allow_implicit_grad=True`.
        """
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed the output gradient
        if gradient is None:
            if self.shape == ():
                self.grad = Tensor(1.0, backend=self.backend)
            elif allow_implicit_grad:
                self.grad = Tensor(self.backend.ones(self.shape), backend=self.backend)
            else:
                raise RuntimeError(
                    "gradient must be specified for non-scalar Tensor. "
                    "Use tensor.backward(gradient=...) or reduce with mean()/sum()."
                )
        else:
            grad_tensor = gradient if isinstance(gradient, Tensor) else Tensor(gradient, backend=self.backend)
            if grad_tensor.shape != self.shape:
                raise RuntimeError(
                    f"Mismatch in shape: grad_output shape {grad_tensor.shape} != output shape {self.shape}"
                )
            self.grad = grad_tensor

        # Traverse DAG in reverse topological order
        for node in reversed(topo):
            node._backward()

    # =========================================================================
    # Representation & Printing
    # =========================================================================

    def __repr__(self) -> str:
        data_str = str(self.tolist())
        req_str = f", requires_grad=True" if self.requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({data_str}, shape={self.shape}{req_str}{op_str})"

    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# Factory Functions
# =============================================================================

def tensor(data: Any, requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a new Tensor."""
    return Tensor(data, requires_grad=requires_grad, backend=backend)

def zeros(shape: Tuple[int, ...], requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor filled with zeros."""
    b = backend or get_backend()
    return Tensor(b.zeros(shape), requires_grad=requires_grad, backend=b)

def ones(shape: Tuple[int, ...], requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor filled with ones."""
    b = backend or get_backend()
    return Tensor(b.ones(shape), requires_grad=requires_grad, backend=b)

def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Create a zero Tensor with the same shape as input tensor."""
    return zeros(t.shape, requires_grad=requires_grad, backend=t.backend)

def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Create a one Tensor with the same shape as input tensor."""
    return ones(t.shape, requires_grad=requires_grad, backend=t.backend)

def randn(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, requires_grad: bool = False, backend: Optional[BaseBackend] = None) -> Tensor:
    """Create a Tensor with Gaussian normal distributed random values."""
    b = backend or get_backend()
    return Tensor(b.randn(shape, mean=mean, std=std), requires_grad=requires_grad, backend=b)
