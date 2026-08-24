"""
termux_train.nn.rope
====================
Rotary Position Embedding (RoPE) for Context-Extrapolating Transformers.
Vectorized C-level implementation for NumPy with Pure Python fallback.
Standardized on Meta LLaMA, Mistral, Gemma, and RoFormer architectures.
Requires 0 learnable parameters (O(0) parameter overhead).
"""

import math
from typing import Tuple, Optional
from .module import Module
from ..tensor import Tensor, _attach_grad_fn

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _rotate_half(x_flat: list, head_dim: int) -> list:
    """Rotates the two halves of head_dim: [-x2, x1]."""
    out = []
    half = head_dim // 2
    for i in range(0, len(x_flat), head_dim):
        x1 = x_flat[i:i + half]
        x2 = x_flat[i + half:i + head_dim]
        out.extend([-v for v in x2])
        out.extend(x1)
    return out


class RotaryEmbedding(Module):
    """
    Rotary Position Embedding (RoPE).
    Computes frequency bands and rotational angles:
      theta_i = base^(-2i / dim)
      RoPE(x, m) = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = float(base)

        # Compute inverse frequencies: 1.0 / (base ** (2i / dim))
        inv_freq = [1.0 / (self.base ** (float(i) / float(dim))) for i in range(0, dim, 2)]
        self.inv_freq = inv_freq

        # Precompute cos and sin tables up to max_seq_len
        cos_table = []
        sin_table = []
        for pos in range(max_seq_len):
            row_cos = []
            row_sin = []
            for freq in inv_freq:
                angle = float(pos) * freq
                c = math.cos(angle)
                s = math.sin(angle)
                row_cos.extend([c, c])
                row_sin.extend([s, s])
            cos_table.append(row_cos)
            sin_table.append(row_sin)

        self._cos_table = cos_table
        self._sin_table = sin_table

        if HAS_NUMPY:
            self._np_cos = np.array(cos_table, dtype=np.float32)
            self._np_sin = np.array(sin_table, dtype=np.float32)
        else:
            self._np_cos = None
            self._np_sin = None

    def _expand_tables(self, required_len: int) -> None:
        """Dynamically expands cos and sin tables to support arbitrarily long contexts."""
        current_len = len(self._cos_table)
        target_len = max(current_len * 2, required_len)
        for pos in range(current_len, target_len):
            row_cos = []
            row_sin = []
            for freq in self.inv_freq:
                angle = float(pos) * freq
                c = math.cos(angle)
                s = math.sin(angle)
                row_cos.extend([c, c])
                row_sin.extend([s, s])
            self._cos_table.append(row_cos)
            self._sin_table.append(row_sin)
        self.max_seq_len = target_len
        if HAS_NUMPY:
            self._np_cos = np.array(self._cos_table, dtype=np.float32)
            self._np_sin = np.array(self._sin_table, dtype=np.float32)

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0
    ) -> Tensor:
        """
        Applies Rotary Embedding to input tensor x of shape (*B, H, S, d_k) or (*B, S, D).
        """
        backend = x.backend
        shape = x.shape
        seq_len = shape[-2]
        head_dim = shape[-1]

        if head_dim != self.dim:
            raise ValueError(f"Input last dimension {head_dim} does not match RoPE dim {self.dim}")

        if position_offset + seq_len > self.max_seq_len:
            self._expand_tables(position_offset + seq_len)

        # NumPy Vectorized Fast Path
        if HAS_NUMPY and isinstance(x._data, np.ndarray) and getattr(backend, "name", "").lower() == "numpy":
            cos = self._np_cos[position_offset:position_offset + seq_len]  # (S, head_dim)
            sin = self._np_sin[position_offset:position_offset + seq_len]  # (S, head_dim)

            # Broadcast cos and sin to x shape
            while cos.ndim < x.ndim:
                cos = np.expand_dims(cos, axis=0)
                sin = np.expand_dims(sin, axis=0)

            half = head_dim // 2
            x_data = x._data
            rot_x = np.concatenate([-x_data[..., half:], x_data[..., :half]], axis=-1)
            out_data = x_data * cos + rot_x * sin
            out = Tensor(out_data, dtype="float32", requires_grad=x.requires_grad, _prev=(x,), _op="rope", backend=backend)

            if out.requires_grad:
                def _backward_np():
                    if out.grad is not None and x.requires_grad:
                        g_data = out.grad._data
                        rot_g = np.concatenate([-g_data[..., half:], g_data[..., :half]], axis=-1)
                        dx_data = g_data * cos - rot_g * sin
                        x._accumulate_grad_data(dx_data)
                _attach_grad_fn(out, (x,), _backward_np)
            return out

        # Pure Python Fallback
        flat_x = backend.to_flat_list(x._data)
        rot_x = _rotate_half(flat_x, head_dim)

        out_flat = []
        total_num_vectors = len(flat_x) // head_dim

        for vec_idx in range(total_num_vectors):
            pos = (vec_idx % seq_len) + position_offset
            cos_row = self._cos_table[pos]
            sin_row = self._sin_table[pos]

            start = vec_idx * head_dim
            for d in range(head_dim):
                val_x = flat_x[start + d]
                val_rot = rot_x[start + d]
                c = cos_row[d]
                s = sin_row[d]
                out_flat.append(val_x * c + val_rot * s)

        out_data = backend.from_data(backend.reshape(out_flat, shape), dtype="float32")
        out = Tensor(
            out_data,
            dtype="float32",
            requires_grad=x.requires_grad,
            _prev=(x,),
            _op="rope",
            backend=backend
        )

        if out.requires_grad:
            def _backward():
                if out.grad is not None and x.requires_grad:
                    g_flat = backend.to_flat_list(out.grad._data)
                    rot_g = _rotate_half(g_flat, head_dim)
                    dx_flat = []

                    for v_i in range(total_num_vectors):
                        pos = (v_i % seq_len) + position_offset
                        cos_r = self._cos_table[pos]
                        sin_r = self._sin_table[pos]

                        g_start = v_i * head_dim
                        for d in range(head_dim):
                            g_val = g_flat[g_start + d]
                            rot_g_val = rot_g[g_start + d]
                            c = cos_r[d]
                            s = sin_r[d]
                            dx_flat.append(g_val * c - rot_g_val * s)

                    dx_data = backend.from_data(backend.reshape(dx_flat, shape), dtype="float32")
                    x._accumulate_grad_data(dx_data)

            _attach_grad_fn(out, (x,), _backward)

        return out

    def __repr__(self) -> str:
        return f"RotaryEmbedding(dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base})"
