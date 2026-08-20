"""
termux_train.nn.attention
=========================
Multi-Head Attention Layer with Causal Masking for Mobile Transformers.
"""

import math
from typing import Optional
from .module import Module
from .linear import Linear
from ..tensor import Tensor


def _create_causal_mask(seq_len: int, backend) -> Tensor:
    """Create lower-triangular causal attention mask where future tokens are -inf."""
    mask_data = []
    for i in range(seq_len):
        row = [0.0 if j <= i else -float('inf') for j in range(seq_len)]
        mask_data.append(row)
    m_data = backend.from_data(mask_data, dtype="float32")
    return Tensor(m_data, dtype="float32", backend=backend)


class MultiHeadAttention(Module):
    """
    Multi-Head Attention with scaled dot-product attention and optional causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True
    ):
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        causal: bool = False
    ) -> Tensor:
        """
        Forward pass of Multi-Head Attention:
          x: shape (B, S, d_model) or (S, d_model)
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            # (S, d_model) -> (1, S, d_model)
            x = x.reshape(1, x.shape[0], x.shape[1])

        if x.ndim != 3:
            raise ValueError(f"MultiHeadAttention expects 2D (S, D) or 3D (B, S, D) input, got shape {x.shape}")

        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Input feature dimension {D} does not match d_model {self.d_model}")

        H = self.num_heads
        d_k = self.d_k

        # 1. Project Q, K, V
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # 2. Reshape and Transpose to (B, H, S, d_k)
        q = q.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        k_t = k.transpose(0, 1, 3, 2)  # (B, H, d_k, S)
        scale = 1.0 / math.sqrt(d_k)
        scores = (q @ k_t) * scale  # (B, H, S, S)

        # 4. Apply Mask (Causal or Custom)
        if causal:
            causal_m = _create_causal_mask(S, backend=x.backend)
            scores = scores + causal_m
        elif mask is not None:
            mask_t = x._ensure_tensor_on_self_backend(mask)
            scores = scores + mask_t

        # 5. Softmax attention probabilities
        attn_weights = scores.softmax(axis=-1)

        # 6. Compute Context: attn_weights @ V -> (B, H, S, d_k)
        context = attn_weights @ v

        # 7. Concatenate heads: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, D)
        context = context.transpose(0, 2, 1, 3).reshape(B, S, D)

        # 8. Output projection
        out = self.out_proj(context)

        if orig_ndim == 2:
            return out.reshape(S, D)
        return out

    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads})"
