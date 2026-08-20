"""
termux_train.nn.attention
=========================
Multi-Head Attention Layer with Causal Mask Caching, Cross-Attention Support,
and N-D Batch Shape Generalization for Production Mobile Transformers.
"""

import math
from typing import Optional
from .module import Module
from .linear import Linear
from ..tensor import Tensor


class MultiHeadAttention(Module):
    """
    Multi-Head Attention with scaled dot-product attention, zero-allocation causal mask caching,
    and cross-attention (Query, Key, Value) support.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        max_seq_len: int = 512
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
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

        self._cached_causal_mask: Optional[Tensor] = None

    def _get_causal_mask(self, seq_len: int, backend) -> Tensor:
        """
        Retrieves or allocates cached causal attention mask to eliminate runtime heap churn.
        """
        if self._cached_causal_mask is None or self._cached_causal_mask.shape[0] < seq_len:
            alloc_len = max(seq_len, self.max_seq_len)
            mask_data = []
            for i in range(alloc_len):
                row = [0.0 if j <= i else -float('inf') for j in range(alloc_len)]
                mask_data.append(row)
            m_data = backend.from_data(mask_data, dtype="float32")
            self._cached_causal_mask = Tensor(m_data, dtype="float32", requires_grad=False, backend=backend)

        if self._cached_causal_mask.shape == (seq_len, seq_len):
            return self._cached_causal_mask

        # Sub-slice mask (seq_len, seq_len)
        flat_cached = backend.to_flat_list(self._cached_causal_mask._data)
        cached_dim = self._cached_causal_mask.shape[0]
        sub_flat = []
        for i in range(seq_len):
            start = i * cached_dim
            sub_flat.extend(flat_cached[start:start + seq_len])
        sub_data = backend.from_data(backend.reshape(sub_flat, (seq_len, seq_len)), dtype="float32")
        return Tensor(sub_data, dtype="float32", requires_grad=False, backend=backend)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        causal: bool = False
    ) -> Tensor:
        """
        Forward pass of Multi-Head Attention:
          query: shape (*B, S_q, d_model) or (S_q, d_model)
          key: shape (*B, S_k, d_model) (defaults to query for self-attention)
          value: shape (*B, S_k, d_model) (defaults to key)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        orig_ndim = query.ndim
        orig_batch_shape = query.shape[:-2] if orig_ndim > 2 else ()

        if orig_ndim == 2:
            q = query.reshape(1, query.shape[0], query.shape[1])
            k = key.reshape(1, key.shape[0], key.shape[1])
            v = value.reshape(1, value.shape[0], value.shape[1])
        elif orig_ndim > 3:
            # Flatten arbitrary leading batch dimensions (*B, S, D) -> (B_flat, S, D)
            b_flat = 1
            for dim in orig_batch_shape:
                b_flat *= dim
            q = query.reshape(b_flat, query.shape[-2], query.shape[-1])
            k = key.reshape(b_flat, key.shape[-2], key.shape[-1])
            v = value.reshape(b_flat, value.shape[-2], value.shape[-1])
        else:
            q, k, v = query, key, value

        B, S_q, D_q = q.shape
        _, S_k, D_k = k.shape

        if D_q != self.d_model or D_k != self.d_model:
            raise ValueError(
                f"Feature dimensions (query={D_q}, key={D_k}) do not match d_model={self.d_model}"
            )

        H = self.num_heads
        d_k = self.d_k

        # 1. Project Q, K, V
        proj_q = self.q_proj(q)  # (B, S_q, D)
        proj_k = self.k_proj(k)  # (B, S_k, D)
        proj_v = self.v_proj(v)  # (B, S_k, D)

        # 2. Reshape and Transpose to (B, H, S, d_k)
        proj_q = proj_q.reshape(B, S_q, H, d_k).transpose(0, 2, 1, 3)
        proj_k = proj_k.reshape(B, S_k, H, d_k).transpose(0, 2, 1, 3)
        proj_v = proj_v.reshape(B, S_k, H, d_k).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        k_t = proj_k.transpose(0, 1, 3, 2)  # (B, H, d_k, S_k)
        scale = 1.0 / math.sqrt(d_k)
        scores = (proj_q @ k_t) * scale  # (B, H, S_q, S_k)

        # 4. Apply Mask (Causal or Custom)
        if causal:
            causal_m = self._get_causal_mask(S_q, backend=q.backend)
            scores = scores + causal_m
        elif mask is not None:
            mask_t = q._ensure_tensor_on_self_backend(mask)
            scores = scores + mask_t

        # 5. Softmax attention probabilities
        attn_weights = scores.softmax(axis=-1)

        # 6. Compute Context: attn_weights @ V -> (B, H, S_q, d_k)
        context = attn_weights @ proj_v

        # 7. Concatenate heads: (B, H, S_q, d_k) -> (B, S_q, H, d_k) -> (B, S_q, D)
        context = context.transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)

        # 8. Output projection
        out = self.out_proj(context)

        if orig_ndim == 2:
            return out.reshape(S_q, self.d_model)
        elif orig_ndim > 3:
            return out.reshape(*(orig_batch_shape + (S_q, self.d_model)))
        return out

    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads})"
