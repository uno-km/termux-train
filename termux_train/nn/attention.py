"""
termux_train.nn.attention
=========================
Multi-Head Attention Layer with Rotary Position Embedding (RoPE),
Universal Causal Trapezoid Masking, Joint Padding Masking, and Incremental KV Caching.
"""

import math
from typing import Optional, Tuple, Union
from .module import Module
from .linear import Linear
from .rope import RotaryEmbedding
from ..tensor import Tensor


def _concat_head_tensors(t1: Tensor, t2: Tensor, backend) -> Tensor:
    """Concatenates two 4D tensors (B, H, S1, d_k) and (B, H, S2, d_k) along axis 2 -> (B, H, S1+S2, d_k)."""
    B, H, S1, d_k = t1.shape
    _, _, S2, _ = t2.shape
    f1 = backend.to_flat_list(t1._data)
    f2 = backend.to_flat_list(t2._data)

    out_flat = []
    # Interleave along axis 2
    for b in range(B):
        for h in range(H):
            start1 = (b * H + h) * S1 * d_k
            start2 = (b * H + h) * S2 * d_k
            out_flat.extend(f1[start1:start1 + S1 * d_k])
            out_flat.extend(f2[start2:start2 + S2 * d_k])

    out_data = backend.from_data(backend.reshape(out_flat, (B, H, S1 + S2, d_k)), dtype="float32")
    return Tensor(out_data, dtype="float32", backend=backend)


class MultiHeadAttention(Module):
    """
    Multi-Head Attention with scaled dot-product attention, zero-allocation causal mask caching,
    universal causal trapezoid masking, native RoPE integration, and incremental KV caching.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        max_seq_len: int = 512,
        rotary_emb: Optional[RotaryEmbedding] = None
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
        self.rotary_emb = rotary_emb

        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

        self._cached_causal_mask: Optional[Tensor] = None

    def _get_causal_mask(self, S_q: int, S_k: int, backend) -> Tensor:
        """
        Retrieves or allocates cached causal attention mask (S_q, S_k).
        Universal trapezoid rule: query i can attend to key j if j <= i + (S_k - S_q).
        """
        offset = S_k - S_q
        if S_q == S_k:
            if self._cached_causal_mask is None or self._cached_causal_mask.shape[0] < S_q:
                alloc_len = max(S_q, self.max_seq_len)
                mask_data = [
                    [0.0 if j <= i else -float('inf') for j in range(alloc_len)]
                    for i in range(alloc_len)
                ]
                m_data = backend.from_data(mask_data, dtype="float32")
                self._cached_causal_mask = Tensor(m_data, dtype="float32", requires_grad=False, backend=backend)

            if self._cached_causal_mask.shape == (S_q, S_q):
                return self._cached_causal_mask

            flat_cached = backend.to_flat_list(self._cached_causal_mask._data)
            cached_dim = self._cached_causal_mask.shape[0]
            sub_flat = []
            for i in range(S_q):
                start = i * cached_dim
                sub_flat.extend(flat_cached[start:start + S_q])
            sub_data = backend.from_data(backend.reshape(sub_flat, (S_q, S_q)), dtype="float32")
            return Tensor(sub_data, dtype="float32", requires_grad=False, backend=backend)

        # Trapezoidal causal mask for arbitrary (S_q, S_k)
        mask_data = [
            [0.0 if j <= (i + offset) else -float('inf') for j in range(S_k)]
            for i in range(S_q)
        ]
        m_data = backend.from_data(mask_data, dtype="float32")
        return Tensor(m_data, dtype="float32", requires_grad=False, backend=backend)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        causal: bool = False,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Forward pass of Multi-Head Attention.
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

        # 3. Apply RoPE if configured
        if self.rotary_emb is not None:
            proj_q = self.rotary_emb(proj_q, position_offset=position_offset)
            proj_k = self.rotary_emb(proj_k, position_offset=position_offset)

        # 4. Incremental KV Cache Concatenation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            proj_k = _concat_head_tensors(past_k, proj_k, backend=q.backend)
            proj_v = _concat_head_tensors(past_v, proj_v, backend=q.backend)

        present_key_value = (proj_k, proj_v) if use_cache else None
        current_s_k = proj_k.shape[2]

        # 5. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        k_t = proj_k.transpose(0, 1, 3, 2)  # (B, H, d_k, current_s_k)
        scale = 1.0 / math.sqrt(d_k)
        scores = (proj_q @ k_t) * scale  # (B, H, S_q, current_s_k)

        # 6. Universal Causal Trapezoid + Padding Mask Application
        if causal:
            causal_m = self._get_causal_mask(S_q, current_s_k, backend=q.backend)
            scores = scores + causal_m

        if mask is not None:
            mask_t = q._ensure_tensor_on_self_backend(mask)
            scores = scores + mask_t

        # 7. Softmax attention probabilities
        attn_weights = scores.softmax(axis=-1)

        # 8. Compute Context: attn_weights @ V -> (B, H, S_q, d_k)
        context = attn_weights @ proj_v

        # 9. Concatenate heads: (B, H, S_q, d_k) -> (B, S_q, H, d_k) -> (B, S_q, D)
        context = context.transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)

        # 10. Output projection
        out = self.out_proj(context)

        if orig_ndim == 2:
            final_out = out.reshape(S_q, self.d_model)
        elif orig_ndim > 3:
            final_out = out.reshape(*(orig_batch_shape + (S_q, self.d_model)))
        else:
            final_out = out

        if use_cache:
            return final_out, present_key_value
        return final_out

    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads}, rotary={self.rotary_emb is not None})"
