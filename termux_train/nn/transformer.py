"""
termux_train.nn.transformer
===========================
Tiny Transformer Architecture and Autoregressive Language Model Primitives for Termux.
Includes Pre-LN Residual Blocks, Weight-Tying, Pre-Allocated Position Buffers,
Incremental KV-Caching, Early-Stopping (<EOS>), and Top-K/Top-P (Nucleus) Sampler.
"""

import math
import random
from typing import Optional, Tuple, List, Union
from .module import Module
from .parameter import Parameter
from .linear import Linear
from .embedding import Embedding
from .layernorm import LayerNorm
from .activations import ReLU
from .attention import MultiHeadAttention
from .sequential import Sequential
from .loss import cross_entropy_loss
from ..tensor import Tensor, zeros, no_grad


def _sample_next_token(
    logits_flat: List[float],
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> int:
    """
    Samples next token with temperature scaling, Top-k truncation, and Top-p (nucleus) filtering.
    """
    v_size = len(logits_flat)
    if temperature < 1e-4 or (top_k == 1):
        return int(max(range(v_size), key=lambda i: logits_flat[i]))

    # 1. Temperature scale with max-shift
    max_val = max(logits_flat)
    safe_temp = max(temperature, 1e-4)
    scaled = [(v - max_val) / safe_temp for v in logits_flat]

    exp_vals = [math.exp(max(-100.0, v)) for v in scaled]
    sum_exp = sum(exp_vals)
    probs = [v / max(1e-12, sum_exp) for v in exp_vals]

    # 2. Sort by descending probability
    indexed_probs = list(enumerate(probs))
    indexed_probs.sort(key=lambda x: x[1], reverse=True)

    # 3. Top-k filtering
    if top_k is not None and 0 < top_k < v_size:
        indexed_probs = indexed_probs[:top_k]

    # 4. Top-p (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        cum_prob = 0.0
        filtered = []
        for idx, p in indexed_probs:
            filtered.append((idx, p))
            cum_prob += p
            if cum_prob >= top_p:
                break
        indexed_probs = filtered

    # 5. Sample from filtered distribution
    tot_p = sum(p for _, p in indexed_probs)
    if tot_p <= 0.0:
        return indexed_probs[0][0]

    r = random.random() * tot_p
    acc = 0.0
    for idx, p in indexed_probs:
        acc += p
        if acc >= r:
            return idx
    return indexed_probs[-1][0]


class FeedForward(Module):
    """
    Position-wise Feed-Forward Network: MLP(x) = ReLU(x W_1 + b_1) W_2 + b_2.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=bias)
        self.act = ReLU()
        self.w2 = Linear(d_ff, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)))

    def __repr__(self) -> str:
        return f"FeedForward(w1={self.w1}, w2={self.w2})"


class TransformerBlock(Module):
    """
    Pre-LayerNorm Transformer Block:
      x = x + MHA(LN1(x), causal=causal, past_key_value=...)
      x = x + FFN(LN2(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model, eps=eps)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len)
        self.ln2 = LayerNorm(d_model, eps=eps)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        causal: bool = True,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        # Pre-LN Self-Attention with Residual & Cache
        norm_x1 = self.ln1(x)
        if use_cache:
            attn_out, present_kv = self.attn(
                norm_x1,
                mask=mask,
                causal=causal,
                past_key_value=past_key_value,
                use_cache=True
            )
        else:
            attn_out = self.attn(
                norm_x1,
                mask=mask,
                causal=causal,
                past_key_value=past_key_value,
                use_cache=False
            )
            present_kv = None

        x = x + attn_out

        # Pre-LN Feed-Forward with Residual
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out

        if use_cache:
            return x, present_kv
        return x

    def __repr__(self) -> str:
        return f"TransformerBlock(d_model={self.attn.d_model}, num_heads={self.attn.num_heads})"


class TinyTransformerLM(Module):
    """
    Full Decoder-Only Autoregressive Language Model:
      - Token Embedding + Learned Positional Embedding
      - Stack of N Pre-LN Transformer Blocks (with KV Cache support)
      - Final LayerNorm + LM Head Projection (with weight tying option)
      - Fast Autoregressive generate() with KV Cache, Early Stopping, and Top-K/Top-P
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 512,
        padding_idx: Optional[int] = None,
        tie_weights: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.tie_weights = tie_weights

        self.tok_emb = Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = Embedding(max_seq_len, d_model)

        self.blocks = Sequential(*[
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(d_model)

        if self.tie_weights:
            self.head = None
        else:
            self.head = Linear(d_model, vocab_size, bias=False)

        backend = self.tok_emb.weight.backend
        pos_list = list(range(max_seq_len))
        pos_data = backend.from_data(pos_list, dtype="int64")
        self._cached_pos_idx = Tensor(pos_data, dtype="int64", requires_grad=False, backend=backend)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        position_offset: int = 0
    ) -> Union[Tuple[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor], List[Tuple[Tensor, Tensor]]]]:
        """
        Forward pass for language model.
        """
        orig_ndim = idx.ndim
        if orig_ndim == 1:
            idx = idx.reshape(1, idx.shape[0])
            if targets is not None and targets.ndim == 1:
                targets = targets.reshape(1, targets.shape[0])

        B, S = idx.shape
        if position_offset + S > self.max_seq_len:
            raise ValueError(f"Sequence position {position_offset + S} exceeds max_seq_len {self.max_seq_len}")

        backend = idx.backend

        # Position embeddings with offset
        pos_flat = backend.to_flat_list(self._cached_pos_idx._data)[position_offset:position_offset + S]
        pos_data = backend.from_data(pos_flat, dtype="int64")
        pos_idx = Tensor(pos_data, dtype="int64", requires_grad=False, backend=backend)

        tok_embeddings = self.tok_emb(idx)        # (B, S, d_model)
        pos_embeddings = self.pos_emb(pos_idx)    # (S, d_model)

        x = tok_embeddings + pos_embeddings

        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present_kv = block(x, mask=mask, causal=True, past_key_value=past_kv, use_cache=True)
                present_kvs.append(present_kv)
            else:
                x = block(x, mask=mask, causal=True, past_key_value=past_kv, use_cache=False)

        x = self.ln_f(x)
        if self.tie_weights:
            logits = x @ self.tok_emb.weight.transpose(1, 0)
        else:
            logits = self.head(x)  # (B, S, vocab_size)

        loss = None
        if targets is not None:
            if targets.shape != (B, S):
                raise ValueError(f"targets shape {targets.shape} does not match logits input shape {(B, S)}")
            logits_flat = logits.reshape(B * S, self.vocab_size)
            targets_flat = targets.reshape(B * S)
            loss = cross_entropy_loss(logits_flat, targets_flat)

        if orig_ndim == 1 and not use_cache:
            logits = logits.reshape(S, self.vocab_size)

        if use_cache:
            return logits, loss, present_kvs
        return logits, loss

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> List[int]:
        """
        Fast autoregressive next-token generation with KV Cache, Early Stopping, and Top-K/Top-P sampling.
        """
        tokens = list(prompt_tokens)
        backend = self.tok_emb.weight.backend
        v_size = self.vocab_size

        with no_grad():
            if use_cache and len(tokens) > 0:
                # 1. Prefill step
                prefill_idx = Tensor(backend.from_data([tokens], dtype="int64"), dtype="int64", backend=backend)
                logits, _, past_kvs = self.forward(prefill_idx, use_cache=True, position_offset=0)
                flat_l = backend.to_flat_list(logits._data)
                last_logits = flat_l[-v_size:]

                next_token = _sample_next_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                tokens.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    return tokens

                # 2. Incremental generation steps (O(1) computation per token step)
                for _ in range(max_new_tokens - 1):
                    if len(tokens) >= self.max_seq_len:
                        break
                    cur_step_idx = Tensor(backend.from_data([[next_token]], dtype="int64"), dtype="int64", backend=backend)
                    pos_offset = len(tokens) - 1
                    logits, _, past_kvs = self.forward(
                        cur_step_idx,
                        past_key_values=past_kvs,
                        use_cache=True,
                        position_offset=pos_offset
                    )
                    flat_l = backend.to_flat_list(logits._data)
                    last_logits = flat_l[-v_size:]

                    next_token = _sample_next_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    tokens.append(next_token)
                    if eos_token_id is not None and next_token == eos_token_id:
                        break
            else:
                # Standard uncached generation fallback
                for _ in range(max_new_tokens):
                    context = tokens[-self.max_seq_len:]
                    idx_data = backend.from_data([context], dtype="int64")
                    idx_tensor = Tensor(idx_data, dtype="int64", backend=backend)

                    logits, _ = self.forward(idx_tensor, use_cache=False)
                    flat_l = backend.to_flat_list(logits._data)
                    last_logits = flat_l[-v_size:]

                    next_token = _sample_next_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    tokens.append(next_token)
                    if eos_token_id is not None and next_token == eos_token_id:
                        break

        return tokens

    def __repr__(self) -> str:
        return (
            f"TinyTransformerLM(vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"num_layers={self.num_layers}, max_seq_len={self.max_seq_len}, tie_weights={self.tie_weights})"
        )
