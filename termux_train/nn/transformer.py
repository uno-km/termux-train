"""
termux_train.nn.transformer
===========================
Tiny Transformer Architecture and Autoregressive Language Model Primitives for Termux.
Includes Pre-LN Residual Blocks, Weight-Tying Option, Pre-Allocated Position Buffers,
and Safe Autoregressive Sampling.
"""

from typing import Optional, Tuple, List
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
      x = x + MHA(LN1(x), causal=causal)
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
        causal: bool = True
    ) -> Tensor:
        # Pre-LN Self-Attention with Residual
        norm_x1 = self.ln1(x)
        attn_out = self.attn(norm_x1, mask=mask, causal=causal)
        x = x + attn_out

        # Pre-LN Feed-Forward with Residual
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out
        return x

    def __repr__(self) -> str:
        return f"TransformerBlock(d_model={self.attn.d_model}, num_heads={self.attn.num_heads})"


class TinyTransformerLM(Module):
    """
    Full Decoder-Only Autoregressive Language Model:
      - Token Embedding + Learned Positional Embedding
      - Stack of N Pre-LN Transformer Blocks
      - Final LayerNorm + LM Head Projection (with optional weight tying)
      - Autoregressive generate() method with safe temperature sampling
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

        # Pre-allocated position indices buffer to avoid per-forward heap allocation
        backend = self.tok_emb.weight.backend
        pos_list = list(range(max_seq_len))
        pos_data = backend.from_data(pos_list, dtype="int64")
        self._cached_pos_idx = Tensor(pos_data, dtype="int64", requires_grad=False, backend=backend)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for language model.
          idx: shape (B, S) or (S,) int64 token indices
          targets: shape (B, S) or (S,) int64 next-token targets
          returns: (logits, loss)
        """
        orig_ndim = idx.ndim
        if orig_ndim == 1:
            idx = idx.reshape(1, idx.shape[0])
            if targets is not None and targets.ndim == 1:
                targets = targets.reshape(1, targets.shape[0])

        B, S = idx.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        backend = idx.backend

        # Zero-allocation slice of cached position tensor
        if self._cached_pos_idx.shape[0] == S:
            pos_idx = self._cached_pos_idx
        else:
            pos_flat = backend.to_flat_list(self._cached_pos_idx._data)[:S]
            pos_data = backend.from_data(pos_flat, dtype="int64")
            pos_idx = Tensor(pos_data, dtype="int64", requires_grad=False, backend=backend)

        tok_embeddings = self.tok_emb(idx)        # (B, S, d_model)
        pos_embeddings = self.pos_emb(pos_idx)    # (S, d_model)

        x = tok_embeddings + pos_embeddings

        for block in self.blocks:
            x = block(x, causal=True)

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

        if orig_ndim == 1:
            logits = logits.reshape(S, self.vocab_size)

        return logits, loss

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Autoregressive next-token generation with safe temperature sampling.
        """
        tokens = list(prompt_tokens)
        backend = self.tok_emb.weight.backend
        v_size = self.vocab_size

        with no_grad():
            for _ in range(max_new_tokens):
                context = tokens[-self.max_seq_len:]
                idx_data = backend.from_data([context], dtype="int64")
                idx_tensor = Tensor(idx_data, dtype="int64", backend=backend)

                logits, _ = self.forward(idx_tensor)
                flat_l = backend.to_flat_list(logits._data)
                last_logits = flat_l[-v_size:]

                if temperature < 1e-4 or temperature == 1.0:
                    best_token = int(max(range(v_size), key=lambda i: last_logits[i]))
                else:
                    safe_temp = max(temperature, 1e-4)
                    max_val = max(last_logits)
                    scaled = [(v - max_val) / safe_temp for v in last_logits]
                    best_token = int(max(range(v_size), key=lambda i: scaled[i]))

                tokens.append(best_token)

        return tokens

    def __repr__(self) -> str:
        return (
            f"TinyTransformerLM(vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"num_layers={self.num_layers}, max_seq_len={self.max_seq_len}, tie_weights={self.tie_weights})"
        )
