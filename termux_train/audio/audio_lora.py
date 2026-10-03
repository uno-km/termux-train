"""
termux_train.audio.audio_lora
=============================
On-device Audio STT (Whisper Cross-Attention LoRA) and TTS (Speaker Style LoRA)
Architectures for native mobile speech adaptation.
Open-Source under Apache License 2.0.
"""

from typing import List, Dict, Any, Optional, Tuple
import math

import termux_train as tt
from termux_train import Tensor, get_backend
from termux_train.nn.module import Module
from termux_train.nn.linear import Linear
from termux_train.nn.lora import LoRALinear
from termux_train.nn.layernorm import LayerNorm
from termux_train.nn.activations import GELU


class WhisperCrossAttentionLoRA(Module):
    """
    Cross-Attention LoRA Module for Whisper STT.
    Query comes from text decoder, Keys and Values come from audio encoder features.
    """

    def __init__(
        self,
        d_model: int,
        audio_dim: int,
        num_heads: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.audio_dim = audio_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Text query projection
        self.q_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        # Audio key & value projections
        self.k_proj = LoRALinear(audio_dim, d_model, rank=rank, alpha=alpha, backend=backend)
        self.v_proj = LoRALinear(audio_dim, d_model, rank=rank, alpha=alpha, backend=backend)
        # Output projection
        self.out_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)

    def forward(self, text_hidden: Tensor, audio_features: Tensor) -> Tensor:
        # text_hidden: (B, S_text, d_model)
        # audio_features: (B, T_audio, audio_dim)
        B, S_text, _ = text_hidden.shape
        _, T_audio, _ = audio_features.shape

        q = self.q_proj(text_hidden)       # (B, S_text, d_model)
        k = self.k_proj(audio_features)    # (B, T_audio, d_model)
        v = self.v_proj(audio_features)    # (B, T_audio, d_model)

        b = text_hidden.backend
        q_heads = q.reshape(B * self.num_heads, S_text, self.head_dim)
        k_heads = k.reshape(B * self.num_heads, T_audio, self.head_dim)
        v_heads = v.reshape(B * self.num_heads, T_audio, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_heads @ k_heads.swapaxes(1, 2)) * scale  # (B*heads, S_text, T_audio)
        attn = scores.softmax(axis=-1)
        out_heads = attn @ v_heads  # (B*heads, S_text, head_dim)

        out = out_heads.reshape(B, S_text, self.d_model)
        return self.out_proj(out)


class WhisperSTTLoRAModel(Module):
    """
    On-device Whisper STT Decoder with Acoustic Cross-Attention LoRA.
    Base weights are frozen (requires_grad=False), adapting only Cross-Attention LoRA layers.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        audio_dim: int = 80,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.audio_dim = audio_dim
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha

        self.tok_emb = Linear(vocab_size, d_model, bias=False, backend=backend)
        self.cross_attns: List[WhisperCrossAttentionLoRA] = []
        for i in range(num_layers):
            attn = WhisperCrossAttentionLoRA(d_model, audio_dim, num_heads=num_heads, rank=rank, alpha=alpha, backend=backend)
            self.cross_attns.append(attn)
            self._modules[f"cross_attn_{i}"] = attn

        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False, backend=backend)

        self._freeze_base_weights()

    def _freeze_base_weights(self) -> None:
        self.tok_emb.weight.requires_grad = False
        self.head.weight.requires_grad = False
        for _, m in self.named_modules():
            if isinstance(m, LoRALinear):
                if hasattr(m, "base") and hasattr(m.base, "weight"):
                    m.base.weight.requires_grad = False
                    if getattr(m.base, "bias", None) is not None:
                        m.base.bias.requires_grad = False

    def forward(self, audio_features: Tensor, text_input_ids: Tensor) -> Tensor:
        # audio_features: (B, T_audio, audio_dim)
        # text_input_ids: (B, S_text)
        b = audio_features.backend
        B = audio_features.shape[0]
        S_text = text_input_ids.shape[1]

        # Embed text tokens
        flat_ids = b.to_flat_list(text_input_ids._data)
        one_hot_flat = [0.0] * (len(flat_ids) * self.vocab_size)
        for i, tid in enumerate(flat_ids):
            valid_id = int(tid) % self.vocab_size
            one_hot_flat[i * self.vocab_size + valid_id] = 1.0

        one_hot_data = b.from_data(b.reshape(one_hot_flat, (B * S_text, self.vocab_size)), dtype="float32")
        one_hot_tensor = Tensor(one_hot_data, dtype="float32", backend=b)
        x = self.tok_emb(one_hot_tensor).reshape(B, S_text, self.d_model)

        # Cross attention layers
        for attn in self.cross_attns:
            x = x + attn(x, audio_features)

        x = self.ln_f(x)
        logits = self.head(x).reshape(B, S_text, self.vocab_size)
        return logits

    def adapter_state_dict(self) -> Dict[str, Any]:
        adapters = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                adapters[name] = {
                    "lora_A": module.lora_A.item() if hasattr(module.lora_A, "item") and module.lora_A.numel == 1 else module.lora_A._data,
                    "lora_B": module.lora_B.item() if hasattr(module.lora_B, "item") and module.lora_B.numel == 1 else module.lora_B._data,
                    "alpha": module.alpha,
                    "rank": module.rank,
                }
        return adapters


class TTSSpeakerAdaptationLoRA(Module):
    """
    On-device TTS Voice Adaptation LoRA Architecture.
    Adapts speaker embeddings and acoustic prosody projection towards a custom voice.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 64,
        n_mels: int = 80,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_mels = n_mels

        self.tok_emb = Linear(vocab_size, d_model, bias=False, backend=backend)
        self.speaker_adapter1 = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.act = GELU()
        self.speaker_adapter2 = LoRALinear(d_model, n_mels, rank=rank, alpha=alpha, backend=backend)

        # Freeze base
        self.tok_emb.weight.requires_grad = False
        for m in (self.speaker_adapter1, self.speaker_adapter2):
            if hasattr(m, "base") and hasattr(m.base, "weight"):
                m.base.weight.requires_grad = False

    def forward(self, text_input_ids: Tensor) -> Tensor:
        # text_input_ids: (B, S) -> returns predicted mel frames (B, S, n_mels)
        b = text_input_ids.backend
        B, S = text_input_ids.shape

        flat_ids = b.to_flat_list(text_input_ids._data)
        one_hot_flat = [0.0] * (len(flat_ids) * self.vocab_size)
        for i, tid in enumerate(flat_ids):
            valid_id = int(tid) % self.vocab_size
            one_hot_flat[i * self.vocab_size + valid_id] = 1.0

        one_hot_data = b.from_data(b.reshape(one_hot_flat, (B * S, self.vocab_size)), dtype="float32")
        x = self.tok_emb(Tensor(one_hot_data, dtype="float32", backend=b)).reshape(B, S, self.d_model)
        x = self.act(self.speaker_adapter1(x))
        mel_out = self.speaker_adapter2(x)
        return mel_out

    def adapter_state_dict(self) -> Dict[str, Any]:
        adapters = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                adapters[name] = {
                    "lora_A": module.lora_A.item() if hasattr(module.lora_A, "item") and module.lora_A.numel == 1 else module.lora_A._data,
                    "lora_B": module.lora_B.item() if hasattr(module.lora_B, "item") and module.lora_B.numel == 1 else module.lora_B._data,
                    "alpha": module.alpha,
                    "rank": module.rank,
                }
        return adapters
