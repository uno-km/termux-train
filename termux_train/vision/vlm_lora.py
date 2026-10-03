"""
termux_train.vision.vlm_lora
============================
On-device Vision-Language Model (VLM) Architecture with LoRA Adaptation.
Includes Multimodal Visual Projector LoRA, Self-Attention LoRA, and Visual Cross-Attention LoRA.
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


class MultimodalProjectorLoRA(Module):
    """
    Multimodal Visual Projector Layer with LoRA Adaptation.
    Maps visual patch tokens from Vision Encoder dimension to LLM hidden dimension.
    """

    def __init__(
        self,
        visual_dim: int,
        llm_dim: int,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim
        self.proj = LoRALinear(visual_dim, llm_dim, rank=rank, alpha=alpha, backend=backend)
        self.act = GELU()
        self.out_proj = LoRALinear(llm_dim, llm_dim, rank=rank, alpha=alpha, backend=backend)

    def forward(self, visual_features: Tensor) -> Tensor:
        # visual_features: (B, N_patches, visual_dim)
        x = self.proj(visual_features)
        x = self.act(x)
        x = self.out_proj(x)
        return x


class VLMAttentionLoRA(Module):
    """Multi-Head Self-Attention with LoRA injected into Q, K, V, and Out projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.k_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.v_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.out_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, S, d_model)
        B, S, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = q.reshape(B * self.num_heads, S, self.head_dim)
        k_heads = k.reshape(B * self.num_heads, S, self.head_dim)
        v_heads = v.reshape(B * self.num_heads, S, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_heads @ k_heads.swapaxes(1, 2)) * scale
        attn = scores.softmax(axis=-1)
        out_heads = attn @ v_heads

        out = out_heads.reshape(B, S, self.d_model)
        return self.out_proj(out)


class VLMCrossAttentionLoRA(Module):
    """
    Visual Cross-Attention with LoRA.
    Query comes from text decoder, Keys and Values come from projected visual tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.k_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.v_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)
        self.out_proj = LoRALinear(d_model, d_model, rank=rank, alpha=alpha, backend=backend)

    def forward(self, text_hidden: Tensor, visual_emb: Tensor) -> Tensor:
        # text_hidden: (B, S_text, d_model)
        # visual_emb: (B, N_patches, d_model)
        B, S_text, _ = text_hidden.shape
        _, N_patches, _ = visual_emb.shape

        q = self.q_proj(text_hidden)
        k = self.k_proj(visual_emb)
        v = self.v_proj(visual_emb)

        q_heads = q.reshape(B * self.num_heads, S_text, self.head_dim)
        k_heads = k.reshape(B * self.num_heads, N_patches, self.head_dim)
        v_heads = v.reshape(B * self.num_heads, N_patches, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_heads @ k_heads.swapaxes(1, 2)) * scale
        attn = scores.softmax(axis=-1)
        out_heads = attn @ v_heads

        out = out_heads.reshape(B, S_text, self.d_model)
        return self.out_proj(out)


class VLMTransformerBlockLoRA(Module):
    """VLM Decoder Block with Self-Attention LoRA, Visual Cross-Attention LoRA, and LayerNorms."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 128,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.self_attn = VLMAttentionLoRA(d_model, num_heads=num_heads, rank=rank, alpha=alpha, backend=backend)
        self.ln2 = LayerNorm(d_model)
        self.cross_attn = VLMCrossAttentionLoRA(d_model, num_heads=num_heads, rank=rank, alpha=alpha, backend=backend)
        self.ln3 = LayerNorm(d_model)
        self.mlp_in = LoRALinear(d_model, d_ff, rank=rank, alpha=alpha, backend=backend)
        self.act = GELU()
        self.mlp_out = LoRALinear(d_ff, d_model, rank=rank, alpha=alpha, backend=backend)

    def forward(self, x: Tensor, visual_emb: Tensor) -> Tensor:
        # Self-Attention
        norm_x1 = self.ln1(x)
        x = x + self.self_attn(norm_x1)

        # Visual Cross-Attention
        norm_x2 = self.ln2(x)
        x = x + self.cross_attn(norm_x2, visual_emb)

        # Feed-Forward
        norm_x3 = self.ln3(x)
        ff = self.mlp_out(self.act(self.mlp_in(norm_x3)))
        return x + ff


class VisionLanguageModelLoRA(Module):
    """
    On-device Vision Language Model (VLM) Architecture.
    Combines Vision Projector LoRA with Visual Cross-Attention LLM Decoder LoRA blocks.
    Base weights are frozen (requires_grad=False), optimizing only LoRA parameters.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        visual_dim: int = 64,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        rank: int = 4,
        alpha: float = 1.0,
        backend=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.visual_dim = visual_dim
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha

        # 1. Visual Projector
        self.projector = MultimodalProjectorLoRA(visual_dim, d_model, rank=rank, alpha=alpha, backend=backend)

        # 2. Text Token Embedding
        self.tok_emb = Linear(vocab_size, d_model, bias=False, backend=backend)

        # 3. Decoder Blocks
        self.blocks: List[VLMTransformerBlockLoRA] = []
        for i in range(num_layers):
            block = VLMTransformerBlockLoRA(d_model, num_heads=num_heads, d_ff=d_ff, rank=rank, alpha=alpha, backend=backend)
            self.blocks.append(block)
            self._modules[f"block_{i}"] = block

        # 4. Final Norm & LM Head
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False, backend=backend)

        self._freeze_base_weights()

    def _freeze_base_weights(self) -> None:
        """Freeze base model weights, keeping LoRA adapters trainable."""
        self.tok_emb.weight.requires_grad = False
        self.head.weight.requires_grad = False

        # In LoRALinear layers, freeze the base weight
        for _, m in self.named_modules():
            if isinstance(m, LoRALinear):
                if hasattr(m, "base") and hasattr(m.base, "weight"):
                    m.base.weight.requires_grad = False
                    if getattr(m.base, "bias", None) is not None:
                        m.base.bias.requires_grad = False

    def forward(self, visual_features: Tensor, text_input_ids: Tensor) -> Tensor:
        """
        Forward pass conditioning text generation on projected visual features.
        visual_features: (B, N_patches, visual_dim)
        text_input_ids: (B, S_text) integer tokens
        Returns: logits: (B, S_text, vocab_size)
        """
        b = visual_features.backend
        B = visual_features.shape[0]
        S_text = text_input_ids.shape[1]

        # 1. Project visual tokens -> (B, N_patches, d_model)
        vis_emb = self.projector(visual_features)

        # 2. Embed text tokens (one-hot conversion for autograd compatibility)
        flat_ids = b.to_flat_list(text_input_ids._data)
        one_hot_flat = [0.0] * (len(flat_ids) * self.vocab_size)
        for i, tid in enumerate(flat_ids):
            valid_id = int(tid) % self.vocab_size
            one_hot_flat[i * self.vocab_size + valid_id] = 1.0

        one_hot_data = b.from_data(b.reshape(one_hot_flat, (B * S_text, self.vocab_size)), dtype="float32")
        one_hot_tensor = Tensor(one_hot_data, dtype="float32", backend=b)
        x = self.tok_emb(one_hot_tensor).reshape(B, S_text, self.d_model)

        # 3. Pass through decoder blocks with visual cross-attention
        for block in self.blocks:
            x = block(x, vis_emb)

        # 4. Final Norm and LM Head projection
        x = self.ln_f(x)
        logits = self.head(x).reshape(B, S_text, self.vocab_size)
        return logits

    def adapter_state_dict(self) -> Dict[str, Any]:
        """Extracts all trainable LoRA parameters for target adapter hub export."""
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
