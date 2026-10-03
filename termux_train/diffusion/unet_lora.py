"""
termux_train.diffusion.unet_lora
================================
Diffusion UNet / DiT Cross-Attention LoRA Injection Architecture.
Implements Cross-Attention LoRA adapters (to_q, to_k, to_v, to_out) conforming
to Stable Diffusion, SDXL, and ComfyUI diffusers-lora format.
Open-Source under Apache License 2.0.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

import termux_train as tt
from termux_train import Tensor, zeros, randn, get_backend
from termux_train.nn.module import Module
from termux_train.nn.linear import Linear
from termux_train.nn.lora import LoRALinear
from termux_train.nn.activations import ReLU, SiLU
from termux_train.nn.layernorm import LayerNorm
from termux_train.diffusion.scheduler import get_timestep_embedding


class CrossAttentionLoRA(Module):
    """
    Multi-Head Cross-Attention Layer with injected LoRALinear adapters.
    Adapts Query, Key, Value, and Output projection matrices while keeping base weights frozen.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rank: int = 4,
        alpha: float = 1.0,
        layer_prefix: str = "attn",
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = 1.0 / math.sqrt(dim_head)
        self.layer_prefix = layer_prefix

        # LoRA Injected Projections
        b = get_backend()
        base_q = Linear(query_dim, self.inner_dim, bias=False, backend=b)
        base_k = Linear(context_dim, self.inner_dim, bias=False, backend=b)
        base_v = Linear(context_dim, self.inner_dim, bias=False, backend=b)
        base_out = Linear(self.inner_dim, query_dim, bias=False, backend=b)

        # Freeze base linear weights to isolate adapter training
        base_q.weight.requires_grad = False
        base_k.weight.requires_grad = False
        base_v.weight.requires_grad = False
        base_out.weight.requires_grad = False

        self.to_q = LoRALinear.from_linear(base_q, rank=rank, alpha=alpha)
        self.to_k = LoRALinear.from_linear(base_k, rank=rank, alpha=alpha)
        self.to_v = LoRALinear.from_linear(base_v, rank=rank, alpha=alpha)
        self.to_out = LoRALinear.from_linear(base_out, rank=rank, alpha=alpha)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, N, query_dim)
        context: (B, S, context_dim) or None (self-attention)
        """
        if context is None:
            context = x

        B, N, _ = x.shape
        _, S, _ = context.shape

        q = self.to_q(x)  # (B, N, inner_dim)
        k = self.to_k(context)  # (B, S, inner_dim)
        v = self.to_v(context)  # (B, S, inner_dim)

        # Attention: Q @ K.T scaled
        # Reshape to (B * heads, N, dim_head)
        q = q.reshape(B * self.heads, N, self.dim_head)
        k = k.reshape(B * self.heads, S, self.dim_head)
        v = v.reshape(B * self.heads, S, self.dim_head)

        attn_scores = (q @ k.swapaxes(1, 2)) * self.scale
        attn_probs = attn_scores.softmax(axis=-1)
        out = attn_probs @ v  # (B * heads, N, dim_head)

        out = out.reshape(B, N, self.inner_dim)
        return self.to_out(out)

    def adapter_state_dict(self) -> Dict[str, Dict[str, Any]]:
        """Returns standard adapter state dict for ComfyUI / Diffusers export."""
        res = {}
        for name, sub in [
            ("to_q", self.to_q),
            ("to_k", self.to_k),
            ("to_v", self.to_v),
            ("to_out", self.to_out),
        ]:
            key = f"{self.layer_prefix}_{name}"
            res[key] = sub.adapter_state_dict()
        return res


class DiffusionUNetLoRA(Module):
    """
    Compact On-Device Latent Diffusion UNet Backbone with Trainable LoRA Adapters.
    Predicts added Gaussian noise eps_theta(z_t, t, c) from noisy latents, timesteps, and text context.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_dim: int = 64,
        context_dim: int = 64,
        time_dim: int = 64,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_dim = model_dim
        self.context_dim = context_dim
        self.time_dim = time_dim

        b = get_backend()
        # 1. Input Projection
        self.conv_in = Linear(in_channels, model_dim, backend=b)

        # 2. Timestep MLP
        self.time_mlp_1 = Linear(time_dim, model_dim, backend=b)
        self.time_act = SiLU()
        self.time_mlp_2 = Linear(model_dim, model_dim, backend=b)

        # 3. Down-Block Cross-Attention LoRA
        self.down_attn = CrossAttentionLoRA(
            query_dim=model_dim,
            context_dim=context_dim,
            heads=2,
            dim_head=model_dim // 2,
            rank=rank,
            alpha=alpha,
            layer_prefix="down_blocks_0_attentions_0",
        )

        # 4. Mid-Block Cross-Attention LoRA
        self.mid_attn = CrossAttentionLoRA(
            query_dim=model_dim,
            context_dim=context_dim,
            heads=2,
            dim_head=model_dim // 2,
            rank=rank,
            alpha=alpha,
            layer_prefix="mid_block_attentions_0",
        )

        # 5. Up-Block Cross-Attention LoRA
        self.up_attn = CrossAttentionLoRA(
            query_dim=model_dim,
            context_dim=context_dim,
            heads=2,
            dim_head=model_dim // 2,
            rank=rank,
            alpha=alpha,
            layer_prefix="up_blocks_0_attentions_0",
        )

        # 6. Output Projection
        self.norm_out = LayerNorm(model_dim)
        self.conv_out = Linear(model_dim, out_channels, backend=b)

    def forward(
        self,
        noisy_latents: Tensor,
        timesteps: Union[int, List[int], Tensor],
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass predicting added noise.
        noisy_latents: (B, C, H, W) or (B, N, C)
        timesteps: scalar or (B,) tensor
        context: (B, S, context_dim) text conditioning
        """
        orig_shape = noisy_latents.shape
        if len(orig_shape) == 4:
            B, C, H, W = orig_shape
            # Permute to (B, H*W, C) for linear transformer attention
            x = noisy_latents.reshape(B, C, H * W).swapaxes(1, 2)
        else:
            B, N, C = orig_shape
            x = noisy_latents

        # 1. Project input latents
        h = self.conv_in(x)

        # 2. Add Timestep Embedding
        t_emb = get_timestep_embedding(timesteps, self.time_dim)
        t_proj = self.time_mlp_2(self.time_act(self.time_mlp_1(t_emb)))  # (B, model_dim)
        # Broadcast add across spatial tokens
        t_proj_expanded = t_proj.reshape(B, 1, self.model_dim)
        h = h + t_proj_expanded

        # 3. Down-Block
        h = h + self.down_attn(h, context=context)

        # 4. Mid-Block
        h = h + self.mid_attn(h, context=context)

        # 5. Up-Block
        h = h + self.up_attn(h, context=context)

        # 6. Final Projection
        h = self.norm_out(h)
        pred_flat = self.conv_out(h)

        if len(orig_shape) == 4:
            # Reshape back to (B, C, H, W)
            return pred_flat.swapaxes(1, 2).reshape(B, self.out_channels, H, W)
        return pred_flat

    def adapter_parameters(self) -> List[Tensor]:
        """Returns trainable parameters across all injected LoRA adapters."""
        params = []
        for attn in (self.down_attn, self.mid_attn, self.up_attn):
            for sub in (attn.to_q, attn.to_k, attn.to_v, attn.to_out):
                params.extend(sub.adapter_parameters())
        return params

    def adapter_state_dict(self) -> Dict[str, Any]:
        """Aggregates all adapter weights into standard diffusers-lora key mappings."""
        state = {}
        for attn in (self.down_attn, self.mid_attn, self.up_attn):
            state.update(attn.adapter_state_dict())
        return {"adapters": state}
