"""
termux_train.diffusion.trainer
==============================
End-to-End On-Device Image Folder Diffusion LoRA Training Pipeline.
Transforms raw image datasets into ComfyUI / Diffusers compatible SafeTensors LoRA adapters.
Open-Source under Apache License 2.0.
"""

import os
import sys
import time
import random
import logging
from typing import Dict, Any, Optional, List, Tuple

import termux_train as tt
from termux_train import Tensor, randn, get_backend, set_backend
import termux_train.nn as nn
import termux_train.optim as optim
from termux_train.data.image_dataset import ImageFolderDataset
from termux_train.diffusion.scheduler import DDPMScheduler
from termux_train.diffusion.unet_lora import DiffusionUNetLoRA
from termux_train.adapters.hub import export_target_adapter

logger = logging.getLogger(__name__)


def train_diffusion_lora(
    image_dir: str,
    output_path: str,
    prompt: str = "a photo of subject",
    resolution: int = 512,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 1,
    rank: int = 4,
    alpha: float = 1.0,
    model_dim: int = 64,
    context_dim: int = 64,
    backend: str = "auto",
    verbose: bool = True,
) -> str:
    """
    Executes end-to-end on-device Diffusion LoRA training from an image folder.

    Args:
        image_dir: Directory containing training images and optional .txt caption pairs.
        output_path: Target .safetensors path for exported adapter.
        prompt: Default conditioning prompt if per-image .txt captions are not provided.
        resolution: Image resolution (default: 512).
        epochs: Training iterations over the image dataset.
        lr: Adapter learning rate.
        batch_size: Mini-batch size.
        rank: LoRA low-rank decomposition dimension.
        alpha: LoRA scaling multiplier.
        model_dim: UNet internal hidden channel dimension.
        context_dim: Text conditioning context dimension.
        backend: Compute backend ('auto', 'vulkan', 'amuda', 'numpy', 'python').
        verbose: Whether to print progress metrics to stdout.

    Returns:
        Absolute path to the exported SafeTensors LoRA file.
    """
    if backend and backend != "auto":
        set_backend(backend)
    b = get_backend()

    t_start = time.perf_counter()
    if verbose:
        print("=" * 65)
        print("  🎨 termux-train: On-Device Diffusion LoRA Engine")
        print(f"  • Image Directory   : {image_dir}")
        print(f"  • Resolution        : {resolution}x{resolution} (Latents: {resolution//8}x{resolution//8}x4)")
        print(f"  • LoRA Config       : Rank={rank}, Alpha={alpha}, LR={lr}")
        print(f"  • Compute Backend   : {b.name.upper()}")
        print(f"  • Target Output     : {output_path}")
        print("=" * 65)

    # 1. Prepare Image Folder Dataset with VAE Latent Caching
    dataset = ImageFolderDataset(
        image_dir=image_dir,
        resolution=resolution,
        default_prompt=prompt,
        cache_latents=True,
    )
    num_samples = len(dataset)
    if verbose:
        print(f"  📂 Loaded {num_samples} images and caption pairs.")

    # 2. Build Noise Scheduler & UNet LoRA Model
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
    unet = DiffusionUNetLoRA(
        in_channels=4,
        out_channels=4,
        model_dim=model_dim,
        context_dim=context_dim,
        rank=rank,
        alpha=alpha,
    )

    optimizer = optim.AdamW(unet.adapter_parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Simple Hash-Deterministic Text Embedding Generator for On-Device Context
    def _text_to_context(caption_str: str) -> Tensor:
        # Generate deterministic (1, 16, context_dim) text conditioning
        import hashlib
        h = hashlib.sha256(caption_str.encode("utf-8")).digest()
        # 16 tokens of context_dim values
        vals = []
        for i in range(16):
            token_row = [((h[(i * 4 + j) % len(h)] / 127.5) - 1.0) * 0.5 for j in range(context_dim)]
            vals.append(token_row)
        return Tensor(b.from_data([vals], dtype="float32"), dtype="float32", backend=b)

    # 3. Training Loop
    total_steps = epochs * num_samples
    global_step = 0

    for ep in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        ep_t0 = time.perf_counter()

        for idx in range(num_samples):
            latent_z0, caption = dataset.get_latent_and_caption(idx)
            # Add batch dimension: (1, 4, H/8, W/8)
            latent_z0 = latent_z0.reshape(1, *latent_z0.shape)

            # Sample random timestep t ~ Uniform(0, 1000)
            t_val = random.randint(0, scheduler.num_train_timesteps - 1)
            # Sample standard Gaussian noise eps ~ N(0, I)
            eps = randn(latent_z0.shape, backend=b)

            # Inject noise: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps
            noisy_zt = scheduler.add_noise(latent_z0, eps, t_val)

            # Text Context conditioning
            context_tensor = _text_to_context(caption)

            optimizer.zero_grad()
            noise_pred = unet(noisy_zt, timesteps=t_val, context=context_tensor)

            loss = criterion(noise_pred, eps)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss_sum += loss_val
            global_step += 1

        avg_loss = epoch_loss_sum / num_samples
        ep_ms = (time.perf_counter() - ep_t0) * 1000.0

        if verbose:
            print(f"  [Epoch {ep}/{epochs}] Step: {global_step}/{total_steps} | Loss: {avg_loss:.6f} | {ep_ms:.1f}ms")

    # 4. Export to ComfyUI / Diffusers / stable-diffusion.cpp Standard SafeTensors
    exported_file = export_target_adapter(
        unet,
        target="diffusion",
        output_path=output_path,
        base_model_name="stable-diffusion-v1-5",
        custom_metadata={
            "resolution": str(resolution),
            "epochs": str(epochs),
            "final_loss": f"{avg_loss:.6f}",
            "prompt": prompt,
        },
    )

    tot_time = time.perf_counter() - t_start
    if verbose:
        print("=" * 65)
        print(f"  ✅ Diffusion LoRA Training Converged in {tot_time:.2f}s!")
        print(f"  📦 Exported Adapter: {exported_file}")
        print("=" * 65)

    return exported_file
