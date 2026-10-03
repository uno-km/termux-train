"""
AMEVA On-Device Image Folder Diffusion LoRA Pipeline Tests.
Validates DDPM Noise Scheduling, Image Dataset VAE Latent Caching,
UNet Cross-Attention LoRA Injection, and ComfyUI/Diffusers SafeTensors Export.
Component: [TRAIN-DIFFUSION-PIPELINE]
"""

import os
import sys
import tempfile
import json
import pytest
from unittest.mock import patch, MagicMock

import termux_train as tt
from termux_train import Tensor, randn
from termux_train.diffusion.scheduler import DDPMScheduler, get_timestep_embedding
from termux_train.diffusion.unet_lora import CrossAttentionLoRA, DiffusionUNetLoRA
from termux_train.data.image_dataset import ImageFolderDataset, create_mock_diffusion_dataset
from termux_train.diffusion.trainer import train_diffusion_lora
from termux_train.checkpoint.safetensors import load_safetensors
from termux_train.cli import main


def test_ddpm_scheduler_and_timestep_embeddings():
    """Verify DDPM scaled-linear noise schedule and sinusoidal PE embeddings."""
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
    assert len(scheduler.betas) == 1000
    assert len(scheduler.alphas_cumprod) == 1000
    assert 0.0 < scheduler.alphas_cumprod[-1] < scheduler.alphas_cumprod[0] <= 1.0

    # Test noise injection
    z0 = randn((2, 4, 16, 16))
    noise = randn((2, 4, 16, 16))
    zt = scheduler.add_noise(z0, noise, timesteps=[100, 500])
    assert zt.shape == (2, 4, 16, 16)

    # Test timestep embedding
    t_emb = get_timestep_embedding([100, 500], embedding_dim=64)
    assert t_emb.shape == (2, 64)


def test_image_folder_dataset_and_latent_caching():
    """Verify scanning, caption pairing, and persistent VAE latent caching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = create_mock_diffusion_dataset(tmp_dir, num_samples=3)

        ds = ImageFolderDataset(dataset_dir, resolution=64, default_prompt="default trigger")
        assert len(ds) == 3

        # First read: creates latent and writes to .cache/latents/
        lat0, cap0 = ds.get_latent_and_caption(0)
        assert lat0.shape == (4, 8, 8)
        assert "photo of item 0" in cap0

        cache_files = os.listdir(ds.cache_dir)
        assert len(cache_files) >= 1

        # Second read: fast cache hit
        lat0_cached, _ = ds.get_latent_and_caption(0)
        assert lat0_cached.shape == (4, 8, 8)


def test_diffusion_unet_lora_isolation_and_backward():
    """Verify base weights are frozen and only LoRA adapter parameters accumulate gradients."""
    unet = DiffusionUNetLoRA(
        in_channels=4,
        out_channels=4,
        model_dim=32,
        context_dim=32,
        time_dim=32,
        rank=2,
    )

    # Base linear weights must be frozen
    assert unet.down_attn.to_q.base.weight.requires_grad is False
    assert unet.down_attn.to_q.lora_A.requires_grad is True
    assert unet.down_attn.to_q.lora_B.requires_grad is True

    # Forward pass
    latents = randn((1, 4, 8, 8))
    context = randn((1, 4, 32))
    noise_pred = unet(latents, timesteps=100, context=context)
    assert noise_pred.shape == (1, 4, 8, 8)

    # Backward pass
    target = randn((1, 4, 8, 8))
    loss = ((noise_pred - target) ** 2).sum()
    loss.backward()

    # Verify adapter parameters received gradients
    assert unet.down_attn.to_q.lora_A.grad is not None
    assert unet.down_attn.to_q.lora_B.grad is not None


def test_train_diffusion_lora_end_to_end():
    """Verify complete end-to-end on-device Image Folder LoRA training and ComfyUI export."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = create_mock_diffusion_dataset(os.path.join(tmp_dir, "images"), num_samples=2)
        out_adapter = os.path.join(tmp_dir, "diffusion_lora.safetensors")

        exported_path = train_diffusion_lora(
            image_dir=dataset_dir,
            output_path=out_adapter,
            prompt="sks style subject",
            resolution=64,
            epochs=2,
            lr=0.01,
            batch_size=1,
            rank=2,
            model_dim=32,
            context_dim=32,
            verbose=False,
        )

        assert os.path.isfile(exported_path)
        assert os.path.isfile(os.path.splitext(exported_path)[0] + "_config.json")

        # Inspect exported SafeTensors keys
        tensors, meta = load_safetensors(exported_path)
        assert len(tensors) > 0
        assert meta.get("ecosystem_target") == "diffusion"

        # Check for ComfyUI / Diffusers standard LoRA keys
        keys = list(tensors.keys())
        has_down = any("lora_down.weight" in k for k in keys)
        has_up = any("lora_up.weight" in k for k in keys)
        assert has_down and has_up


def test_cli_diffusion_train_args(monkeypatch):
    """Verify CLI accepts diffusion-train subcommand."""
    test_args = [
        "termux-train",
        "diffusion-train",
        "--image-dir",
        "./samples/images",
        "--prompt",
        "a photo of cybernetic cat",
        "--resolution",
        "512",
        "--epochs",
        "3",
        "--rank",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_train.cli.cmd_diffusion_train") as mock_diff_train:
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None
        assert mock_diff_train.call_count == 1
        args = mock_diff_train.call_args[0][0]
        assert args.image_dir == "./samples/images"
        assert args.prompt == "a photo of cybernetic cat"
        assert args.resolution == 512
        assert args.epochs == 3
        assert args.rank == 8
