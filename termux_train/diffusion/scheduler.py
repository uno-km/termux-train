"""
termux_train.diffusion.scheduler
================================
Production-Grade Diffusion Noise Scheduler & Timestep Embedding for On-Device Training.
Implements DDPM Linear & Scaled-Linear Schedules conforming to Stable Diffusion & DiT.
Open-Source under Apache License 2.0.
"""

import math
from typing import Tuple, Optional, Union, List
import termux_train as tt
from termux_train import Tensor, zeros, randn, get_backend


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) Training Scheduler.
    Computes closed-form noisy latents z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        # 1. Compute Betas
        if beta_schedule == "linear":
            self.betas = [
                beta_start + i * (beta_end - beta_start) / (num_train_timesteps - 1)
                for i in range(num_train_timesteps)
            ]
        elif beta_schedule == "scaled_linear":
            # Stable Diffusion standard: sqrt(beta) linearly spaced
            start_sq = math.sqrt(beta_start)
            end_sq = math.sqrt(beta_end)
            self.betas = [
                (start_sq + i * (end_sq - start_sq) / (num_train_timesteps - 1)) ** 2
                for i in range(num_train_timesteps)
            ]
        else:
            raise ValueError(f"Unsupported beta_schedule '{beta_schedule}'. Choose 'linear' or 'scaled_linear'.")

        # 2. Alphas and Cumulative Product
        self.alphas = [1.0 - b for b in self.betas]
        self.alphas_cumprod = []
        curr = 1.0
        for a in self.alphas:
            curr *= a
            self.alphas_cumprod.append(curr)

        # Precompute sqrt factors
        self.sqrt_alphas_cumprod = [math.sqrt(ac) for ac in self.alphas_cumprod]
        self.sqrt_one_minus_alphas_cumprod = [math.sqrt(1.0 - ac) for ac in self.alphas_cumprod]

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Union[int, List[int], Tensor],
    ) -> Tensor:
        """
        Adds noise to original latents at specified timestep:
        z_t = sqrt(alpha_bar_t) * original_samples + sqrt(1 - alpha_bar_t) * noise
        """
        b = original_samples.backend
        if isinstance(timesteps, int):
            t_idx = min(max(0, timesteps), self.num_train_timesteps - 1)
            scale_orig = self.sqrt_alphas_cumprod[t_idx]
            scale_noise = self.sqrt_one_minus_alphas_cumprod[t_idx]
            return original_samples * scale_orig + noise * scale_noise

        # Multi-sample timesteps
        if isinstance(timesteps, Tensor):
            t_list = [int(v) for v in b.to_flat_list(timesteps._data)]
        else:
            t_list = [int(v) for v in timesteps]

        batch_size = original_samples.shape[0]
        # Bounded scaling per batch index
        out_chunks = []
        for i in range(batch_size):
            t_val = t_list[i % len(t_list)]
            t_idx = min(max(0, t_val), self.num_train_timesteps - 1)
            so = self.sqrt_alphas_cumprod[t_idx]
            sn = self.sqrt_one_minus_alphas_cumprod[t_idx]
            sample_slice = original_samples[i] * so + noise[i] * sn
            out_chunks.append(sample_slice.tolist())

        return Tensor(b.from_data(out_chunks, dtype="float32"), dtype="float32", backend=b)


def get_timestep_embedding(
    timesteps: Union[int, List[int], Tensor],
    embedding_dim: int,
    max_period: int = 10000,
) -> Tensor:
    """
    Creates sinusoidal timestep embeddings conforming to Transformer / Diffusion standard.
    PE(t, 2i)   = sin(t / max_period^(2i / dim))
    PE(t, 2i+1) = cos(t / max_period^(2i / dim))
    """
    if isinstance(timesteps, int):
        t_vals = [float(timesteps)]
    elif isinstance(timesteps, Tensor):
        t_vals = [float(x) for x in timesteps.backend.to_flat_list(timesteps._data)]
    else:
        t_vals = [float(x) for x in timesteps]

    half = embedding_dim // 2
    freqs = [math.exp(-math.log(max_period) * i / max(1, half - 1)) for i in range(half)]

    embs = []
    for t in t_vals:
        row = []
        for f in freqs:
            row.append(math.sin(t * f))
        for f in freqs:
            row.append(math.cos(t * f))
        if len(row) < embedding_dim:
            row.append(0.0)
        embs.append(row)

    backend = get_backend()
    return Tensor(backend.from_data(embs, dtype="float32"), dtype="float32", backend=backend)
