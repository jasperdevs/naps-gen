"""
Precomputed noise schedule coefficients used by both DDPM and DDIM samplers.

All math follows the original DDPM paper (Ho et al. 2020):
  beta_t           = variance schedule
  alpha_t          = 1 - beta_t
  alpha_bar_t      = cumulative product of alpha_t
  posterior_mean, posterior_variance = q(x_{t-1} | x_t, x_0)
"""

from __future__ import annotations

import numpy as np
import torch


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal 2021)."""
    steps = np.arange(num_timesteps + 1, dtype=np.float64)
    f = np.cos((steps / num_timesteps + s) / (1 + s) * (np.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 0.0, 0.999)


class NoiseSchedule:
    """Pre-computes and stores all diffusion coefficients on a given device."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ):
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Store everything as float32 tensors on the target device.
        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float32, device=device)

        self.betas = _t(betas)
        self.alphas = _t(alphas)
        self.alphas_cumprod = _t(alphas_cumprod)
        self.alphas_cumprod_prev = _t(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = _t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = _t(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = _t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = _t(np.sqrt(1.0 / alphas_cumprod - 1.0))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = _t(posterior_variance)
        self.posterior_log_variance_clipped = _t(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = _t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = _t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

    def to(self, device: torch.device | str) -> "NoiseSchedule":
        """Move all tensors to a different device (returns self for chaining)."""
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self
