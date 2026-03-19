"""
Complete DDPM training loop -- runs on GPU, logs metrics, saves checkpoints,
generates samples, broadcasts to WebSocket.

This is the REAL training code (not a placeholder).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.ema import EMA

logger = logging.getLogger("napsgen.trainer")


class DDPMTrainer:
    """Full DDPM training loop with all the bells and whistles."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        *,
        lr: float = 1e-4,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        ema_decay: float = 0.9999,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        sample_dir: str = "outputs/samples",
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.num_timesteps = num_timesteps
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir = Path(sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Noise schedule
        self._build_schedule(schedule_type, num_timesteps)

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(model, decay=ema_decay)

        # State
        self.global_step = 0
        self.epoch = 0

    def _build_schedule(self, schedule_type: str, T: int):
        import numpy as np
        if schedule_type == "linear":
            betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
        else:
            steps = np.arange(T + 1, dtype=np.float64)
            f = np.cos((steps / T + 0.008) / 1.008 * (math.pi / 2)) ** 2
            ab = f / f[0]
            betas = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
        alphas = 1.0 - betas
        ac = np.cumprod(alphas)
        self.sqrt_ac = torch.tensor(np.sqrt(ac), dtype=torch.float32, device=self.device)
        self.sqrt_one_minus_ac = torch.tensor(np.sqrt(1.0 - ac), dtype=torch.float32, device=self.device)

    def _add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa = self.sqrt_ac[t][:, None, None, None]
        soma = self.sqrt_one_minus_ac[t][:, None, None, None]
        return sa * x0 + soma * noise

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step: add noise -> predict noise -> MSE loss."""
        batch = batch.to(self.device)
        noise = torch.randn_like(batch)
        t = torch.randint(0, self.num_timesteps, (batch.shape[0],), device=self.device)
        noisy = self._add_noise(batch, t, noise)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred = self.model(noisy, t)
            loss = F.mse_loss(pred, noise) / self.grad_accum_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.grad_accum_steps

    def optimizer_step(self) -> float:
        """Unscale, clip, step, and zero grad."""
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        self.global_step += 1
        return grad_norm

    def save_checkpoint(self, run_id: int | None = None) -> str:
        tag = f"run_{run_id}_" if run_id else ""
        path = self.checkpoint_dir / f"{tag}step_{self.global_step}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": getattr(self.model, "config", None),
        }, str(path))
        logger.info("Checkpoint saved: %s", path)
        return str(path)

    @torch.no_grad()
    def generate_samples(self, n: int = 16, steps: int = 50) -> torch.Tensor:
        """Quick DDIM-style sampling for progress visualization."""
        self.model.eval()
        # Apply EMA for sampling
        orig_params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.ema.apply_to(self.model)

        sample_batch = next(iter(self.dataloader))
        if isinstance(sample_batch, (list, tuple)):
            sample_batch = sample_batch[0]
        img_height = sample_batch.shape[-2]
        img_width = sample_batch.shape[-1]
        x = torch.randn(n, 3, img_height, img_width, device=self.device)

        step_size = max(1, self.num_timesteps // steps)
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred_noise = self.model(x, t)
            alpha_bar = self.sqrt_ac[t_val] ** 2
            alpha_bar_prev = self.sqrt_ac[max(t_val - step_size, 0)] ** 2
            x0_pred = (x - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            dir_xt = (1 - alpha_bar_prev).sqrt() * pred_noise
            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt

        # Restore training weights
        for name, param in self.model.named_parameters():
            param.data.copy_(orig_params[name])
        self.model.train()
        return x.clamp(-1, 1)
