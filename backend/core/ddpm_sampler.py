"""
DDPM reverse-diffusion sampler (Ho et al. 2020).

Walks *every* timestep from T-1 down to 0.  Slow (typically 1000 NFEs)
but produces the highest-quality samples.  Supports returning intermediate
frames for denoising visualisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from backend.core.noise_schedule import NoiseSchedule


def _extract(coeff: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Index ``coeff[t]`` and reshape for broadcasting against an ndim tensor."""
    out = coeff.gather(0, t)
    return out.view(-1, *([1] * (ndim - 1)))


@torch.no_grad()
def ddpm_sample(
    model: torch.nn.Module,
    schedule: "NoiseSchedule",
    shape: tuple[int, ...],
    *,
    device: torch.device | str = "cuda",
    clip_denoised: bool = True,
    return_intermediates: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Full DDPM ancestral sampling loop.

    Parameters
    ----------
    model : nn.Module
        Noise-prediction network  eps_theta(x_t, t) -> predicted noise.
    schedule : NoiseSchedule
        Pre-computed diffusion coefficients.
    shape : tuple
        (B, C, H, W) of the output batch.
    device : torch.device
        Where to run inference.
    clip_denoised : bool
        Clamp predicted x_0 to [-1, 1] at each step (stabilises sampling).
    return_intermediates : int
        If >0, capture an intermediate image every N steps for visualisation.

    Returns
    -------
    x_0 : Tensor  (B, C, H, W) -- final denoised images in [-1, 1].
    intermediates : list[Tensor] -- snapshots collected during sampling.
    """
    batch_size = shape[0]
    ndim = len(shape)
    intermediates: list[torch.Tensor] = []

    # x_T ~ N(0, I)
    x = torch.randn(shape, device=device)

    for step in reversed(range(schedule.num_timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        # --- predict x_0 from the noise prediction --------------------------
        predicted_noise = model(x, t)
        x0_pred = (
            _extract(schedule.sqrt_recip_alphas_cumprod, t, ndim) * x
            - _extract(schedule.sqrt_recipm1_alphas_cumprod, t, ndim) * predicted_noise
        )
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # --- posterior mean  q(x_{t-1} | x_t, x_0) -------------------------
        mean = (
            _extract(schedule.posterior_mean_coef1, t, ndim) * x0_pred
            + _extract(schedule.posterior_mean_coef2, t, ndim) * x
        )
        log_var = _extract(schedule.posterior_log_variance_clipped, t, ndim)

        # --- sample x_{t-1} (no noise at t=0) ------------------------------
        noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
        x = mean + noise * torch.exp(0.5 * log_var)

        # --- optionally capture intermediates --------------------------------
        if return_intermediates > 0 and (step % return_intermediates == 0 or step == 0):
            intermediates.append(x.detach().cpu())

    return x, intermediates
