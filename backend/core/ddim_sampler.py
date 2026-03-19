"""
DDIM sampler (Song et al. 2020) -- accelerated deterministic/stochastic sampling.

Key insight: DDIM defines a *non-Markovian* forward process that shares the same
training objective as DDPM but allows skipping timesteps during the reverse pass.

  eta = 0  -> fully deterministic (same seed = same image)
  eta = 1  -> variance matches DDPM (stochastic, but still uses a sub-sequence)
  0 < eta < 1 -> interpolation between the two

Typical step counts: 20, 50, 100, 250 (vs. 1000 for DDPM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from backend.core.noise_schedule import NoiseSchedule


def _make_ddim_timestep_sequence(num_train_steps: int, num_ddim_steps: int) -> list[int]:
    """
    Build an evenly-spaced sub-sequence of training timesteps.

    Example: 1000 training steps, 50 DDIM steps -> [0, 20, 40, ..., 980].
    """
    step = num_train_steps // num_ddim_steps
    return list(range(0, num_train_steps, step))[:num_ddim_steps]


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    schedule: "NoiseSchedule",
    shape: tuple[int, ...],
    *,
    num_steps: int = 50,
    eta: float = 0.0,
    device: torch.device | str = "cuda",
    clip_denoised: bool = True,
    return_intermediates: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    DDIM sampling with configurable step count and stochasticity.

    Parameters
    ----------
    model : nn.Module
        Noise-prediction network  eps_theta(x_t, t).
    schedule : NoiseSchedule
        Pre-computed diffusion coefficients (trained with T steps).
    shape : tuple
        (B, C, H, W) of the output batch.
    num_steps : int
        Number of DDIM denoising steps (fewer = faster, more = better quality).
    eta : float
        Stochasticity control. 0 = deterministic, 1 = DDPM-like variance.
    device : torch.device | str
        Compute device.
    clip_denoised : bool
        Clamp predicted x_0 to [-1, 1].
    return_intermediates : int
        If >0, save a snapshot every N *DDIM* steps.

    Returns
    -------
    x_0 : Tensor  (B, C, H, W) -- denoised images in [-1, 1].
    intermediates : list[Tensor] -- optional snapshots.
    """
    batch_size = shape[0]
    intermediates: list[torch.Tensor] = []

    # Build timestep sub-sequence and prepend -1 so the last step lands at t=0.
    timesteps = _make_ddim_timestep_sequence(schedule.num_timesteps, num_steps)
    timestep_pairs = list(zip(reversed(timesteps), [-1] + list(reversed(timesteps))[:-1]))

    # Convenience: index alpha_bar safely (alpha_bar_{-1} = 1.0 by convention).
    alpha_bar = schedule.alphas_cumprod
    alpha_bar_with_start = torch.cat([torch.tensor([1.0], device=device), alpha_bar])

    x = torch.randn(shape, device=device)

    for step_idx, (t_cur, t_prev) in enumerate(timestep_pairs):
        t_batch = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)

        # Model predicts noise.
        eps_pred = model(x, t_batch)

        # Predicted x_0.
        a_t = alpha_bar[t_cur]
        x0_pred = (x - eps_pred * (1 - a_t).sqrt()) / a_t.sqrt()
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Coefficients for x_{t-1}.
        a_prev = alpha_bar_with_start[t_prev + 1]  # +1 because we prepended 1.0

        # Equation 12 from DDIM paper:
        #   sigma_t = eta * sqrt((1 - a_{t-1}) / (1 - a_t)) * sqrt(1 - a_t / a_{t-1})
        if eta > 0:
            # Guard against numerical issues with very small denominators
            sigma_sq = eta**2 * ((1 - a_prev) / (1 - a_t).clamp(min=1e-8) * (1 - a_t / a_prev.clamp(min=1e-8))).clamp(min=0)
            sigma = sigma_sq.sqrt()
            dir_xt = (1 - a_prev - sigma_sq).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)
        else:
            sigma = torch.zeros((), device=device)
            dir_xt = (1 - a_prev).sqrt() * eps_pred
            noise = torch.zeros_like(x)

        x = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        if return_intermediates > 0 and (step_idx % return_intermediates == 0 or t_prev < 0):
            intermediates.append(x.detach().cpu())

    return x, intermediates
