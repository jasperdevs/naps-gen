"""
Model loading utilities: checkpoint handling, EMA weight selection, warm-up.

Supports two checkpoint formats:
  1. Full training checkpoint  { "model", "ema", "optimizer", "config", "step" }
  2. Weights-only state_dict   (plain OrderedDict of parameters)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint file and return the raw dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return {"model": load_file(str(path))}

    return torch.load(str(path), map_location="cpu", weights_only=False)


def extract_state_dict(
    checkpoint: dict[str, Any],
    use_ema: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Pull the right state_dict out of a checkpoint.

    EMA weights generally produce sharper images because they average parameters
    over many training steps, smoothing out gradient noise.
    """
    # Full training checkpoint with separate EMA key.
    if use_ema and "ema" in checkpoint:
        logger.info("Using EMA weights from checkpoint")
        return checkpoint["ema"]

    if "model" in checkpoint:
        logger.info("Using %s weights from checkpoint", "training" if not use_ema else "model (no EMA found)")
        return checkpoint["model"]

    # Assume the dict *is* the state_dict.
    logger.info("Loaded bare state_dict (no wrapper keys)")
    return checkpoint


def extract_config(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Return embedded model/diffusion config if present."""
    return checkpoint.get("config")


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = True,
    strict: bool = True,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """
    End-to-end: load checkpoint -> select weights -> load into model -> move to device.
    """
    ckpt = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(ckpt, use_ema=use_ema)
    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and moved to %s", device)
    return model


def warmup_model(model: torch.nn.Module, shape: tuple[int, ...], device: torch.device | str) -> None:
    """
    Run a single dummy forward pass so that CUDA kernels, memory allocators,
    and torch.compile graphs are fully initialised before real requests arrive.

    Without this, the first real inference call takes 2-10x longer.
    """
    logger.info("Running model warm-up forward pass ...")
    with torch.no_grad():
        dummy_x = torch.randn(shape, device=device)
        dummy_t = torch.zeros(shape[0], device=device, dtype=torch.long)
        _ = model(dummy_x, dummy_t)
    # Free the dummy tensors immediately.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Warm-up complete")
