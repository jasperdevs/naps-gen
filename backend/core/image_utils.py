"""
Post-processing utilities for diffusion outputs.

Converts raw [-1, 1] tensors into PIL images, base64 strings, grids,
and saved files -- with optional metadata embedding.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, PngImagePlugin


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """
    Convert a (B, C, H, W) float tensor in [-1, 1] to a list of PIL RGB images.
    """
    # Clamp, rescale to [0, 255], move to uint8 on CPU.
    images = tensor.detach().cpu().clamp(-1, 1)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).numpy()  # B, H, W, C
    return [Image.fromarray(img) for img in images]


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a data-URI-ready base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def make_image_grid(images: list[Image.Image], cols: int | None = None, padding: int = 2) -> Image.Image:
    """
    Arrange a list of same-size PIL images into a single grid image.

    If *cols* is None, pick the nearest square layout.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images to grid")

    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    w, h = images[0].size
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * (w + padding), r * (h + padding)))

    return grid


def save_with_metadata(
    image: Image.Image,
    output_dir: str | Path,
    *,
    seed: int | None = None,
    steps: int | None = None,
    sampler: str | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Save a PNG with generation metadata embedded in the tEXt chunk.

    Returns the written file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.png"
    filepath = output_dir / filename

    meta = PngImagePlugin.PngInfo()
    if seed is not None:
        meta.add_text("seed", str(seed))
    if steps is not None:
        meta.add_text("steps", str(steps))
    if sampler is not None:
        meta.add_text("sampler", sampler)
    for k, v in (extra or {}).items():
        meta.add_text(k, str(v))

    image.save(str(filepath), pnginfo=meta)
    return filepath
