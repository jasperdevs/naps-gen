"""
Pydantic schemas for the generation API request / response cycle.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SamplerType(str, Enum):
    ddpm = "ddpm"
    ddim = "ddim"


class GenerationRequest(BaseModel):
    """Parameters the caller can control."""

    # Sampling
    sampler: SamplerType = Field(SamplerType.ddim, description="Sampling algorithm")
    num_steps: int = Field(50, ge=1, le=1000, description="Denoising steps (DDIM: 20-250, DDPM: use 1000)")
    eta: float = Field(0.0, ge=0.0, le=1.0, description="DDIM stochasticity (0 = deterministic)")

    # Reproducibility
    seed: int | None = Field(None, description="RNG seed (None = random)")

    # Output
    batch_size: int = Field(1, ge=1, le=16, description="Images per request")
    image_size: int = Field(64, ge=32, le=512, description="Output resolution (must match training)")

    # Conditional generation
    guidance_scale: float = Field(1.0, ge=1.0, le=20.0, description="CFG scale (1.0 = unconditional)")
    class_label: int | None = Field(None, description="Class index for conditional models")

    # Performance
    use_fp16: bool = Field(False, description="Run inference in float16")

    # Visualisation
    return_intermediates: int = Field(0, ge=0, description="Capture every N steps (0 = off)")


class GenerationResult(BaseModel):
    """Returned to the caller."""

    images_base64: list[str] = Field(description="Generated images as data-URI base64 PNGs")
    grid_base64: str | None = Field(None, description="Grid image when batch_size > 1")
    seed: int = Field(description="Seed used (for reproducibility)")
    steps: int = Field(description="Actual denoising steps used")
    sampler: str = Field(description="Sampler that was used")
    elapsed_seconds: float = Field(description="Wall-clock generation time")
    intermediates_base64: list[str] = Field(default_factory=list, description="Intermediate snapshots")
    saved_paths: list[str] = Field(default_factory=list, description="Paths of saved PNGs on disk")
