"""
Orchestration layer: ties together model loading, sampling, and post-processing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from backend.core.ddim_sampler import ddim_sample
from backend.core.ddpm_sampler import ddpm_sample
from backend.core.image_utils import (
    make_image_grid,
    pil_to_base64,
    save_with_metadata,
    tensor_to_pil,
)
from backend.core.noise_schedule import NoiseSchedule
from backend.models.generation_params import GenerationRequest, GenerationResult, SamplerType

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/generated")


class GenerationService:
    """Stateful service: holds the model + noise schedule in GPU memory."""

    def __init__(
        self,
        model: torch.nn.Module,
        schedule: NoiseSchedule,
        device: torch.device | str = "cuda",
        compile_model: bool = False,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.schedule = schedule.to(self.device)
        self.model = model.to(self.device).eval()

        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile (first call will be slow) ...")
            self.model = torch.compile(self.model)

    # ------------------------------------------------------------------

    def generate(self, req: GenerationRequest) -> GenerationResult:
        """Run the full generation pipeline and return results."""
        seed = req.seed if req.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # Global seed for randn calls inside samplers.
        torch.manual_seed(seed)

        shape = (req.batch_size, 3, req.image_size, req.image_size)
        autocast = self._get_autocast_ctx(req.use_fp16)
        start = time.perf_counter()

        with autocast:
            if req.sampler == SamplerType.ddpm:
                samples, intermediates = ddpm_sample(
                    model=self.model,
                    schedule=self.schedule,
                    shape=shape,
                    device=self.device,
                    return_intermediates=req.return_intermediates,
                )
            else:
                samples, intermediates = ddim_sample(
                    model=self.model,
                    schedule=self.schedule,
                    shape=shape,
                    num_steps=req.num_steps,
                    eta=req.eta,
                    device=self.device,
                    return_intermediates=req.return_intermediates,
                )

        elapsed = time.perf_counter() - start
        return self._build_result(samples, intermediates, req, seed, elapsed)

    # ------------------------------------------------------------------

    def _build_result(
        self,
        samples: torch.Tensor,
        intermediates: list[torch.Tensor],
        req: GenerationRequest,
        seed: int,
        elapsed: float,
    ) -> GenerationResult:
        pil_images = tensor_to_pil(samples)

        # Save to disk.
        saved: list[str] = []
        for img in pil_images:
            p = save_with_metadata(
                img,
                OUTPUT_DIR,
                seed=seed,
                steps=req.num_steps,
                sampler=req.sampler.value,
                extra={"eta": req.eta, "guidance_scale": req.guidance_scale},
            )
            saved.append(str(p))

        images_b64 = [pil_to_base64(img) for img in pil_images]
        grid_b64 = pil_to_base64(make_image_grid(pil_images)) if len(pil_images) > 1 else None

        # Intermediate snapshots.
        inter_b64: list[str] = []
        for t in intermediates:
            frame_images = tensor_to_pil(t)
            inter_b64.append(pil_to_base64(make_image_grid(frame_images) if len(frame_images) > 1 else frame_images[0]))

        return GenerationResult(
            images_base64=images_b64,
            grid_base64=grid_b64,
            seed=seed,
            steps=req.num_steps,
            sampler=req.sampler.value,
            elapsed_seconds=round(elapsed, 3),
            intermediates_base64=inter_b64,
            saved_paths=saved,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _get_autocast_ctx(use_fp16: bool):
        """Return the appropriate autocast context (or a no-op)."""
        if use_fp16 and torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return _NullContext()


class _NullContext:
    """Minimal no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
