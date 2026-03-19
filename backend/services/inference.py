"""
Inference service: loads the custom DDPM model and generates images.

Uses the real U-Net + DDIM sampler for fast generation.
Thread-safe via asyncio lock.
"""

import asyncio
import logging
import time
import uuid
import io
import base64
from pathlib import Path

import torch
from PIL import Image

from backend.core.config import settings
from backend.core.noise_schedule import NoiseSchedule
from backend.core.ddim_sampler import ddim_sample
from backend.core.ddpm_sampler import ddpm_sample
from backend.core.image_utils import tensor_to_pil, pil_to_base64, make_image_grid, save_with_metadata
from backend.core.ws_manager import ws_manager

logger = logging.getLogger("napsgen.inference")


class InferenceService:
    def __init__(self):
        self._model = None
        self._schedule: NoiseSchedule | None = None
        self._loaded_checkpoint: str | None = None
        self._lock = asyncio.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._image_width = 64
        self._image_height = 64

    async def generate(
        self,
        seed: int | None = None,
        num_steps: int = 50,
        sampler: str = "ddim",
        eta: float = 0.0,
        batch_size: int = 1,
        checkpoint: str | None = None,
        return_base64: bool = True,
        **kwargs,
    ) -> dict:
        async with self._lock:
            start = time.time()
            self._ensure_model(checkpoint)

            if self._model is None:
                return {"error": "No model loaded. Train a model first or provide a checkpoint.", "images": []}

            if seed is None:
                seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)

            shape = (batch_size, 3, self._image_height, self._image_width)

            with torch.no_grad():
                if sampler == "ddpm":
                    samples, intermediates = ddpm_sample(
                        model=self._model,
                        schedule=self._schedule,
                        shape=shape,
                        device=self._device,
                    )
                else:
                    samples, intermediates = ddim_sample(
                        model=self._model,
                        schedule=self._schedule,
                        shape=shape,
                        num_steps=num_steps,
                        eta=eta,
                        device=self._device,
                    )

            pil_images = tensor_to_pil(samples)

            output_paths = []
            base64_images = []
            for img in pil_images:
                p = save_with_metadata(
                    img, settings.output_dir,
                    seed=seed, steps=num_steps, sampler=sampler,
                    extra={"eta": eta},
                )
                output_paths.append(f"/static/generated/{p.name}")
                if return_base64:
                    base64_images.append(pil_to_base64(img))

            grid_b64 = None
            if len(pil_images) > 1:
                grid_b64 = pil_to_base64(make_image_grid(pil_images))

            elapsed = round(time.time() - start, 3)
            logger.info("Generated %d images in %.2fs (seed=%d, sampler=%s, steps=%d)",
                        batch_size, elapsed, seed, sampler, num_steps)

            # Broadcast generation event
            await ws_manager.broadcast({
                "type": "generation_complete",
                "seed": seed,
                "sampler": sampler,
                "steps": num_steps,
                "batch_size": batch_size,
                "elapsed": elapsed,
                "images": base64_images[:4],  # limit broadcast size
            })

            return {
                "images": base64_images if return_base64 else output_paths,
                "grid": grid_b64,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "sampler": sampler,
                "steps": num_steps,
            }

    def _ensure_model(self, checkpoint: str | None = None):
        target = checkpoint or self._find_latest_checkpoint()
        if target is None:
            logger.warning("No checkpoint found")
            return

        if self._model is not None and self._loaded_checkpoint == target:
            return

        logger.info("Loading model from: %s", target)

        from model.unet import UNet
        from model.config import ModelConfig
        from backend.core.model_loader import load_checkpoint, extract_state_dict, extract_config

        ckpt_data = load_checkpoint(target)

        # Try to get config from checkpoint
        config_dict = extract_config(ckpt_data)
        if config_dict and isinstance(config_dict, ModelConfig):
            model_config = config_dict
        elif config_dict and isinstance(config_dict, dict):
            model_config = ModelConfig.from_dict(config_dict)
        else:
            model_config = ModelConfig()

        self._image_width = model_config.sample_width
        self._image_height = model_config.sample_height

        model = UNet(model_config)
        state_dict = extract_state_dict(ckpt_data, use_ema=True)

        # Handle EMA state dict format
        if isinstance(state_dict, dict) and "shadow" in state_dict:
            model.load_state_dict(state_dict["shadow"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self._device).eval()

        self._schedule = NoiseSchedule(
            num_timesteps=model_config.num_timesteps,
            schedule_type="cosine",
            device=self._device,
        )

        self._model = model
        self._loaded_checkpoint = target
        logger.info("Model loaded on %s (%dx%d)", self._device, self._image_width, self._image_height)

    def _find_latest_checkpoint(self) -> str | None:
        ckpt_dir = settings.checkpoint_dir
        pts = sorted(ckpt_dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pts[0]) if pts else None

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._schedule = None
            self._loaded_checkpoint = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")


inference_service = InferenceService()
