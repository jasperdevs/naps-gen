"""
FastAPI router that exposes the generation pipeline over HTTP.
"""

from __future__ import annotations

import logging

import torch
from fastapi import APIRouter, HTTPException

from backend.models.generation_params import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generation"])

# Injected at startup by the application factory (see main.py).
_service = None


def set_service(service) -> None:
    """Called once during app startup to inject the GenerationService."""
    global _service
    _service = service


@router.post("/", response_model=GenerationResult)
async def generate_images(req: GenerationRequest) -> GenerationResult:
    """
    Generate images using the loaded diffusion model.

    All heavy work runs synchronously on the GPU.  For production, wrap
    in ``run_in_executor`` or use a task queue, but for single-user / dev
    this is fine.
    """
    if _service is None:
        raise HTTPException(503, "Model not loaded yet -- server is still starting up")

    try:
        return _service.generate(req)
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(507, "CUDA out of memory -- reduce batch_size or image_size")
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(500, "Internal generation error")

