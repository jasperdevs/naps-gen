"""Generation endpoint -- uses real DDPM/DDIM inference."""

import torch
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.inference import inference_service
from backend.core.config import settings

router = APIRouter(prefix="/generate")


class GenerateParams(BaseModel):
    seed: int | None = None
    num_steps: int = Field(50, ge=5, le=500)
    sampler: str = "ddim"
    eta: float = Field(0.0, ge=0.0, le=1.0)
    batch_size: int = Field(1, ge=1, le=16)
    checkpoint: str | None = None


@router.post("")
async def generate_image(params: GenerateParams):
    """Generate images using the trained DDPM model."""
    try:
        result = await inference_service.generate(
            seed=params.seed,
            num_steps=params.num_steps,
            sampler=params.sampler,
            eta=params.eta,
            batch_size=params.batch_size,
            checkpoint=params.checkpoint,
            return_base64=True,
        )
        return result
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(507, "CUDA out of memory -- reduce batch_size")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/history")
async def generation_history(page: int = 1, page_size: int = 50):
    """List previously generated images from disk."""
    output_dir = settings.output_dir
    if not output_dir.exists():
        return {"images": [], "total": 0}

    files = sorted(
        [f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    total = len(files)
    start = (page - 1) * page_size
    page_files = files[start:start + page_size]

    images = []
    for f in page_files:
        images.append({
            "path": f"/static/generated/{f.name}",
            "filename": f.name,
            "size_bytes": f.stat().st_size,
        })

    return {"images": images, "total": total, "page": page, "page_size": page_size}
