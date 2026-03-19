"""Dataset management endpoints."""

import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from backend.services.dataset import get_stats, get_images_page, save_upload, cleanup_square_images
from backend.services.dedup import run_dedup
from backend.core.ws_manager import ws_manager

router = APIRouter(prefix="/dataset")


@router.get("/stats")
async def dataset_stats():
    return get_stats()


@router.get("/images")
async def dataset_images(page: int = 1, page_size: int = 20):
    return get_images_page(page, page_size)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    data = await file.read()
    result = await save_upload(file.filename, data)

    # Broadcast dataset change to all connected clients
    await ws_manager.broadcast({
        "type": "dataset_update",
        "action": "upload",
        "filename": file.filename,
    })

    return result


class DedupRequest(BaseModel):
    threshold: int = Field(8, ge=1, le=20)
    dry_run: bool = False


@router.post("/deduplicate")
async def deduplicate_dataset(req: DedupRequest = DedupRequest()):
    """Remove perceptual duplicates from dataset."""
    result = await asyncio.to_thread(run_dedup, threshold=req.threshold, dry_run=req.dry_run)
    if not req.dry_run and result["duplicates_removed"] > 0:
        await ws_manager.broadcast({
            "type": "dataset_update",
            "action": "dedup_complete",
            "filename": "",
        })
    return result


@router.post("/cleanup")
async def cleanup_dataset():
    """Remove square/icon images from dataset (keeps only wide thumbnails)."""
    result = await asyncio.to_thread(cleanup_square_images)
    if result["removed"] > 0:
        await ws_manager.broadcast({
            "type": "dataset_update",
            "action": "cleanup_complete",
            "filename": "",
        })
    return result
