"""Settings endpoints -- archive and delete data, training runs, images."""

import asyncio
import shutil
import time
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.ws_manager import ws_manager

logger = logging.getLogger("napsgen.settings")
router = APIRouter(prefix="/settings")

ARCHIVE_DIR = settings.base_dir / "archives"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ensure_archive_dir() -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR


def _count_images(d: Path) -> int:
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)


# ── Overview ──

@router.get("/storage")
async def storage_overview():
    """Get disk usage of all data directories."""
    def _dir_size(d: Path) -> float:
        return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024), 2)

    return {
        "dataset": {"path": str(settings.dataset_dir), "size_mb": _dir_size(settings.dataset_dir), "images": _count_images(settings.dataset_dir)},
        "checkpoints": {"path": str(settings.checkpoint_dir), "size_mb": _dir_size(settings.checkpoint_dir)},
        "outputs": {"path": str(settings.output_dir), "size_mb": _dir_size(settings.output_dir)},
        "samples": {"path": str(settings.sample_dir), "size_mb": _dir_size(settings.sample_dir)},
        "archives": {"path": str(ARCHIVE_DIR), "size_mb": _dir_size(ARCHIVE_DIR) if ARCHIVE_DIR.exists() else 0},
    }


# ── Archive ──

class ArchiveRequest(BaseModel):
    target: str  # "dataset" | "checkpoints" | "outputs" | "samples" | "all"


@router.post("/archive")
async def archive_data(req: ArchiveRequest):
    """Move data to timestamped archive folder instead of deleting."""
    archive_base = _ensure_archive_dir()
    tag = time.strftime("%Y%m%d_%H%M%S")
    results = {}

    targets = _resolve_targets(req.target)
    for name, src_dir in targets.items():
        if not src_dir.exists() or not any(src_dir.iterdir()):
            results[name] = "empty"
            continue
        dest = archive_base / f"{name}_{tag}"
        await asyncio.to_thread(shutil.copytree, str(src_dir), str(dest))
        # Clear the source
        for f in src_dir.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        results[name] = str(dest)
        logger.info("Archived %s -> %s", name, dest)

    await ws_manager.broadcast({"type": "dataset_update", "action": "archived", "filename": ""})
    return {"archived": results, "timestamp": tag}


# ── Delete ──

class DeleteRequest(BaseModel):
    target: str  # "dataset" | "checkpoints" | "outputs" | "samples" | "training_history" | "all"
    confirm: bool = False


@router.post("/delete")
async def delete_data(req: DeleteRequest):
    """Permanently delete data. Requires confirm=true."""
    if not req.confirm:
        raise HTTPException(400, "Set confirm=true to delete permanently.")

    results = {}

    if req.target in ("training_history", "all"):
        db = await get_db()
        try:
            await db.execute("DELETE FROM training_metrics")
            await db.execute("DELETE FROM checkpoints")
            await db.execute("DELETE FROM training_runs")
            await db.commit()
            results["training_history"] = "deleted"
        finally:
            await db.close()

    targets = _resolve_targets(req.target)
    for name, src_dir in targets.items():
        if not src_dir.exists():
            results[name] = "not found"
            continue
        count = 0
        for f in src_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
            elif f.is_dir():
                shutil.rmtree(f)
                count += 1
        results[name] = f"deleted {count} items"
        logger.info("Deleted %d items from %s", count, name)

    # Also clear the URL log if deleting dataset
    if req.target in ("dataset", "all"):
        url_log = settings.dataset_dir / ".seen_urls.txt"
        if url_log.exists():
            url_log.unlink()
            results["url_log"] = "deleted"

    await ws_manager.broadcast({"type": "dataset_update", "action": "deleted", "filename": ""})
    return {"deleted": results}


# ── Delete Archives ──

@router.post("/delete-archives")
async def delete_archives():
    """Permanently delete all archived data."""
    if not ARCHIVE_DIR.exists():
        return {"deleted": 0, "freed_mb": 0}

    size = sum(f.stat().st_size for f in ARCHIVE_DIR.rglob("*") if f.is_file())
    freed_mb = round(size / (1024 * 1024), 2)
    count = 0
    for item in ARCHIVE_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
            count += 1
        elif item.is_file():
            item.unlink()
            count += 1

    logger.info("Deleted %d archive items, freed %.2f MB", count, freed_mb)
    return {"deleted": count, "freed_mb": freed_mb}


# ── Helpers ──

def _resolve_targets(target: str) -> dict[str, Path]:
    mapping = {
        "dataset": settings.dataset_dir,
        "checkpoints": settings.checkpoint_dir,
        "outputs": settings.output_dir,
        "samples": settings.sample_dir,
    }
    if target == "all":
        return mapping
    if target == "training_history":
        return {"checkpoints": settings.checkpoint_dir, "samples": settings.sample_dir}
    if target in mapping:
        return {target: mapping[target]}
    return {}
