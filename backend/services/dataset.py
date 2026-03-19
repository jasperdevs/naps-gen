"""
Dataset service: manages training images on disk.
"""

import os
import logging
from pathlib import Path

from PIL import Image

from backend.core.config import settings
from backend.models.schemas import DatasetStats, DatasetImage, DatasetPage

logger = logging.getLogger("napsgen.dataset")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_images() -> list[Path]:
    return sorted(
        p
        for p in settings.dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def get_stats() -> DatasetStats:
    images = _list_images()
    total_bytes = 0
    resolutions: dict[str, int] = {}
    formats: dict[str, int] = {}

    for p in images:
        total_bytes += p.stat().st_size
        ext = p.suffix.lower().lstrip(".")
        formats[ext] = formats.get(ext, 0) + 1
        try:
            with Image.open(p) as img:
                key = f"{img.width}x{img.height}"
                resolutions[key] = resolutions.get(key, 0) + 1
        except Exception:
            pass

    return DatasetStats(
        image_count=len(images),
        total_size_mb=round(total_bytes / (1024 * 1024), 2),
        resolutions=resolutions,
        formats=formats,
    )


def get_images_page(page: int = 1, page_size: int = 20) -> DatasetPage:
    all_images = _list_images()
    total = len(all_images)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_images[start:end]

    result: list[DatasetImage] = []
    for p in page_items:
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            w, h = 0, 0

        result.append(
            DatasetImage(
                filename=p.name,
                path=f"/static/dataset/{p.name}",
                size_bytes=p.stat().st_size,
                width=w,
                height=h,
            )
        )

    return DatasetPage(images=result, total=total, page=page, page_size=page_size)


def cleanup_square_images(min_ratio: float = 1.3) -> dict:
    """Remove square/icon images from dataset. Real thumbnails are 16:9 (~1.78 ratio)."""
    images = _list_images()
    removed = 0
    kept = 0

    for p in images:
        try:
            with Image.open(p) as img:
                w, h = img.size
                if h == 0 or (w / h) < min_ratio:
                    p.unlink()
                    removed += 1
                    logger.info("Removed square image: %s (%dx%d)", p.name, w, h)
                else:
                    kept += 1
        except Exception:
            # Can't read image, remove it
            p.unlink()
            removed += 1

    logger.info("Cleanup complete: removed %d square images, kept %d", removed, kept)
    return {"removed": removed, "remaining": kept, "total_scanned": removed + kept}


async def save_upload(filename: str, data: bytes) -> DatasetImage:
    dest = settings.dataset_dir / filename
    dest.write_bytes(data)

    try:
        with Image.open(dest) as img:
            w, h = img.size
    except Exception:
        w, h = 0, 0

    logger.info("Uploaded dataset image: %s (%dx%d)", filename, w, h)
    return DatasetImage(
        filename=filename,
        path=f"/static/dataset/{filename}",
        size_bytes=len(data),
        width=w,
        height=h,
    )
