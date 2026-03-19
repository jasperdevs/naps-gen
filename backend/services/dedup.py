"""Deduplication service: perceptual hash based duplicate removal."""

import logging
from pathlib import Path

from backend.core.config import settings

logger = logging.getLogger("napsgen.dedup")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def run_dedup(threshold: int = 8, dry_run: bool = False) -> dict:
    """Run perceptual dedup on the dataset directory."""
    from dataset.quality_filter import find_duplicates

    image_paths = sorted([
        p for p in settings.dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ])

    if len(image_paths) < 2:
        return {
            "total_images": len(image_paths),
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "remaining": len(image_paths),
            "dry_run": dry_run,
        }

    duplicates = find_duplicates(image_paths, threshold=threshold)

    if not dry_run:
        for dup in duplicates:
            dup.unlink(missing_ok=True)

    logger.info("Dedup %s: %d dupes out of %d images",
                "dry-run" if dry_run else "removed", len(duplicates), len(image_paths))

    return {
        "total_images": len(image_paths),
        "duplicates_found": len(duplicates),
        "duplicates_removed": 0 if dry_run else len(duplicates),
        "remaining": len(image_paths) - (0 if dry_run else len(duplicates)),
        "dry_run": dry_run,
    }
