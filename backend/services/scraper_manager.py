"""
Scraper manager: runs Roblox scraping in a background thread.

- Passes a should_stop callback into the scraper so it exits fast
- Broadcasts live progress over WebSocket
- Runs perceptual dedup after scraping finishes
"""

import asyncio
import threading
import time
import logging

from backend.core.config import settings
from backend.core.ws_manager import ws_manager

logger = logging.getLogger("napsgen.scraper")


class ScraperManager:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self.is_running: bool = False
        self.downloaded: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.target: int = 0
        self.started_at: float = 0.0
        self.scrape_type: str = "all"

    async def start(self, count: int = 5000, scrape_type: str = "all") -> dict:
        if self.is_running:
            raise RuntimeError("Scraper is already running")

        self._stop_event.clear()
        self._loop = asyncio.get_running_loop()
        self.is_running = True
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.target = count
        self.scrape_type = scrape_type
        self.started_at = time.time()

        self._thread = threading.Thread(
            target=self._scrape_loop,
            args=(count, scrape_type),
            daemon=True,
            name="scraper-thread",
        )
        self._thread.start()
        logger.info("Scraper started: %d images, type=%s", count, scrape_type)
        return {"status": "started", "target": count, "type": scrape_type}

    async def stop(self):
        if not self.is_running:
            return
        logger.info("Scraper stop requested")
        self._stop_event.set()

    def get_status(self) -> dict:
        elapsed = time.time() - self.started_at if self.is_running else 0.0
        return {
            "is_running": self.is_running,
            "downloaded": self.downloaded,
            "failed": self.failed,
            "skipped": self.skipped,
            "target": self.target,
            "type": self.scrape_type,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _scrape_loop(self, count: int, scrape_type: str):
        try:
            asyncio.run(self._async_scrape(count, scrape_type))
        except Exception as e:
            logger.exception("Scraper failed: %s", e)
            self._broadcast({"type": "scraper_error", "error": str(e)})
        finally:
            self.is_running = False
            self._broadcast({
                "type": "scraper_stopped",
                "downloaded": self.downloaded,
            })

    async def _async_scrape(self, count: int, scrape_type: str):
        from data.scraper import RobloxScraper

        output_dir = str(settings.dataset_dir)

        # Pass the stop event check as a callback so the scraper exits fast
        scraper = RobloxScraper(
            output_dir=output_dir,
            max_concurrent=20,
            should_stop=self._stop_event.is_set,
        )

        # Wrap download to track progress + broadcast
        original_download = scraper._download_image

        async def tracked_download(session, sem, url, prefix):
            if self._stop_event.is_set():
                return False
            result = await original_download(session, sem, url, prefix)
            self.downloaded = scraper.downloaded
            self.failed = scraper.failed
            self.skipped = scraper.skipped

            if scraper.downloaded % 5 == 0 or scraper.downloaded <= 3:
                self._broadcast({
                    "type": "scraper_progress",
                    "downloaded": scraper.downloaded,
                    "failed": scraper.failed,
                    "skipped": scraper.skipped,
                    "target": count,
                })
            return result

        scraper._download_image = tracked_download

        self._broadcast({
            "type": "scraper_started",
            "target": count,
            "scrape_type": scrape_type,
        })

        # Single unified scrape -- discovers games then fetches their thumbnails
        if not self._stop_event.is_set():
            await scraper.scrape_game_thumbnails(count)

        self.downloaded = scraper.downloaded
        self.failed = scraper.failed
        self.skipped = scraper.skipped

        was_stopped = self._stop_event.is_set()

        # Run dedup if we actually downloaded anything and weren't stopped
        dedup_removed = 0
        if self.downloaded > 0 and not was_stopped:
            self._broadcast({"type": "scraper_progress", "downloaded": self.downloaded,
                             "failed": self.failed, "skipped": self.skipped,
                             "target": count, "status": "deduplicating"})
            try:
                dedup_removed = await asyncio.to_thread(self._run_dedup, output_dir)
            except Exception as e:
                logger.warning("Post-scrape dedup failed: %s", e)

        self._broadcast({
            "type": "scraper_complete",
            "downloaded": self.downloaded,
            "failed": self.failed,
            "skipped": self.skipped,
            "dedup_removed": dedup_removed,
        })
        self._broadcast({"type": "dataset_update", "action": "scrape_complete", "filename": ""})

        logger.info("Scraper done: %d downloaded, %d failed, %d skipped, %d deduped",
                     self.downloaded, self.failed, self.skipped, dedup_removed)

    def _run_dedup(self, image_dir: str) -> int:
        from pathlib import Path
        from dataset.quality_filter import find_duplicates

        dir_path = Path(image_dir)
        image_paths = sorted([
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ])
        if len(image_paths) < 2:
            return 0

        duplicates = find_duplicates(image_paths, threshold=8)
        for dup in duplicates:
            dup.unlink(missing_ok=True)
        logger.info("Dedup removed %d images", len(duplicates))
        return len(duplicates)

    def _broadcast(self, data: dict):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(data), self._loop)


scraper_manager = ScraperManager()
