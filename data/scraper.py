"""
Roblox image scraper -- collects real game thumbnails (promotional screenshots)
and game icons from actual popular Roblox games.

Uses:
- apis.roblox.com/explore-api -- discover trending/popular games
- apis.roblox.com/search-api -- search games by genre keywords
- thumbnails.roblox.com -- fetch promotional screenshots (768x432) and icons (512x512)

All endpoints are public, no auth required.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Callable

import aiohttp

logger = logging.getLogger("napsgen.scraper")

# Discovery APIs
EXPLORE_API = "https://apis.roblox.com/explore-api/v1/get-sorts?sessionId=0"
EXPLORE_API_PAGED = "https://apis.roblox.com/explore-api/v1/get-sorts?sessionId=0&sortsPageToken={token}"
SEARCH_API = "https://apis.roblox.com/search-api/omni-search?searchQuery={query}&pageToken={token}&sessionId=none"

# Thumbnail APIs
GAME_THUMBS_API = "https://thumbnails.roblox.com/v1/games/multiget/thumbnails?universeIds={ids}&countPerUniverse=10&defaults=true&size=768x432&format=Png&isCircular=false"
GAME_ICONS_API = "https://thumbnails.roblox.com/v1/games/icons?universeIds={ids}&returnPolicy=PlaceHolder&size=512x512&format=Png&isCircular=false"

# Genre keywords for diverse game discovery
SEARCH_GENRES = [
    "simulator", "tycoon", "obby", "roleplay", "shooter", "horror",
    "racing", "fighting", "adventure", "rpg", "anime", "survival",
    "tower defense", "cooking", "pet", "superhero", "zombie",
    "building", "sandbox", "escape", "mystery", "sport",
]


class RobloxScraper:
    """Async scraper for real Roblox game thumbnails."""

    def __init__(
        self,
        output_dir: str = "data/images",
        max_concurrent: int = 20,
        should_stop: Callable[[], bool] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self._should_stop = should_stop or (lambda: False)

        self._url_log_path = self.output_dir / ".seen_urls.txt"
        self._seen_urls: set[str] = self._load_seen_urls()
        self._seen_universe_ids: set[int] = set()

    def _load_seen_urls(self) -> set[str]:
        if self._url_log_path.exists():
            try:
                return set(self._url_log_path.read_text().splitlines())
            except Exception:
                return set()
        return set()

    def _persist_url(self, url: str):
        with open(self._url_log_path, "a") as f:
            f.write(url + "\n")

    async def scrape_game_thumbnails(self, count: int = 5000) -> int:
        """Main entry: discover games then fetch their promotional thumbnails."""
        logger.info("Scraping up to %d real game thumbnails...", count)
        sem = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Phase 1: Discover games from explore API (trending, popular, etc.)
            logger.info("Phase 1: Discovering games from explore API...")
            universe_ids = await self._discover_from_explore(session)
            logger.info("Found %d games from explore API", len(universe_ids))

            # Phase 2: Discover more games via search by genre
            if len(universe_ids) < count and not self._should_stop():
                logger.info("Phase 2: Searching by genre keywords...")
                search_ids = await self._discover_from_search(session, count)
                universe_ids.update(search_ids)
                logger.info("Total unique games: %d", len(universe_ids))

            # Phase 3: Fetch thumbnails in batches of 50
            logger.info("Phase 3: Fetching thumbnails for %d games...", len(universe_ids))
            id_list = list(universe_ids)

            for i in range(0, len(id_list), 50):
                if self._should_stop() or self.downloaded >= count:
                    break

                batch = id_list[i:i + 50]

                # Get promotional screenshots only (768x432, up to 10 per game)
                thumb_urls = await self._fetch_game_thumbnails(session, batch)

                remaining = max(0, count - self.downloaded)
                if remaining == 0:
                    break

                # Only schedule up to the remaining image budget for this pass.
                # This keeps the scraper from overshooting when one game returns
                # many promotional thumbnails at once.
                tasks = [self._download_image(session, sem, url, "game") for url in thumb_urls[:remaining]]
                await asyncio.gather(*tasks)

                logger.info("Progress: %d downloaded, %d failed, batch %d/%d",
                            self.downloaded, self.failed, i // 50 + 1, (len(id_list) + 49) // 50)
                await asyncio.sleep(0.2)

        return self.downloaded

    async def _discover_from_explore(self, session: aiohttp.ClientSession) -> set[int]:
        """Paginate through the explore API to get featured/trending game IDs."""
        ids: set[int] = set()
        next_token = ""

        for page in range(10):  # Up to 10 pages
            if self._should_stop():
                break

            url = EXPLORE_API if not next_token else EXPLORE_API_PAGED.format(token=next_token)
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning("Explore API returned %d", resp.status)
                        break
                    data = await resp.json()

                for sort in data.get("sorts", []):
                    for game in sort.get("games", []):
                        uid = game.get("universeId")
                        if uid:
                            ids.add(uid)

                next_token = data.get("nextSortsPageToken", "")
                if not next_token:
                    break

                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning("Explore API error: %s", e)
                break

        return ids

    async def _discover_from_search(self, session: aiohttp.ClientSession, target: int) -> set[int]:
        """Search by genre keywords to find diverse games."""
        ids: set[int] = set()

        for genre in SEARCH_GENRES:
            if self._should_stop() or len(ids) >= target:
                break

            page_token = ""
            for page in range(5):  # Up to 5 pages per genre
                if self._should_stop():
                    break

                url = SEARCH_API.format(query=genre, token=page_token)
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()

                    for group in data.get("searchResults", []):
                        for game in group.get("contents", []):
                            uid = game.get("universeId")
                            if uid:
                                ids.add(uid)

                    page_token = data.get("nextPageToken", "")
                    if not page_token:
                        break

                    await asyncio.sleep(0.15)
                except Exception as e:
                    logger.warning("Search API error for '%s': %s", genre, e)
                    break

            logger.info("Genre '%s': %d total unique games so far", genre, len(ids))

        return ids

    async def _fetch_game_thumbnails(self, session: aiohttp.ClientSession, universe_ids: list[int]) -> list[str]:
        """Fetch promotional screenshots (wide 768x432) for a batch of games."""
        ids_str = ",".join(str(i) for i in universe_ids)
        url = GAME_THUMBS_API.format(ids=ids_str)
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                urls = []
                for item in data.get("data", []):
                    for thumb in item.get("thumbnails", []):
                        img_url = thumb.get("imageUrl")
                        if img_url and thumb.get("state") == "Completed":
                            # Filter out auto-generated placeholders (t0-t7.rbxcdn.com)
                            # Keep only real developer-uploaded images (tr.rbxcdn.com)
                            if "tr.rbxcdn.com" in img_url:
                                urls.append(img_url)
                return urls
        except Exception as e:
            logger.warning("Thumbnail fetch failed: %s", e)
            return []

    async def _fetch_game_icons(self, session: aiohttp.ClientSession, universe_ids: list[int]) -> list[str]:
        """Fetch game icons (square 512x512) for a batch of games."""
        ids_str = ",".join(str(i) for i in universe_ids)
        url = GAME_ICONS_API.format(ids=ids_str)
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [
                    item["imageUrl"]
                    for item in data.get("data", [])
                    if item.get("imageUrl") and item.get("state") == "Completed"
                ]
        except Exception as e:
            logger.warning("Icon fetch failed: %s", e)
            return []

    @staticmethod
    def _is_valid_aspect_ratio(data: bytes, min_ratio: float = 1.3) -> bool:
        """Reject square-ish images (icons). Real thumbnails are 16:9 (~1.78 ratio)."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as img:
                w, h = img.size
                if h == 0:
                    return False
                ratio = w / h
                # Real thumbnails: 768x432 = 1.78 ratio
                # Icons: 512x512 = 1.0 ratio
                # Reject anything with ratio < 1.3 (catches 1:1 and near-square)
                return ratio >= min_ratio
        except Exception:
            return False

    async def _download_image(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str, prefix: str
    ) -> bool:
        if self._should_stop():
            return False
        if url in self._seen_urls:
            self.skipped += 1
            return False
        self._seen_urls.add(url)
        self._persist_url(url)

        async with sem:
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        self.failed += 1
                        return False
                    data = await resp.read()
                    if len(data) < 1000:
                        self.skipped += 1
                        return False

                    # Reject square icons by checking aspect ratio
                    if not self._is_valid_aspect_ratio(data):
                        self.skipped += 1
                        return False

                    h = hashlib.md5(data).hexdigest()
                    path = self.output_dir / f"{prefix}_{h}.png"
                    if path.exists():
                        self.skipped += 1
                        return False

                    path.write_bytes(data)
                    self.downloaded += 1
                    return True
            except Exception:
                self.failed += 1
                return False

    # Legacy method name for compatibility with scraper_manager
    async def scrape_asset_thumbnails(self, start_id: int = 1, count: int = 5000) -> int:
        """Redirects to scrape_game_thumbnails -- no more random asset ID scraping."""
        return await self.scrape_game_thumbnails(count)


async def main():
    parser = argparse.ArgumentParser(description="Scrape real Roblox game thumbnails")
    parser.add_argument("--output", default="data/images")
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    scraper = RobloxScraper(output_dir=args.output)
    await scraper.scrape_game_thumbnails(args.count)
    logger.info("Done! Downloaded: %d, Failed: %d, Skipped: %d",
                scraper.downloaded, scraper.failed, scraper.skipped)


if __name__ == "__main__":
    asyncio.run(main())
