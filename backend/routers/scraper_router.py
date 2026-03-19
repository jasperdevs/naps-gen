"""Scraper endpoints -- start/stop/status, all from the dashboard."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.scraper_manager import scraper_manager

router = APIRouter(prefix="/scraper")


class ScrapeRequest(BaseModel):
    count: int = Field(5000, ge=10, le=100000)
    scrape_type: str = Field("all", pattern="^(games|assets|all)$")


@router.get("/status")
async def scraper_status():
    return scraper_manager.get_status()


@router.post("/start")
async def start_scraper(req: ScrapeRequest):
    try:
        result = await scraper_manager.start(count=req.count, scrape_type=req.scrape_type)
        return result
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/stop")
async def stop_scraper():
    await scraper_manager.stop()
    return {"status": "stopping"}
