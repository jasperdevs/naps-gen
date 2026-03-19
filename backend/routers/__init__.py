"""Unified API router -- mounts all sub-routers."""

from fastapi import APIRouter

from backend.routers.generate import router as generate_router
from backend.routers.training import router as training_router
from backend.routers.dataset_router import router as dataset_router
from backend.routers.scraper_router import router as scraper_router
from backend.routers.settings_router import router as settings_router

api_router = APIRouter()
api_router.include_router(generate_router, tags=["generation"])
api_router.include_router(training_router, tags=["training"])
api_router.include_router(dataset_router, tags=["dataset"])
api_router.include_router(scraper_router, tags=["scraper"])
api_router.include_router(settings_router, tags=["settings"])
