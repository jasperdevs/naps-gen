"""
NapsGen -- FastAPI application.

Serves the DDPM model for inference, manages training, dataset ops,
and streams live metrics over WebSocket.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.database import init_db, mark_running_runs_failed
from backend.core.ws_manager import ws_manager
from backend.routers import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: cleanup."""
    logger.info("Initializing database...")
    await init_db()
    await mark_running_runs_failed()
    logger.info("NapsGen backend ready on port 8000")
    yield
    logger.info("Shutdown complete")


app = FastAPI(title="NapsGen", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for generated images and samples
app.mount("/static/generated", StaticFiles(directory=str(settings.output_dir)), name="generated")
app.mount("/static/samples", StaticFiles(directory=str(settings.sample_dir)), name="samples")
app.mount("/static/dataset", StaticFiles(directory=str(settings.dataset_dir)), name="dataset")

# API routes
app.include_router(api_router, prefix="/api")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for live metric streaming."""
    await ws_manager.connect(websocket)
    logger.info("WebSocket client connected (%d total)", ws_manager.client_count)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send ping/subscribe messages if needed
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", ws_manager.client_count)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ws_clients": ws_manager.client_count,
    }
