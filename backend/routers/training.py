"""Training endpoints -- start/stop/status + history via DB."""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import TrainingConfig, TrainingStatus
from backend.services.training_manager import training_manager
from pathlib import Path

from backend.core.config import settings
from backend.core.database import get_run_metrics, get_all_runs, get_latest_run, list_checkpoint_records

router = APIRouter(prefix="/training")


@router.get("/status", response_model=TrainingStatus)
async def training_status():
    status = training_manager.get_status()
    if status.run_id is not None or status.is_running:
        return status

    latest = await get_latest_run()
    if not latest:
        return status

    status.run_id = latest.get("id")
    status.run_status = latest.get("status")
    status.current_loss = latest.get("final_loss")
    status.error = latest.get("error")
    return status


@router.post("/start")
async def start_training(config: TrainingConfig):
    try:
        run_id = await training_manager.start(config)
        return {"run_id": run_id, "status": "started"}
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.post("/stop")
async def stop_training():
    await training_manager.stop()
    return {"status": "stopping"}


@router.get("/history")
async def training_history():
    runs = await get_all_runs()
    return {"runs": runs}


@router.get("/runs/{run_id}/metrics")
async def run_metrics(run_id: int):
    metrics = await get_run_metrics(run_id)
    return {"run_id": run_id, "metrics": metrics}


@router.get("/runs/{run_id}/samples")
async def run_samples(run_id: int):
    files = sorted(
        settings.sample_dir.glob(f"run_{run_id}_step_*_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    grouped: dict[int, list[str]] = {}
    for file in files:
        if file.name.endswith("_grid.png"):
            continue
        parts = file.stem.split("_")
        try:
            step = int(parts[3])
        except Exception:
            continue
        grouped.setdefault(step, []).append(f"/static/samples/{file.name}")

    items = [
        {"step": step, "images": sorted(images)}
        for step, images in sorted(grouped.items(), reverse=True)
    ]
    return {"run_id": run_id, "samples": items[:10]}


@router.get("/checkpoints")
async def checkpoints(run_id: int | None = None):
    records = await list_checkpoint_records(run_id)
    return {"checkpoints": records}
