import aiosqlite
import json
import time
from pathlib import Path

from backend.core.config import settings

DB_PATH = str(settings.db_path)

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    status      TEXT    NOT NULL DEFAULT 'running',
    config      TEXT    NOT NULL,
    final_loss  REAL,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS training_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    epoch       INTEGER NOT NULL,
    loss        REAL    NOT NULL,
    lr          REAL    NOT NULL,
    timestamp   REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    path        TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES training_runs(id)
);
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    db = await get_db()
    try:
        await db.executescript(SCHEMA)
        await db.execute("ALTER TABLE training_runs ADD COLUMN error TEXT")
    except Exception:
        pass
    try:
        await db.commit()
    finally:
        await db.close()


async def create_run(config: dict) -> int:
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO training_runs (started_at, status, config) VALUES (?, ?, ?)",
            (time.time(), "running", json.dumps(config)),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def end_run(run_id: int, status: str, final_loss: float | None = None, error: str | None = None):
    db = await get_db()
    try:
        await db.execute(
            "UPDATE training_runs SET ended_at=?, status=?, final_loss=?, error=? WHERE id=?",
            (time.time(), status, final_loss, error, run_id),
        )
        await db.commit()
    finally:
        await db.close()


async def log_metric(run_id: int, step: int, epoch: int, loss: float, lr: float):
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO training_metrics (run_id, step, epoch, loss, lr, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, step, epoch, loss, lr, time.time()),
        )
        await db.commit()
    finally:
        await db.close()


async def save_checkpoint_record(run_id: int, step: int, path: str):
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO checkpoints (run_id, step, path, created_at) VALUES (?, ?, ?, ?)",
            (run_id, step, path, time.time()),
        )
        await db.commit()
    finally:
        await db.close()


async def get_run_metrics(run_id: int) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT step, epoch, loss, lr, timestamp FROM training_metrics "
            "WHERE run_id=? ORDER BY step",
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_all_runs() -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM training_runs ORDER BY id DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_latest_run() -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def mark_running_runs_failed():
    db = await get_db()
    try:
        await db.execute(
            "UPDATE training_runs SET ended_at=?, status='failed' WHERE status='running'",
            (time.time(),),
        )
        await db.commit()
    finally:
        await db.close()


async def list_checkpoint_records(run_id: int | None = None) -> list[dict]:
    db = await get_db()
    try:
        if run_id:
            cursor = await db.execute(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step", (run_id,)
            )
        else:
            cursor = await db.execute("SELECT * FROM checkpoints ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()
