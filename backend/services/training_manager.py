"""
Training manager: runs DDPM training in a background thread with real forward/backward.

Auto-configures all hyperparameters from the dataset when the user just hits Start.
Broadcasts every metric over WebSocket in real-time.
"""

import asyncio
import math
import threading
import time
import logging
from pathlib import Path

import torch
from PIL import Image

from backend.core.config import settings
from backend.core.database import (
    create_run,
    end_run,
    log_metric,
    save_checkpoint_record,
    get_latest_run,
)
from backend.core.ws_manager import ws_manager
from backend.core.image_utils import tensor_to_pil, make_image_grid, pil_to_base64
from backend.models.schemas import TrainingConfig, TrainingStatus

logger = logging.getLogger("napsgen.training")


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _get_gpu_vram_gb() -> float:
    """Return available GPU VRAM in GB, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def _auto_configure(config: TrainingConfig, dataset_dir: Path) -> TrainingConfig:
    """Auto-configure everything from dataset size + GPU capabilities."""
    from PIL import Image as PILImage

    image_files = [f for f in dataset_dir.rglob("*") if f.suffix.lower() in _IMG_EXTS]
    n = len(image_files)
    if n == 0:
        raise RuntimeError(f"No images found in {dataset_dir}. Scrape or upload images first.")

    vram = _get_gpu_vram_gb()
    has_gpu = torch.cuda.is_available()

    # ── Resolution ──
    # Small datasets: 64, medium: 128, large+high-res source: 256
    if config.resolution is not None:
        res = config.resolution
    elif n < 500:
        res = 64
    elif n < 2000:
        res = 128 if vram >= 6 else 64
    else:
        res = 256 if vram >= 10 else (128 if vram >= 6 else 64)

    # ── Batch size ──
    # Maximize GPU utilization. Preloaded dataset removes I/O bottleneck,
    # so bigger batches = better GPU throughput.
    if config.batch_size is not None:
        bs = config.batch_size
    elif res <= 64:
        bs = min(64, max(8, n)) if vram >= 8 else min(32, max(4, n))
    elif res <= 128:
        bs = min(32, max(4, n)) if vram >= 8 else min(16, max(2, n))
    else:
        bs = min(8, max(1, n // 50)) if vram >= 10 else min(4, max(1, n // 100))

    # ── Gradient accumulation ──
    # With big batch sizes and preloaded data, accumulation just slows things down
    ga = config.gradient_accumulation_steps if config.gradient_accumulation_steps is not None else 1

    # ── Epochs ──
    # More images = fewer epochs needed
    if config.epochs is not None:
        ep = config.epochs
    elif n < 100:
        ep = 80
    elif n < 500:
        ep = 60
    elif n < 2000:
        ep = 40
    elif n < 5000:
        ep = 30
    else:
        ep = 20

    # ── Learning rate ──
    # Scale with effective batch size
    effective_bs = bs * ga
    lr = config.learning_rate if config.learning_rate is not None else max(1e-5, min(5e-4, 1e-4 * math.sqrt(effective_bs)))

    # ── Sample/save intervals ──
    spe = max(1, n // (bs * ga))
    total = spe * ep
    samp = config.sample_every_n_steps if config.sample_every_n_steps is not None else max(10, total // 8)
    save = config.save_every_n_steps if config.save_every_n_steps is not None else max(25, total // 4)

    resolved = TrainingConfig(
        learning_rate=lr, batch_size=bs, epochs=ep, resolution=res,
        gradient_accumulation_steps=ga, sample_every_n_steps=samp,
        save_every_n_steps=save, resume_from_checkpoint=config.resume_from_checkpoint,
    )
    logger.info(
        "Auto-config: %d imgs, res=%d, bs=%d, ga=%d, ep=%d, lr=%.2e, vram=%.1fGB",
        n, res, bs, ga, ep, lr, vram,
    )
    return resolved


class TrainingManager:
    """Singleton manager that owns the training thread."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self.is_running: bool = False
        self.run_id: int | None = None
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.total_epochs: int = 0
        self.current_loss: float | None = None
        self.learning_rate: float | None = None
        self.latest_samples: list[str] = []
        self.started_at: float = 0.0
        self._config: TrainingConfig | None = None
        self.run_status: str | None = None
        self.error: str | None = None

    async def start(self, config: TrainingConfig) -> int:
        if self.is_running:
            raise RuntimeError("Training is already running")

        config = _auto_configure(config, settings.dataset_dir)

        self._stop_event.clear()
        self._config = config
        self._loop = asyncio.get_running_loop()

        run_id = await create_run(config.model_dump())
        self.run_id = run_id
        self.is_running = True
        self.started_at = time.time()
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = None
        self.latest_samples = []
        self.run_status = "running"
        self.error = None

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(run_id, config),
            daemon=True,
            name="training-thread",
        )
        self._thread.start()
        logger.info("Training started (run_id=%d)", run_id)
        return run_id

    async def stop(self):
        if not self.is_running:
            return
        logger.info("Graceful stop requested for run_id=%d", self.run_id)
        self._stop_event.set()

    def get_status(self) -> TrainingStatus:
        elapsed = time.time() - self.started_at if self.is_running else 0.0
        return TrainingStatus(
            is_running=self.is_running,
            run_id=self.run_id,
            run_status=self.run_status,
            current_epoch=self.current_epoch,
            current_step=self.current_step,
            total_steps=self.total_steps,
            total_epochs=self.total_epochs,
            current_loss=self.current_loss,
            learning_rate=self.learning_rate,
            samples=self.latest_samples,
            elapsed_seconds=round(elapsed, 2),
            error=self.error,
        )

    def _train_loop(self, run_id: int, config: TrainingConfig):
        try:
            self._run_training(run_id, config)
        except Exception as e:
            logger.exception("Training failed: %s", e)
            self.run_status = "failed"
            self.error = str(e)
            self._fire_and_forget(end_run(run_id, "failed", self.current_loss, str(e)))
            self._broadcast({
                "type": "training_error",
                "run_id": run_id,
                "error": str(e),
            })
        finally:
            self.is_running = False
            self._broadcast({"type": "training_stopped", "run_id": run_id})

    def _run_training(self, run_id: int, config: TrainingConfig):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Training on device: %s", device)

        # ── GPU optimizations ──
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled: cudnn.benchmark, tf32 matmul, tf32 cudnn")

        # ── 1. Build model ──
        from model.unet import UNet
        from model.config import ModelConfig
        from model.trainer import DDPMTrainer

        # ── 2. Build dataset + dataloader (preloaded into GPU for small datasets) ──
        from backend.services.training_dataset import DatasetConfig, create_dataloader

        # Width = resolution, height = 9/16 ratio, rounded to nearest multiple of 16
        # so the U-Net downsampling never hits odd dimension mismatches
        target_w = config.resolution
        target_h = max(16, round((config.resolution * 9) / 16 / 16) * 16)

        ds_config = DatasetConfig(
            data_dir=str(settings.dataset_dir),
            width=target_w,
            height=target_h,
            batch_size=config.batch_size,
            random_flip=True,
            center_crop=True,
        )
        dataloader = create_dataloader(ds_config, split="train", device=device)
        if len(dataloader.dataset) == 0:
            raise RuntimeError("No images found in dataset directory")

        n_images = len(dataloader.dataset)

        if n_images < 200:
            model_config = ModelConfig(
                image_size=config.resolution,
                image_width=target_w,
                image_height=target_h,
                base_channels=32,
                channel_mult=(1, 2, 2),
                num_res_blocks=1,
                attention_resolutions=(),
                num_heads=2,
            )
        elif n_images < 1000:
            model_config = ModelConfig(
                image_size=config.resolution,
                image_width=target_w,
                image_height=target_h,
                base_channels=64,
                channel_mult=(1, 2, 2, 2),
                num_res_blocks=2,
                attention_resolutions=(16,),
                num_heads=4,
            )
        else:
            model_config = ModelConfig(
                image_size=config.resolution,
                image_width=target_w,
                image_height=target_h,
                base_channels=128 if config.resolution >= 128 else 64,
                channel_mult=(1, 2, 2, 2) if config.resolution <= 128 else (1, 1, 2, 2, 4, 4),
                attention_resolutions=(16,) if config.resolution <= 128 else (16, 32),
                num_heads=4 if config.resolution <= 128 else 8,
            )

        model = UNet(model_config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Model: %d parameters (%.1fM)", param_count, param_count / 1e6)

        steps_per_epoch = max(1, len(dataloader) // config.gradient_accumulation_steps)
        self.total_epochs = config.epochs
        self.total_steps = steps_per_epoch * config.epochs
        self.learning_rate = config.learning_rate

        # ── 3. Build trainer ──
        trainer = DDPMTrainer(
            model=model,
            dataloader=dataloader,
            lr=config.learning_rate,
            grad_accum_steps=config.gradient_accumulation_steps,
            use_amp=device == "cuda",
            device=device,
            checkpoint_dir=str(settings.checkpoint_dir),
            sample_dir=str(settings.sample_dir),
        )

        self._broadcast({
            "type": "training_started",
            "run_id": run_id,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "param_count": param_count,
            "n_images": n_images,
        })

        logger.info("Starting: %d images, %d epochs, %d steps", n_images, config.epochs, self.total_steps)

        # ── 4. Training loop ──
        global_step = 0
        last_loss = 0.0

        for epoch in range(config.epochs):
            self.current_epoch = epoch + 1
            trainer.epoch = epoch + 1
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(dataloader):
                if self._stop_event.is_set():
                    logger.info("Stop flag at step %d", global_step)
                    ckpt = trainer.save_checkpoint(run_id)
                    self._fire_and_forget(save_checkpoint_record(run_id, global_step, ckpt))
                    self.run_status = "stopped"
                    self._fire_and_forget(end_run(run_id, "stopped", last_loss))
                    return

                # Handle dataset returning (images, labels) or just images
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                loss = trainer.train_step(batch)
                epoch_loss += loss
                batch_count += 1

                # Accumulate gradients
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    grad_norm = trainer.optimizer_step()
                    global_step += 1
                    self.current_step = global_step
                    last_loss = loss
                    self.current_loss = loss

                    # Log to DB
                    self._fire_and_forget(
                        log_metric(run_id, global_step, epoch + 1, loss, config.learning_rate)
                    )

                    # Broadcast to all WS clients
                    self._broadcast({
                        "type": "metric",
                        "run_id": run_id,
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(loss, 6),
                        "lr": config.learning_rate,
                        "grad_norm": round(grad_norm, 4),
                        "total_steps": self.total_steps,
                        "images_per_sec": round(config.batch_size * config.gradient_accumulation_steps / max(0.001, time.time() - self.started_at) * global_step, 1),
                    })

                    # Save checkpoint
                    if global_step % config.save_every_n_steps == 0:
                        ckpt = trainer.save_checkpoint(run_id)
                        self._fire_and_forget(save_checkpoint_record(run_id, global_step, ckpt))
                        self._broadcast({
                            "type": "checkpoint",
                            "run_id": run_id,
                            "step": global_step,
                            "path": ckpt,
                        })

                    # Generate samples
                    if global_step % config.sample_every_n_steps == 0:
                        self._generate_and_broadcast(trainer, run_id, global_step, config)

        # ── Training complete ──
        ckpt = trainer.save_checkpoint(run_id)
        self._fire_and_forget(save_checkpoint_record(run_id, global_step, ckpt))
        self._generate_and_broadcast(trainer, run_id, global_step, config)
        self.run_status = "completed"
        self._fire_and_forget(end_run(run_id, "completed", last_loss))
        logger.info("Training completed at step %d, loss=%.6f", global_step, last_loss)

    def _generate_and_broadcast(self, trainer, run_id: int, step: int, config: TrainingConfig):
        """Generate sample images and broadcast them over WS."""
        try:
            samples = trainer.generate_samples(n=4, steps=50)
            pil_images = tensor_to_pil(samples)

            paths = []
            b64_images = []
            for i, img in enumerate(pil_images):
                p = settings.sample_dir / f"run_{run_id}_step_{step}_{i}.png"
                img.save(str(p))
                paths.append(str(p))
                b64_images.append(pil_to_base64(img))

            # Grid
            grid = make_image_grid(pil_images)
            grid_path = settings.sample_dir / f"run_{run_id}_step_{step}_grid.png"
            grid.save(str(grid_path))

            self.latest_samples = paths

            self._broadcast({
                "type": "sample",
                "run_id": run_id,
                "step": step,
                "images": b64_images,
                "grid": pil_to_base64(grid),
                "paths": paths,
            })
        except Exception as e:
            logger.warning("Sample generation failed at step %d: %s", step, e)

    def _broadcast(self, data: dict):
        self._fire_and_forget(ws_manager.broadcast(data))

    def _fire_and_forget(self, coro):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)


training_manager = TrainingManager()
