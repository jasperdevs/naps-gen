from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NapsGen"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    checkpoint_dir: Path = base_dir / "checkpoints"
    output_dir: Path = base_dir / "outputs" / "generated"
    sample_dir: Path = base_dir / "outputs" / "samples"
    dataset_dir: Path = base_dir / "dataset"
    log_dir: Path = base_dir / "logs"
    db_path: Path = base_dir / "logs" / "training.db"

    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Training defaults
    default_lr: float = 1e-5
    default_batch_size: int = 1
    default_epochs: int = 100
    sample_every_n_steps: int = 50
    save_every_n_steps: int = 200

    # Inference defaults
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_width: int = 512
    default_height: int = 512

    # Model
    base_model_id: str = "stabilityai/stable-diffusion-2-1-base"

    class Config:
        env_prefix = "NAPSGEN_"


settings = Settings()

# Ensure directories exist
for d in [
    settings.checkpoint_dir,
    settings.output_dir,
    settings.sample_dir,
    settings.dataset_dir,
    settings.log_dir,
]:
    d.mkdir(parents=True, exist_ok=True)
