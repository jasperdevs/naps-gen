from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    steps: int = Field(30, ge=1, le=200)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: int | None = None
    batch_size: int = Field(1, ge=1, le=8)
    checkpoint: str | None = None


class GenerateResponse(BaseModel):
    images: list[str]
    seed: int
    elapsed_seconds: float


class TrainingConfig(BaseModel):
    """All fields optional -- backend auto-configures from dataset."""
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int | None = None
    resolution: int | None = None
    gradient_accumulation_steps: int | None = None
    use_8bit_adam: bool = False
    sample_every_n_steps: int | None = None
    save_every_n_steps: int | None = None
    prompt_for_samples: str = ""
    resume_from_checkpoint: str | None = None


class TrainingStatus(BaseModel):
    is_running: bool
    run_id: int | None = None
    run_status: str | None = None
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    total_epochs: int = 0
    current_loss: float | None = None
    learning_rate: float | None = None
    samples: list[str] = []
    elapsed_seconds: float = 0.0
    error: str | None = None


class TrainingMetric(BaseModel):
    step: int
    epoch: int
    loss: float
    lr: float
    timestamp: float


class TrainingHistory(BaseModel):
    run_id: int
    status: str
    config: dict
    metrics: list[TrainingMetric]


class DatasetStats(BaseModel):
    image_count: int
    total_size_mb: float
    resolutions: dict[str, int]
    formats: dict[str, int]


class DatasetImage(BaseModel):
    filename: str
    path: str
    size_bytes: int
    width: int
    height: int


class DatasetPage(BaseModel):
    images: list[DatasetImage]
    total: int
    page: int
    page_size: int


class CheckpointInfo(BaseModel):
    id: int
    run_id: int
    step: int
    path: str
    created_at: float
