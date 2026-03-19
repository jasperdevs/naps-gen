/** Shared types matching the FastAPI backend schemas. */

export interface TrainingStatus {
  is_running: boolean;
  run_id: number | null;
  run_status?: string | null;
  current_epoch: number;
  current_step: number;
  total_steps: number;
  total_epochs: number;
  current_loss: number | null;
  learning_rate: number | null;
  samples: string[];
  elapsed_seconds: number;
  error?: string | null;
}

export interface TrainingConfig {
  learning_rate: number;
  batch_size: number;
  epochs: number;
  resolution: number;
  gradient_accumulation_steps: number;
  use_8bit_adam: boolean;
  sample_every_n_steps: number;
  save_every_n_steps: number;
}

export interface TrainingMetric {
  type?: "metric";
  run_id?: number;
  step: number;
  epoch: number;
  loss: number;
  lr: number;
  grad_norm?: number;
  total_steps?: number;
  timestamp?: number;
}

export interface TrainingSampleHistoryItem {
  step: number;
  images: string[];
}

export interface DatasetStats {
  image_count: number;
  total_size_mb: number;
  resolutions: Record<string, number>;
  formats: Record<string, number>;
}

export interface DatasetImage {
  filename: string;
  path: string;
  size_bytes: number;
  width: number;
  height: number;
}

export interface DatasetPage {
  images: DatasetImage[];
  total: number;
  page: number;
  page_size: number;
}

export interface GenerationResult {
  images: string[];
  grid?: string | null;
  seed: number;
  elapsed_seconds: number;
  sampler: string;
  steps: number;
}

/** WebSocket message types from the backend. */
export type WSMessage =
  | { type: "metric"; run_id: number; step: number; epoch: number; loss: number; lr: number; grad_norm?: number; total_steps: number; images_per_sec?: number }
  | { type: "sample"; run_id: number; step: number; images: string[]; grid?: string; paths?: string[] }
  | { type: "checkpoint"; run_id: number; step: number; path: string }
  | { type: "training_started"; run_id: number; total_steps: number; total_epochs: number; param_count: number; n_images: number }
  | { type: "training_stopped"; run_id: number }
  | { type: "training_error"; run_id: number; error: string }
  | { type: "generation_complete"; seed: number; sampler: string; steps: number; batch_size: number; elapsed: number; images: string[] }
  | { type: "dataset_update"; action: string; filename: string }
  | { type: "scraper_started"; target: number; scrape_type: string }
  | { type: "scraper_progress"; downloaded: number; failed: number; skipped: number; target: number }
  | { type: "scraper_complete"; downloaded: number; failed: number; skipped: number; dedup_removed?: number }
  | { type: "scraper_stopped"; downloaded: number }
  | { type: "scraper_error"; error: string };

export interface ScraperStatus {
  is_running: boolean;
  downloaded: number;
  failed: number;
  skipped: number;
  target: number;
  type: string;
  elapsed_seconds: number;
}
