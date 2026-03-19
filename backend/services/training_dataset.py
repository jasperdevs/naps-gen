"""Torch dataset utilities for DDPM training.

For small datasets (< ~5000 images), preloads everything into GPU memory
so training loops never touch disk. This eliminates the biggest bottleneck
on fast GPUs like RTX 30xx/40xx/50xx.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("napsgen.dataset")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Preload threshold: datasets under this size get loaded into GPU/CPU RAM
PRELOAD_THRESHOLD = 10_000


@dataclass
class DatasetConfig:
    data_dir: str
    width: int
    height: int
    batch_size: int
    random_flip: bool = True
    center_crop: bool = True
    num_workers: int = 0


def _list_images(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _crop_to_aspect(image: Image.Image, target_ratio: float, center_crop: bool) -> Image.Image:
    width, height = image.size
    current_ratio = width / max(height, 1)

    if abs(current_ratio - target_ratio) < 0.01:
        return image

    if current_ratio > target_ratio:
        new_width = max(1, int(height * target_ratio))
        left = max(0, (width - new_width) // 2) if center_crop else random.randint(0, max(0, width - new_width))
        return image.crop((left, 0, left + new_width, height))

    new_height = max(1, int(width / target_ratio))
    top = max(0, (height - new_height) // 2) if center_crop else random.randint(0, max(0, height - new_height))
    return image.crop((0, top, width, top + new_height))


class PreloadedDataset(Dataset[Tensor]):
    """All images preloaded as tensors in memory. No disk I/O during training."""

    def __init__(self, config: DatasetConfig, device: str = "cpu"):
        paths = _list_images(config.data_dir)
        target_ratio = config.width / config.height

        base_transform = transforms.Compose([
            transforms.Resize((config.height, config.width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        tensors: list[Tensor] = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    img = _crop_to_aspect(img, target_ratio, config.center_crop)
                    t = base_transform(img)
                    tensors.append(t)
                    # Also add flipped version for data augmentation
                    if config.random_flip:
                        tensors.append(torch.flip(t, [-1]))
            except Exception as e:
                logger.warning("Skipping %s: %s", path.name, e)

        if len(tensors) == 0:
            self.data = torch.empty(0)
        else:
            self.data = torch.stack(tensors).to(device)

        logger.info(
            "Preloaded %d tensors (%d images) to %s (%.1f MB)",
            len(tensors), len(paths), device,
            self.data.nelement() * self.data.element_size() / 1024 / 1024,
        )

    def __len__(self) -> int:
        return self.data.shape[0] if self.data.ndim == 4 else 0

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]


class DiskDataset(Dataset[Tensor]):
    """Standard disk-based dataset for large collections."""

    def __init__(self, config: DatasetConfig):
        self.image_paths = _list_images(config.data_dir)
        self.center_crop = config.center_crop
        self.target_width = config.width
        self.target_height = config.height
        self.transform = transforms.Compose([
            transforms.Resize((config.height, config.width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5) if config.random_flip else transforms.Lambda(lambda img: img),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = ImageOps.exif_transpose(image)
            image = _crop_to_aspect(image, self.target_width / self.target_height, self.center_crop)
            return self.transform(image)


def create_dataloader(
    config: DatasetConfig,
    split: str = "train",
    device: str = "cpu",
) -> DataLoader[Tensor]:
    n_images = len(_list_images(config.data_dir))

    if n_images <= PRELOAD_THRESHOLD:
        dataset = PreloadedDataset(config, device=device)
        # Preloaded data is already on device, no need for workers or pin_memory
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=False,
            drop_last=len(dataset) > config.batch_size,
        )

    dataset = DiskDataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=split == "train",
        num_workers=0,  # Windows multiprocessing is unreliable from threads
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) > config.batch_size,
    )
