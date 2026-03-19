"""Exponential Moving Average of model parameters for smoother generation."""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def _current_decay(self) -> float:
        if self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self._current_decay()
        self.step_count += 1
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into the model (for sampling)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.step_count = state["step_count"]
        self.decay = state.get("decay", self.decay)
