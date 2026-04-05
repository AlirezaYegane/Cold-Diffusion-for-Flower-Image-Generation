from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class EMA:
    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay

    @torch.no_grad()
    def update(self, ema_model: torch.nn.Module, model: torch.nn.Module) -> None:
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_metrics_csv(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)