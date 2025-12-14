"""Utility helpers for demo/inference."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch

from src.data.module import Batch


def make_dummy_batch(batch_size: int, num_numeric: int, num_categorical: int, hash_bucket_size: int, device: torch.device) -> Batch:
    numerical = torch.randn(batch_size, num_numeric, device=device)
    categorical = torch.randint(0, hash_bucket_size, (batch_size, num_categorical), device=device)
    labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=device)
    return Batch(numerical=numerical, categorical=categorical, label=labels)


def save_predictions(path: Path, preds: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "probability"])
        for idx, p in enumerate(preds):
            writer.writerow([idx, p])


__all__ = ["make_dummy_batch", "save_predictions"]

