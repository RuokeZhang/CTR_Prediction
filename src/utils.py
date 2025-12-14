"""Utility helpers: data iterator, metrics, demo helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

# Columns for processed Criteo data
NUMERIC_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
TARGET_COL = "label"


@dataclass
class Batch:
    numerical: torch.Tensor
    categorical: torch.Tensor
    label: torch.Tensor


class CriteoBatchIterator:
    """Streams torch tensors from cached parquet files split by manifest."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        batch_size: int = 2048,
        device: Optional[torch.device] = None,
        drop_last: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.drop_last = drop_last
        self._file_map = self._load_manifest()

    def __iter__(self) -> Iterator[Batch]:
        files = self._file_map.get(self.split)
        if not files:
            raise ValueError(f"No files mapped to split '{self.split}'.")

        for path_str in files:
            df = pd.read_parquet(path_str)
            if df.empty:
                continue
            for batch in self._yield_batches(df):
                yield batch

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Split manifest missing at {self.manifest_path}.")
        with self.manifest_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _yield_batches(self, df: pd.DataFrame) -> Iterator[Batch]:
        numeric_np = df[NUMERIC_COLS].to_numpy(dtype="float32", copy=False)
        categorical_np = df[CATEGORICAL_COLS].to_numpy(dtype="int64", copy=False)
        labels_np = df[TARGET_COL].to_numpy(dtype="float32", copy=False)

        total = len(df)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            if self.drop_last and (end - start) < self.batch_size:
                break
            yield Batch(
                numerical=torch.as_tensor(numeric_np[start:end], device=self.device),
                categorical=torch.as_tensor(categorical_np[start:end], dtype=torch.long, device=self.device),
                label=torch.as_tensor(labels_np[start:end], device=self.device).unsqueeze(-1),
            )


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


def compute_auc(labels, predictions) -> float:
    y_true = torch.cat([torch.as_tensor(t).view(-1) for t in labels]).cpu().numpy()
    y_pred = torch.cat([torch.as_tensor(p).view(-1) for p in predictions]).cpu().numpy()
    return float(roc_auc_score(y_true, y_pred))


def compute_logloss(labels, predictions) -> float:
    y_true = torch.cat([torch.as_tensor(t).view(-1) for t in labels]).cpu().numpy()
    y_pred = torch.cat([torch.as_tensor(p).view(-1) for p in predictions]).cpu().numpy()
    y_pred = y_pred.clip(1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_pred))


__all__ = [
    "Batch",
    "CriteoBatchIterator",
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "make_dummy_batch",
    "save_predictions",
    "compute_auc",
    "compute_logloss",
]

