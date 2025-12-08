"""Lightweight batching utilities built on top of cached parquet chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd
import torch

from .pipeline import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, List[str]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest missing at {self.manifest_path}. "
                "Run CriteoDataProcessor.build_split_manifest() first."
            )
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
                numerical=self._to_tensor(numeric_np[start:end]),
                categorical=self._to_tensor(categorical_np[start:end], dtype=torch.long),
                label=self._to_tensor(labels_np[start:end]).unsqueeze(-1),
            )

    def _to_tensor(self, array, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=dtype, device=self.device)
        return tensor


__all__ = ["Batch", "CriteoBatchIterator"]

