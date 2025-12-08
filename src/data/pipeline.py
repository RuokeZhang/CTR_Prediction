"""Data ingestion and preprocessing utilities for the Criteo CTR dataset.

This module focuses on memory-efficient reading of the 5GB+ Criteo click logs by
chunking the CSV file, downcasting dtypes, and caching processed chunks to disk.
It also wires up lightweight feature preprocessing (numeric scaling +
categorical hashing) so downstream models can consume ready-to-train parquet
files without reprocessing the raw text file each time.
"""

from __future__ import annotations

import logging
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Column layout follows the published Criteo schema.
TARGET_COL = "label"
NUMERIC_COLS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
COLUMN_NAMES = [TARGET_COL] + NUMERIC_COLS + CATEGORICAL_COLS

try:
    _STRING_DTYPE = pd.StringDtype()
except AttributeError:  # pragma: no cover - older pandas fallback
    _STRING_DTYPE = "object"


def _build_dtype_map(columns: Sequence[str]) -> Dict[str, str]:
    """Returns a dtype mapping that downcasts numerics and enforces strings."""
    dtype_map: Dict[str, str] = {}
    for col in columns:
        if col == TARGET_COL:
            dtype_map[col] = "int8"
        elif col in NUMERIC_COLS:
            dtype_map[col] = "float32"
        else:
            dtype_map[col] = _STRING_DTYPE
    return dtype_map


@dataclass
class CriteoDataConfig:
    """Runtime configuration for ingesting and caching the dataset."""

    raw_csv_path: Path = Path("data/raw/criteo_train.txt")
    processed_dir: Path = Path("data/processed")
    chunk_size: int = 400_000
    hash_bucket_size: int = 1 << 18  # 262,144 buckets per categorical feature
    cache_prefix: str = "criteo_chunk"

    def __post_init__(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "metadata").mkdir(parents=True, exist_ok=True)

    @property
    def scaler_path(self) -> Path:
        return self.processed_dir / "metadata" / "numeric_scaler.joblib"

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "metadata" / "split_manifest.json"

    def parquet_path_for(self, chunk_idx: int) -> Path:
        return self.processed_dir / f"{self.cache_prefix}_{chunk_idx:04d}.parquet"


class CriteoDataProcessor:
    """Handles chunked ingestion, preprocessing, and caching of Criteo data."""

    def __init__(self, config: Optional[CriteoDataConfig] = None) -> None:
        self.config = config or CriteoDataConfig()
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_numeric_scaler(self, max_chunks: Optional[int] = None) -> None:
        """One pass over the data to estimate numeric scaling statistics."""
        scaler = StandardScaler(with_mean=True, with_std=True)
        for idx, chunk in enumerate(
            self._iterate_raw_chunks(columns=[TARGET_COL] + NUMERIC_COLS)
        ):
            numerics = self._prepare_numeric_features(chunk[NUMERIC_COLS])
            scaler.partial_fit(numerics)
            if max_chunks and idx + 1 >= max_chunks:
                break

        joblib.dump(scaler, self.config.scaler_path)
        self._scaler = scaler
        logger.info("Saved numeric scaler metadata to %s", self.config.scaler_path)

    def cache_processed_dataset(
        self,
        file_format: str = "parquet",
        limit_chunks: Optional[int] = None,
        overwrite: bool = False,
        ensure_scaler: bool = True,
    ) -> List[Path]:
        """Transforms the raw CSV into cached chunks on disk."""
        if ensure_scaler and not self._scaler_exists():
            logger.info("Numeric scaler missing; fitting before caching.")
            self.fit_numeric_scaler(max_chunks=limit_chunks)
        else:
            self._scaler = joblib.load(self.config.scaler_path)

        written_paths: List[Path] = []
        for idx, chunk in enumerate(self._iterate_raw_chunks()):
            #加载每一小块的dataframe
            processed = self._transform_chunk(chunk)
            out_path = self._write_chunk(processed, idx, file_format, overwrite)
            written_paths.append(out_path)
            logger.info("Cached chunk %d -> %s", idx, out_path)

            if limit_chunks and idx + 1 >= limit_chunks:
                break

        return written_paths

    def build_split_manifest(
        self,
        splits: Optional[Dict[str, float]] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> Path:
        """Assigns cached parquet files to train/val/test splits."""
        splits = splits or {"train": 0.8, "val": 0.1, "test": 0.1}
        if not np.isclose(sum(splits.values()), 1.0):
            raise ValueError("Split ratios must sum to 1.0")

        parquet_files = sorted(self.config.processed_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No cached parquet files found in {self.config.processed_dir}."
            )

        file_paths = [str(path) for path in parquet_files]
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(file_paths)

        counts = {
            split: max(1, int(ratio * len(file_paths))) for split, ratio in splits.items()
        }
        # Adjust counts to ensure total matches.
        assigned = sum(counts.values())
        while assigned > len(file_paths):
            largest = max(counts, key=counts.get)
            counts[largest] -= 1
            assigned -= 1
        while assigned < len(file_paths):
            smallest = min(counts, key=counts.get)
            counts[smallest] += 1
            assigned += 1

        manifest: Dict[str, List[str]] = {split: [] for split in splits}
        cursor = 0
        for split, count in counts.items():
            manifest[split] = file_paths[cursor : cursor + count]
            cursor += count

        with self.config.manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

        logger.info("Saved split manifest to %s", self.config.manifest_path)
        return self.config.manifest_path

    def load_split_manifest(self) -> Dict[str, List[str]]:
        if not self.config.manifest_path.exists():
            raise FileNotFoundError(
                "Split manifest not found. Run build_split_manifest() first."
            )
        with self.config.manifest_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def iter_cached_parquet(self) -> Iterator[pd.DataFrame]:
        """Streams cached parquet files back into memory."""
        for parquet_file in sorted(self.config.processed_dir.glob("*.parquet")):
            yield pd.read_parquet(parquet_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iterate_raw_chunks(
        self, columns: Optional[Sequence[str]] = None
    ) -> Iterator[pd.DataFrame]:
        usecols = list(columns) if columns else COLUMN_NAMES
        dtype_map = _build_dtype_map(usecols)

        if not self.config.raw_csv_path.exists():
            raise FileNotFoundError(
                f"Missing raw dataset at {self.config.raw_csv_path}. "
                "Download the Criteo training set and place it there."
            )

        reader = pd.read_csv(
            self.config.raw_csv_path,
            sep="\t",
            names=COLUMN_NAMES,
            header=None,
            usecols=usecols,
            dtype=dtype_map,
            chunksize=self.config.chunk_size,
            iterator=True,
            na_values=["", " ", "NA"],
            low_memory=True,
        )
        for chunk in reader:
            yield chunk

    def _prepare_numeric_features(self, df: pd.DataFrame) -> np.ndarray:
        filled = df.fillna(0.0).astype(np.float32)
        return filled.values

    def _hash_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df:
                continue
            series = df[col].fillna("UNK").astype("string")
            hashed = (
                pd.util.hash_pandas_object(series, index=False).astype(np.uint64)
                % self.config.hash_bucket_size
            ).astype(np.int32)
            df[col] = hashed
        return df

    def _transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk[NUMERIC_COLS] = self._apply_numeric_scaler(chunk[NUMERIC_COLS])
        chunk = self._hash_categorical_features(chunk)
        chunk[TARGET_COL] = chunk[TARGET_COL].fillna(0).astype(np.int8)
        return chunk

    def _apply_numeric_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        numerics = self._prepare_numeric_features(df)
        if not self._scaler:
            raise RuntimeError(
                "Numeric scaler not fitted. Call fit_numeric_scaler() first."
            )
        transformed = self._scaler.transform(numerics)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

    def _write_chunk(
        self,
        processed: pd.DataFrame,
        chunk_idx: int,
        file_format: str,
        overwrite: bool,
    ) -> Path:
        ext = file_format.lower()
        if ext not in {"parquet", "feather", "csv"}:
            raise ValueError(f"Unsupported cache format: {file_format}")

        if ext == "parquet":
            out_path = self.config.parquet_path_for(chunk_idx)
        else:
            suffix = {"feather": "feather", "csv": "csv"}[ext]
            out_path = self.config.processed_dir / f"{self.config.cache_prefix}_{chunk_idx:04d}.{suffix}"

        if out_path.exists() and not overwrite:
            logger.warning("Skipping existing cache file %s", out_path)
            return out_path

        if ext == "parquet":
            processed.to_parquet(out_path, index=False)
        elif ext == "feather":
            processed.to_feather(out_path)
        else:
            processed.to_csv(out_path, index=False)
        return out_path

    def _scaler_exists(self) -> bool:
        return self.config.scaler_path.exists()


__all__ = [
    "CriteoDataConfig",
    "CriteoDataProcessor",
    "COLUMN_NAMES",
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
]

