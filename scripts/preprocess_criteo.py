#!/usr/bin/env python3
"""CLI entry point for ingesting and caching the Criteo dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.data import CriteoDataConfig, CriteoDataProcessor

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream, preprocess, and cache the Criteo CTR dataset."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/train.txt"),
        help="Path to the raw Criteo TSV file.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed chunks will be stored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400_000,
        help="Number of rows to process per chunk (smaller -> lower RAM).",
    )
    parser.add_argument(
        "--hash-buckets",
        type=int,
        default=1 << 18,
        help="Number of buckets for categorical hashing.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional cap on number of chunks (useful for smoke tests).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "feather", "csv"],
        default="parquet",
        help="On-disk cache format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached files when they already exist.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Only fit and persist numeric scaler statistics.",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="After caching, assign cached files to train/val/test splits.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    cli_args = args or parse_args()

    config = CriteoDataConfig(
        raw_csv_path=cli_args.raw_path,
        processed_dir=cli_args.processed_dir,
        chunk_size=cli_args.chunk_size,
        hash_bucket_size=cli_args.hash_buckets,
    )
    processor = CriteoDataProcessor(config)

    if cli_args.fit_only:
        processor.fit_numeric_scaler(max_chunks=cli_args.limit_chunks)
    else:
        processor.cache_processed_dataset(
            file_format=cli_args.format,
            limit_chunks=cli_args.limit_chunks,
            overwrite=cli_args.overwrite,
            ensure_scaler=True,
        )
        if cli_args.build_manifest:
            processor.build_split_manifest()


if __name__ == "__main__":
    main()

