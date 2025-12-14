#!/usr/bin/env python3
"""Train the DeepFM model on cached Criteo data."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn
import numpy as np
import random

from src.data.module import CriteoBatchIterator
from src.data.pipeline import CATEGORICAL_COLS, NUMERIC_COLS
from src.eval.metrics import compute_auc, compute_logloss
from src.models.baselines import evaluate
from src.models.deepfm import DeepFM, DeepFMConfig, train_deepfm_epoch

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    level=logging.INFO,
)


def parse_layers(layers: str) -> Sequence[int]:
    return tuple(int(dim) for dim in layers.split(",") if dim)


def iterator_factory(
    manifest: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    limit_chunks: int | None,
    data_root: Path | None,
) -> Callable[[], CriteoBatchIterator]:
    def factory() -> CriteoBatchIterator:
        iterator = CriteoBatchIterator(
            manifest_path=manifest,
            split=split,
            batch_size=batch_size,
            device=device,
        )
        # If data_root provided, rewrite manifest file paths to that directory (use filename basename)
        if data_root is not None:
            data_root_abs = data_root.expanduser().resolve()
            file_list = iterator._file_map.get(split, [])
            iterator._file_map[split] = [str(data_root_abs / Path(p).name) for p in file_list]
        if limit_chunks:
            iterator._file_map[split] = iterator._file_map[split][:limit_chunks]
        return iterator

    return factory


def parse_dropout(dropout_str: str) -> Sequence[float]:
    """Parse dropout string like '0.5,0.5' into tuple of floats."""
    return tuple(float(d) for d in dropout_str.split(",") if d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepFM on Criteo data.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/metadata/split_manifest.json"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-size", type=int, default=4)
    parser.add_argument("--hidden-dims", type=str, default="32,32")
    parser.add_argument("--dropout", type=str, default="0.5,0.5")
    parser.add_argument("--hash-buckets", type=int, default=1 << 18)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None, help="Optional directory containing parquet chunks (overrides paths in manifest)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepfm.pt"))
    parser.add_argument("--metrics-file", type=Path, default=Path("results/deepfm_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr-step-size", type=int, default=1, help="Step size (epochs) for LR decay")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="LR decay factor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    config = DeepFMConfig.from_data_dims(
        num_numeric=len(NUMERIC_COLS),
        num_categorical=len(CATEGORICAL_COLS),
        hash_bucket_size=args.hash_buckets,
        embedding_size=args.embedding_size,
        hidden_dims=parse_layers(args.hidden_dims),
        dropout=parse_dropout(args.dropout),
        use_cuda=(device.type == "cuda"),
    )
    model = DeepFM(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    train_iter_factory = iterator_factory(args.manifest, "train", args.batch_size, device, args.limit_chunks, args.data_root)
    test_iter_factory = iterator_factory(args.manifest, "test", args.batch_size, device, args.limit_chunks, args.data_root)

    best_test_auc = 0.0
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_file.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "test_loss", "test_auc", "test_logloss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            logging.info("Epoch %d/%d", epoch, args.epochs)
            train_loss = train_deepfm_epoch(model, train_iter_factory(), optimizer, loss_fn, scaler)
            scheduler.step()

            logging.info("Epoch %d train_loss=%.4f lr=%.6f", epoch, train_loss, scheduler.get_last_lr()[0])

        # Final test evaluation
        test_stats = evaluate(
            model,
            test_iter_factory(),
            loss_fn,
            metric_fns={"auc": compute_auc, "logloss": compute_logloss},
        )
        test_auc = test_stats.metrics.get("auc", 0.0)
        test_logloss = test_stats.metrics.get("logloss", 0.0)
        logging.info(
            "Test set: loss=%.4f auc=%.4f logloss=%.4f",
            test_stats.loss,
            test_auc,
            test_logloss,
        )
        writer.writerow(
            {
                "epoch": "test",
                "train_loss": "",
                "test_loss": f"{test_stats.loss:.6f}",
                "test_auc": f"{test_auc:.6f}",
                "test_logloss": f"{test_logloss:.6f}",
            }
        )
        # Save checkpoint on best test AUC
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": args.epochs,
                    "test_auc": test_auc,
                    "test_logloss": test_logloss,
                },
                args.checkpoint,
            )
            logging.info("Saved test-best checkpoint to %s (test AUC=%.4f)", args.checkpoint, test_auc)


if __name__ == "__main__":
    main()

