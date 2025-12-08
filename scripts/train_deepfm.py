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
    manifest: Path, split: str, batch_size: int, device: torch.device, limit_chunks: int | None
) -> Callable[[], CriteoBatchIterator]:
    def factory() -> CriteoBatchIterator:
        iterator = CriteoBatchIterator(
            manifest_path=manifest,
            split=split,
            batch_size=batch_size,
            device=device,
        )
        if limit_chunks:
            iterator._file_map[split] = iterator._file_map[split][:limit_chunks]
        return iterator

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepFM on Criteo data.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/metadata/split_manifest.json"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--deep-layers", type=str, default="256,128")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hash-buckets", type=int, default=1 << 18)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/deepfm.pt"))
    parser.add_argument("--metrics-file", type=Path, default=Path("reports/deepfm_metrics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = DeepFMConfig(
        num_numeric=len(NUMERIC_COLS),
        num_categorical=len(CATEGORICAL_COLS),
        hash_bucket_size=args.hash_buckets,
        embedding_dim=args.embedding_dim,
        deep_layers=parse_layers(args.deep_layers),
        dropout=args.dropout,
    )
    model = DeepFM(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    train_iter_factory = iterator_factory(args.manifest, "train", args.batch_size, device, args.limit_chunks)
    val_iter_factory = iterator_factory(args.manifest, "val", args.batch_size, device, args.limit_chunks)

    best_auc = 0.0
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_file.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "val_loss", "val_auc", "val_logloss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            logging.info("Epoch %d/%d", epoch, args.epochs)
            train_loss = train_deepfm_epoch(model, train_iter_factory(), optimizer, loss_fn, scaler)
            val_stats = evaluate(
                model,
                val_iter_factory(),
                loss_fn,
                metric_fns={"auc": compute_auc, "logloss": compute_logloss},
            )
            val_auc = val_stats.metrics.get("auc", 0.0)
            val_logloss = val_stats.metrics.get("logloss", 0.0)

            logging.info(
                "Epoch %d train_loss=%.4f val_loss=%.4f val_auc=%.4f val_logloss=%.4f",
                epoch,
                train_loss,
                val_stats.loss,
                val_auc,
                val_logloss,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_stats.loss:.6f}",
                    "val_auc": f"{val_auc:.6f}",
                    "val_logloss": f"{val_logloss:.6f}",
                }
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config.__dict__,
                        "epoch": epoch,
                        "val_auc": val_auc,
                    },
                    args.checkpoint,
                )
                logging.info("Saved new best checkpoint to %s (AUC=%.4f)", args.checkpoint, val_auc)


if __name__ == "__main__":
    main()

