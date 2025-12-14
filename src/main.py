"""Entry point for DeepModel training (simple LR-style with embeddings)."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from src.utils import (
    CriteoBatchIterator,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    compute_auc,
    compute_logloss,
    Batch,
)
from src.model import DeepModel

# Ensure project root is on sys.path to import config.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as cfg_mod  # type: ignore


def iterator_factory(manifest: Path, split: str, batch_size: int, device: torch.device, limit_chunks: int | None, data_root: Path | None):
    def factory() -> CriteoBatchIterator:
        iterator = CriteoBatchIterator(
            manifest_path=manifest,
            split=split,
            batch_size=batch_size,
            device=device,
        )
        if data_root is not None:
            data_root_abs = data_root.expanduser().resolve()
            file_list = iterator._file_map.get(split, [])
            iterator._file_map[split] = [str(data_root_abs / Path(p).name) for p in file_list]
        if limit_chunks:
            iterator._file_map[split] = iterator._file_map[split][:limit_chunks]
        return iterator

    return factory


def parse_args() -> argparse.Namespace:
    cfg = cfg_mod.DeepModelConfig()
    parser = argparse.ArgumentParser(description="Train DeepModel on Criteo data.")
    parser.add_argument("--manifest", type=Path, default=Path(cfg.manifest))
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--embedding-dim", type=int, default=cfg.embedding_dim)
    parser.add_argument("--hash-buckets", type=int, default=cfg.hash_buckets)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path(cfg.data_root) if cfg.data_root else None, help="Optional directory containing parquet chunks (overrides paths in manifest)")
    parser.add_argument("--checkpoint", type=Path, default=Path(cfg.checkpoint))
    parser.add_argument("--metrics-file", type=Path, default=Path(cfg.metrics_file))
    parser.add_argument("--seed", type=int, default=cfg.seed, help="Random seed for reproducibility")
    parser.add_argument("--lr-step-size", type=int, default=cfg.lr_step_size, help="Step size (epochs) for LR decay")
    parser.add_argument("--lr-gamma", type=float, default=cfg.lr_gamma, help="LR decay factor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s", level=logging.INFO)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    model = DeepModel(
        num_numeric=len(NUMERIC_COLS),
        num_categorical=len(CATEGORICAL_COLS),
        hash_bucket_size=args.hash_buckets,
        embedding_dim=args.embedding_dim,
        use_embeddings=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    loss_fn = nn.BCELoss()

    train_iter_factory = iterator_factory(args.manifest, "train", args.batch_size, device, args.limit_chunks, args.data_root)
    test_iter_factory = iterator_factory(args.manifest, "test", args.batch_size, device, args.limit_chunks, args.data_root)

    best_test_auc = 0.0
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with args.metrics_file.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "train_loss", "test_loss", "test_auc", "test_logloss"])
        writer.writeheader()
        def train_epoch_fn():
            model.train()
            total_loss = 0.0
            steps = 0
            for batch in train_iter_factory():
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch)
                loss = loss_fn(preds, batch.label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            return total_loss / max(steps, 1)

        def evaluate_fn():
            model.eval()
            total_loss = 0.0
            steps = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_iter_factory():
                    preds = model(batch)
                    loss = loss_fn(preds, batch.label)
                    total_loss += loss.item()
                    steps += 1
                    all_preds.append(preds.detach().cpu())
                    all_labels.append(batch.label.detach().cpu())
            metrics = {}
            auc = compute_auc(all_labels, all_preds) if all_preds else 0.0
            logloss = compute_logloss(all_labels, all_preds) if all_preds else 0.0
            metrics["auc"] = auc
            metrics["logloss"] = logloss
            return total_loss / max(steps, 1), metrics

        for epoch in range(1, args.epochs + 1):
            logging.info("Epoch %d/%d", epoch, args.epochs)
            train_loss = train_epoch_fn()
            scheduler.step()

            # Evaluate on test each epoch (monitoring)
            test_loss, metrics = evaluate_fn()
            test_auc = metrics.get("auc", 0.0)
            test_logloss = metrics.get("logloss", 0.0)

            logging.info(
                "Epoch %d train_loss=%.4f lr=%.6f | test_loss=%.4f test_auc=%.4f test_logloss=%.4f",
                epoch,
                train_loss,
                scheduler.get_last_lr()[0],
                test_loss,
                test_auc,
                test_logloss,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "test_loss": f"{test_loss:.6f}",
                    "test_auc": f"{test_auc:.6f}",
                    "test_logloss": f"{test_logloss:.6f}",
                }
            )

            # Save checkpoint on best test AUC so far
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "test_auc": test_auc,
                        "test_logloss": test_logloss,
                    },
                    args.checkpoint,
                )
                logging.info("Saved test-best checkpoint to %s (test AUC=%.4f)", args.checkpoint, test_auc)


if __name__ == "__main__":
    main()
