"""Entry point for quick DeepFM demo inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import load_deepfm_from_checkpoint
from src.utils import make_dummy_batch, save_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick DeepFM demo inference.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/deepfm.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/demo_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hash-buckets", type=int, default=1 << 18)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_deepfm_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_numeric=13,
        num_categorical=26,
        hash_bucket_size=args.hash_buckets,
        device=device,
    )
    model.eval()

    batch = make_dummy_batch(
        batch_size=args.batch_size,
        num_numeric=13,
        num_categorical=26,
        hash_bucket_size=args.hash_buckets,
        device=device,
    )
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

    save_predictions(args.output, probs)
    print(f"Demo completed. Saved probabilities to {args.output}")


if __name__ == "__main__":
    main()

