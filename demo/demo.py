"""Simplified demo: load DeepFM checkpoint and run on dummy inputs."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model import load_deepfm_from_checkpoint
from src.utils import make_dummy_batch, save_predictions


def run_demo(
    checkpoint: Path = Path("checkpoints/deepfm.pt"),
    output: Path = Path("results/demo_predictions.csv"),
    batch_size: int = 8,
    hash_buckets: int = 1 << 18,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deepfm_from_checkpoint(
        checkpoint_path=checkpoint,
        num_numeric=13,
        num_categorical=26,
        hash_bucket_size=hash_buckets,
        device=device,
    )
    model.eval()
    batch = make_dummy_batch(batch_size, 13, 26, hash_buckets, device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).squeeze(-1).cpu().tolist()
    save_predictions(output, probs)
    print(f"Demo finished. Predictions saved to {output}")


if __name__ == "__main__":
    run_demo()

