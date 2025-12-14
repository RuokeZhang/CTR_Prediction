"""Model entrypoints for DeepFM demo/training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.models.deepfm import DeepFM, DeepFMConfig


def load_deepfm_from_checkpoint(
    checkpoint_path: Path,
    num_numeric: int = 13,
    num_categorical: int = 26,
    hash_bucket_size: int = 1 << 18,
    device: Optional[torch.device] = None,
) -> DeepFM:
    """
    Load DeepFM model from checkpoint if it exists; otherwise init fresh model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg_dict = ckpt.get("config") or {}
        config = DeepFMConfig(
            feature_sizes=cfg_dict.get("feature_sizes", [hash_bucket_size] * (num_numeric + num_categorical)),
            num_numeric=cfg_dict.get("num_numeric", num_numeric),
            num_categorical=cfg_dict.get("num_categorical", num_categorical),
            embedding_size=cfg_dict.get("embedding_size", 8),
            hidden_dims=tuple(cfg_dict.get("hidden_dims", (64, 32))),
            num_classes=cfg_dict.get("num_classes", 1),
            dropout=tuple(cfg_dict.get("dropout", (0.1, 0.1))),
            use_cuda=device.type == "cuda",
        )
        model = DeepFM(config).to(device)
        model.load_state_dict(ckpt["model_state"])
    else:
        config = DeepFMConfig.from_data_dims(
            num_numeric=num_numeric,
            num_categorical=num_categorical,
            hash_bucket_size=hash_bucket_size,
            embedding_size=8,
            hidden_dims=(64, 32),
            dropout=(0.1, 0.1),
            use_cuda=device.type == "cuda",
        )
        model = DeepFM(config).to(device)
    return model


__all__ = ["load_deepfm_from_checkpoint", "DeepFM", "DeepFMConfig"]

