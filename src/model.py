"""Model entrypoints for DeepFM demo/training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

class DeepModel(torch.nn.Module):

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        hash_bucket_size: int,
        embedding_dim: int = 8,
        use_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.hash_bucket_size = hash_bucket_size
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

        if use_embeddings:
            self.embeddings = torch.nn.ModuleList(
                [torch.nn.Embedding(hash_bucket_size, embedding_dim) for _ in range(num_categorical)]
            )
            input_dim = num_numeric + num_categorical * embedding_dim
        else:
            self.embeddings = None
            input_dim = num_numeric + num_categorical

        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        dense = batch.numerical
        sparse = batch.categorical

        if self.use_embeddings:
            emb_list = []
            for idx, emb_layer in enumerate(self.embeddings):
                emb_list.append(emb_layer(sparse[:, idx]))
            categorical_features = torch.cat(emb_list, dim=-1)
        else:
            categorical_features = sparse.float()

        features = torch.cat([dense, categorical_features], dim=-1)
        logits = self.linear(features)
        return torch.sigmoid(logits)


def load_deep_model_from_checkpoint(
    checkpoint_path: Path,
    num_numeric: int = 13,
    num_categorical: int = 26,
    hash_bucket_size: int = 1 << 18,
    embedding_dim: int = 8,
    use_embeddings: bool = True,
    device: Optional[torch.device] = None,
) -> DeepModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepModel(
        num_numeric=num_numeric,
        num_categorical=num_categorical,
        hash_bucket_size=hash_bucket_size,
        embedding_dim=embedding_dim,
        use_embeddings=use_embeddings,
    ).to(device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    return model


__all__ = ["DeepModel", "load_deep_model_from_checkpoint"]

