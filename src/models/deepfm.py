"""DeepFM implementation for CTR prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from src.data.module import Batch


@dataclass
class DeepFMConfig:
    num_numeric: int
    num_categorical: int
    hash_bucket_size: int
    embedding_dim: int = 16
    deep_layers: Sequence[int] = (256, 128)
    dropout: float = 0.2


class DeepFM(nn.Module):
    def __init__(self, config: DeepFMConfig) -> None:
        super().__init__()
        self.config = config

        # First-order linear terms.
        self.linear_dense = nn.Linear(config.num_numeric, 1)
        self.linear_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.hash_bucket_size, 1)
                for _ in range(config.num_categorical)
            ]
        )

        # Second-order + deep component share embeddings.
        self.feature_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.hash_bucket_size, config.embedding_dim)
                for _ in range(config.num_categorical)
            ]
        )

        deep_input_dim = config.num_numeric + config.num_categorical * config.embedding_dim
        layers = []
        prev_dim = deep_input_dim
        for dim in config.deep_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, batch: Batch) -> torch.Tensor:
        dense = batch.numerical
        sparse = batch.categorical

        linear_term = self.linear_dense(dense)
        for idx, embed in enumerate(self.linear_embeddings):
            linear_term = linear_term + embed(sparse[:, idx])

        # FM second-order term
        embed_stack = []
        for idx, embed in enumerate(self.feature_embeddings):
            embed_stack.append(embed(sparse[:, idx]))
        embeddings = torch.stack(embed_stack, dim=1)  # (B, F, D)

        sum_of_embeddings = embeddings.sum(dim=1)
        sum_square = sum_of_embeddings * sum_of_embeddings
        square_sum = (embeddings * embeddings).sum(dim=1)
        fm_term = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)

        deep_input = torch.cat([dense, embeddings.view(dense.size(0), -1)], dim=1)
        deep_term = self.deep(deep_input)

        logits = linear_term + fm_term + deep_term
        return torch.sigmoid(logits)


def train_deepfm_epoch(
    model: DeepFM,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    autocast = torch.cuda.amp.autocast
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            preds = model(batch)
            loss = loss_fn(preds, batch.label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


__all__ = ["DeepFM", "DeepFMConfig", "train_deepfm_epoch"]

