"""Baseline CTR models built with PyTorch.

Updated to support multi-dimensional embeddings for categorical features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch
from torch import nn

from src.data.module import Batch





class LogisticRegressionModel(nn.Module):
    """Logistic regression with embedding support for categorical features.

    Architecture:
        - Numerical features: direct linear transformation
        - Categorical features: embedding lookup -> pooling -> linear
        - Output: sigmoid(linear(concat([numerical, categorical_embeddings])))
    """

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        hash_bucket_size: int,
        embedding_dim: int = 8,
        use_embeddings: bool = True,
    ) -> None:
        """Initialize Logistic Regression model.

        Args:
            num_numeric: Number of numerical features (13 for Criteo)
            num_categorical: Number of categorical features (26 for Criteo)
            hash_bucket_size: Size of hash buckets for categorical features (262144)
            embedding_dim: Dimension of categorical embeddings (default: 8)
            use_embeddings: If True, use embeddings; if False, use legacy normalization
        """
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.hash_bucket_size = hash_bucket_size
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

        if use_embeddings:
            # Create one embedding table per categorical feature
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(hash_bucket_size, embedding_dim)
                    for _ in range(num_categorical)
                ]
            )
            # Input dimension: numerical + (categorical_fields * embedding_dim)
            input_dim = num_numeric + num_categorical * embedding_dim


        self.linear = nn.Linear(input_dim, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Batch with numerical, categorical, and label tensors

        Returns:
            Predicted click probabilities [batch_size, 1]
        """
        dense = batch.numerical  # [B, num_numeric]
        sparse = batch.categorical  # [B, num_categorical]

        if self.use_embeddings:
            # Lookup embeddings for each categorical field
            emb_list = []
            for idx, emb_layer in enumerate(self.embeddings):
                emb_list.append(emb_layer(sparse[:, idx]))  # [B, embedding_dim]

            # Concatenate all embeddings
            categorical_features = torch.cat(emb_list, dim=-1)  # [B, num_cat * emb_dim]


        # Concatenate numerical and categorical features
        features = torch.cat([dense, categorical_features], dim=-1)

        # Linear transformation + sigmoid
        logits = self.linear(features)
        return torch.sigmoid(logits)


class ShallowMLP(nn.Module):
    """Two-hidden-layer MLP with embedding support for categorical features.

    Architecture:
        Input -> [Embeddings] -> Concat -> MLP(128->64->1) -> Sigmoid
    """

    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        hash_bucket_size: int,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_embeddings: bool = True,
    ) -> None:
        """Initialize Shallow MLP model.

        Args:
            num_numeric: Number of numerical features
            num_categorical: Number of categorical features
            hash_bucket_size: Size of hash buckets
            embedding_dim: Dimension of categorical embeddings (default: 8)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout: Dropout probability (default: 0.1)
            use_embeddings: If True, use embeddings; if False, use legacy normalization
        """
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.hash_bucket_size = hash_bucket_size
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

        if use_embeddings:
            # Create embedding layers
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(hash_bucket_size, embedding_dim)
                    for _ in range(num_categorical)
                ]
            )
            input_dim = num_numeric + num_categorical * embedding_dim
        else:
            self.embeddings = None
            input_dim = num_numeric + num_categorical

        # MLP layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Batch with numerical, categorical, and label tensors

        Returns:
            Predicted click probabilities [batch_size, 1]
        """
        dense = batch.numerical
        sparse = batch.categorical

        if self.use_embeddings:
            # Lookup embeddings
            emb_list = []
            for idx, emb_layer in enumerate(self.embeddings):
                emb_list.append(emb_layer(sparse[:, idx]))
            categorical_features = torch.cat(emb_list, dim=-1)
        else:
            categorical_features = _cat_to_float(sparse, self.hash_bucket_size)

        # Concatenate and pass through MLP
        features = torch.cat([dense, categorical_features], dim=-1)
        logits = self.net(features)
        return torch.sigmoid(logits)


@dataclass
class TrainingStats:
    """Container for training statistics."""
    loss: float
    auc: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


def train_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> TrainingStats:
    """Generic training loop for baseline models.

    Args:
        model: PyTorch model to train
        batches: Iterator of Batch objects
        optimizer: PyTorch optimizer
        loss_fn: Loss function (e.g., BCELoss)

    Returns:
        TrainingStats with average training loss
    """
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch)
        loss = loss_fn(preds, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return TrainingStats(loss=total_loss / max(steps, 1))


def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    loss_fn: nn.Module,
    metric_fns: Optional[Dict[str, callable]] = None,
) -> TrainingStats:
    """Evaluate model on validation/test set.

    Args:
        model: PyTorch model to evaluate
        batches: Iterator of Batch objects
        loss_fn: Loss function
        metric_fns: Dictionary of metric functions (e.g., {'auc': compute_auc})

    Returns:
        TrainingStats with loss, AUC, and other metrics
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in batches:
            preds = model(batch)
            loss = loss_fn(preds, batch.label)
            total_loss += loss.item()
            steps += 1
            all_preds.append(preds.detach().cpu())
            all_labels.append(batch.label.detach().cpu())

    metrics = {}
    auc = None
    if metric_fns:
        # Always convert logits to probabilities for metrics
        import torch.nn.functional as F
        all_preds_probs = [F.sigmoid(p) for p in all_preds]

        for name, fn in metric_fns.items():
            metrics[name] = fn(all_labels, all_preds_probs)
        auc = metrics.get("auc")

    return TrainingStats(loss=total_loss / max(steps, 1), auc=auc, metrics=metrics)


__all__ = [
    "LogisticRegressionModel",
    "ShallowMLP",
    "train_epoch",
    "evaluate",
    "TrainingStats",
]
