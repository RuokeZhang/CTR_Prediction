"""DeepFM implementation for CTR prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from src.data.module import Batch


@dataclass
class DeepFMConfig:
    feature_sizes: Sequence[int]
    embedding_size: int = 4
    hidden_dims: Sequence[int] = (32, 32)
    num_classes: int = 1
    dropout: Sequence[float] = (0.5, 0.5)
    use_cuda: bool = True

    @classmethod
    def from_data_dims(
        cls,
        num_numeric: int,
        num_categorical: int,
        hash_bucket_size: int,
        embedding_size: int = 4,
        hidden_dims: Sequence[int] = (32, 32),
        dropout: Sequence[float] = (0.5, 0.5),
        use_cuda: bool = True,
    ):
        """Create config from data dimensions (compatible with Batch format)."""
        # All features (numeric + categorical) use the same hash bucket size
        feature_sizes = [hash_bucket_size] * (num_numeric + num_categorical)
        return cls(
            feature_sizes=feature_sizes,
            embedding_size=embedding_size,
            hidden_dims=hidden_dims,
            num_classes=1,
            dropout=dropout,
            use_cuda=use_cuda,
        )


class DeepFM(nn.Module):
    def __init__(self, config: DeepFMConfig) -> None:
        super().__init__()
        self.field_size = len(config.feature_sizes)
        self.feature_sizes = config.feature_sizes
        self.embedding_size = config.embedding_size
        self.hidden_dims = config.hidden_dims
        self.num_classes = config.num_classes
        self.dtype = torch.long
        self.bias = nn.Parameter(torch.randn(1))

        if config.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # FM first order embeddings
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes]
        )
        # FM second order embeddings
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )

        # Deep part
        all_dims = [self.field_size * self.embedding_size] + list(self.hidden_dims) + [self.num_classes]
        for i in range(1, len(self.hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(config.dropout[i-1]))

    def _batch_to_xi_xv(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert Batch object to Xi/Xv format.

        Args:
            batch: Batch with numerical and categorical features

        Returns:
            Xi: Feature indices [N, field_size, 1]
            Xv: Feature values [N, field_size, 1]
        """
        # For numerical features: use normalized values as Xv, indices as 0,1,2...
        # For categorical features: use the hashed indices as Xi, values as 1.0
        N = batch.numerical.size(0)
        num_numeric = batch.numerical.size(1)
        num_categorical = batch.categorical.size(1)

        # Create Xi (indices)
        Xi_numeric = torch.arange(num_numeric, device=batch.numerical.device).unsqueeze(0).unsqueeze(-1).expand(N, -1, 1)
        Xi_categorical = batch.categorical.unsqueeze(-1)
        Xi = torch.cat([Xi_numeric, Xi_categorical], dim=1)  # [N, field_size, 1]

        # Create Xv (values)
        Xv_numeric = batch.numerical.unsqueeze(-1)
        Xv_categorical = torch.ones_like(Xi_categorical, dtype=torch.float32)
        Xv = torch.cat([Xv_numeric, Xv_categorical], dim=1)  # [N, field_size, 1]
        Xv = Xv.squeeze(-1)  # [N, field_size]

        return Xi, Xv

    def forward(self, input_data, Xv: torch.Tensor = None) -> torch.Tensor:
        """
        Forward process of network.

        Inputs (two modes):
        Mode 1 - Batch object:
            - input_data: Batch object with numerical and categorical fields
        Mode 2 - Xi/Xv tensors:
            - input_data (Xi): A tensor of input's index, shape of (N, field_size, 1)
            - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        # Handle Batch input
        if isinstance(input_data, Batch):
            Xi, Xv = self._batch_to_xi_xv(input_data)
        else:
            Xi = input_data
            if Xv is None:
                raise ValueError("Xv must be provided when input_data is Xi tensor")

        # FM first order
        fm_first_order_emb_arr = [
            (torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
            for i, emb in enumerate(self.fm_first_order_embeddings)
        ]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

        # FM second order
        fm_second_order_emb_arr = [
            (torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
            for i, emb in enumerate(self.fm_second_order_embeddings)
        ]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        # Deep part
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        # Sum all parts
        total_sum = (torch.sum(fm_first_order, 1) +
                     torch.sum(fm_second_order, 1) +
                     torch.sum(deep_out, 1) +
                     self.bias)

        return total_sum.unsqueeze(-1)


def train_deepfm_epoch(
    model: DeepFM,
    batches,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler=None,
) -> float:
    """Train DeepFM for one epoch (compatible with baselines.train_epoch).

    Args:
        model: DeepFM model
        batches: Iterator of Batch objects
        optimizer: PyTorch optimizer
        loss_fn: Loss function (e.g., BCELoss)
        scaler: Optional GradScaler for mixed precision training

    Returns:
        Average training loss for the epoch
    """
    from src.data.module import Batch

    model.train()
    total_loss = 0.0
    steps = 0

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    autocast = torch.cuda.amp.autocast
    for batch in batches:
        if not isinstance(batch, Batch):
            raise TypeError(f"Expected Batch object, got {type(batch)}")

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

