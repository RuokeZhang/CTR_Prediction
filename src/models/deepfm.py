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
    num_numeric: int
    num_categorical: int
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
            num_numeric=num_numeric,
            num_categorical=num_categorical,
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
        self.num_numeric = config.num_numeric
        self.num_categorical = config.num_categorical
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
        # Linear term for dense numeric features
        self.linear_dense = nn.Linear(self.num_numeric, 1, bias=False)
        # FM second order embeddings
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )

        # Deep part (MLP)
        deep_layers = []
        input_dim = self.num_numeric + self.num_categorical * self.embedding_size
        for i, out_dim in enumerate(self.hidden_dims):
            deep_layers.append(nn.Linear(input_dim, out_dim))
            deep_layers.append(nn.BatchNorm1d(out_dim))
            deep_layers.append(nn.ReLU())
            dropout_prob = config.dropout[i] if i < len(config.dropout) else 0.0
            deep_layers.append(nn.Dropout(dropout_prob))
            input_dim = out_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.deep_output = nn.Linear(input_dim, self.num_classes)

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
        # Handle Batch input (DeepFM requires dense features for deep/linear parts)
        if isinstance(input_data, Batch):
            Xi, Xv = self._batch_to_xi_xv(input_data)
            dense_input = input_data.numerical  # [N, num_numeric]
        else:
            raise TypeError("DeepFM forward expects a Batch object as input_data.")

        # FM first order
        fm_first_order_emb_arr = [
            (torch.sum(emb(Xi[:, idx, :]), 1).t() * Xv[:, idx]).t()
            for idx, emb in enumerate(
                self.fm_first_order_embeddings[self.num_numeric :], start=self.num_numeric
            )
        ]
        if fm_first_order_emb_arr:
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            first_order_cat = torch.sum(fm_first_order, dim=1, keepdim=True)
        else:
            first_order_cat = torch.zeros(
                (dense_input.size(0), 1), device=dense_input.device, dtype=dense_input.dtype
            )
        first_order_term = self.linear_dense(dense_input) + first_order_cat

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

        # Deep part: concat dense numerical + categorical embeddings
        if not isinstance(input_data, Batch):
            raise TypeError("DeepFM forward expects a Batch object for deep part input.")
        dense_input = input_data.numerical  # [N, num_numeric]
        cat_second_order_emb_arr = fm_second_order_emb_arr[self.num_numeric :]  # categorical only
        if cat_second_order_emb_arr:
            cat_deep = torch.cat(cat_second_order_emb_arr, dim=1)  # [N, num_categorical * embed]
        else:
            cat_deep = torch.zeros(
                (dense_input.size(0), 0), device=dense_input.device, dtype=dense_input.dtype
            )
        deep_input = torch.cat([dense_input, cat_deep], dim=1)
        deep_out = self.deep_layers(deep_input)
        deep_logits = self.deep_output(deep_out)

        # Sum all parts (keep dims for clean broadcasting)
        second_order_term = torch.sum(fm_second_order, dim=1, keepdim=True)
        total_sum = first_order_term + second_order_term + deep_logits + self.bias.view(1, 1)

        return total_sum


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
        # Unscale before gradient clipping to avoid inf/nan from AMP
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


__all__ = ["DeepFM", "DeepFMConfig", "train_deepfm_epoch"]

