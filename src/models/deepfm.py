"""DeepFM implementation for CTR prediction (PyTorch).

对齐论文《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》。
包含三部分：线性项（一阶）、FM 二阶交互、深层 DNN，三者共享同一套 embedding。
输出为 logits，训练请使用 BCEWithLogitsLoss。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
        self.bias = nn.Parameter(torch.zeros(1))

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
        # Linear term for dense numeric features
        self.linear_dense = nn.Linear(self.num_numeric, 1, bias=False)

        # Deep part (MLP)
        deep_layers = []
        input_dim = self.field_size * self.embedding_size
        for i, out_dim in enumerate(self.hidden_dims):
            deep_layers.append(nn.Linear(input_dim, out_dim))
            deep_layers.append(nn.BatchNorm1d(out_dim))
            deep_layers.append(nn.ReLU())
            dropout_prob = config.dropout[i] if i < len(config.dropout) else 0.0
            deep_layers.append(nn.Dropout(dropout_prob))
            input_dim = out_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.deep_output = nn.Linear(input_dim, self.num_classes, bias=False)

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
        # Handle Batch input (preferred)
        if isinstance(input_data, Batch):
            Xi, Xv = self._batch_to_xi_xv(input_data)
            dense_input = input_data.numerical  # [N, num_numeric]
        else:
            Xi = input_data
            if Xv is None:
                raise ValueError("Xv must be provided when input_data is Xi tensor")
            dense_input = None

        Xi = Xi.long()

        # ---------------------------
        # First-order (linear) part
        # ---------------------------
        first_order_list = []
        for i, emb in enumerate(self.fm_first_order_embeddings):
            # emb -> [N, 1, 1]; value -> [N,1]
            w_i = emb(Xi[:, i, :]).squeeze(-1)  # [N,1]
            v_i = Xv[:, i : i + 1]
            first_order_list.append(w_i * v_i)
        first_order_cat = torch.sum(torch.cat(first_order_list, dim=1), dim=1, keepdim=True)
        if dense_input is not None:
            first_order_term = first_order_cat + self.linear_dense(dense_input)
        else:
            first_order_term = first_order_cat

        # ---------------------------
        # FM second-order part
        # ---------------------------
        second_order_list = []
        for i, emb in enumerate(self.fm_second_order_embeddings):
            v_i = Xv[:, i : i + 1]  # [N,1]
            e_i = emb(Xi[:, i, :]).squeeze(1) * v_i  # [N, embed]
            second_order_list.append(e_i)
        stacked = torch.stack(second_order_list, dim=1)  # [N, field, embed]
        sum_emb = torch.sum(stacked, dim=1)  # [N, embed]
        sum_emb_square = sum_emb * sum_emb
        square_sum_emb = torch.sum(stacked * stacked, dim=1)
        fm_second = 0.5 * (sum_emb_square - square_sum_emb)  # [N, embed]
        fm_second_term = torch.sum(fm_second, dim=1, keepdim=True)  # [N,1]

        # ---------------------------
        # Deep part
        # ---------------------------
        deep_input = stacked.reshape(stacked.size(0), -1)  # [N, field*embed]
        deep_out = self.deep_layers(deep_input)
        deep_logits = self.deep_output(deep_out)  # [N,1]

        # ---------------------------
        # Final logit
        # ---------------------------
        total_sum = first_order_term + fm_second_term + deep_logits + self.bias.view(1, 1)
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

