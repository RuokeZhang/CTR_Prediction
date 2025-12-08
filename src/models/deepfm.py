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

    def forward(self, Xi: torch.Tensor, Xv: torch.Tensor) -> torch.Tensor:
        """
        Forward process of network.

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
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

        return total_sum


def fit(model: DeepFM, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
    """
    Training a model and valid accuracy.

    Inputs:
    - loader_train: Training data loader
    - loader_val: Validation data loader
    - optimizer: Optimizer used in training process
    - epochs: Number of epochs
    - verbose: If print during training
    - print_every: Print after every number of iterations
    """
    import torch.nn.functional as F

    model = model.train().to(device=model.device)
    criterion = F.binary_cross_entropy_with_logits

    for _ in range(epochs):
        for t, (xi, xv, y) in enumerate(loader_train):
            xi = xi.to(device=model.device, dtype=model.dtype)
            xv = xv.to(device=model.device, dtype=torch.float)
            y = y.to(device=model.device, dtype=torch.float)

            total = model(xi, xv)
            loss = criterion(total, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()


def check_accuracy(loader, model: DeepFM):
    """Check accuracy on validation or test set."""
    import torch.nn.functional as F

    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for xi, xv, y in loader:
            xi = xi.to(device=model.device, dtype=model.dtype)
            xv = xv.to(device=model.device, dtype=torch.float)
            y = y.to(device=model.device, dtype=torch.bool)
            total = model(xi, xv)
            preds = (F.sigmoid(total) > 0.5)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


__all__ = ["DeepFM", "DeepFMConfig", "fit", "check_accuracy"]

