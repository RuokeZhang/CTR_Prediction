"""Evaluation helpers for CTR models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import torch


def tensor_to_numpy(tensor_list) -> np.ndarray:
    tensors = [torch.as_tensor(t).view(-1) for t in tensor_list]
    return torch.cat(tensors).numpy()


def compute_auc(labels, predictions) -> float:
    y_true = tensor_to_numpy(labels)
    y_pred = tensor_to_numpy(predictions)
    return float(roc_auc_score(y_true, y_pred))


def compute_logloss(labels, predictions) -> float:
    y_true = tensor_to_numpy(labels)
    y_pred = tensor_to_numpy(predictions)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_pred))


__all__ = ["compute_auc", "compute_logloss"]

