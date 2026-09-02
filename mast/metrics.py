"""Metrics for forecasting (regression) and classification quality."""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score


def mse(y_true, y_pred):
    """Mean squared error, ignoring positions where the prediction is NaN."""
    mask = ~torch.isnan(y_pred)
    if not mask.any():
        return torch.tensor(float('nan'), device=y_pred.device)
    return F.mse_loss(y_true[mask], y_pred[mask], reduction='mean')


def mae(y_true, y_pred):
    """Mean absolute error, ignoring positions where the prediction is NaN."""
    mask = ~torch.isnan(y_pred)
    if not mask.any():
        return torch.tensor(float('nan'), device=y_pred.device)
    return F.l1_loss(y_true[mask], y_pred[mask], reduction='mean')


def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))


# --- classification metrics (take raw logits; targets are int64 class indices) ---

def _argmax_preds(y_true, y_pred):
    """Reduce logit tensors `[N, n_classes]` to predicted class indices `[N]`."""
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(dim=-1)
    return y_true, y_pred


def accuracy(y_true, y_pred):
    y_true, y_pred = _argmax_preds(y_true, y_pred)
    return (y_pred == y_true).float().mean()


def precision_macro(y_true, y_pred):
    y_true, y_pred = _argmax_preds(y_true, y_pred)
    return torch.tensor(precision_score(y_true.cpu(), y_pred.cpu(),
                                        average='macro', zero_division=0))


def recall_macro(y_true, y_pred):
    y_true, y_pred = _argmax_preds(y_true, y_pred)
    return torch.tensor(recall_score(y_true.cpu(), y_pred.cpu(),
                                     average='macro', zero_division=0))


def f1_macro(y_true, y_pred):
    y_true, y_pred = _argmax_preds(y_true, y_pred)
    return torch.tensor(f1_score(y_true.cpu(), y_pred.cpu(),
                                 average='macro', zero_division=0))
