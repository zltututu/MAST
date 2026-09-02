"""Regression metrics used to report forecasting quality."""

import torch
import torch.nn.functional as F


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
