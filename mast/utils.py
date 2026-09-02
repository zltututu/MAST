"""Checkpoint helpers."""

from pathlib import Path

import torch
from torch import nn


def get_model(model):
    """Unwrap a model held inside `nn.DataParallel`."""
    return model.module if isinstance(model, nn.DataParallel) else model


def join_path_file(file, path, ext=''):
    if not isinstance(file, (str, Path)):
        return file
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{file}{ext}'


def save_model(path, model, opt, with_opt=True, pickle_protocol=2):
    """Save `model` (and the optimizer state when available) to `path`."""
    if opt is None:
        with_opt = False
    state = get_model(model).state_dict()
    if with_opt:
        state = {'model': state, 'opt': opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)


def load_model(path, model, opt=None, with_opt=False, device='cpu', strict=True):
    """Load a checkpoint written by `save_model` into `model`."""
    state = torch.load(path, map_location=device)
    if not opt:
        with_opt = False
    model_state = state['model'] if with_opt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if with_opt:
        opt.load_state_dict(state['opt'])
    model.to(device)
