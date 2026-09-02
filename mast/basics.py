"""Small utilities shared across MAST: attribute delegation, device handling and tensor helpers."""

import random

import numpy as np
import torch


class GetAttr:
    """Base class delegating unknown attribute lookups to `self.default` (used by Learner/callbacks)."""

    _default = 'default'

    def _component_attr_filter(self, k):
        if k.startswith('__') or k in ('_xtra', self._default):
            return False
        xtra = getattr(self, '_xtra', None)
        return xtra is None or k in xtra

    def _dir(self):
        return [k for k in dir(getattr(self, self._default)) if self._component_attr_filter(k)]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None:
                return getattr(attr, k)

    def __dir__(self):
        return custom_dir(self, self._dir())

    def __setstate__(self, data):
        self.__dict__.update(data)


def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add


def set_seed(seed=2021):
    """Seed python, numpy and torch RNGs so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_available_cuda(usage=10):
    """Return the ids of GPUs whose utilisation is below `usage` percent."""
    if not torch.cuda.is_available():
        return []
    return [i for i in range(torch.cuda.device_count()) if torch.cuda.utilization(i) < usage]


def default_device(use_cuda=True):
    """Pick cuda:0 when available, otherwise cpu."""
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


def select_device(require_cuda=False):
    """Resolve the training device. Falls back to CPU unless `require_cuda` is set."""
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    if require_cuda:
        raise RuntimeError('CUDA is not available; install a CUDA-enabled PyTorch or drop --require_cuda.')
    return torch.device('cpu')


def to_device(b, device=None, non_blocking=False):
    """Recursively move the tensors inside `b` to `device`."""
    if device is None:
        device = default_device()
    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)
    return b.to(device, non_blocking=non_blocking)


def to_numpy(b):
    """Recursively convert the tensors inside `b` to numpy arrays."""
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)
    return b.detach().cpu().numpy()
