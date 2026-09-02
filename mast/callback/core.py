"""Callback base class and the callbacks that glue batches, devices and predictions together.

Callback ordering used by the Learner::

    before_fit
        before_epoch
            before_epoch_train
                before_batch_train / after_batch_train
            after_epoch_train
            before_epoch_valid
                before_batch_valid / after_batch_valid
            after_epoch_valid
        after_epoch
    after_fit

    before_predict / before_batch_predict / after_batch_predict / after_predict
    before_test / before_batch_test / after_batch_test / after_test
"""

import torch

from ..basics import GetAttr, default_device, to_device

__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB']


class Callback(GetAttr):
    _default = 'learner'


class SetupLearnerCB(Callback):
    def __init__(self, device=None):
        self.device = device if device is not None else default_device()

    def before_batch_train(self):
        self._to_device()

    def before_batch_valid(self):
        self._to_device()

    def before_batch_predict(self):
        self._to_device()

    def before_batch_test(self):
        self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)
        if self.n_inp > 1:
            xb, yb = batch
        else:
            xb, yb = batch, None
        self.learner.batch = xb, yb

    def before_fit(self):
        self.learner.model.to(self.device)
        self.learner.device = self.device


class GetPredictionsCB(Callback):
    def before_predict(self):
        self.preds = []

    def after_batch_predict(self):
        self.preds.append(self.pred)

    def after_predict(self):
        self.preds = torch.concat(self.preds)


class GetTestCB(Callback):
    def before_test(self):
        self.preds, self.targets = [], []

    def after_batch_test(self):
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)
