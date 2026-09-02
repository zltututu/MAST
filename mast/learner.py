"""A minimal training loop (Learner) with the callback hook system MAST relies on."""

from typing import List

import torch
from torch.optim import Adam

from .basics import GetAttr, to_numpy
from .callback.core import GetTestCB, SetupLearnerCB
from .callback.scheduler import OneCycleLR
from .callback.tracking import PrintResultsCB, TrackTimerCB, TrackTrainingCB
from .utils import get_model, join_path_file, load_model, save_model


class Learner(GetAttr):
    """Owns the model, the data loaders and the callbacks, and drives training/validation/testing."""

    def __init__(self, dls, model, loss_func=None, lr=1e-3, cbs=None, metrics=None, opt_func=Adam, device=None):
        self.model, self.dls, self.loss_func, self.lr = model, dls, loss_func, lr
        self.opt_func = opt_func
        self.metrics = metrics
        self.n_inp = 2
        self.set_opt()
        self.device = device if device is not None else next(model.parameters()).device

        if cbs and not isinstance(cbs, List):
            cbs = [cbs]
        self.initialize_callbacks(cbs)

    def set_opt(self):
        if self.model:
            self.opt = self.opt_func(self.model.parameters(), self.lr)
        else:
            self.opt = None

    def default_callback(self):
        return [SetupLearnerCB(self.device), TrackTimerCB(),
                TrackTrainingCB(train_metrics=False, valid_metrics=True)]

    def initialize_callbacks(self, cbs):
        default_cbs = self.default_callback()
        self.cbs = update_callbacks(cbs, default_cbs) if cbs else default_cbs
        self.cbs += [PrintResultsCB()]
        for cb in self.cbs:
            cb.learner = self
        self('init_cb')

    def add_callback(self, cb):
        if not cb:
            return
        cb.learner = self
        self.cbs = update_callback(cb, self.cbs)

    def add_callbacks(self, cbs):
        if not isinstance(cbs, list):
            cbs = [cbs]
        for cb in cbs:
            self.add_callback(cb)

    def remove_callback(self, cb):
        cb.learner = None
        self.cbs, removed_cb = remove_callback(cb, self.cbs)
        return removed_cb

    def fit(self, n_epochs, lr=None, cbs=None, do_valid=True):
        self.n_epochs = n_epochs
        if not self.dls.valid:
            do_valid = False
        if cbs:
            self.add_callbacks(cbs)
        if lr:
            self.opt = self.opt_func(self.model.parameters(), lr)

        self('before_fit')
        try:
            for self.epoch in range(n_epochs):
                self('before_epoch')
                self.one_epoch(train=True)
                if do_valid:
                    self.one_epoch(train=False)
                self('after_epoch')
        except KeyboardInterrupt:
            pass
        self('after_fit')

    def fit_one_cycle(self, n_epochs, lr_max=None, pct_start=0.3):
        self.n_epochs = n_epochs
        self.lr_max = lr_max if lr_max else self.lr
        self.fit(self.n_epochs, cbs=OneCycleLR(lr_max=self.lr_max, pct_start=pct_start))

    def one_epoch(self, train):
        self.epoch_train() if train else self.epoch_validate()

    def epoch_train(self):
        self('before_epoch_train')
        self.model.train()
        self.dl = self.dls.train
        self.all_batches('train')
        self('after_epoch_train')

    def epoch_validate(self, dl=None):
        self('before_epoch_valid')
        self.model.eval()
        self.dl = dl if dl else self.dls.valid
        if self.dl:
            with torch.no_grad():
                self.all_batches('valid')
        self('after_epoch_valid')

    def all_batches(self, type_):
        for num, batch in enumerate(self.dl):
            self.iter, self.batch = num, batch
            if type_ == 'train':
                self.batch_train()
            elif type_ == 'valid':
                self.batch_validate()
            elif type_ == 'predict':
                self.batch_predict()
            elif type_ == 'test':
                self.batch_test()

    def batch_train(self):
        self('before_batch_train')
        self._do_batch_train()
        self('after_batch_train')

    def batch_validate(self):
        self('before_batch_valid')
        self._do_batch_validate()
        self('after_batch_valid')

    def batch_test(self):
        self('before_batch_test')
        self._do_batch_test()
        self('after_batch_test')

    def _do_batch_train(self):
        self.pred, self.loss = self.train_step(self.batch)
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

    def train_step(self, batch):
        self.xb, self.yb = batch
        if isinstance(self.yb, torch.Tensor) and self.yb.dtype != torch.float32:
            self.yb = self.yb.float()
        pred = self.model_forward()
        return pred, self.loss_func(pred, self.yb)

    def model_forward(self):
        self('before_forward')
        self.pred = self.model(self.xb)
        self('after_forward')
        return self.pred

    def _do_batch_validate(self):
        self.pred, self.loss = self.valid_step(self.batch)

    def valid_step(self, batch):
        self.xb, self.yb = batch
        if isinstance(self.yb, torch.Tensor) and self.yb.dtype != torch.float32:
            self.yb = self.yb.float()
        pred = self.model_forward()
        return pred, self.loss_func(pred, self.yb)

    def _do_batch_test(self):
        self.pred, self.yb = self.test_step(self.batch)

    def test_step(self, batch):
        self.xb, self.yb = batch
        return self.model_forward(), self.yb

    def test(self, dl, weight_path=None, scores=None):
        """Run inference on `dl` and return `(preds, targets, scores)`."""
        if dl is None:
            return
        self.dl = dl
        if weight_path is not None:
            self.load(weight_path)
        cb = GetTestCB()
        self.add_callback(cb)
        self('before_test')
        self.model.eval()
        with torch.no_grad():
            self.all_batches('test')
        self('after_test')
        self.preds, self.targets = to_numpy([cb.preds, cb.targets])

        if scores:
            s_vals = [score(cb.targets, cb.preds).to('cpu').numpy() for score in scores]
            return self.preds, self.targets, s_vals
        return self.preds, self.targets

    def fine_tune(self, n_epochs, base_lr=None, freeze_epochs=1, pct_start=0.3):
        """Train the head for `freeze_epochs`, then unfreeze and train the whole network."""
        assert n_epochs > 0 or freeze_epochs > 0, "Either n_epochs or freeze_epochs has to be > 0"
        if not base_lr:
            base_lr = self.lr
        if freeze_epochs > 0:
            print('Finetune the head')
            self.freeze()
            self.fit_one_cycle(freeze_epochs, lr_max=base_lr, pct_start=pct_start)
        if n_epochs > 0:
            print('Finetune the entire network')
            self.unfreeze()
            self.fit_one_cycle(n_epochs, lr_max=base_lr / 2, pct_start=pct_start)

    def linear_probe(self, n_epochs, base_lr=None, pct_start=0.3):
        """Freeze the backbone and train only the prediction head."""
        assert n_epochs > 0, "n_epochs has to be > 0"
        if not base_lr:
            base_lr = self.lr
        print('Finetune the head')
        self.freeze()
        self.fit_one_cycle(n_epochs, lr_max=base_lr, pct_start=pct_start)

    def freeze(self):
        model = get_model(self.model)
        if hasattr(model, 'head'):
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

    def unfreeze(self):
        for param in get_model(self.model).parameters():
            param.requires_grad = True

    def __call__(self, name):
        for cb in self.cbs:
            attr = getattr(cb, name)
            if attr is not None:
                attr()

    def save(self, fname, path, **kwargs):
        fname = join_path_file(fname, path, ext='.pth')
        save_model(fname, self.model, getattr(self, 'opt', None), **kwargs)
        return fname

    def load(self, fname, with_opt=False, device=None, strict=True, **kwargs):
        load_model(fname, self.model, self.opt, with_opt,
                   device=device or self.device, strict=strict)


def update_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) == type(cb):
            list_cbs.remove(cb_)
    list_cbs += [cb]
    return list_cbs


def update_callbacks(list_cbs, default_cbs):
    for cb in list_cbs:
        default_cbs = update_callback(cb, default_cbs)
    return default_cbs


def remove_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) == type(cb):
            list_cbs.remove(cb_)
            break
    return list_cbs, cb_
