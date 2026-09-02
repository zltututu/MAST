"""Data loading for MAST: TSLib-style forecasting CSVs and UEA classification .ts files.

Forecasting datasets follow the Time-Series-Library conventions: a CSV whose first
column is a timestamp and whose remaining columns are the variates (the last one is
usually the prediction `target`). ETTh1/ETTh2/ETTm1/ETTm2 use the fixed 12/4/4-month
ETT split; every other CSV (`--data custom`: weather, electricity, traffic, exchange,
illness, ...) uses a 70/10/20 ratio split. Batches are returned as `(x, y)` float
tensors with shapes `[seq_len, n_vars]` and `[pred_len, n_vars]`.

Classification uses the 10 UEA datasets distributed with Time-Series-Library in the
sktime `.ts` format (`<Name>_TRAIN.ts` / `<Name>_TEST.ts`). Series are zero-padded to
a common length and returned as `(x [seq_len, n_vars], y int64 label)`.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

__all__ = ['Dataset_ETT_hour', 'Dataset_ETT_minute', 'Dataset_Custom', 'UEADataset',
           'DataLoaders', 'data_dict', 'get_dls', 'get_uea_dls', 'DATASET_ROOT', 'DATASET_PATH']

# ETTh1 ships with the repository in ./dataset/ETT-small/; other datasets are
# downloaded from Time-Series-Library and placed in ./dataset/<name>/.
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'dataset', 'ETT-small')
DATASET_PATH = 'ETTh1.csv'


class _WindowDataset(Dataset):
    """Shared sliding-window logic for the forecasting CSV datasets."""

    def __init__(self, root_path, split, size, features, data_path, target, scale):
        self.seq_len, self.label_len, self.pred_len = size if size is not None else (384, 96, 96)
        assert split in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path

    def _select_variates(self, df_raw):
        if self.features in ('M', 'MS'):
            return df_raw[df_raw.columns[1:]]
        if self.features == 'S':
            return df_raw[[self.target]]
        raise ValueError(f"features must be one of M, MS, S; got {self.features!r}")

    def _fit_and_scale(self, df_data, border1s, border2s):
        """Standardise with statistics fitted on the training split only."""
        self.scaler = StandardScaler()
        if not self.scale:
            return df_data.values
        self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
        return self.scaler.transform(df_data.values)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(_WindowDataset):
    """Sliding-window hourly ETT dataset (ETTh1 / ETTh2) with the standard 12/4/4 month split.

    Args:
        root_path: directory holding the CSV file.
        split: one of ``train``, ``val``, ``test``.
        size: ``[seq_len, label_len, pred_len]``; `label_len` is kept for compatibility
            with the TSLib-style signature and is subtracted from the target window start.
        features: ``M`` uses every variate, ``S`` only `target`.
    """

    _steps_per_day = 24   # hourly sampling

    def __init__(self, root_path=DATASET_ROOT, split='train', size=None,
                 features='M', data_path=DATASET_PATH, target='OT', scale=True):
        super().__init__(root_path, split, size, features, data_path, target, scale)
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 12 / 4 / 4 months
        s = self._steps_per_day
        border1s = [0, 12 * 30 * s - self.seq_len, (12 + 4) * 30 * s - self.seq_len]
        border2s = [12 * 30 * s, (12 + 4) * 30 * s, (12 + 8) * 30 * s]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_data = self._select_variates(df_raw)
        data = self._fit_and_scale(df_data, border1s, border2s)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


class Dataset_ETT_minute(Dataset_ETT_hour):
    """Sliding-window 15-minute ETT dataset (ETTm1 / ETTm2), same interface as ETTh1/ETTh2."""

    _steps_per_day = 96   # 15-minute sampling

    def __init__(self, root_path=DATASET_ROOT, split='train', size=None,
                 features='M', data_path='ETTm1.csv', target='OT', scale=True):
        super().__init__(root_path, split, size, features, data_path, target, scale)


class Dataset_Custom(_WindowDataset):
    """Sliding-window CSV dataset with a 70/10/20 ratio split.

    Used for every non-ETT TSLib forecasting dataset (weather, electricity, traffic,
    exchange_rate, national_illness, ...). Expects the TSLib CSV layout: the first
    column is a timestamp, the remaining columns are the variates; column order is
    kept as-is and `target` selects the variate for ``features='S'``.
    """

    def __init__(self, root_path=DATASET_ROOT, split='train', size=None,
                 features='M', data_path='weather.csv', target='OT', scale=True,
                 train_split=0.7, test_split=0.2):
        super().__init__(root_path, split, size, features, data_path, target, scale)
        self.train_split, self.test_split = train_split, test_split
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_data = self._select_variates(df_raw)
        data = self._fit_and_scale(df_data, border1s, border2s)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


# dataset flag -> dataset class (see get_dls)
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


class DataLoaders:
    """Holds the train / validation / test DataLoaders for one configuration."""

    def __init__(self, datasetCls, dataset_kwargs: dict, batch_size: int,
                 workers: int = 0, collate_fn=None, shuffle_train=True, shuffle_val=False):
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        self.dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k != 'split'}
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

        self.train = self._make_dloader('train', shuffle=self.shuffle_train)
        self.valid = self._make_dloader('val', shuffle=self.shuffle_val)
        self.test = self._make_dloader('test', shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0:
            return None
        return DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size,
                          num_workers=self.workers, collate_fn=self.collate_fn)


class _SimpleDLS:
    """Minimal loader holder exposing the train / valid / test attributes the Learner expects."""

    def __init__(self, train, valid, test):
        self.train, self.valid, self.test = train, valid, test


def get_dls(params, root_path=None):
    """Build the DataLoaders for the dataset selected by ``params.data``.

    ``--data UEA`` routes to the classification loader (`get_uea_dls`); every other
    flag maps to a forecasting dataset class via `data_dict`. The loaders are
    annotated with `vars` (number of variates) and `len` (look-back length), both
    needed to build the model before the first batch is seen.
    """
    if params.data == 'UEA':
        return get_uea_dls(params, root_path=root_path)

    root_path = root_path or getattr(params, 'root_path', None) or DATASET_ROOT
    data_path = getattr(params, 'data_path', None) or DATASET_PATH
    datasetCls = data_dict[params.data]

    full_path = os.path.join(root_path, data_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Dataset file not found: {full_path}. Download it from the "
            "Time-Series-Library dataset folder and place it there "
            "(see the README for per-dataset download links)."
        )

    dls = DataLoaders(
        datasetCls=datasetCls,
        dataset_kwargs={
            'root_path': root_path,
            'data_path': data_path,
            'features': params.features,
            'target': getattr(params, 'target', 'OT'),
            'scale': True,
            'size': [params.context_points, 0, params.target_points],
        },
        batch_size=params.batch_size,
        workers=params.num_workers,
    )
    dls.vars = dls.train.dataset[0][0].shape[1]
    dls.len = params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


# --- UEA classification datasets (sktime .ts format) -------------------------

def _interpolate_nan(x):
    """Linearly interpolate NaNs per channel; leading/trailing NaNs become 0."""
    if not np.isnan(x).any():
        return x
    df = pd.DataFrame(x)
    df = df.interpolate(method='linear', limit_direction='both')
    return df.fillna(0.0).values.astype(np.float64)


def _parse_ts_file(path):
    """Parse one sktime ``.ts`` file into `(samples, labels, class_names)`.

    Handles the plain ``v1,v2,...:v1,v2,...:label`` layout used by the UEA archive as
    well as the parenthesised ``(...):(...):label`` variant. Data lines may also be
    wrapped in double quotes. ``NaN`` entries are linearly interpolated per channel.
    """
    class_names, samples, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip().strip('"')
            if not line or line.startswith('#') or line.startswith('@'):
                if line.startswith('@classLabel'):
                    parts = line.split()
                    if len(parts) > 2 and parts[1].lower() == 'true':
                        class_names = parts[2:]
                continue

            fields = line.split(':')
            channels = []
            for dim in fields[:-1]:
                dim = dim.strip()
                if dim.startswith('(') and dim.endswith(')'):
                    dim = dim[1:-1]
                channels.append([float(v) for v in dim.split(',') if v != ''])
            samples.append(np.asarray(channels, dtype=np.float64).T)   # [length, n_vars]
            labels.append(fields[-1].strip())

    if not samples:
        raise ValueError(f"No data lines found in {path}")
    if not class_names:
        class_names = sorted(set(labels))
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = [_interpolate_nan(s) for s in samples]
    y = [label_to_idx[label] for label in labels]
    return samples, y, class_names


class UEADataset(Dataset):
    """One split of a UEA classification dataset, padded to a common length.

    Args:
        samples: list of `[length, n_vars]` arrays of variable length.
        labels: list of int class indices.
        max_len: every sample is zero-padded (or truncated) to this length.
        mean / std: per-channel statistics from the train split; computed from
            `samples` themselves when omitted (i.e. for the train split).
    """

    def __init__(self, samples, labels, max_len, mean=None, std=None):
        self.samples, self.labels, self.max_len = samples, labels, max_len
        self.n_vars = samples[0].shape[1]

        if mean is None:
            stacked = np.concatenate(samples, axis=0)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            std[std == 0] = 1.0
        self.mean, self.std = mean, std

    def __getitem__(self, index):
        x = (self.samples[index] - self.mean) / self.std
        length = min(len(x), self.max_len)
        out = torch.zeros(self.max_len, self.n_vars)
        out[:length] = torch.from_numpy(x[:length]).float()
        return out, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.samples)


def get_uea_dls(params, root_path=None):
    """Build the train/valid/test loaders for a UEA classification dataset.

    ``root_path`` must contain exactly one ``<Name>_TRAIN.ts`` and the matching
    ``<Name>_TEST.ts``. The UEA archive only defines TRAIN and TEST splits, so —
    following the TSLib convention — the TEST split doubles as the validation set.

    A ``context_points`` of 0 (or less) means auto-detect: the padded length becomes
    the longest series across both splits, and ``params.context_points`` is updated
    accordingly so the model is built with the right number of patches.
    """
    root_path = root_path or getattr(params, 'root_path', None)
    if not root_path:
        raise ValueError("UEA datasets need --root_path pointing at the directory "
                         "holding <Name>_TRAIN.ts / <Name>_TEST.ts")

    train_files = glob.glob(os.path.join(root_path, '*_TRAIN.ts'))
    if len(train_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one <Name>_TRAIN.ts in {root_path!r} "
            f"(found {len(train_files)}). Download a UEA classification dataset from "
            "the Time-Series-Library and place its .ts files in that directory."
        )
    name = os.path.basename(train_files[0])[:-len('_TRAIN.ts')]

    train_X, train_y, class_names = _parse_ts_file(train_files[0])
    test_X, test_y, test_class_names = _parse_ts_file(os.path.join(root_path, f'{name}_TEST.ts'))
    if test_class_names != class_names:
        raise ValueError(f"Class labels differ between the TRAIN and TEST files of {name}")

    max_len = params.context_points
    if not max_len or max_len <= 0:
        max_len = max(len(x) for x in train_X + test_X)
        params.context_points = max_len
        print(f'UEA dataset {name}: {len(train_X)} train / {len(test_X)} test samples, '
              f'{len(class_names)} classes, auto-detected context_points = {max_len}')

    train_ds = UEADataset(train_X, train_y, max_len)
    test_ds = UEADataset(test_X, test_y, max_len, mean=train_ds.mean, std=train_ds.std)

    batch_size, workers = params.batch_size, params.num_workers
    dls = _SimpleDLS(
        train=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
        valid=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
        test=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers),
    )
    dls.vars = train_ds.n_vars
    dls.len = max_len
    dls.num_classes = len(class_names)
    dls.dataset_name = name
    return dls
