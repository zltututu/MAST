"""ETTh1 data loading for MAST.

ETTh1 is the hourly Electricity Transformer Temperature dataset: 17,420 hourly steps
over 7 variates (6 load features plus the oil temperature `OT`). Batches are returned as
`(x, y)` float tensors with shapes `[seq_len x n_vars]` and `[seq_len x n_vars]`.
"""

import os

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

__all__ = ['Dataset_ETT_hour', 'DataLoaders', 'get_dls', 'DATASET_ROOT', 'DATASET_PATH']

# ETTh1 lives in ./dataset/ETT-small/ relative to the repository root
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'ETT-small')
DATASET_PATH = 'ETTh1.csv'


class Dataset_ETT_hour(Dataset):
    """Sliding-window ETTh1 dataset with the standard 12/4/4 month train/val/test split.

    Args:
        root_path: directory holding the CSV file.
        split: one of ``train``, ``val``, ``test``.
        size: ``[seq_len, label_len, pred_len]``; `label_len` is kept for compatibility
            with the TSLib-style signature and is subtracted from the target window start.
        features: ``M`` uses every variate, ``S`` only `target`.
    """

    def __init__(self, root_path=DATASET_ROOT, split='train', size=None,
                 features='M', data_path=DATASET_PATH, target='OT', scale=True):
        self.seq_len, self.label_len, self.pred_len = size if size is not None else (384, 96, 96)
        assert split in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[split]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 12 / 4 / 4 months of hourly data
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"features must be one of M, MS, S; got {self.features!r}")

        if self.scale:
            # fit the scaler on the training split only
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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


def get_dls(params, root_path=None):
    """Build the DataLoaders for ETTh1 and annotate them with `vars`, `len` and `c`.

    `vars` is the number of variates and `len` the look-back length; both are needed to
    build the model before the first batch is seen.
    """
    if not os.path.exists(os.path.join(root_path or DATASET_ROOT, DATASET_PATH)):
        raise FileNotFoundError(
            f"ETTh1 not found at {os.path.join(root_path or DATASET_ROOT, DATASET_PATH)}. "
            "Download it from the Time-Series-Library dataset folder and place it there."
        )

    dls = DataLoaders(
        datasetCls=Dataset_ETT_hour,
        dataset_kwargs={
            'root_path': root_path or DATASET_ROOT,
            'data_path': DATASET_PATH,
            'features': params.features,
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
