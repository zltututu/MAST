#!/usr/bin/env python
"""MAST self-supervised pretraining on ETTh1.

Example:
    python pretrain.py --n_epochs 20 --lr 1e-3
"""

import argparse

from mast.cli import add_common_args, ensure_save_dir
from mast.pipeline import run_pretrain


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAST self-supervised pretraining',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)

    mask = parser.add_argument_group('masking')
    mask.add_argument('--patch_mask_ratio', type=float, default=0.2,
                      help='fraction of patches removed in the time view')
    mask.add_argument('--point_mask_ratio', type=float, default=0.3,
                      help='fraction of points removed inside each surviving patch')
    mask.add_argument('--freq_mask_min', type=float, default=0.0,
                      help='lower bound of the frequency mask ratio')
    mask.add_argument('--freq_mask_max', type=float, default=0.7,
                      help='upper bound of the frequency mask ratio')
    mask.add_argument('--mask_seed', type=int, default=None,
                      help='seed for the masking RNG (None: non-deterministic masks)')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_name', type=str, default='mast_pretrain_etth1')

    return ensure_save_dir(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    print('pretraining configuration:')
    for key in sorted(vars(args)):
        print(f'  {key}: {getattr(args, key)}')

    learn = run_pretrain(args)
    print(f'\npretraining done, checkpoint: {args.save_dir}/{args.save_name}.pth')
