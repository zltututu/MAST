#!/usr/bin/env python
"""MAST downstream forecasting on ETTh1: linear probing or end-to-end finetuning.

Examples:
    python forecast.py --finetune_mode linear_probe --pretrained_model saved_models/mast_pretrain_etth1.pth
    python forecast.py --finetune_mode finetune    --pretrained_model saved_models/mast_pretrain_etth1.pth
"""

import argparse

from mast.cli import add_common_args, ensure_save_dir
from mast.pipeline import run_forecast


def parse_args():
    parser = argparse.ArgumentParser(
        description='MAST downstream forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)

    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='checkpoint produced by pretrain.py')
    parser.add_argument('--finetune_mode', type=str, default='linear_probe',
                        choices=['linear_probe', 'finetune'],
                        help='linear_probe: train the head only; finetune: unfreeze everything')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='peak learning rate of the one-cycle schedule')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='finetune mode only: head-only epochs before unfreezing')
    parser.add_argument('--handle_missing', type=int, default=1, choices=[0, 1],
                        help='substitute NaN inputs with the learned patch_token')
    parser.add_argument('--save_name', type=str, default=None,
                        help='defaults to mast_<finetune_mode>_etth1')

    args = ensure_save_dir(parser.parse_args())
    if args.save_name is None:
        args.save_name = f'mast_{args.finetune_mode}_etth1'
    return args


if __name__ == '__main__':
    args = parse_args()
    print('forecasting configuration:')
    for key in sorted(vars(args)):
        print(f'  {key}: {getattr(args, key)}')

    run_forecast(args)
