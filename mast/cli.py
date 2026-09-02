"""Argument definitions shared by `pretrain.py` and `forecast.py`.

Both stages must agree on the architecture hyper-parameters, otherwise the pretrained
checkpoint will not line up with the downstream model; they are therefore defined once here.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_common_args(parser):
    """Add the arguments that both stages share."""
    # --- data -------------------------------------------------------------
    data = parser.add_argument_group('data')
    data.add_argument('--data', type=str, default='ETTh1',
                      choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom', 'UEA'],
                      help='dataset flag: the four ETT variants use their fixed split; '
                           'custom = any other TSLib forecasting CSV (weather, electricity, '
                           'traffic, exchange_rate, national_illness, ...); UEA = the TSLib '
                           'classification datasets (.ts format)')
    data.add_argument('--root_path', type=str,
                      default=os.path.join(_REPO_ROOT, 'dataset', 'ETT-small'),
                      help='directory holding the dataset file')
    data.add_argument('--data_path', type=str, default='ETTh1.csv',
                      help='dataset file name (ignored for --data UEA)')
    data.add_argument('--target', type=str, default='OT',
                      help='target variate, used with --features S and by Dataset_Custom')
    data.add_argument('--context_points', type=int, default=336,
                      help='look-back window length (seq_len); 0 = auto-detect (UEA only)')
    data.add_argument('--target_points', type=int, default=96,
                      help='forecast horizon (pred_len); ignored by classify.py')
    data.add_argument('--batch_size', type=int, default=64)
    data.add_argument('--num_workers', type=int, default=0)
    data.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'],
                      help='M: all variates, S: only --target, MS: all inputs, --target output')

    # --- patching ---------------------------------------------------------
    patch = parser.add_argument_group('patching')
    patch.add_argument('--patch_len', type=int, default=16)
    patch.add_argument('--stride', type=int, default=8)
    patch.add_argument('--revin', type=int, default=1, choices=[0, 1],
                       help='apply reversible instance normalisation')

    # --- model ------------------------------------------------------------
    model = parser.add_argument_group('model')
    model.add_argument('--n_layers', type=int, default=3)
    model.add_argument('--n_heads', type=int, default=16)
    model.add_argument('--d_model', type=int, default=128)
    model.add_argument('--d_ff', type=int, default=256)
    model.add_argument('--dropout', type=float, default=0.2)
    model.add_argument('--head_dropout', type=float, default=0.2)

    # --- optimisation -----------------------------------------------------
    opt = parser.add_argument_group('optimisation')
    opt.add_argument('--n_epochs', type=int, default=20)
    opt.add_argument('--seed', type=int, default=2021)
    opt.add_argument('--require_cuda', action='store_true',
                     help='fail instead of falling back to CPU when no GPU is visible')

    # --- output -----------------------------------------------------------
    out = parser.add_argument_group('output')
    out.add_argument('--save_dir', type=str, default=os.path.join('saved_models'),
                     help='directory where checkpoints and metrics are written')
    return parser


def ensure_save_dir(args):
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_path = args.save_dir
    return args
