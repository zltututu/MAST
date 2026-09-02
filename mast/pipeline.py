"""Shared plumbing for the two entry points: model construction, checkpoint transfer, run bookkeeping."""

import os

import pandas as pd
import torch

from .basics import select_device, set_seed
from .callback.revin import RevInCB
from .callback.tracking import SaveModelCB
from .data import get_dls
from .learner import Learner
from .masking import MissingTokenCB, PatchCB, TimeFreqSequentialMaskCB
from .metrics import mae, mse
from .model import MAST, MASTModel


def num_patches(seq_len, patch_len, stride):
    """Number of whole patches a look-back window of `seq_len` is split into."""
    if seq_len < patch_len:
        raise ValueError(f"context_points ({seq_len}) must be at least patch_len ({patch_len})")
    return (seq_len - patch_len) // stride + 1


def build_model(c_in, args, head_type='pretrain') -> MASTModel:
    """Build the MAST backbone with the requested head and wrap it with the MAST tokens."""
    backbone = MAST(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patches(args.context_points, args.patch_len, args.stride),
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type=head_type,
    )
    return MASTModel(backbone, patch_len=args.patch_len)


def load_pretrained_weights(model, weight_path, exclude_head=True, device='cpu'):
    """Copy the matching backbone weights of `weight_path` into `model`.

    Keys are matched by name, so the pretraining-only modules (`ss_proj`, `ss_pred`,
    `ss_mproj`) and the randomly initialised prediction head are simply skipped. `W_pos`
    is the one tensor whose size depends on the look-back length, so it is truncated or
    zero-padded when the pretraining and downstream windows differ.
    """
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = model.state_dict()
    matched, unmatched = 0, []

    for name, param in state_dict.items():
        if exclude_head and 'head' in name:
            continue
        if name not in checkpoint:
            unmatched.append(name)
            continue

        source = checkpoint[name]
        matched += 1
        if source.shape == param.shape:
            param.copy_(source)
        elif 'W_pos' in name and source.dim() == 2 and param.dim() == 2:
            # W_pos holds one row per patch: crop or pad with zeros
            if source.shape[0] > param.shape[0]:
                param.copy_(source[:param.shape[0], :])
                print(f'  W_pos cropped: {tuple(source.shape)} -> {tuple(param.shape)} '
                      f'(pretrain num_patch={source.shape[0]}, downstream num_patch={param.shape[0]})')
            else:
                padding = torch.zeros(param.shape[0] - source.shape[0], param.shape[1],
                                      dtype=source.dtype, device=source.device)
                param.copy_(torch.cat([source, padding], dim=0))
                print(f'  W_pos zero-padded: {tuple(source.shape)} -> {tuple(param.shape)} '
                      f'(pretrain num_patch={source.shape[0]}, downstream num_patch={param.shape[0]})')
        else:
            unmatched.append(f'{name} (shape mismatch: {tuple(param.shape)} vs {tuple(source.shape)})')

    if matched == 0:
        raise RuntimeError(
            f'No shared weights between the checkpoint and the model. '
            f'Model keys (first 10): {list(state_dict)[:10]}; '
            f'checkpoint keys (first 10): {list(checkpoint)[:10]}'
        )
    if unmatched:
        print(f'  Layers left at their initialisation: {unmatched[:10]}')
    return model.to(device)


def save_losses(learn, path):
    """Write the per-epoch train/validation loss to CSV."""
    pd.DataFrame({'train_loss': learn.recorder['train_loss'],
                  'valid_loss': learn.recorder['valid_loss']}).to_csv(path, float_format='%.6f', index=False)


def run_pretrain(args):
    """Self-supervised pretraining on ETTh1. Returns the fitted Learner."""
    set_seed(args.seed)
    device = select_device(require_cuda=args.require_cuda)
    print(f'device: {device}')

    dls = get_dls(args)
    print(f'train/val/test batches: {len(dls.train)}/{len(dls.valid)}/{len(dls.test)}, variates: {dls.vars}')

    model = build_model(dls.vars, args, head_type='pretrain').to(device)
    print(f'trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        TimeFreqSequentialMaskCB(
            patch_len=args.patch_len,
            stride=args.stride,
            patch_mask_ratio=args.patch_mask_ratio,
            point_mask_ratio=args.point_mask_ratio,
            freq_mask_min=args.freq_mask_min,
            freq_mask_max=args.freq_mask_max,
            mask_seed=args.mask_seed,
        ),
        SaveModelCB(monitor='valid_loss', fname=args.save_name, path=args.save_path),
    ]

    learn = Learner(dls, model, torch.nn.MSELoss(reduction='mean'), lr=args.lr, cbs=cbs, device=device)
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=args.lr)

    save_losses(learn, os.path.join(args.save_path, f'{args.save_name}_losses.csv'))
    return learn


def run_forecast(args):
    """Downstream forecasting: linear probe or end-to-end finetuning of a pretrained MAST."""
    set_seed(args.seed)
    device = select_device(require_cuda=args.require_cuda)
    print(f'device: {device}')

    dls = get_dls(args)
    print(f'train/val/test batches: {len(dls.train)}/{len(dls.valid)}/{len(dls.test)}, variates: {dls.vars}')

    model = build_model(dls.vars, args, head_type='prediction').to(device)
    model = load_pretrained_weights(model, args.pretrained_model, exclude_head=True, device=device)

    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    if args.handle_missing:
        cbs += [MissingTokenCB(model.patch_token.data)]
    cbs += [SaveModelCB(monitor='valid_loss', fname=args.save_name, path=args.save_path)]

    learn = Learner(dls, model, torch.nn.MSELoss(reduction='mean'), lr=args.lr, cbs=cbs,
                    metrics=[mse], device=device)

    if args.finetune_mode == 'linear_probe':
        learn.linear_probe(n_epochs=args.n_epochs, base_lr=args.lr)
    else:
        learn.fine_tune(n_epochs=args.n_epochs, base_lr=args.lr, freeze_epochs=args.freeze_epochs)

    save_losses(learn, os.path.join(args.save_path, f'{args.save_name}_losses.csv'))

    weight_path = os.path.join(args.save_path, f'{args.save_name}.pth')
    _, _, scores = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])
    mse_score, mae_score = float(scores[0]), float(scores[1])

    print(f'\n[{args.finetune_mode}] test MSE: {mse_score:.6f}   MAE: {mae_score:.6f}   '
          f'RMSE: {mse_score ** 0.5:.6f}')
    pd.DataFrame({'mse': [mse_score], 'mae': [mae_score], 'rmse': [mse_score ** 0.5]}).to_csv(
        os.path.join(args.save_path, f'{args.save_name}_acc.csv'), float_format='%.6f', index=False
    )
    return learn, (mse_score, mae_score)
