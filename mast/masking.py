"""Patching and the MAST self-supervised masking strategy.

The pretext task builds two corrupted views of every window and asks the encoder to
reconstruct the clean signal from both:

1. a **time view** - a hybrid mask that drops whole patches *and* individual points,
   with every dropped position replaced by the learnable `patch_token`;
2. a **frequency view** - the time-masked signal with a random subset of its rFFT
   coefficients zeroed out, again with masked positions replaced by `patch_token`.

Both views are encoded, and on top of the reconstruction loss the latent of the
frequency view is asked to predict the latent of the time view (with a covariance
regulariser keeping the projected latents from collapsing).
"""

import torch
from torch import nn

from .callback.core import Callback

__all__ = ['create_patch', 'PatchCB', 'MissingTokenCB', 'TimeFreqSequentialMaskCB',
           'hybrid_masking', 'frequency_masking', 'FreqMaskProj']


def create_patch(xb, patch_len, stride):
    """Split [bs x seq_len x n_vars] into [bs x num_patch x n_vars x patch_len].

    Leading time steps that do not fit into a whole patch are dropped, so the number of
    patches is deterministic given the look-back length, patch length and stride.
    """
    seq_len = xb.shape[1]
    if seq_len < patch_len:
        return xb.unfold(dimension=1, size=patch_len, step=stride), 0

    num_patch = (seq_len - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    remainder = seq_len - tgt_len
    if remainder > 0:
        xb = xb[:, remainder:, :]
    return xb.unfold(dimension=1, size=patch_len, step=stride), num_patch


class PatchCB(Callback):
    """Convert each batch into patches before the forward pass."""

    def __init__(self, patch_len, stride):
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self):
        xb_patch, _ = create_patch(self.xb, self.patch_len, self.stride)
        self.learner.xb = xb_patch


class MissingTokenCB(Callback):
    """Replace NaN/Inf values in patched data with the learnable `patch_token`.

    Used downstream so that a model pretrained with `patch_token` sees the same
    placeholder for genuine missing values. Must run *after* `PatchCB`.
    """

    def __init__(self, patch_token: torch.Tensor):
        self.patch_token = patch_token.data.clone() if isinstance(patch_token, nn.Parameter) else patch_token.clone()

    def before_forward(self):
        xb_patch = self.learner.xb
        if xb_patch.dim() != 4:
            raise ValueError(f"MissingTokenCB expects patched 4D input, got shape {tuple(xb_patch.shape)}")

        missing_mask = torch.isnan(xb_patch) | torch.isinf(xb_patch)
        if not missing_mask.any():
            return

        token = self.patch_token.view(1, 1, 1, -1).to(xb_patch.device)
        self.learner.xb = torch.where(missing_mask, token, xb_patch)


class FreqMaskProj(nn.Module):
    """Embed the frequency mask, normalised by the number of masked coefficients.

    ``M^f_vec = (M^f W^f) / sqrt(k)`` with ``k = number of ones in M^f``.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_f = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, M_f: torch.Tensor) -> torch.Tensor:
        M_f = M_f.float()
        k = M_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return self.W_f(M_f / torch.sqrt(k))


def _rand(shape, device, generator=None):
    kwargs = {"device": device}
    if generator is not None:
        kwargs["generator"] = generator
    try:
        return torch.rand(shape, **kwargs)
    except RuntimeError:
        if generator is None:
            raise
        return torch.rand(shape, generator=generator).to(device)


def hybrid_masking(xb, patch_mask_ratio, point_mask_ratio, replace_val=0.0, generator=None):
    """Hybrid time-domain masking.

    Args:
        xb: patched data [bs x num_patch x n_vars x patch_len].
        patch_mask_ratio: fraction of patches removed per (sample, variate).
        point_mask_ratio: fraction of points removed inside each surviving patch.

    Returns:
        ``(x_masked, mask_patch, mask_point, mask_combined)`` where the masks are boolean
        and share the shape of `xb` (patch mask broadcast over the patch length).
    """
    bs, L, nvars, D = xb.shape
    device = xb.device

    # patch-level masking: shuffle noise so that the kept patches are contiguous
    noise_patch = _rand((bs, L, nvars), device, generator=generator)
    ids_shuffle = torch.argsort(noise_patch, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    len_keep = int(L * (1 - patch_mask_ratio))

    mask_patch = torch.ones([bs, L, nvars], device=device)
    mask_patch[:, :len_keep, :] = 0
    mask_patch = torch.gather(mask_patch, dim=1, index=ids_restore)
    mask_patch_exp = mask_patch.unsqueeze(-1).expand(-1, -1, -1, D).bool()

    # point-level masking, only inside the surviving patches
    num_points_to_mask = int(D * point_mask_ratio)
    if num_points_to_mask > 0:
        point_noise = _rand((bs, L, nvars, D), device, generator=generator)
        num_keep = D - num_points_to_mask
        _, ids_keep = torch.topk(point_noise, num_keep, dim=-1, largest=False)

        mask_point = torch.ones([bs, L, nvars, D], device=device)
        bidx = torch.arange(bs, device=device).view(-1, 1, 1, 1).expand(-1, L, nvars, num_keep)
        pidx = torch.arange(L, device=device).view(1, -1, 1, 1).expand(bs, -1, nvars, num_keep)
        vidx = torch.arange(nvars, device=device).view(1, 1, -1, 1).expand(bs, L, -1, num_keep)
        mask_point[bidx, pidx, vidx, ids_keep] = 0
        mask_point = (mask_point * (~mask_patch_exp)).bool()
    else:
        mask_point = torch.zeros([bs, L, nvars, D], device=device, dtype=torch.bool)

    combined_mask = (mask_patch_exp | mask_point).bool()
    return xb.masked_fill(combined_mask, replace_val), mask_patch_exp, mask_point, combined_mask


def frequency_masking(xb, freq_mask_min=0.1, freq_mask_max=0.5, generator=None):
    """Zero out a random subset of rFFT coefficients.

    Args:
        xb: patched data [bs x num_patch x n_vars x patch_len].
        freq_mask_min / freq_mask_max: per-sample mask ratio is drawn uniformly from this range.

    Returns:
        ``(x_masked, freq_mask)`` with `freq_mask` of shape [bs x L_fft].
    """
    bs, P, C, Lp = xb.shape
    L_seq = P * Lp
    device = xb.device

    x_seq = xb.permute(0, 2, 1, 3).reshape(bs, C, L_seq)
    Xf = torch.fft.rfft(x_seq, dim=-1)
    L_fft = Xf.size(-1)

    r = _rand((bs, 1), device, generator=generator) * (freq_mask_max - freq_mask_min) + freq_mask_min
    freq_mask = _rand((bs, L_fft), device, generator=generator) < r

    Xf_masked = torch.where(freq_mask.unsqueeze(1), torch.zeros_like(Xf), Xf)
    x_masked_seq = torch.fft.irfft(Xf_masked, n=L_seq, dim=-1)
    xb_masked = x_masked_seq.reshape(bs, C, P, Lp).permute(0, 2, 1, 3)

    return xb_masked, freq_mask.bool()


class TimeFreqSequentialMaskCB(Callback):
    """Build the two MAST views and own the pretraining objective.

    The callback takes over the Learner's loss function in `before_fit`, so it has to be
    listed among the callbacks of any pretraining run.
    """

    def __init__(self, patch_len, stride, patch_mask_ratio, point_mask_ratio,
                 freq_mask_min=0.0, freq_mask_max=0.7, proj_dim=128,
                 lambda_rec=1.0, lambda_pred=1.0, lambda_sig=0.01, mask_seed=None):
        self.patch_len = patch_len
        self.stride = stride
        self.patch_mask_ratio = patch_mask_ratio
        self.point_mask_ratio = point_mask_ratio
        self.freq_mask_min = freq_mask_min
        self.freq_mask_max = freq_mask_max
        self.proj_dim = proj_dim
        self.lambda_rec = float(lambda_rec)
        self.lambda_pred = float(lambda_pred)
        self.lambda_sig = float(lambda_sig)
        self.mask_seed = mask_seed
        self.mask_generator = None

    def before_fit(self):
        self.learner.loss_func = self._loss
        model = self.learner.model
        if not hasattr(model, "patch_token"):
            raise AttributeError("the model must expose `patch_token` for MAST masking")

        self.patch_token = model.patch_token
        device = self.patch_token.device
        d_model = model.encoder.backbone.d_model
        self.num_patch = model.encoder.backbone.num_patch
        self.L_fft = (self.num_patch * self.patch_len) // 2 + 1

        # latent projector, latent predictor and frequency-mask embedding
        self.H_proj = nn.Sequential(
            nn.Linear(d_model, self.proj_dim, bias=False),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
        ).to(device)
        self.H_pred = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim, bias=False),
        ).to(device)
        self.M_proj = FreqMaskProj(self.L_fft, self.proj_dim).to(device)

        # register so they are optimised and saved with the model
        model.add_module('ss_proj', self.H_proj)
        model.add_module('ss_pred', self.H_pred)
        model.add_module('ss_mproj', self.M_proj)

        if self.mask_seed is not None:
            try:
                self.mask_generator = torch.Generator(device=device)
            except (TypeError, RuntimeError):
                self.mask_generator = torch.Generator()
            self.mask_generator.manual_seed(int(self.mask_seed))

    def before_forward(self):
        self.patch_masking()

    def patch_masking(self):
        xb_patch, _ = create_patch(self.xb, self.patch_len, self.stride)

        # genuine missing values (NaN) are zeroed before the FFT and remembered, so the
        # reconstruction loss is only computed on positions we actually masked ourselves
        original_missing_mask = torch.isnan(xb_patch)
        if original_missing_mask.any():
            xb_patch = torch.where(original_missing_mask, torch.zeros_like(xb_patch), xb_patch)

        xb_time_masked, _, _, mask_t = hybrid_masking(
            xb_patch, self.patch_mask_ratio, self.point_mask_ratio,
            replace_val=0.0, generator=self.mask_generator,
        )
        xb_freq_masked, mask_f = frequency_masking(
            xb_time_masked, freq_mask_min=self.freq_mask_min, freq_mask_max=self.freq_mask_max,
            generator=self.mask_generator,
        )

        # both views share the same learnable placeholder at every missing position
        all_missing_mask = mask_t | original_missing_mask
        token = self.patch_token.view(1, 1, 1, -1).to(xb_patch.device)
        xb_time_masked = torch.where(all_missing_mask, token, xb_time_masked)
        xb_freq_masked = torch.where(all_missing_mask, token, xb_freq_masked)

        self.xb_patch = xb_patch
        self.mask_t = mask_t
        self.mask_f = mask_f
        self.original_missing_mask = original_missing_mask

        # hand the missingness map to the model for missing-state-aware patch dropping;
        # the two views are concatenated along the batch axis, so the mask is too
        self.learner.model.current_missing_mask = torch.cat([mask_t.float(), mask_t.float()], dim=0)

        self.learner.xb = torch.cat([xb_time_masked, xb_freq_masked], dim=0)
        self.learner.yb = torch.cat([xb_patch, xb_patch], dim=0)

    def _loss(self, preds, target):
        z, out = preds
        z_time, z_freq = z.chunk(2, dim=0)
        out_time, _ = out.chunk(2, dim=0)

        # reconstruction on the positions we masked ourselves
        valid_mask = self.mask_t & (~self.original_missing_mask)
        recon = (out_time - self.xb_patch) ** 2
        recon = (recon * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # latent prediction: frequency view -> time view (time branch is detached)
        H_t = self.H_proj(z_time.mean(1).mean(-1))
        H_f = self.H_proj(z_freq.mean(1).mean(-1))
        pred_loss = torch.nn.functional.mse_loss(self.H_pred(H_f + self.M_proj(self.mask_f.float())), H_t.detach())

        # covariance regulariser on the concatenated latents
        H_concat = torch.cat([H_t, H_f], dim=0)
        H_concat = H_concat - H_concat.mean(0, keepdim=True)
        cov = (H_concat.T @ H_concat) / H_concat.size(0)
        sig_loss = ((cov - torch.eye(self.proj_dim, device=cov.device)) ** 2).mean()

        return self.lambda_rec * recon + self.lambda_pred * (pred_loss + self.lambda_sig * sig_loss)
