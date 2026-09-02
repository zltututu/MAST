"""MAST model definition.

The architecture is a PatchTST-style channel-independent Transformer encoder
(`MASTEncoder`) with two interchangeable heads:

* `PretrainHead`   - reconstructs every masked patch, used during self-supervised pretraining.
* `PredictionHead` - maps the encoded patches to the forecast horizon, used downstream.

`MASTModel` wraps the encoder and adds the learnable parameters that MAST
introduces on top of PatchTST:

* `patch_token`  - learnable placeholder substituted for missing / masked values.
* `mem_token`    - two learnable memory patches prepended to the sequence and discarded afterwards.
* missing-state-aware patch dropping - a small MLP that reads the per-patch missing
  indicator and randomly zeroes out patches whose missingness is high (pretraining only).
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .layers.attention import MultiheadAttention
from .layers.basics import Transpose, get_activation_fn
from .layers.pos_encoding import positional_encoding

__all__ = ['MAST', 'MASTModel', 'MASTEncoder', 'PretrainHead', 'PredictionHead']


class MAST(nn.Module):
    """PatchTST-style encoder with a task head.

    Args:
        c_in: number of input variates.
        target_dim: forecast horizon (prediction head) - unused by the pretraining head.
        patch_len: number of time steps per patch.
        stride: stride between consecutive patches.
        num_patch: number of patches the look-back window is split into.
        head_type: one of ``pretrain`` or ``prediction``.
    """

    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, num_patch: int,
                 n_layers: int = 3, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 head_type="prediction", individual=False, verbose: bool = False, **kwargs):

        super().__init__()

        if head_type not in ('pretrain', 'prediction'):
            raise ValueError(f"head_type must be 'pretrain' or 'prediction', got {head_type!r}")

        self.backbone = MASTEncoder(c_in, num_patch=num_patch, patch_len=patch_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                    shared_embedding=shared_embedding, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act,
                                    res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout)
        else:
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)

    def forward(self, z):
        """`z`: [bs x num_patch x n_vars x patch_len]."""
        z = self.backbone(z)   # [bs x n_vars x d_model x num_patch]
        return self.head(z)


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """`x`: [bs x n_vars x d_model x num_patch] -> [bs x num_patch x n_vars x patch_len]."""
        x = x.transpose(2, 3)                  # [bs x n_vars x num_patch x d_model]
        x = self.linear(self.dropout(x))       # [bs x n_vars x num_patch x patch_len]
        return x.permute(0, 2, 1, 3)           # [bs x num_patch x n_vars x patch_len]


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        head_dim = d_model * num_patch

        if self.individual:
            self.flattens = nn.ModuleList()
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """`x`: [bs x n_vars x d_model x num_patch] -> [bs x forecast_len x n_vars]."""
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])   # [bs x d_model * num_patch]
                z = self.linears[i](z)                # [bs x forecast_len]
                x_out.append(self.dropouts[i](z))
            x = torch.stack(x_out, dim=1)             # [bs x n_vars x forecast_len]
        else:
            x = self.flatten(x)                       # [bs x n_vars x (d_model * num_patch)]
            x = self.linear(self.dropout(x))          # [bs x n_vars x forecast_len]
        return x.transpose(2, 1)                      # [bs x forecast_len x n_vars]


class MASTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Input encoding: project each patch onto the d_model dimensional space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, pre_norm=pre_norm, activation='gelu',
                                  res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x, pos_encoding: Optional[Tensor] = None) -> Tensor:
        """`x`: [bs x num_patch x n_vars x patch_len] -> [bs x n_vars x d_model x num_patch].

        `pos_encoding` overrides the learned positional encoding, which is how the memory
        tokens of `MASTModel` are prepended without changing the `W_pos` shape.
        """
        bs, num_patch, n_vars, _ = x.shape

        if not self.shared_embedding:
            x = torch.stack([self.W_P[i](x[:, :, i, :]) for i in range(n_vars)], dim=2)
        else:
            x = self.W_P(x)                                     # [bs x num_patch x n_vars x d_model]
        x = x.transpose(1, 2)                                   # [bs x n_vars x num_patch x d_model]

        u = torch.reshape(x, (bs * n_vars, num_patch, self.d_model))
        pos_emb = pos_encoding if pos_encoding is not None else self.W_pos
        u = self.dropout(u + pos_emb)                           # [bs * n_vars x num_patch x d_model]

        z = self.encoder(u)                                     # [bs * n_vars x num_patch x d_model]
        z = torch.reshape(z, (-1, n_vars, num_patch, self.d_model))
        return z.permute(0, 1, 3, 2)                            # [bs x n_vars x d_model x num_patch]


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn)
                                     for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor):
        """`src`: [bs x q_len x d_model]."""
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
            return output
        for mod in self.layers:
            output = mod(output)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads

        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_k, attn_dropout=attn_dropout,
                                            proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = _make_norm(norm, d_model)

        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = _make_norm(norm, d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        """`src`: [bs x q_len x d_model]."""
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src = src + self.dropout_ffn(self.ff(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


def _make_norm(norm: str, d_model: int) -> nn.Module:
    if "batch" in norm.lower():
        return nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
    return nn.LayerNorm(d_model)


class MASTModel(nn.Module):
    """MAST wrapper: `MAST` encoder plus the learnable tokens and missing-state patch dropping.

    Forward pass returns ``(z, out)`` for the pretraining head (the latent representation is
    needed by the self-supervised objective) and just ``out`` for the prediction head.
    """

    def __init__(self, encoder: MAST, patch_len: int, device=None,
                 use_missing_state_drop: bool = True, use_memory_token: bool = True):
        super().__init__()
        self.encoder = encoder
        self.patch_len = patch_len
        self.use_missing_state_drop = use_missing_state_drop
        self.use_memory_token = use_memory_token
        if device is None:
            device = next(encoder.parameters()).device

        self.patch_token = nn.Parameter(torch.randn(patch_len, device=device) * 0.01)
        self.mem_token = nn.Parameter(torch.randn(patch_len * 2, device=device) * 0.01)

        # Missing-state-aware patch dropping:
        #   m_n^(i) (length patch_len) -> W^t -> sigmoid(w^T . + b) -> Bernoulli drop
        d_model = getattr(self.encoder.backbone, "d_model", None)
        if d_model is None:
            raise AttributeError("encoder.backbone must expose `d_model` for patch dropping.")
        self.missing_state_proj = nn.Linear(patch_len, d_model, bias=False).to(device)
        self.drop_prob_head = nn.Linear(d_model, 1, bias=True).to(device)

        # Set externally by the masking callback. Shape
        # [bs x num_patch x n_vars x patch_len], 1 marks a missing/masked value.
        self.current_missing_mask = None

    def forward(self, x):
        """`x`: [bs x num_patch x n_vars x patch_len]."""
        bs, _, n_vars, _ = x.shape

        if (self.use_missing_state_drop and self.training
                and getattr(self.encoder, "head_type", None) == "pretrain"
                and self.current_missing_mask is not None):
            m = self.current_missing_mask.to(x.device)
            if m.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                m = m.float()
            m_embed = self.missing_state_proj(m)                       # [bs x num_patch x n_vars x d_model]
            p = torch.sigmoid(self.drop_prob_head(m_embed).squeeze(-1))  # [bs x num_patch x n_vars]
            keep = (1.0 - torch.bernoulli(p)).to(x.dtype)
            # never drop the first patch, otherwise a variate could lose all of its context
            keep[:, 0, :] = 1.0
            x = x * keep.unsqueeze(-1)

        if self.use_memory_token:
            mem = self.mem_token.view(1, 2, 1, self.patch_len).expand(bs, 2, n_vars, self.patch_len)
            x_cat = torch.cat([mem, x], dim=1)                         # [bs x num_patch + 2 x n_vars x patch_len]

            W_pos = self.encoder.backbone.W_pos                        # [num_patch x d_model]
            num_patch = self.encoder.backbone.num_patch
            zero_pe = torch.zeros(2, W_pos.size(1), dtype=W_pos.dtype, device=W_pos.device)
            extended_W_pos = torch.cat([zero_pe, W_pos], dim=0)        # [num_patch + 2 x d_model]

            self.encoder.backbone.num_patch = num_patch + 2
            z_all = self.encoder.backbone(x_cat, pos_encoding=extended_W_pos)
            self.encoder.backbone.num_patch = num_patch

            z = z_all[:, :, :, 2:]                                     # drop the memory positions
        else:
            z = self.encoder.backbone(x)

        out = self.encoder.head(z)
        return (z, out) if self.encoder.head_type == 'pretrain' else out
