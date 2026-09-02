import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        # Handle NaN and Inf values: use nanmean and nanvar to ignore NaN when computing statistics
        # If all values are NaN, fall back to 0 mean and 1 stdev
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        
        if has_nan or has_inf:
            # Compute statistics ignoring NaN and Inf
            # Replace NaN and Inf with 0 for computation
            x_valid = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
            valid_mask = ~(torch.isnan(x) | torch.isinf(x))
            valid_count = valid_mask.sum(dim=dim2reduce, keepdim=True).float()
            
            # If all values are NaN/Inf (valid_count == 0), use 0 mean and 1 stdev
            self.mean = torch.where(
                valid_count > 0,
                (x_valid.sum(dim=dim2reduce, keepdim=True) / valid_count),
                torch.zeros_like(valid_count)
            ).detach()
            
            # Ensure mean is finite
            self.mean = torch.where(torch.isfinite(self.mean), self.mean, torch.zeros_like(self.mean))
            
            x_centered = x_valid - self.mean
            x_centered = torch.where(valid_mask, x_centered, torch.zeros_like(x_centered))
            variance = torch.where(
                valid_count > 0,
                (x_centered ** 2).sum(dim=dim2reduce, keepdim=True) / valid_count,
                torch.ones_like(valid_count)  # If all NaN/Inf, use variance = 1
            )
            # Ensure variance is finite and non-negative
            variance = torch.clamp(variance, min=0.0)
            variance = torch.where(torch.isfinite(variance), variance, torch.ones_like(variance))
            
            self.stdev = torch.sqrt(variance + self.eps).detach()
            # Ensure stdev is finite, not too small, and not too large
            self.stdev = torch.clamp(self.stdev, min=self.eps, max=1e6)
            self.stdev = torch.where(torch.isfinite(self.stdev), self.stdev, torch.ones_like(self.stdev))
        else:
            # Normal computation when no NaN/Inf
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
            self.stdev = torch.sqrt(variance + self.eps).detach()
            # Ensure stdev is not too small to avoid division by very small numbers
            self.stdev = torch.clamp(self.stdev, min=self.eps)
            # Ensure mean and stdev are finite
            self.mean = torch.where(torch.isfinite(self.mean), self.mean, torch.zeros_like(self.mean))
            self.stdev = torch.where(torch.isfinite(self.stdev), self.stdev, torch.ones_like(self.stdev))

    def _normalize(self, x):
        # Store NaN and Inf mask before normalization
        nan_inf_mask = torch.isnan(x) | torch.isinf(x)
        
        # Normalize only valid (non-NaN, non-Inf) values
        # For NaN/Inf positions, we'll preserve them so MissingTokenCB can handle them
        x_normalized = x.clone()
        valid_mask = ~nan_inf_mask
        
        # Only normalize valid positions
        if valid_mask.any():
            x_normalized = torch.where(
                valid_mask,
                (x - self.mean) / self.stdev,
                x  # Keep NaN/Inf as is
            )
            
            # Apply affine transformation if enabled (only to valid positions)
            if self.affine:
                x_normalized = torch.where(
                    valid_mask,
                    x_normalized * self.affine_weight + self.affine_bias,
                    x_normalized  # Keep NaN/Inf as is
                )
        
        # Ensure we didn't accidentally create new NaN/Inf from normalization
        # Check if any valid values became NaN/Inf after normalization
        new_nan_inf = (~nan_inf_mask) & (torch.isnan(x_normalized) | torch.isinf(x_normalized))
        if new_nan_inf.any():
            # If normalization created new NaN/Inf, keep original NaN/Inf positions
            # and set new problematic positions to NaN (will be handled by MissingTokenCB)
            x_normalized = torch.where(new_nan_inf, torch.tensor(float('nan'), device=x.device, dtype=x.dtype), x_normalized)
        
        # Preserve original NaN/Inf positions
        x_normalized = torch.where(nan_inf_mask, x, x_normalized)
        
        return x_normalized

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
