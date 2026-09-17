"""Length-aware patch Transformer for packet time-series classification."""
import math
import torch
from torch import nn


class TransformerTSC(nn.Module):
    """Classify ``(B, 1, 3000)`` packets, optionally ignoring padded suffixes."""
    def __init__(self, input_shape, num_classes, patch_size=16, d_model=32, nhead=4,
                 num_layers=2, dim_feedforward=64, dropout=.2, length_aware=True):
        super().__init__()
        if len(input_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in input_shape):
            raise ValueError("input_shape must contain positive (channels, length)")
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0: raise ValueError("patch_size must be a positive integer")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (num_classes, d_model, nhead, num_layers, dim_feedforward)):
            raise ValueError("transformer dimensions must be positive integers")
        if d_model % nhead: raise ValueError("d_model must be divisible by nhead")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1: raise ValueError("dropout must be in [0, 1)")
        self.input_shape, self.patch_size, self.length_aware = tuple(input_shape), patch_size, length_aware
        self.num_tokens = math.ceil(input_shape[1] / patch_size)
        self.input_projection = nn.Linear(input_shape[0] * patch_size, d_model)
        positions = torch.arange(self.num_tokens, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.) / d_model))
        pe = torch.zeros(1, self.num_tokens, d_model); pe[0, :, 0::2] = torch.sin(positions * div)
        pe[0, :, 1::2] = torch.cos(positions * div[:d_model // 2])
        self.register_buffer("position_encoding", pe)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                      activation="gelu", batch_first=True, norm_first=True) for _ in range(num_layers)])
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        if x.ndim == 2: x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != self.input_shape[0]: raise ValueError("expected (batch, channels, length)")
        if x.shape[-1] > self.input_shape[1]: raise ValueError("input length cannot exceed configured length")
        if self.length_aware and lengths is None: raise ValueError("lengths are required when length_aware is enabled")
        if lengths is None:
            if x.shape[-1] != self.input_shape[1]: raise ValueError("fixed-width input must have configured length")
        else:
            if lengths.ndim != 1 or len(lengths) != len(x) or lengths.dtype not in (torch.int32, torch.int64): raise ValueError("lengths must be one integer per example")
            if torch.any(lengths < 1) or torch.any(lengths > x.shape[-1]): raise ValueError("lengths must be within input width")
            width = int(lengths.max().item()); x = x[..., :width]
            valid = torch.arange(width, device=x.device)[None] < lengths.to(x.device)[:, None]
            x = x.masked_fill(~valid[:, None], 0)
        tokens = math.ceil(x.shape[-1] / self.patch_size)
        x = nn.functional.pad(x, (0, tokens * self.patch_size - x.shape[-1]))
        x = x.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3).reshape(len(x), tokens, -1)
        x = self.input_dropout(self.input_projection(x) + self.position_encoding[:, :tokens].to(x.dtype))
        mask = None
        if lengths is not None:
            token_lengths = (lengths.to(x.device) + self.patch_size - 1) // self.patch_size
            mask = torch.arange(tokens, device=x.device)[None] >= token_lengths[:, None]
        for layer in self.layers: x = layer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        if mask is None: pooled = x.mean(1)
        else: pooled = x.masked_fill(mask[:, :, None], 0).sum(1) / (~mask).sum(1, keepdim=True)
        return self.head(pooled)
