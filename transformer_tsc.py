"""Small, fully supervised transformer for fixed-length time-series classification.

Inspired by Zerveas et al., KDD 2021: https://arxiv.org/abs/2010.02803
Reference implementation: https://github.com/gzerveas/mvts_transformer
This is a lightweight adaptation, not a reproduction: pre-LayerNorm encoder,
fixed sinusoidal positions, and mean pooling instead of a flattened head.

Each non-overlapping patch is a token; attention is global and bidirectional.
Patching adds local grouping but no convolutions. Set patch_size=1 to use every
timestep as a token. The final partial patch is zero-padded. Attention cost is
quadratic in token count, so few parameters do not guarantee fast long-series training.
"""

import math

import torch
from torch import nn

from sequence_lengths import trim_to_lengths


class TransformerTSC(nn.Module):
    """Accept (batch, length) or (batch, channels, length); return class logits.

    ``input_shape`` is (channels, length), matching the ResNet constructor.
    With lengths, ignore padding and accept inputs up to the configured maximum length.
    Defaults use ~18k parameters plus the class head for univariate input.
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        patch_size=16,
        length_aware=False,
    ):
        super().__init__()
        if len(input_shape) != 2 or any(n <= 0 for n in input_shape):
            raise ValueError("input_shape must be (channels, length) with positive sizes")
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if num_classes <= 0 or num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("num_classes, num_layers, and dim_feedforward must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self.length_aware = length_aware
        self.patch_size = patch_size
        self.num_tokens = math.ceil(input_shape[1] / patch_size)
        self.input_shape = tuple(input_shape)
        self.model_config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "patch_size": patch_size,
        }
        self.input_projection = nn.Linear(input_shape[0] * patch_size, d_model)
        position = torch.arange(self.num_tokens, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, self.num_tokens, d_model)
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        encoding[0, :, 1::2] = torch.cos(position * frequency[: d_model // 2])
        self.register_buffer("position_encoding", encoding)
        self.input_dropout = nn.Dropout(dropout)
        # Construct layers independently rather than cloning identical initial weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != self.input_shape[0]:
            raise ValueError("Expected (batch, channels, length) or (batch, length)")
        if self.length_aware and lengths is None:
            raise ValueError("This model requires original frame lengths")
        if lengths is None:
            if x.shape[-1] != self.input_shape[1]:
                raise ValueError("Unmasked input must have the configured length")
        else:
            if x.shape[-1] > self.input_shape[1]:
                raise ValueError("Input exceeds the configured maximum length")
            x, _ = trim_to_lengths(x, lengths)
        num_tokens = math.ceil(x.shape[-1] / self.patch_size)
        padding = num_tokens * self.patch_size - x.shape[-1]
        x = nn.functional.pad(x, (0, padding))
        x = x.unfold(-1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], num_tokens, -1)
        x = self.input_projection(x)
        x = self.input_dropout(x + self.position_encoding[:, :num_tokens].to(dtype=x.dtype))
        padding_mask = None
        if lengths is not None:
            # Keep the final partial patch; its missing samples were zeroed above.
            token_lengths = (lengths.to(x.device) + self.patch_size - 1) // self.patch_size
            padding_mask = torch.arange(num_tokens, device=x.device)[None, :] >= token_lengths[:, None]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            x = x.masked_fill(padding_mask[:, :, None], 0)
            pooled = x.sum(dim=1) / (~padding_mask).sum(dim=1, keepdim=True)
        return self.head(pooled)
