"""Flat, video-aware patch Transformer for raw WPP ``row_bytes`` tensors.

This model deliberately has no dataset reader, normalization, encoder, or training
entry point.  It consumes the explicit masks produced by ``collate_wpp_videos``;
observed zeros are data, never an implicit padding signal.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TIME_STEPS = 3000


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _axis_sincos(length: int, dimension: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return standard sin/cos positions for one axis without a learned limit."""
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.empty(length, dimension, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


def two_dimensional_sincos_positions(
    rows: int, patches_per_row: int, d_model: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create ``[rows, patches_per_row, d_model]`` distinct row/time positions.

    Half the channels encode the WPP vertical row and half encode the temporal
    patch index.  Concatenation, rather than adding two identical encodings,
    keeps ``(row=a, time=b)`` distinct from ``(row=b, time=a)``.  Positions are
    generated for the actual input row count, so there is no learned maximum-R
    table or row truncation.
    """
    if d_model % 4:
        raise ValueError("d_model must be divisible by 4 for 2-D sin/cos positions")
    row_encoding = _axis_sincos(rows, d_model // 2, device=device, dtype=dtype)
    time_encoding = _axis_sincos(patches_per_row, d_model // 2, device=device, dtype=dtype)
    return torch.cat(
        (
            row_encoding[:, None, :].expand(rows, patches_per_row, -1),
            time_encoding[None, :, :].expand(rows, patches_per_row, -1),
        ),
        dim=-1,
    )


class WPPTransformer(nn.Module):
    """Classify one video from flat attention over all of its WPP row patches.

    Args:
        num_classes: Number of output classes.
        patch_size: Non-overlapping temporal width for each row.  The final
            temporal patch is right-zero-padded; patches never cross row bounds.
        d_model, nhead, num_layers, dim_feedforward, dropout: Original-style
            patch Transformer hyperparameters.

    ``forward(values, row_mask, time_mask)`` requires ``values`` shaped
    ``[batch, rows, 3000]`` plus Boolean prefix masks shaped ``[batch, rows]``
    and ``[batch, 3000]``.  A token is valid when its row is valid and its patch
    contains any valid time value.  All valid tokens of a video are flattened
    into one attention sequence and masked-mean pooled after the encoder.
    """

    time_steps = TIME_STEPS

    def __init__(
        self,
        num_classes: int,
        *,
        patch_size: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        _positive_integer(num_classes, "num_classes")
        _positive_integer(patch_size, "patch_size")
        for name, value in (("d_model", d_model), ("nhead", nhead), ("num_layers", num_layers),
                            ("dim_feedforward", dim_feedforward)):
            _positive_integer(value, name)
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if d_model % 4:
            raise ValueError("d_model must be divisible by 4 for 2-D sin/cos positions")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a real value in [0, 1)")

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.d_model = d_model
        self.patches_per_row = math.ceil(TIME_STEPS / patch_size)
        self.input_projection = nn.Linear(patch_size, d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _require_prefix(mask: Tensor, name: str) -> None:
        # The WPP collator creates a contiguous true prefix.  Enforcing this
        # catches a mask/data contract mismatch instead of silently reordering
        # the meaning of patches or padded rows.
        if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError(f"{name} must be a contiguous true prefix")

    def _validate_inputs(self, values: Tensor, row_mask: Tensor, time_mask: Tensor, *, normalized: bool) -> None:
        if not isinstance(values, Tensor) or values.ndim != 3 or values.shape[2] != TIME_STEPS:
            raise ValueError("values must be a torch Tensor shaped [batch, rows, 3000]")
        if not values.is_floating_point():
            raise ValueError("values must have a floating-point dtype")
        if values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("values must have positive batch and row dimensions")
        if not isinstance(row_mask, Tensor) or row_mask.dtype is not torch.bool or row_mask.shape != values.shape[:2]:
            raise ValueError("row_mask must be a bool Tensor shaped [batch, rows]")
        if (not isinstance(time_mask, Tensor) or time_mask.dtype is not torch.bool
                or time_mask.shape != (values.shape[0], TIME_STEPS)):
            raise ValueError("time_mask must be a bool Tensor shaped [batch, 3000]")
        if row_mask.device != values.device or time_mask.device != values.device:
            raise ValueError("values, row_mask, and time_mask must be on the same device")
        self._require_prefix(row_mask, "row_mask")
        self._require_prefix(time_mask, "time_mask")
        valid_cells = row_mask[:, :, None] & time_mask[:, None, :]
        if not torch.all(valid_cells.any(dim=(1, 2))):
            raise ValueError("each video must contain at least one valid row and frame")
        valid_values = values[valid_cells]
        if not torch.isfinite(valid_values).all():
            raise ValueError("values must be finite where row_mask and time_mask are valid")
        if not normalized and torch.any(valid_values < 0):
            raise ValueError("raw row_bytes values must be nonnegative where valid")

    def _token_mask(self, row_mask: Tensor, time_mask: Tensor) -> Tensor:
        """Return flattened ``[batch, rows * patches_per_row]`` valid-token mask."""
        patch_time_mask = F.pad(
            time_mask,
            (0, self.patches_per_row * self.patch_size - TIME_STEPS),
            value=False,
        ).unfold(1, self.patch_size, self.patch_size).any(dim=-1)
        return (row_mask[:, :, None] & patch_time_mask[:, None, :]).reshape(row_mask.shape[0], -1)

    def forward(self, values: Tensor, row_mask: Tensor, time_mask: Tensor, *, normalized: bool = False) -> Tensor:
        """Classify raw nonnegative values, or runner-normalized valid cells.

        ``normalized=True`` is reserved for the native runner after its frozen
        train-only transform; masks retain the same semantics in either mode.
        """
        self._validate_inputs(values, row_mask, time_mask, normalized=normalized)
        valid_cells = row_mask[:, :, None] & time_mask[:, None, :]
        # ``where`` is essential: invalid padding may intentionally carry NaN or
        # Inf in callers/tests, and multiplication by zero would retain NaNs.
        clean_values = torch.where(valid_cells, values, torch.zeros((), device=values.device, dtype=values.dtype))
        padded = F.pad(clean_values, (0, self.patches_per_row * self.patch_size - TIME_STEPS))
        patches = padded.unfold(2, self.patch_size, self.patch_size)
        token_mask = self._token_mask(row_mask, time_mask)
        projected = self.input_projection(patches.to(dtype=self.input_projection.weight.dtype))
        positions = two_dimensional_sincos_positions(
            values.shape[1], self.patches_per_row, self.d_model,
            device=projected.device, dtype=projected.dtype,
        )
        tokens = (projected + positions.unsqueeze(0)).reshape(values.shape[0], -1, self.d_model)
        tokens = torch.where(token_mask[:, :, None], tokens, torch.zeros((), device=tokens.device, dtype=tokens.dtype))
        tokens = self.input_dropout(tokens)
        padding_mask = ~token_mask
        # Every layer receives one flattened R*ceil(3000/P) sequence, never a
        # per-row classification or hierarchical pre-pooling stage.
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=padding_mask)
            tokens = torch.where(token_mask[:, :, None], tokens, torch.zeros((), device=tokens.device, dtype=tokens.dtype))
        tokens = self.norm(tokens)
        tokens = torch.where(token_mask[:, :, None], tokens, torch.zeros((), device=tokens.device, dtype=tokens.dtype))
        pooled = tokens.sum(dim=1) / token_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
        return self.head(pooled)
