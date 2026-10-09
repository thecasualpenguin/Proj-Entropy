"""Native-row WPP models.  These are intentionally independent of packet models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class EncodedWindow:
    logits: torch.Tensor
    duration_seconds: float


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1)


def _sinusoidal(values: torch.Tensor, width: int) -> torch.Tensor:
    """Structural native-position encoding; values are not measured features."""
    half = width // 2
    frequencies = torch.exp(torch.arange(half, device=values.device, dtype=values.dtype) *
                            (-math.log(10000.0) / max(half - 1, 1)))
    angle = values.unsqueeze(-1) * frequencies
    output = torch.cat((angle.sin(), angle.cos()), dim=-1)
    return nn.functional.pad(output, (0, width - output.shape[-1]))


class MaskedResidualBlock(nn.Module):
    """Conv/LayerNorm residual block that zeros padding after every operation."""
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _norm(x: torch.Tensor, norm: nn.LayerNorm, mask: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2) * mask[:, None, :].to(x.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask[:, None, :].to(x.dtype)
        residual = x * valid
        x = self.conv1(residual) * valid
        x = nn.functional.gelu(self._norm(x, self.norm1, mask))
        x = self.dropout(x) * valid
        x = self.conv2(x) * valid
        x = self._norm(x, self.norm2, mask)
        return nn.functional.gelu((x + residual) * valid) * valid


class WPPResNet(nn.Module):
    """Row-residual then frame-temporal-residual native-window classifier."""
    architecture = "wpp_resnet"
    def __init__(self, width: int = 64, dropout: float = .1, blocks: int = 3,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.width, self.gradient_checkpointing = width, gradient_checkpointing
        self.row_input = nn.Conv1d(2, width, kernel_size=1)
        self.row_blocks = nn.ModuleList(MaskedResidualBlock(width, dropout) for _ in range(blocks))
        self.temporal_input = nn.Conv1d(width, width, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(MaskedResidualBlock(width, dropout) for _ in range(blocks))
        self.head = nn.Linear(width, 2)

    def _blocks(self, x: torch.Tensor, mask: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else: x = block(x, mask)
        return x

    def forward_window(self, rows: torch.Tensor, row_mask: torch.Tensor, frame_pts_seconds: torch.Tensor) -> torch.Tensor:
        """``rows`` is (frames, max_native_rows, 2); zero values remain valid via mask."""
        if rows.ndim != 3 or rows.shape[-1] != 2 or row_mask.shape != rows.shape[:2]:
            raise ValueError("rows must be (frames, native_rows, 2) with matching mask")
        row_x = self.row_input(rows.transpose(1, 2))
        row_x = self._blocks(row_x, row_mask, self.row_blocks)
        frames = _masked_mean(row_x.transpose(1, 2), row_mask, 1)  # (frames,width)
        temporal_mask = torch.ones((1, len(frames)), dtype=torch.bool, device=rows.device)
        temporal = self.temporal_input(frames.unsqueeze(0).transpose(1, 2))
        temporal = self._blocks(temporal, temporal_mask, self.temporal_blocks)
        pooled = _masked_mean(temporal.transpose(1, 2), temporal_mask, 1)
        return self.head(pooled).squeeze(0)


class _MaskedTransformer(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, width * 2, dropout,
                                                                activation="gelu", batch_first=True,
                                                                norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Transformer masks use True for ignored positions; zero after every layer
        invalid = ~valid
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=invalid)
            x = x * valid.unsqueeze(-1).to(x.dtype)
        return self.norm(x) * valid.unsqueeze(-1).to(x.dtype)


class WPPTransformer(nn.Module):
    """Row attention and distinct native-PTS temporal attention classifier."""
    architecture = "wpp_transformer"
    def __init__(self, width: int = 64, heads: int = 4, dropout: float = .1,
                 row_layers: int = 2, temporal_layers: int = 2,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if width % heads: raise ValueError("width must be divisible by heads")
        self.width, self.gradient_checkpointing = width, gradient_checkpointing
        self.row_input = nn.Linear(2, width)
        self.row_encoder = _MaskedTransformer(width, heads, row_layers, dropout)
        self.temporal_encoder = _MaskedTransformer(width, heads, temporal_layers, dropout)
        self.dropout, self.head = nn.Dropout(dropout), nn.Linear(width, 2)

    def _row(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.row_encoder(x, mask)

    def _temporal(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.temporal_encoder(x, mask)

    def forward_window(self, rows: torch.Tensor, row_mask: torch.Tensor, frame_pts_seconds: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 3 or rows.shape[-1] != 2 or row_mask.shape != rows.shape[:2]:
            raise ValueError("rows must be (frames, native_rows, 2) with matching mask")
        if frame_pts_seconds.ndim != 1 or len(frame_pts_seconds) != len(rows):
            raise ValueError("one native PTS position is required per frame")
        count, native_rows = rows.shape[:2]
        positions = _sinusoidal(torch.arange(native_rows, device=rows.device, dtype=rows.dtype), self.width)
        x = self.row_input(rows) + positions.unsqueeze(0)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self._row, x, row_mask, use_reentrant=False)
        else: x = self._row(x, row_mask)
        frames = _masked_mean(x, row_mask, 1).unsqueeze(0)
        frame_mask = torch.ones((1, count), dtype=torch.bool, device=rows.device)
        native_time = frame_pts_seconds - frame_pts_seconds[0]
        temporal = frames + _sinusoidal(native_time, self.width).unsqueeze(0)
        if self.gradient_checkpointing and self.training:
            temporal = checkpoint(self._temporal, temporal, frame_mask, use_reentrant=False)
        else: temporal = self._temporal(temporal, frame_mask)
        return self.head(self.dropout(_masked_mean(temporal, frame_mask, 1))).squeeze(0)


def padded_window_tensors(video, window, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize only one native window; valid numeric zeros are never inferred as padding."""
    frames = [video.frames[index] for index in window.frame_indices]
    max_rows = max(len(frame.rows) for frame in frames)
    rows = torch.zeros((len(frames), max_rows, 2), dtype=torch.float32, device=device)
    mask = torch.zeros((len(frames), max_rows), dtype=torch.bool, device=device)
    for index, frame in enumerate(frames):
        n = len(frame.rows); rows[index, :n] = torch.from_numpy(frame.rows).to(device); mask[index, :n] = True
    # Exact rationals are converted only at the final model boundary for sinusoidal positions.
    first = frames[0].pts
    pts = torch.tensor([float((frame.pts - first) * video.time_base) for frame in frames], dtype=torch.float32, device=device)
    return rows, mask, pts


def video_logits(model: nn.Module, video, windows: Sequence, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Duration-weighted, differentiable aggregation of all native window logits."""
    logits: list[torch.Tensor] = []; weights: list[float] = []
    for window in windows:
        rows, mask, pts = padded_window_tensors(video, window, device)
        logits.append(model.forward_window(rows, mask, pts))
        weights.append(float(window.duration))
    if not logits or any(weight <= 0 for weight in weights): raise ValueError("video requires positive-duration windows")
    stacked = torch.stack(logits)
    duration = torch.tensor(weights, dtype=stacked.dtype, device=device)
    return (stacked * duration[:, None]).sum(0) / duration.sum(), logits


def make_wpp_model(config: dict) -> nn.Module:
    if config["architecture"] == "resnet":
        return WPPResNet(width=config["width"], dropout=config["dropout"], blocks=3,
                          gradient_checkpointing=config.get("gradient_checkpointing", True))
    if config["architecture"] == "transformer":
        return WPPTransformer(width=config["width"], heads=config["heads"], dropout=config["dropout"],
                              row_layers=2, temporal_layers=2,
                              gradient_checkpointing=config.get("gradient_checkpointing", True))
    raise ValueError("WPP architecture must be resnet or transformer")
