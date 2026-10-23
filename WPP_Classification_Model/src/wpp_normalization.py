"""Masked, train-only normalization for native WPP video batches."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch

POLICY = {
    "method": "global_valid_train_zscore",
    "statistics_dtype": "float64",
    "standard_deviation": "population",
    "ddof": 0,
    "fit_split": "train",
    "valid_cells": "explicit_row_mask_and_time_mask",
    "observed_zero": "included",
    "padding": "zero",
    "log_transform": None,
    "clipping": None,
    "per_row_normalization": False,
}


class NormalizationError(ValueError):
    """Raised when native WPP normalization cannot be safely applied."""


def _mask(values: np.ndarray, row_mask: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    rows, times = np.asarray(row_mask), np.asarray(time_mask)
    if values.ndim != 3 or rows.shape != values.shape[:2] or times.shape != (values.shape[0], values.shape[2]):
        raise NormalizationError("values/row_mask/time_mask shapes are incompatible")
    if rows.dtype != np.bool_ or times.dtype != np.bool_:
        raise NormalizationError("row_mask and time_mask must be boolean")
    return rows[:, :, None] & times[:, None, :]


@dataclass
class WelfordPopulation:
    """Mergeable float64 population moments, without an E[x^2] shortcut."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray, row_mask: np.ndarray, time_mask: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        valid = _mask(x, row_mask, time_mask)
        selected = x[valid]
        if selected.size == 0:
            return
        if not np.isfinite(selected).all():
            raise NormalizationError("valid WPP values must be finite")
        # Batch Welford moments are calculated around its own mean, then merged.
        n = int(selected.size)
        batch_mean = float(selected.mean(dtype=np.float64))
        centered = selected - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + n
        self.m2 += batch_m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def freeze(self, *, expected_count: int | None = None) -> dict[str, Any]:
        if expected_count is not None and self.count != expected_count:
            raise NormalizationError(f"valid training-cell count mismatch: {self.count} != {expected_count}")
        if self.count <= 0:
            raise NormalizationError("training split has no valid WPP cells")
        variance = self.m2 / self.count
        if not math.isfinite(self.mean) or not math.isfinite(variance) or variance <= 0:
            raise NormalizationError("training WPP population variance must be finite and nonzero")
        std = math.sqrt(variance)
        if not math.isfinite(std) or std <= 0:
            raise NormalizationError("training WPP population standard deviation must be finite and nonzero")
        return {**POLICY, "count": self.count, "mean": self.mean, "std": std}


def validate_statistics(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(POLICY) | {"count", "mean", "std"}:
        raise NormalizationError("normalization statistics do not match native WPP policy")
    result = dict(value)
    for key, expected in POLICY.items():
        if result[key] != expected:
            raise NormalizationError(f"normalization policy mismatch for {key}")
    if isinstance(result["count"], bool) or not isinstance(result["count"], int) or result["count"] <= 0:
        raise NormalizationError("normalization count must be a positive integer")
    for key in ("mean", "std"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(float(result[key])):
            raise NormalizationError(f"normalization {key} must be finite")
        result[key] = float(result[key])
    if result["std"] <= 0:
        raise NormalizationError("normalization std must be positive")
    return result


def normalize_numpy(values: np.ndarray, row_mask: np.ndarray, time_mask: np.ndarray,
                    statistics: Mapping[str, Any]) -> np.ndarray:
    stats = validate_statistics(statistics)
    x = np.asarray(values, dtype=np.float64)
    valid = _mask(x, row_mask, time_mask)
    if not np.isfinite(x[valid]).all():
        raise NormalizationError("valid WPP values must be finite")
    out = np.zeros(x.shape, dtype=np.float32)
    out[valid] = ((x[valid] - stats["mean"]) / stats["std"]).astype(np.float32)
    return out


def normalize_torch(values: torch.Tensor, row_mask: torch.Tensor, time_mask: torch.Tensor,
                    statistics: Mapping[str, Any]) -> torch.Tensor:
    stats = validate_statistics(statistics)
    if (not isinstance(values, torch.Tensor) or values.ndim != 3 or not values.is_floating_point()
            or not isinstance(row_mask, torch.Tensor) or row_mask.dtype is not torch.bool
            or not isinstance(time_mask, torch.Tensor) or time_mask.dtype is not torch.bool
            or row_mask.shape != values.shape[:2] or time_mask.shape != (values.shape[0], values.shape[2])):
        raise NormalizationError("invalid torch WPP batch or masks")
    valid = row_mask[:, :, None] & time_mask[:, None, :]
    if not torch.isfinite(values[valid]).all():
        raise NormalizationError("valid WPP values must be finite")
    # where avoids carrying arbitrary NaN/Inf padding through the output.
    normalized = (values - values.new_tensor(stats["mean"])) / values.new_tensor(stats["std"])
    return torch.where(valid, normalized, torch.zeros((), dtype=values.dtype, device=values.device))
