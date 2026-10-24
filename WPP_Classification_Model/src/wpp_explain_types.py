"""Public, UI-safe value types for read-only native WPP explanations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

Partition = Literal["train", "validation", "test"]
DisplayMode = Literal["positive", "signed"]
DevicePreference = Literal["auto", "cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DataRoots:
    """Explicit, relocatable read-only inputs for an explanation session."""

    packed_bundle: str | Path
    source_manifest: str | Path
    index: str | Path
    source_video_root: str | Path
    media_url_resolver: Callable[[Path], str] | None = None


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    message: str
    completed: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class VideoSummary:
    video_id: str
    partition: Partition
    label: str
    class_id: int
    source_relative_path: str
    original_frame_count: int
    used_frame_count: int
    row_count: int
    time_truncated: bool


@dataclass(frozen=True)
class Prediction:
    predicted_class_id: int
    predicted_label: str
    predicted_logit: float
    predicted_probability: float
    ground_truth_class_id: int
    ground_truth_label: str
    correct: bool


@dataclass(frozen=True)
class MediaTiming:
    source_path: Path
    media_url: str | None
    width: int
    height: int
    sample_aspect_ratio: str
    rotation_degrees: int
    pts_seconds: np.ndarray
    media_seconds: np.ndarray
    covered_end_pts_seconds: float
    covered_end_media_seconds: float
    source_sha256_verified: bool
    timing_source: str


@dataclass(frozen=True)
class AttentionSummary:
    """Attention-only diagnostics; these values are not class-specific."""

    rollout_tokens: np.ndarray
    rollout_frames: np.ndarray
    frame_strip: np.ndarray
    incoming_tokens: np.ndarray
    patch_size: int
    layers: int
    heads: int


@dataclass(frozen=True)
class VideoExplanation:
    video: VideoSummary
    prediction: Prediction
    media: MediaTiming
    # Signed raw-input IG values [WPP row, retained presentation frame].
    ig_signed: np.ndarray
    ig_frame_strip_signed: np.ndarray
    attention: AttentionSummary
    metadata: Mapping[str, Any]
    display_mode: DisplayMode


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class ExplainerSession:
    """Stateful read-only inference session. Construct with ``open_explainer``."""

    model: Any
    bundle: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    normalization: Mapping[str, Any]
    class_names: tuple[str, ...]
    roots: DataRoots
    requested_device: str
    device: Any
    fallback_reason: str | None = None
    default_attention_chunk_queries: int = 128
    metadata: dict[str, Any] = field(default_factory=dict)
