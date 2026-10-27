"""Read-only source-video verification and presentation-time metadata for WPP UI."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .native_wpp_workflow import PresentationProbe, WorkflowError, probe_presentation
from .wpp_explain_types import MediaTiming


class MediaError(ValueError):
    """A source video cannot be safely synchronized with the WPP artifact."""


@dataclass(frozen=True)
class ValidatedSourceMediaFrames:
    """Source-frame identity, timeline, and display geometry without an overlay endpoint."""

    source_path: Path
    probe: PresentationProbe
    artifact_presentation_times: tuple[Fraction, ...]
    rotation_degrees: int
    sample_aspect_ratio: str
    frame_durations: Mapping[Fraction, Fraction]

    @property
    def width(self) -> int:
        return int(self.probe.decoded_width)

    @property
    def height(self) -> int:
        return int(self.probe.decoded_height)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_video(source_root: str | Path, relative_path: str) -> Path:
    root = Path(source_root).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MediaError("source_relative_path escapes source_video_root") from exc
    if not candidate.is_file():
        raise MediaError(f"source video is missing: {candidate}")
    return candidate


def verify_source_video(path: str | Path, expected_sha256: str) -> Path:
    source = Path(path).expanduser().resolve()
    if _sha256(source) != expected_sha256:
        raise MediaError(f"source video SHA-256 differs from the manifest: {source}")
    return source


def _artifact_presentation_times(artifact_dir: Path) -> tuple[tuple[Fraction, ...], int]:
    try:
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        time_base = Fraction(str(metadata["time_base"]))
        with np.load(artifact_dir / "features.npz", allow_pickle=False) as archive:
            pts = np.asarray(archive["pts"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError(f"cannot read native artifact presentation metadata: {artifact_dir}") from exc
    if pts.ndim != 1 or not len(pts) or pts.dtype.kind not in "iu":
        raise MediaError("native artifact PTS are invalid")
    integers = tuple(int(value) for value in pts)
    if any(later <= earlier for earlier, later in zip(integers, integers[1:])):
        raise MediaError("native artifact PTS are not strictly increasing")
    return tuple(Fraction(value) * time_base for value in integers), int(metadata.get("frames", -1))


def _ffprobe_display_metadata(source: Path) -> tuple[int, str, dict[Fraction, Fraction]]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise MediaError("ffprobe is required to validate source presentation timing; install FFmpeg or provide a supported environment")
    command = [
        executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=sample_aspect_ratio,time_base:stream_tags=rotate:stream_side_data=rotation:frame=media_type,best_effort_timestamp,duration,pkt_duration",
        "-of", "json", str(source),
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        value = json.loads(result.stdout)
        stream = value["streams"][0]
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise MediaError(f"cannot probe source display metadata: {source}") from exc
    rotation = 0
    tag = stream.get("tags", {}).get("rotate")
    side_data = stream.get("side_data_list", [])
    raw_rotation = tag if tag is not None else next((item.get("rotation") for item in side_data if "rotation" in item), 0)
    try:
        rotation = int(round(float(raw_rotation))) % 360
    except (TypeError, ValueError) as exc:
        raise MediaError("source video rotation metadata is invalid") from exc
    sar = str(stream.get("sample_aspect_ratio") or "1:1")
    if sar in ("N/A", "0:1", "1:0"):
        sar = "1:1"
    try:
        numerator, denominator = sar.split(":")
        if int(numerator) <= 0 or int(denominator) <= 0:
            raise ValueError
    except ValueError as exc:
        raise MediaError("source video sample aspect ratio is invalid") from exc
    try:
        time_base = Fraction(str(stream["time_base"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise MediaError("source video time base is invalid") from exc
    durations: dict[Fraction, Fraction] = {}
    for frame in value.get("frames", []):
        if frame.get("media_type") != "video":
            continue
        try:
            timestamp = int(frame["best_effort_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        # Newer FFmpeg exposes AVFrame.duration; older releases use
        # pkt_duration. Both are integer ticks in the stream time base.
        for field in ("duration", "pkt_duration"):
            try:
                duration = int(frame[field])
            except (KeyError, TypeError, ValueError):
                continue
            if duration > 0:
                durations[Fraction(timestamp) * time_base] = Fraction(duration) * time_base
                break
    return rotation, sar, durations


def validate_source_media_frames(*, source_path: str | Path, expected_sha256: str, artifact_dir: str | Path,
                                 expected_frames: int, retained_frames: int) -> ValidatedSourceMediaFrames:
    """Validate original-source frame identity and display geometry without deriving an endpoint.

    This deliberately stops before requiring a final decoded-frame duration.
    Spatial consumers use retained presentation-frame indices rather than elapsed
    time, while browser overlays must call :func:`source_media_timing` for its
    stricter final-frame endpoint validation.
    """
    if retained_frames <= 0 or expected_frames < retained_frames:
        raise MediaError("invalid retained/native frame counts")
    source = verify_source_video(source_path, expected_sha256)
    artifact_times, artifact_frames = _artifact_presentation_times(Path(artifact_dir))
    if artifact_frames != expected_frames or len(artifact_times) != expected_frames:
        raise MediaError("native artifact frame count disagrees with bundle record")
    try:
        probe, _ = probe_presentation(source)
    except (WorkflowError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise MediaError(f"cannot obtain exact source presentation timestamps: {source}") from exc
    if len(probe.presentation_times) != expected_frames:
        raise MediaError("source presentation frame count differs from validated WPP artifact")
    # The native build permits one encoded output tick after origin normalization.
    tolerance = Fraction(str(json.loads((Path(artifact_dir) / "metadata.json").read_text())["time_base"]))
    source_zero, artifact_zero = probe.presentation_times[0], artifact_times[0]
    if any(abs((left - source_zero) - (right - artifact_zero)) > tolerance
           for left, right in zip(probe.presentation_times, artifact_times)):
        raise MediaError("source presentation timeline does not match the validated WPP artifact")
    rotation, sar, frame_durations = _ffprobe_display_metadata(source)
    if rotation != 0:
        raise MediaError("source video has rotation metadata; explicitly create a browser media derivative with rotation baked in before overlaying WPP rows")
    if sar != "1:1":
        raise MediaError("source video has non-square sample aspect ratio; explicitly create a square-pixel browser derivative before overlaying WPP rows")
    return ValidatedSourceMediaFrames(source_path=source, probe=probe, artifact_presentation_times=artifact_times,
                                      rotation_degrees=rotation, sample_aspect_ratio=sar,
                                      frame_durations=frame_durations)


def source_media_timing(*, source_path: str | Path, expected_sha256: str, artifact_dir: str | Path,
                        expected_frames: int, retained_frames: int, media_url: str | None = None) -> MediaTiming:
    """Return strict browser-overlay timing for a validated original source.

    The model's first ``retained_frames`` display frames are synchronized by
    actual presentation timestamps, not an assumed frame rate.  A next timestamp
    (when truncating) or validated decoded-frame duration defines the last
    covered frame's end, so a browser never silently overlays unmodeled trailing
    video.
    """
    frames = validate_source_media_frames(source_path=source_path, expected_sha256=expected_sha256,
                                          artifact_dir=artifact_dir, expected_frames=expected_frames,
                                          retained_frames=retained_frames)
    pts = frames.probe.presentation_times[:retained_frames]
    if retained_frames < expected_frames:
        end = frames.probe.presentation_times[retained_frames]
    elif pts[-1] in frames.frame_durations:
        # This is a decoded frame duration in the same raw PTS time base.  Do
        # not substitute a previous VFR gap or a container duration with an
        # unknown origin.
        end = pts[-1] + frames.frame_durations[pts[-1]]
    else:
        raise MediaError("cannot establish the final retained frame end from decoded timestamp metadata")
    if end <= pts[-1]:
        raise MediaError("validated media duration does not extend through the final retained frame")
    pts_seconds = np.asarray([float(item) for item in pts], dtype=np.float64)
    media_seconds = np.asarray([float(item - pts[0]) for item in pts], dtype=np.float64)
    # One extra endpoint is intentionally recorded for clipping browser overlays.
    media_end = float(end - pts[0])
    return MediaTiming(source_path=frames.source_path, media_url=media_url, width=frames.width, height=frames.height,
                       sample_aspect_ratio=frames.sample_aspect_ratio, rotation_degrees=frames.rotation_degrees,
                       pts_seconds=pts_seconds, media_seconds=media_seconds, covered_end_pts_seconds=float(end),
                       covered_end_media_seconds=media_end, source_sha256_verified=True,
                       timing_source="ffprobe_source_pts_validated_against_native_artifact")


def wpp_row_boxes(*, rows: int, frame_height: int, ctu_size: int = 64) -> np.ndarray:
    """Return source-pixel [y, height] bands, clipping the final WPP row."""
    if rows <= 0 or frame_height <= 0 or ctu_size <= 0:
        raise MediaError("rows, frame_height, and ctu_size must be positive")
    boxes = np.empty((rows, 2), dtype=np.int64)
    for row in range(rows):
        y = row * ctu_size
        boxes[row] = (y, max(0, min(ctu_size, frame_height - y)))
    if np.any(boxes[:, 1] == 0):
        raise MediaError("WPP row count exceeds source frame height; cannot map 64px row overlays")
    return boxes
