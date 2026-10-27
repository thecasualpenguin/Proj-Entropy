"""Read-only access to one complete native-WPP attribution cache record.

This module never opens an explainer, acquires the cache writer lock, writes a
cache file, or starts attribution.  It deliberately reuses the class-cache
validator so a completed record can be read safely while another record is
being atomically published by the cache writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .wpp_class_attribution import (
    CACHE_SCHEMA_VERSION,
    ClassAttributionError,
    _read_json,
    _safe_video_name,
    _validate_expected_entry,
    _validated_cache_record,
)


@dataclass(frozen=True)
class SingleVideoRecord:
    """Listing metadata for one fully validated record."""

    video_id: str
    split: str
    truth_class_id: int
    truth_label: str
    predicted_class_id: int
    predicted_label: str
    correct: bool
    native_wpp_rows: int
    retained_frame_count: int


@dataclass(frozen=True)
class SingleVideoCacheListing:
    """A truthful inventory of a potentially incomplete/live cache.

    ``records`` contains only records passing the cache's full integrity and
    provenance validation.  An absent NPZ/JSON pair is counted as unavailable
    (including a writer's NPZ-first atomic-publication window), while a present
    but invalid pair is reported in ``invalid_records`` rather than silently
    making the cache appear empty.
    """

    records: tuple[SingleVideoRecord, ...]
    cache_status: str
    expected_videos: int
    available_videos: int
    unavailable_videos: int
    publication_in_progress_or_missing: int
    invalid_records: Mapping[str, str]


@dataclass(frozen=True)
class SingleVideoAttribution:
    """One native-resolution attribution record, without resampling.

    The first two dimensions of ``ig_values`` and ``attention_frame_values``
    are exactly ``[native WPP row, retained presentation-frame index]``.
    ``attention_frame_values`` is token relevance repeated for display through
    each valid patch frame.  It is piecewise-constant relevance, not conserved
    per-frame probability/mass; use ``attention_token_mass`` for native
    conserved token mass or ``attention_mass_per_frame`` when a frame-split
    mass representation is required.
    """

    record: SingleVideoRecord
    row_indices: np.ndarray
    frame_indices: np.ndarray
    ig_target: str
    truth_target_source: str
    sign: str
    value_mode: str
    ig_values: np.ndarray
    attention_frame_values: np.ndarray
    attention_token_mass: np.ndarray
    attention_mass_per_frame: np.ndarray
    ig_display_scale: float
    attention_display_scale: float
    # These are deliberately explicit raw/native companions for numerical
    # export.  They do not change the declared displayed-value fields above.
    ig_native_signed: np.ndarray
    attention_frame_raw: np.ndarray
    valid_mask: np.ndarray
    metadata: Mapping[str, Any]


def _load_manifest(cache_dir: str | Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]], bool]:
    """Validate non-scientific manifest structure needed by the shared validator."""
    root = Path(cache_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA_VERSION:
        raise ClassAttributionError("unsupported cache schema")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ClassAttributionError("cache fingerprint is missing")
    class_names = list(fingerprint.get("class_names", []))
    if (not class_names or any(not isinstance(name, str) for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ClassAttributionError("cache class mapping is invalid")
    raw_expected = manifest.get("expected_videos")
    if not isinstance(raw_expected, list):
        raise ClassAttributionError("cache expected_videos must be a list")
    expected = [_validate_expected_entry(value, class_names) for value in raw_expected]
    identities = [(entry["partition"], entry["video_id"]) for entry in expected]
    video_ids = [entry["video_id"] for entry in expected]
    if len(set(identities)) != len(identities) or len(set(video_ids)) != len(video_ids):
        raise ClassAttributionError("cache manifest has duplicate video identities")
    total, completed, status = manifest.get("total"), manifest.get("completed"), manifest.get("status")
    if (not isinstance(total, int) or isinstance(total, bool) or total != len(expected)
            or not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= total
            or status not in {"building", "paused", "cancelled", "complete"}):
        raise ClassAttributionError("cache manifest status/counts are invalid")
    return root, manifest, fingerprint, expected, status == "complete" and completed == total


def _paths(root: Path, entry: Mapping[str, Any]) -> tuple[Path, Path]:
    stem = _safe_video_name(str(entry["video_id"]))
    return root / "videos" / f"{stem}.json", root / "videos" / f"{stem}.npz"


def _record(meta: Mapping[str, Any]) -> SingleVideoRecord:
    return SingleVideoRecord(
        video_id=str(meta["video_id"]), split=str(meta["partition"]),
        truth_class_id=int(meta["ground_truth_class_id"]), truth_label=str(meta["ground_truth_label"]),
        predicted_class_id=int(meta["predicted_class_id"]), predicted_label=str(meta["predicted_label"]),
        correct=bool(meta["correct"]), native_wpp_rows=int(meta["row_count"]),
        retained_frame_count=int(meta["used_frames"]),
    )


def list_single_video_records(cache_dir: str | Path) -> SingleVideoCacheListing:
    """List valid records read-only; cache completion is not required.

    This does not take the writer lock.  Missing or half-published records are
    unavailable, not zero-valued explanations.  Malformed present records are
    visible in ``invalid_records`` for an actionable UI status.
    """
    root, manifest, fingerprint, expected, _complete = _load_manifest(cache_dir)
    records: list[SingleVideoRecord] = []
    missing = 0
    invalid: dict[str, str] = {}
    for entry in expected:
        meta_path, npz_path = _paths(root, entry)
        if not meta_path.is_file() or not npz_path.is_file():
            missing += 1
            continue
        try:
            meta, _arrays = _validated_cache_record(meta_path, npz_path, fingerprint=fingerprint, expected=entry)
        except (OSError, ValueError, ClassAttributionError) as exc:
            invalid[str(entry["video_id"])] = str(exc)
            continue
        records.append(_record(meta))
    unavailable = missing + len(invalid)
    return SingleVideoCacheListing(
        records=tuple(records), cache_status=str(manifest["status"]), expected_videos=len(expected),
        available_videos=len(records), unavailable_videos=unavailable,
        publication_in_progress_or_missing=missing, invalid_records=invalid,
    )


def _expand_token_relevance(token_mass: np.ndarray, *, frame_count: int, patch_size: int) -> np.ndarray:
    """Repeat token relevance over valid frames, cropping a partial final patch."""
    valid_tokens = int(np.ceil(frame_count / patch_size))
    return np.repeat(token_mass[:, :valid_tokens], patch_size, axis=1)[:, :frame_count].copy()


def _split_token_mass_over_frames(token_mass: np.ndarray, *, frame_count: int, patch_size: int) -> np.ndarray:
    """Produce an optional frame representation whose sum preserves token mass."""
    result = np.zeros((token_mass.shape[0], frame_count), dtype=token_mass.dtype)
    for token in range(int(np.ceil(frame_count / patch_size))):
        start = token * patch_size
        end = min(start + patch_size, frame_count)
        result[:, start:end] = token_mass[:, token:token + 1] / (end - start)
    return result


def _scale(values: np.ndarray, maximum: float) -> np.ndarray:
    # Existing explorer turns an all-zero overlay transparent.  Numeric API
    # returns defined zeros rather than NaN/Inf while preserving that semantics.
    return np.zeros_like(values) if maximum <= 0 else values / np.asarray(maximum, dtype=values.dtype)


def load_single_video_attribution(
    cache_dir: str | Path,
    video_id: str,
    *,
    ig_target: Literal["predicted", "truth"] = "predicted",
    sign: Literal["abs", "signed"] = "abs",
    value_mode: Literal["display_normalized", "raw"] = "display_normalized",
) -> SingleVideoAttribution:
    """Load exactly one current, checksum-validated record without computation.

    ``display_normalized`` uses existing explorer-style fixed per-video scales:
    IG divides by ``max(abs(native signed IG))`` and rollout display relevance
    divides by ``max(native token relevance)``.  ``raw`` leaves selected values
    unscaled.  Neither mode resizes, re-bins, normalizes coordinates, or
    averages native values.
    """
    if not isinstance(video_id, str) or not video_id:
        raise ClassAttributionError("video_id must be a non-empty string")
    if ig_target not in {"predicted", "truth"}:
        raise ClassAttributionError("ig_target must be predicted or truth")
    if sign not in {"abs", "signed"}:
        raise ClassAttributionError("sign must be abs or signed")
    if value_mode not in {"display_normalized", "raw"}:
        raise ClassAttributionError("value_mode must be display_normalized or raw")
    root, manifest, fingerprint, expected, complete = _load_manifest(cache_dir)
    selected = next((entry for entry in expected if entry["video_id"] == video_id), None)
    if selected is None:
        raise ClassAttributionError(f"video_id is not declared by this cache manifest: {video_id}")
    meta_path, npz_path = _paths(root, selected)
    if not meta_path.is_file() or not npz_path.is_file():
        raise ClassAttributionError(
            f"selected video is not available: record is missing or atomically publishing ({video_id})"
        )
    # A selected present-but-corrupt record must reject, not disappear as it may
    # in a listing intended to remain usable during a writer's live session.
    meta, arrays = _validated_cache_record(meta_path, npz_path, fingerprint=fingerprint, expected=selected)
    record = _record(meta)
    native_signed = arrays["ig_pred_signed"].copy()
    truth_source = str(meta["ig_truth_source"])
    if ig_target == "truth" and not record.correct:
        try:
            native_signed = arrays["ig_truth_signed"].copy()
        except KeyError as exc:  # shared validator normally makes this unreachable
            raise ClassAttributionError("misclassified cache lacks truth-logit IG") from exc
    ig_values_raw = np.abs(native_signed) if sign == "abs" else native_signed.copy()
    attention_token_mass = arrays["attention_token_mass"].copy()
    patch_size = int(meta["patch_size"])
    attention_frame_raw = _expand_token_relevance(
        attention_token_mass, frame_count=record.retained_frame_count, patch_size=patch_size,
    )
    attention_mass_per_frame = _split_token_mass_over_frames(
        attention_token_mass, frame_count=record.retained_frame_count, patch_size=patch_size,
    )
    ig_display_scale = float(np.max(np.abs(native_signed)))
    attention_display_scale = float(np.max(attention_frame_raw))
    ig_values = (_scale(ig_values_raw, ig_display_scale) if value_mode == "display_normalized"
                 else ig_values_raw.copy())
    attention_values = (_scale(attention_frame_raw, attention_display_scale) if value_mode == "display_normalized"
                        else attention_frame_raw.copy())
    metadata = {
        "cache_status": manifest["status"], "cache_complete": complete,
        "ig_target": ig_target, "ig_truth_source": truth_source, "sign": sign, "value_mode": value_mode,
        "ig_display_scale": ig_display_scale, "attention_display_scale": attention_display_scale,
        "axes": {"x": "actual retained presentation-frame index", "y": "native WPP row index"},
        "ig_units": "native predicted- or truth-logit integrated gradients; ig_native_signed is always signed",
        "attention_units": "piecewise-constant native token relevance repeated for display; not per-frame conserved mass; do not sum expanded frames as probability",
        "attention_token_mass_units": "native non-class-specific rollout token mass; use for conserved mass",
        "scale_formula": {
            "ig": "max(abs(native signed IG)); values are divided by it only in display_normalized mode",
            "attention": "max(native token relevance); repeated display values are divided by it only in display_normalized mode",
            "zero": "defined all-zero numeric output (explorer renders this transparent)",
            "clipping": "explorer color mapping clamps normalized non-signed values to [0,1] and signed values to [-1,1]; these validated values need no clipping",
        },
        "provenance": {"cache_fingerprint": fingerprint, "record_npz_sha256": meta["npz_sha256"]},
    }
    return SingleVideoAttribution(
        record=record,
        row_indices=np.arange(record.native_wpp_rows, dtype=np.int64),
        frame_indices=np.arange(record.retained_frame_count, dtype=np.int64),
        ig_target=ig_target, truth_target_source=truth_source, sign=sign, value_mode=value_mode,
        ig_values=ig_values, attention_frame_values=attention_values,
        attention_token_mass=attention_token_mass, attention_mass_per_frame=attention_mass_per_frame,
        ig_display_scale=ig_display_scale, attention_display_scale=attention_display_scale,
        ig_native_signed=native_signed, attention_frame_raw=attention_frame_raw,
        valid_mask=arrays["valid_mask"].copy(), metadata=metadata,
    )


__all__ = [
    "SingleVideoAttribution", "SingleVideoCacheListing", "SingleVideoRecord",
    "list_single_video_records", "load_single_video_attribution",
]
