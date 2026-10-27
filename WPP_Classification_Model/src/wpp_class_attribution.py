"""Read-only, resumable class-average native-WPP attribution cache and aggregation.

The cache stores native signed IG cells (never a display transform) and native
attention-token mass.  Aggregation deliberately resamples *mass*, rather than
interpolating intensities: a source cell/token is split by its overlap with a
common grid.  Consequently each video's total signed and absolute IG mass is
preserved by resampling.  Absolute IG is taken before resampling; signed
per-video normalization uses the native L1 mass, never a cancelling coarse bin.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import socket
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import torch

from .wpp_explain import (_accelerator_runtime_error, _attention_summary, _batch_tensors,
                          _integrated_gradients, _move_to_cpu, _sha256, open_explainer)
from .wpp_explain_types import DataRoots, ProgressCallback, ProgressEvent
from .wpp_media import MediaError, resolve_source_video, validate_source_media_frames

CACHE_SCHEMA_VERSION = "wpp-class-attribution-cache-2"
AGGREGATION_SCHEMA_VERSION = "wpp-class-attribution-aggregate-1"
_SPLITS = ("train", "test")


class ClassAttributionError(ValueError):
    """A class-attribution cache or aggregation request is invalid."""


@dataclass(frozen=True)
class CacheBuildResult:
    cache_dir: Path
    complete: bool
    completed: int
    total: int
    reused: int
    computed: int
    checkpoint_sha256: str


@dataclass(frozen=True)
class AttributionGroup:
    """One equal-video average.  Empty groups have ``count == 0`` and zero grids."""

    key: str
    split: str
    label: str
    count: int
    ig_grid: np.ndarray
    attention_grid: np.ndarray


@dataclass(frozen=True)
class AttributionAggregate:
    """Numeric common-grid result, intentionally independent of any notebook/UI."""

    groups: Mapping[str, AttributionGroup]
    metadata: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _emit(callback: ProgressCallback | None, stage: str, message: str, done: int, total: int) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, message, done, total))


class _CacheWriterLock:
    """Advisory OS-released lock: exactly one cache builder may mutate a destination."""

    def __init__(self, cache_dir: Path) -> None:
        self.path = cache_dir.parent / f".{cache_dir.name}.build.lock"
        self.handle: Any | None = None

    def acquire(self) -> None:
        # Kept lazy so import remains portable; supported production hosts are POSIX/Linux.
        try:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close(); self.handle = None
                raise ClassAttributionError(f"another class-attribution cache builder holds {self.path}") from exc
            self.handle.seek(0); self.handle.truncate(); self.handle.write(json.dumps({"pid": os.getpid(), "hostname": socket.gethostname()}) + "\n"); self.handle.flush()
        except ImportError as exc:  # pragma: no cover - project deployment is POSIX, but do not silently race elsewhere.
            raise ClassAttributionError("cache builder locking requires POSIX fcntl") from exc

    def release(self) -> None:
        if self.handle is not None:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close(); self.handle = None

    def __del__(self) -> None:  # exceptional exits still release the process-owned descriptor promptly
        try: self.release()
        except Exception: pass


def _safe_video_name(video_id: str) -> str:
    return hashlib.sha256(video_id.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassAttributionError(f"cannot read cache metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ClassAttributionError(f"cache metadata is not an object: {path}")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ClassAttributionError(f"{name} must be a positive integer")
    return value


def _cache_fingerprint(session: Any, *, splits: tuple[str, ...], ig_steps: int,
                       attention_chunk_queries: int) -> dict[str, Any]:
    """Every costly-input setting; incompatible caches are rejected, never mixed."""
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": session.checkpoint_sha256,
        "checkpoint_contract": session.metadata.get("checkpoint_contract"),
        "checkpoint_version": session.metadata.get("checkpoint_version"),
        "provenance": session.metadata.get("provenance"),
        "model_config": session.metadata.get("model_config"),
        "class_names": list(session.class_names),
        "splits": list(splits),
        "ig": {"method": "trapezoid_normalized_zero_baseline", "steps": ig_steps,
               "predicted_always": True, "truth_only_if_misclassified": True},
        "attention": {"method": "masked_mean_head_residual_rollout", "chunk_queries": attention_chunk_queries,
                      "class_specific": False, "stored": "native_token_mass"},
        "normalization": dict(session.normalization),
        "geometry": {"vertical": "physical source/display-pixel CTU bands, clipped by validated decoded display height",
                     "temporal": "normalized retained presentation-frame index/progress (not elapsed time)"},
    }


def _record_items(session: Any, splits: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for split in splits:
        if split not in _SPLITS:
            raise ClassAttributionError("splits may contain only train and test (validation is intentionally excluded)")
        dataset = session.bundle.partition_view(split)
        result.extend((split, dataset[index]) for index in range(len(dataset)))
    return result


def _artifact_geometry(session: Any, item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return display/source-pixel WPP bands, validated against the media contract.

    HEVC coded dimensions may include padding.  They must never define display
    bands: source-frame validation verifies the source hash/timeline and supplies
    the decoded display height, so the final CTU band is clipped correctly.
    """
    root_ref = session.bundle.extraction["metadata"].get("artifact_root_ref")
    if not isinstance(root_ref, str):
        raise ClassAttributionError("extraction index lacks artifact_root_ref")
    artifact = (session.bundle.index.parent / root_ref / str(item["artifact_relpath"])).resolve()
    try:
        raw = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        ctu = int(raw.get("ctu_size", 64))
        # Production sessions always take this branch: it validates source bytes, presentation timing,
        # and decoded display geometry, but does not derive a browser-overlay endpoint. The metadata
        # fallback is solely for already-validated native contract fixtures that carry decoded_display_height directly.
        if hasattr(session, "roots"):
            source = resolve_source_video(session.roots.source_video_root, str(item["source_relative_path"]))
            frames = validate_source_media_frames(source_path=source, expected_sha256=str(item["source_sha256"]), artifact_dir=artifact,
                                                  expected_frames=int(item["original_frame_count"]), retained_frames=int(item["used_time_length"]))
            height = frames.height
        else:
            height = int(raw.get("decoded_display_height", raw["coded_height"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, MediaError) as exc:
        raise ClassAttributionError(f"cannot validate source-pixel WPP geometry for {item.get('video_id', artifact)}: {exc}") from exc
    rows = int(item["row_count"])
    expected_rows = (height + ctu - 1) // ctu
    if height <= 0 or ctu <= 0 or rows != expected_rows:
        raise ClassAttributionError(f"WPP rows ({rows}) do not equal ceil(display_height/ctu_size) ({expected_rows})")
    low = np.arange(rows, dtype=np.float64) * ctu / height
    high = np.minimum(np.arange(1, rows + 1, dtype=np.float64) * ctu, height) / height
    return low, high, ctu


def _validate_expected_entry(entry: Any, class_names: list[str]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ClassAttributionError("cache expected-video entry is not an object")
    required = {"video_id", "partition", "source_sha256", "class_id"}
    optional = {"label", "row_count", "used_frames"}
    if not required.issubset(entry) or not set(entry).issubset(required | optional) or not isinstance(entry["video_id"], str) or entry["partition"] not in _SPLITS:
        raise ClassAttributionError("cache expected-video entry has an invalid schema")
    class_id = entry["class_id"]
    if (not isinstance(class_id, int) or isinstance(class_id, bool) or not 0 <= class_id < len(class_names) or
            not isinstance(entry["source_sha256"], str)):
        raise ClassAttributionError("cache expected-video identity is invalid")
    result = dict(entry); result.setdefault("label", class_names[class_id])
    if result["label"] != class_names[class_id]: raise ClassAttributionError("cache expected-video label is invalid")
    for key in ("row_count", "used_frames"):
        if key in result and (not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] <= 0 or (key == "used_frames" and result[key] > 3000)):
            raise ClassAttributionError("cache expected-video shape is invalid")
    return result


def _validated_cache_record(meta_path: Path, npz_path: Path, *, fingerprint: Mapping[str, Any],
                            expected: Mapping[str, Any], geometry_fingerprint: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Strictly validate one immutable record.  Callers may recompute or fail, never coerce it."""
    meta = _read_json(meta_path)
    class_names = list(fingerprint.get("class_names", [])); expected = _validate_expected_entry(expected, class_names)
    if (meta.get("cache_fingerprint") != fingerprint or meta.get("video_id") != expected["video_id"] or
            meta.get("partition") != expected["partition"] or meta.get("source_sha256") != expected["source_sha256"] or
            meta.get("ground_truth_class_id") != expected["class_id"] or meta.get("ground_truth_label") != expected["label"] or
            ("row_count" in expected and meta.get("row_count") != expected["row_count"]) or
            ("used_frames" in expected and meta.get("used_frames") != expected["used_frames"]) or
            (geometry_fingerprint is not None and meta.get("geometry_fingerprint") != geometry_fingerprint)):
        raise ClassAttributionError("per-video cache metadata identity does not match its manifest")
    predicted = meta.get("predicted_class_id")
    if (not isinstance(predicted, int) or isinstance(predicted, bool) or not 0 <= predicted < len(class_names) or
            meta.get("predicted_label") != class_names[predicted] or not isinstance(meta.get("correct"), bool) or
            meta["correct"] != (predicted == expected["class_id"])):
        raise ClassAttributionError("per-video prediction metadata is inconsistent")
    expected_truth_source = "predicted_reused" if meta["correct"] else "separate_truth_logit"
    if meta.get("ig_truth_source") != expected_truth_source:
        raise ClassAttributionError("per-video IG target provenance is inconsistent")
    digest = meta.get("npz_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != _sha256(npz_path):
        raise ClassAttributionError("per-video NPZ checksum is missing, malformed, or differs")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ClassAttributionError(f"cannot read per-video cache: {npz_path}") from exc
    required = {"ig_pred_signed", "valid_mask", "attention_token_mass", "attention_token_mask", "vertical_low", "vertical_high"}
    expected_names = required | (set() if meta["correct"] else {"ig_truth_signed"})
    if set(arrays) != expected_names:
        raise ClassAttributionError("per-video NPZ array schema is inconsistent")
    rows, frames = meta.get("row_count"), meta.get("used_frames")
    if (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0 or not isinstance(frames, int) or
            isinstance(frames, bool) or not 0 < frames <= 3000):
        raise ClassAttributionError("per-video native shape metadata is invalid")
    predicted_ig = arrays["ig_pred_signed"]
    if predicted_ig.dtype != np.float32 or predicted_ig.ndim != 2 or not predicted_ig.shape[0] or not predicted_ig.shape[1] or not np.isfinite(predicted_ig).all():
        raise ClassAttributionError("ig_pred_signed must be finite float32 native cells")
    if predicted_ig.shape != (rows, frames): raise ClassAttributionError("native IG shape disagrees with metadata")
    if "row_count" in expected and (rows, frames) != (expected["row_count"], expected["used_frames"]):
        raise ClassAttributionError("native IG shape disagrees with manifest")
    if not meta["correct"]:
        value = arrays["ig_truth_signed"]
        if value.dtype != np.float32 or value.shape != (rows, frames) or not np.isfinite(value).all():
            raise ClassAttributionError("ig_truth_signed must be finite float32 native cells with the expected shape")
    valid = arrays["valid_mask"]
    if valid.dtype != np.bool_ or valid.shape != (rows, frames) or not valid.all():
        raise ClassAttributionError("valid_mask must exactly cover trimmed native cells")
    patch_size = fingerprint.get("model_config", {}).get("patch_size", 16)
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
        raise ClassAttributionError("cache model patch size is invalid")
    metadata_patch_size = meta.get("patch_size")
    if (not isinstance(metadata_patch_size, int) or isinstance(metadata_patch_size, bool)
            or metadata_patch_size != patch_size):
        raise ClassAttributionError("per-video cache patch size does not match cache fingerprint")
    total_tokens = int(np.ceil(3000 / patch_size)); valid_tokens = int(np.ceil(frames / patch_size))
    attention, token_mask = arrays["attention_token_mass"], arrays["attention_token_mask"]
    if (attention.dtype != np.float32 or attention.shape not in {(rows, total_tokens), (rows, valid_tokens)} or not np.isfinite(attention).all() or
            np.any(attention < 0) or token_mask.dtype != np.bool_ or token_mask.shape != attention.shape or
            not np.all(token_mask[:, :valid_tokens]) or np.any(token_mask[:, valid_tokens:]) or
            np.any(attention[~token_mask] != 0) or not np.isclose(float(attention[token_mask].sum()), 1.0, rtol=1e-4, atol=1e-6)): 
        raise ClassAttributionError("attention native-token mass/mask is invalid")
    low, high = arrays["vertical_low"], arrays["vertical_high"]
    if (low.dtype != np.float64 or high.dtype != np.float64 or low.shape != (rows,) or high.shape != (rows,) or
            not np.isfinite(low).all() or not np.isfinite(high).all() or not np.isclose(low[0], 0.0) or
            not np.isclose(high[-1], 1.0) or np.any(high <= low) or not np.allclose(low[1:], high[:-1], rtol=0, atol=1e-12)):
        raise ClassAttributionError("physical vertical-band intervals are invalid")
    return meta, arrays


def _geometry_fingerprint_from(geometry: tuple[np.ndarray, np.ndarray, int]) -> str:
    low, high, ctu = geometry
    encoded = json.dumps({"low": low.tolist(), "high": high.tolist(), "ctu_size": ctu},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_fingerprint(session: Any, item: Mapping[str, Any]) -> str:
    return _geometry_fingerprint_from(_artifact_geometry(session, item))


def _compute_one(session: Any, item: Mapping[str, Any], *, ig_steps: int, chunk_queries: int,
                 callback: ProgressCallback | None, geometry: tuple[np.ndarray, np.ndarray, int] | None = None,
                 geometry_fingerprint: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _, normalized, rows, times = _batch_tensors(item, session)
    with torch.no_grad():
        logits = session.model(normalized, rows, times, normalized=True)
        predicted = int(logits.argmax(dim=1).item())
        probabilities = torch.softmax(logits, dim=1)
    truth = int(item["class_id"])
    pred_ig, pred_diagnostics = _integrated_gradients(session.model, normalized, rows, times, predicted,
                                                       ig_steps, callback)
    truth_ig: torch.Tensor | None = None
    truth_diagnostics: Mapping[str, float] | None = None
    if predicted != truth:
        truth_ig, truth_diagnostics = _integrated_gradients(session.model, normalized, rows, times, truth,
                                                             ig_steps, callback)
    traced_logits, attention = _attention_summary(session.model, normalized, rows, times,
                                                  row_count=int(item["row_count"]),
                                                  used_frames=int(item["used_time_length"]),
                                                  chunk_queries=chunk_queries, callback=callback)
    if not torch.allclose(logits, traced_logits, rtol=1e-5, atol=1e-6):
        raise ClassAttributionError("attention trace logits differ from prediction logits")
    row_count, frames = int(item["row_count"]), int(item["used_time_length"])
    vertical_low, vertical_high, ctu = geometry if geometry is not None else _artifact_geometry(session, item)
    geometry_fingerprint = geometry_fingerprint or _geometry_fingerprint_from((vertical_low, vertical_high, ctu))
    arrays: dict[str, np.ndarray] = {
        "ig_pred_signed": pred_ig[0, :row_count, :frames].detach().cpu().numpy().astype(np.float32, copy=False),
        "valid_mask": np.ones((row_count, frames), dtype=np.bool_),
        # Native tokens, not repeated frames: each temporal patch is split over its valid progress interval later.
        "attention_token_mass": attention.rollout_tokens[:row_count].astype(np.float32, copy=False),
        "attention_token_mask": (np.arange(attention.rollout_tokens.shape[1])[None, :]
                                 < int(np.ceil(frames / attention.patch_size))).repeat(row_count, axis=0),
        "vertical_low": vertical_low.astype(np.float64), "vertical_high": vertical_high.astype(np.float64),
    }
    if truth_ig is not None:
        arrays["ig_truth_signed"] = truth_ig[0, :row_count, :frames].detach().cpu().numpy().astype(np.float32, copy=False)
    metadata = {
        "video_id": str(item["video_id"]), "partition": str(item["partition"]),
        "source_sha256": str(item["source_sha256"]),
        "ground_truth_class_id": truth, "ground_truth_label": session.class_names[truth],
        "predicted_class_id": predicted, "predicted_label": session.class_names[predicted],
        "predicted_logit": float(logits[0, predicted].item()),
        "predicted_probability": float(probabilities[0, predicted].item()),
        "correct": predicted == truth, "ig_truth_source": "predicted_reused" if predicted == truth else "separate_truth_logit",
        "row_count": row_count, "used_frames": frames, "patch_size": int(attention.patch_size), "ctu_size": ctu,
        "ig_pred_diagnostics": pred_diagnostics, "ig_truth_diagnostics": truth_diagnostics,
        # Rechecked on resume so a changed native physical-geometry artifact cannot be silently reused.
        "geometry_fingerprint": geometry_fingerprint,
    }
    return metadata, arrays


def _build_class_attribution_cache_unlocked(training_root: str | Path, *, roots: DataRoots,
                                  cache_dir: str | Path | None = None,
                                  splits: Iterable[Literal["train", "test"]] = _SPLITS,
                                  ig_steps: int = 32, attention_chunk_queries: int = 128,
                                  device: str = "auto", progress: ProgressCallback | None = None,
                                  should_cancel: Callable[[], bool] | None = None,
                                  max_new_videos: int | None = None,
                                  prioritize_video_ids: Iterable[str] = ()) -> CacheBuildResult:
    """Populate a resumable cache from ``training_root/checkpoints/best.pt``.

    Existing cache files are reused only when the full checkpoint/data/settings
    fingerprint and per-video metadata match.  Cancellation or
    ``max_new_videos`` leaves a valid explicitly incomplete manifest;
    aggregation rejects it unless ``preview`` is explicitly requested.
    ``prioritize_video_ids`` changes only compute order, never the cache's
    canonical expected-video set, so a one-video pilot resumes into the same
    full train/test cache.
    """
    ig_steps = _validate_positive_int(ig_steps, "ig_steps")
    if ig_steps < 2: raise ClassAttributionError("ig_steps must be at least 2")
    attention_chunk_queries = _validate_positive_int(attention_chunk_queries, "attention_chunk_queries")
    if max_new_videos is not None and (not isinstance(max_new_videos, int) or isinstance(max_new_videos, bool) or max_new_videos < 0):
        raise ClassAttributionError("max_new_videos must be a nonnegative integer or None")
    priorities = tuple(prioritize_video_ids)
    if any(not isinstance(video_id, str) for video_id in priorities):
        raise ClassAttributionError("prioritize_video_ids must contain strings")
    split_tuple = tuple(splits)
    if not split_tuple or len(set(split_tuple)) != len(split_tuple):
        raise ClassAttributionError("splits must be a nonempty unique selection")
    training = Path(training_root).expanduser().resolve()
    destination = (Path(cache_dir).expanduser().resolve() if cache_dir is not None
                   else training / "class_attributions")
    session = open_explainer(training / "checkpoints" / "best.pt", roots=roots, device=device, progress=progress)
    fingerprint = _cache_fingerprint(session, splits=split_tuple, ig_steps=ig_steps,
                                     attention_chunk_queries=attention_chunk_queries)
    items = _record_items(session, split_tuple)
    expected = [{"video_id": str(item["video_id"]), "partition": split,
                 "source_sha256": str(item.get("source_sha256", "")), "class_id": int(item["class_id"]),
                 "label": str(item.get("label", session.class_names[int(item["class_id"])])),
                 "row_count": int(item["row_count"]), "used_frames": int(item["used_time_length"])} for split, item in items]
    expected_by_identity = {(entry["partition"], entry["video_id"]): entry for entry in expected}
    if len(expected_by_identity) != len(expected):
        raise ClassAttributionError("train/test cache cannot contain duplicate video/partition identities")
    priority_order = {video_id: position for position, video_id in enumerate(priorities)}
    work_items = sorted(items, key=lambda value: priority_order.get(str(value[1]["video_id"]), len(priority_order)))
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()):
        existing = _read_json(manifest_path) if manifest_path.is_file() else None
        if existing is None or existing.get("fingerprint") != fingerprint or existing.get("expected_videos") != expected:
            raise ClassAttributionError("cache exists with different checkpoint, data, splits, or settings; choose a fresh cache_dir")
    else:
        destination.mkdir(parents=True, exist_ok=True)
    # Count every on-disk record that passes the full cache schema/checksum contract before
    # traversal. This makes progress truthful when a prioritized pilot cached a later video.
    # Geometry/source-media validation remains once per traversed item below; this scan never
    # hashes source media or probes it again.
    valid_on_disk: set[tuple[str, str]] = set()
    for identity, expected_entry in expected_by_identity.items():
        stem = _safe_video_name(expected_entry["video_id"])
        meta_path, npz_path = destination / "videos" / f"{stem}.json", destination / "videos" / f"{stem}.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            continue
        try:
            _validated_cache_record(meta_path, npz_path, fingerprint=fingerprint, expected=expected_entry)
            valid_on_disk.add(identity)
        except (OSError, ValueError, ClassAttributionError):
            # It is deliberately not counted. The normal traversal will atomically replace it.
            continue
    manifest = {"schema": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint, "expected_videos": expected,
                "status": "building", "completed": len(valid_on_disk), "total": len(items)}
    _atomic_json(manifest_path, manifest)
    completed, reused, computed = len(valid_on_disk), 0, 0
    for ordinal, (split, item) in enumerate(work_items, 1):
        if should_cancel is not None and should_cancel():
            manifest.update(status="cancelled", completed=completed)
            _atomic_json(manifest_path, manifest)
            return CacheBuildResult(destination, False, completed, len(items), reused, computed, session.checkpoint_sha256)
        stem = _safe_video_name(str(item["video_id"])); meta_path = destination / "videos" / f"{stem}.json"; npz_path = destination / "videos" / f"{stem}.npz"
        valid_existing = False
        # Dataset provenance binds model values; this separately binds the physical geometry used for display aggregation.
        geometry = _artifact_geometry(session, item)
        geometry_fingerprint = _geometry_fingerprint_from(geometry)
        identity_key = (split, str(item["video_id"]))
        if meta_path.is_file() and npz_path.is_file():
            try:
                _validated_cache_record(meta_path, npz_path, fingerprint=fingerprint,
                                        expected=expected_by_identity[identity_key], geometry_fingerprint=geometry_fingerprint)
                valid_existing = True
            except (OSError, ValueError, ClassAttributionError):
                valid_existing = False
        if valid_existing:
            reused += 1; _emit(progress, "class_attribution_cache", "reused cached video", ordinal, len(items)); continue
        # A record may have passed the no-media pre-scan but fail current source geometry validation.
        if identity_key in valid_on_disk:
            valid_on_disk.remove(identity_key); completed -= 1
        if max_new_videos is not None and computed >= max_new_videos:
            manifest.update(status="paused", completed=completed)
            _atomic_json(manifest_path, manifest)
            return CacheBuildResult(destination, False, completed, len(items), reused, computed, session.checkpoint_sha256)
        try:
            metadata, arrays = _compute_one(session, item, ig_steps=ig_steps, chunk_queries=attention_chunk_queries,
                                            callback=progress, geometry=geometry, geometry_fingerprint=geometry_fingerprint)
        except RuntimeError as exc:
            if not _accelerator_runtime_error(exc, session.device): raise
            reason = f"{session.device}: {exc}"; _move_to_cpu(session); session.fallback_reason = reason
            metadata, arrays = _compute_one(session, item, ig_steps=ig_steps, chunk_queries=attention_chunk_queries,
                                            callback=progress, geometry=geometry, geometry_fingerprint=geometry_fingerprint)
        identity = expected_by_identity[(split, str(item["video_id"]))]
        # Make the cache identity explicit even if a test/different compute implementation omitted it.
        metadata.update(source_sha256=identity["source_sha256"], ground_truth_class_id=identity["class_id"],
                        ground_truth_label=identity["label"], row_count=identity["row_count"],
                        used_frames=identity["used_frames"])
        metadata["cache_fingerprint"] = fingerprint
        metadata["geometry_fingerprint"] = geometry_fingerprint
        # Arrays first then metadata: an interruption cannot advertise an incomplete NPZ as ready.
        _atomic_npz(npz_path, **arrays)
        metadata["npz_sha256"] = _sha256(npz_path)
        _atomic_json(meta_path, metadata)
        valid_on_disk.add(identity_key); completed += 1; computed += 1
        manifest.update(completed=completed); _atomic_json(manifest_path, manifest)
        _emit(progress, "class_attribution_cache", "cached video", ordinal, len(items))
    manifest.update(status="complete", completed=completed); _atomic_json(manifest_path, manifest)
    return CacheBuildResult(destination, True, completed, len(items), reused, computed, session.checkpoint_sha256)


def build_class_attribution_cache(training_root: str | Path, *, roots: DataRoots,
                                  cache_dir: str | Path | None = None,
                                  splits: Iterable[Literal["train", "test"]] = _SPLITS,
                                  ig_steps: int = 32, attention_chunk_queries: int = 128,
                                  device: str = "auto", progress: ProgressCallback | None = None,
                                  should_cancel: Callable[[], bool] | None = None,
                                  max_new_videos: int | None = None,
                                  prioritize_video_ids: Iterable[str] = ()) -> CacheBuildResult:
    """Build/resume a cache under an exclusive destination writer lock."""
    training = Path(training_root).expanduser().resolve()
    destination = Path(cache_dir).expanduser().resolve() if cache_dir is not None else training / "class_attributions"
    lock = _CacheWriterLock(destination); lock.acquire()
    try:
        return _build_class_attribution_cache_unlocked(
            training, roots=roots, cache_dir=destination, splits=splits, ig_steps=ig_steps,
            attention_chunk_queries=attention_chunk_queries, device=device, progress=progress,
            should_cancel=should_cancel, max_new_videos=max_new_videos,
            prioritize_video_ids=prioritize_video_ids)
    finally:
        lock.release()


def _overlap_weights(low: np.ndarray, high: np.ndarray, bins: int) -> np.ndarray:
    """Rows are destination bins; each source interval's weights sum exactly one."""
    if low.ndim != 1 or high.shape != low.shape or np.any(high <= low) or np.any(low < 0) or np.any(high > 1):
        raise ClassAttributionError("source coordinate intervals must be nonempty normalized intervals")
    starts = np.arange(bins, dtype=np.float64) / bins; ends = (np.arange(bins, dtype=np.float64) + 1) / bins
    widths = high - low
    return np.maximum(0.0, np.minimum(ends[:, None], high[None, :]) - np.maximum(starts[:, None], low[None, :])) / widths[None, :]


def _resample_native_mass(values: np.ndarray, vertical_low: np.ndarray, vertical_high: np.ndarray,
                          *, time_bins: int, band_bins: int, token_width: int | None = None) -> np.ndarray:
    """Split native cell/token mass by overlap; no value interpolation or duplication."""
    if values.ndim != 2 or values.shape[0] != len(vertical_low):
        raise ClassAttributionError("native attribution array and vertical geometry disagree")
    vertical = _overlap_weights(np.asarray(vertical_low), np.asarray(vertical_high), band_bins)
    if token_width is None:
        time_low = np.arange(values.shape[1], dtype=np.float64) / values.shape[1]
        time_high = (np.arange(values.shape[1], dtype=np.float64) + 1) / values.shape[1]
    else:
        if token_width <= 0: raise ClassAttributionError("token_width must be positive")
        # Final valid patch ends at the presentation end, not at its padded 16-frame end.
        frames = (values.shape[1] - 1) * token_width + min(token_width, token_width)
        # Caller trims invalid tokens, so actual final interval is supplied by its count only if full;
        # derive the true progress endpoint from native valid frame count in aggregation instead.
        time_low = np.arange(values.shape[1], dtype=np.float64) / values.shape[1]
        time_high = (np.arange(values.shape[1], dtype=np.float64) + 1) / values.shape[1]
    temporal = _overlap_weights(time_low, time_high, time_bins)
    result = vertical @ np.asarray(values, dtype=np.float64) @ temporal.T
    return result.astype(np.float64, copy=False)


def _resample_attention(values: np.ndarray, vertical_low: np.ndarray, vertical_high: np.ndarray, *,
                        frames: int, patch_size: int, time_bins: int, band_bins: int) -> np.ndarray:
    tokens = int(np.ceil(frames / patch_size)); native = values[:, :tokens]
    low = np.arange(tokens, dtype=np.float64) * patch_size / frames
    high = np.minimum(np.arange(1, tokens + 1, dtype=np.float64) * patch_size, frames) / frames
    return _overlap_weights(vertical_low, vertical_high, band_bins) @ native @ _overlap_weights(low, high, time_bins).T


def _selected_classes(classes: Iterable[int | str] | None, class_names: list[str]) -> set[int] | None:
    if classes is None: return None
    result: set[int] = set()
    for value in classes:
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(class_names): result.add(value)
        elif isinstance(value, str) and value in class_names: result.add(class_names.index(value))
        else: raise ClassAttributionError(f"unknown class selection: {value!r}")
    return result


def aggregate_class_attributions(cache_dir: str | Path, *, split: Literal["separate", "combined"] = "separate",
                                 group_by: Literal["gt", "pred"] = "gt",
                                 correctness: Literal["all", "correct", "incorrect"] = "all",
                                 classes: Iterable[int | str] | None = None,
                                 ig_target: Literal["predicted", "pred", "truth"] = "predicted",
                                 sign: Literal["abs", "signed"] = "abs",
                                 magnitude: Literal["raw", "per_video_normalized"] = "raw",
                                 grid: tuple[int, int] = (32, 192), include_pooled_misclassified: bool = True,
                                 preview: bool = False) -> AttributionAggregate:
    """Aggregate cached native values without model inference.

    ``grid`` is ``(physical_vertical_band_bins, presentation_progress_bins)``.
    Its default therefore returns a visible ``[32, 192]`` [band, progress]
    grid. In ``raw`` mode each group is the mean native mass per video.  In ``per_video_normalized``
    mode each video is divided by native L1 mass before resampling, so signed
    values retain sign and abs values cannot cancel within a coarse grid cell.
    """
    if split not in ("separate", "combined") or group_by not in ("gt", "pred") or correctness not in ("all", "correct", "incorrect"):
        raise ClassAttributionError("invalid split, group_by, or correctness option")
    if ig_target not in ("predicted", "pred", "truth") or sign not in ("abs", "signed") or magnitude not in ("raw", "per_video_normalized"):
        raise ClassAttributionError("invalid IG target, sign, or magnitude option")
    # ``pred`` remains a compact backwards-compatible spelling; public/UI text uses ``predicted``.
    ig_target = "predicted" if ig_target == "pred" else ig_target
    if not isinstance(grid, tuple) or len(grid) != 2: raise ClassAttributionError("grid must be (band_bins, time_bins)")
    band_bins, time_bins = (_validate_positive_int(grid[0], "band_bins"), _validate_positive_int(grid[1], "time_bins"))
    root = Path(cache_dir).expanduser().resolve(); manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA_VERSION: raise ClassAttributionError("unsupported cache schema")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, Mapping): raise ClassAttributionError("cache fingerprint is missing")
    class_names = list(fingerprint.get("class_names", []))
    if not class_names or any(not isinstance(name, str) for name in class_names) or len(set(class_names)) != len(class_names):
        raise ClassAttributionError("cache class mapping is invalid")
    expected_raw = manifest.get("expected_videos")
    if not isinstance(expected_raw, list): raise ClassAttributionError("cache expected_videos must be a list")
    expected = [_validate_expected_entry(entry, class_names) for entry in expected_raw]
    identities = [(entry["partition"], entry["video_id"]) for entry in expected]
    if len(set(identities)) != len(identities) or len({entry["video_id"] for entry in expected}) != len(expected):
        raise ClassAttributionError("cache manifest has duplicate video identities")
    total, completed, status = manifest.get("total"), manifest.get("completed"), manifest.get("status")
    if (not isinstance(total, int) or isinstance(total, bool) or total != len(expected) or
            not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= total or
            status not in {"building", "paused", "cancelled", "complete"}):
        raise ClassAttributionError("cache manifest status/counts are invalid")
    complete = status == "complete" and completed == total
    if not complete and not preview: raise ClassAttributionError("cache is incomplete; resume it or request preview=True explicitly")
    selected = _selected_classes(classes, class_names)
    sources: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    missing = 0
    for entry in expected:
        stem = _safe_video_name(entry["video_id"]); meta_path = root / "videos" / f"{stem}.json"; data_path = root / "videos" / f"{stem}.npz"
        if not meta_path.is_file() or not data_path.is_file():
            missing += 1; continue
        # A present malformed/tampered record is a hard error even in preview; preview permits only absence.
        meta, arrays = _validated_cache_record(meta_path, data_path, fingerprint=fingerprint, expected=entry)
        sources.append((meta, arrays))
    if complete and missing: raise ClassAttributionError("cache manifest is complete but video cache files are missing")
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
    for meta, arrays in sources:
        is_correct = bool(meta["correct"])
        if correctness == "correct" and not is_correct: continue
        if correctness == "incorrect" and is_correct: continue
        class_id = int(meta["ground_truth_class_id"] if group_by == "gt" else meta["predicted_class_id"])
        if selected is not None and class_id not in selected: continue
        section = str(meta["partition"]) if split == "separate" else "combined"
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        buckets.setdefault(f"{section}|{group_by}:{label}", []).append((meta, arrays))
    # Requested selected classes are explicit even when empty.
    sections = list(fingerprint.get("splits", [])) if split == "separate" else ["combined"]
    candidate_ids = sorted(selected if selected is not None else range(len(class_names)))
    for section in sections:
        for cid in candidate_ids:
            buckets.setdefault(f"{section}|{group_by}:{class_names[cid]}", [])
    if include_pooled_misclassified:
        for section in sections:
            pooled = [(m, a) for m, a in sources if not bool(m["correct"]) and (split == "combined" or m["partition"] == section)]
            if selected is not None:
                pooled = [(m, a) for m, a in pooled if int(m["ground_truth_class_id"] if group_by == "gt" else m["predicted_class_id"]) in selected]
            buckets[f"{section}|pooled_misclassified"] = pooled if correctness != "correct" else []
    groups: dict[str, AttributionGroup] = {}
    for key, entries in buckets.items():
        ig_total = np.zeros((band_bins, time_bins), dtype=np.float64); attention_total = np.zeros_like(ig_total)
        for meta, arrays in entries:
            required = {"ig_pred_signed", "valid_mask", "attention_token_mass", "attention_token_mask", "vertical_low", "vertical_high"}
            if not required.issubset(arrays): raise ClassAttributionError(f"cache video is missing required arrays: {meta['video_id']}")
            raw = arrays["ig_pred_signed"] if ig_target == "predicted" or bool(meta["correct"]) else arrays.get("ig_truth_signed")
            if raw is None: raise ClassAttributionError("misclassified cache lacks truth-logit IG")
            valid = arrays["valid_mask"]
            if raw.shape != valid.shape or valid.dtype != np.bool_ or not valid.all():
                raise ClassAttributionError("cached IG valid mask does not exactly cover native cache values")
            native = np.abs(raw, dtype=np.float64) if sign == "abs" else raw.astype(np.float64, copy=False)
            denominator = float(np.abs(raw, dtype=np.float64).sum())
            if magnitude == "per_video_normalized" and denominator > 0: native = native / denominator
            ig_total += _resample_native_mass(native, arrays["vertical_low"], arrays["vertical_high"],
                                              time_bins=time_bins, band_bins=band_bins)
            attention = arrays["attention_token_mass"].astype(np.float64, copy=False)
            token_mask = arrays["attention_token_mask"]
            expected_tokens = int(np.ceil(int(meta["used_frames"]) / int(meta["patch_size"])))
            if (token_mask.shape != attention.shape or token_mask.dtype != np.bool_ or
                    not np.all(token_mask[:, :expected_tokens]) or np.any(token_mask[:, expected_tokens:])):
                raise ClassAttributionError("cached attention token mask does not exactly cover valid presentation patches")
            attention_denom = float(attention[:, :expected_tokens].sum())
            if magnitude == "per_video_normalized" and attention_denom > 0: attention = attention / attention_denom
            attention_total += _resample_attention(attention[:, :expected_tokens], arrays["vertical_low"], arrays["vertical_high"],
                                                   frames=int(meta["used_frames"]), patch_size=int(meta["patch_size"]),
                                                   time_bins=time_bins, band_bins=band_bins)
        count = len(entries)
        if count: ig_total /= count; attention_total /= count
        section, label = key.split("|", 1)
        groups[key] = AttributionGroup(key, section, label, count, ig_total, attention_total)
    metadata = {"schema": AGGREGATION_SCHEMA_VERSION, "cache_complete": complete, "preview": preview,
                "expected_videos": len(expected), "available_videos": len(sources), "missing_videos": missing,
                "settings": {"split": split, "group_by": group_by, "correctness": correctness,
                             "classes": None if selected is None else sorted(selected), "ig_target": ig_target,
                             "sign": sign, "magnitude": magnitude, "grid": {"band_bins": band_bins, "time_bins": time_bins}}, 
                "semantics": {"averaging": "equal per-video mean", "ig": "absolute value is applied to native signed cells before overlap resampling",
                              "resampling": "source mass is split by physical source-pixel vertical-band and normalized retained presentation-frame-index interval overlap (not elapsed time)",
                              "signed_unit_total": "native IG L1 denominator", "attention": "native token mass; no repeated-frame inflation"},
                "cache_fingerprint": fingerprint}
    return AttributionAggregate(groups, metadata)


def export_class_attributions(result: AttributionAggregate, output_dir: str | Path, *, zip_output: bool = False) -> Path:
    """Export numeric CSV grids plus metadata; no plotting/UI dependency is required."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists(): raise ClassAttributionError("export output_dir must be fresh")
    destination.mkdir(parents=True); data = destination / "data"; data.mkdir()
    group_manifest: list[dict[str, Any]] = []
    for ordinal, (key, group) in enumerate(result.groups.items()):
        stem = f"group-{ordinal:04d}"
        np.savetxt(data / f"{stem}-ig.csv", group.ig_grid, delimiter=",")
        np.savetxt(data / f"{stem}-attention.csv", group.attention_grid, delimiter=",")
        group_manifest.append({"key": key, "split": group.split, "label": group.label, "count": group.count,
                               "ig_csv": f"data/{stem}-ig.csv", "attention_csv": f"data/{stem}-attention.csv"})
    _atomic_json(destination / "metadata.json", {"metadata": result.metadata, "groups": group_manifest})
    if zip_output:
        archive = destination.with_suffix(".zip")
        if archive.exists(): raise ClassAttributionError("ZIP output already exists")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(destination.rglob("*")):
                if path.is_file(): handle.write(path, path.relative_to(destination))
    return destination
