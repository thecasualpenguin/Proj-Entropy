"""Native-geometry/timing WPP extraction workflow.

This module intentionally stops at an extraction-index.  It does not create a
model-ready packet tensor or pad/truncate the producer's frame axis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .manifest_contract import (
    ContractError, load_extraction_index, load_source_manifest, portable_relative_path, sha256_file,
    validate_extraction_index_records, write_extraction_index,
)

PRODUCER_SCHEMA_VERSION = "2.0"
PROFILE_NAME = "entropy-cabac-wpp encode_wpp.sh defaults (gop=16, qp=28, bframes=0)"
FRAME_KEYS = (
    "display_index", "decode_index", "pts", "pts_seconds", "slice_type",
    "slice_qp", "temporal_id", "nal_type", "vcl_nal_bytes", "header_bytes",
    "payload_bytes", "cu_qp_delta_enabled",
)
ROW_ARRAY_KEYS = (
    "row_bytes", "row_share", "log1p_row_bytes", "row_bits_per_coded_pixel",
    "delta_row_share",
)


class WorkflowError(RuntimeError):
    """A configuration, provenance, or per-video workflow error."""


class UnsafeOutputError(WorkflowError):
    """A public output directory changed while a child held a pinned FD.

    This is a build-level safety failure, not an ordinary source failure: do not
    attempt pathname cleanup or publish a failure receipt after it.
    """


@dataclass(frozen=True)
class PresentationProbe:
    decoded_width: int
    decoded_height: int
    coded_width: int | None
    coded_height: int | None
    sample_aspect_ratio: str
    time_base: Fraction
    presentation_times: tuple[Fraction, ...]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _reject_symlink_ancestors(path: Path, *, include_leaf: bool = True, field: str = "path") -> Path:
    """Check raw absolute components without resolving through a symlink."""
    raw = Path(path).absolute()
    current = Path(raw.anchor)
    parts = raw.parts[1:]
    limit = len(parts) if include_leaf else max(0, len(parts) - 1)
    for part in parts[:limit]:
        current /= part
        if current.is_symlink():  # includes dangling links
            raise WorkflowError(f"{field} contains a symlink component: {current}")
    return raw


def _safe_source_path(source_root: Path, relative: str) -> Path:
    """Return a regular, non-symlink source under a non-symlink root."""
    root = _reject_symlink_ancestors(Path(source_root), field="source root")
    if root.is_symlink() or not root.is_dir():
        raise WorkflowError(f"source root must be a non-symlink directory: {root}")
    relative = portable_relative_path(relative, "source_relative_path")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise WorkflowError(f"source path contains a symlink component: {candidate}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"source path is missing or escapes source root: {relative}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise WorkflowError(f"source must be a non-symlink regular file: {relative}")
    return resolved


def _stable_hash(path: Path) -> tuple[str, os.stat_result]:
    """Hash a file while detecting a simple concurrent modification race."""
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise WorkflowError(f"source changed while hashing: {path}")
    return digest, after


def verified_stage_copy(source: Path, expected_sha256: str, expected_bytes: int, destination: Path) -> dict[str, Any]:
    """Hash original immediately before copying, then prove the staged bytes match.

    The final source stat check cannot prevent every hostile filesystem race, but it
    prevents ordinary mutation during hash/copy and the encoder only receives the
    verified staged copy.
    """
    actual, initial_stat = _stable_hash(source)
    if actual != expected_sha256:
        raise WorkflowError(f"source SHA-256 mismatch: {source}")
    if initial_stat.st_size != expected_bytes:
        raise WorkflowError(f"source byte count mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    staged_sha = sha256_file(destination)
    if staged_sha != expected_sha256:
        raise WorkflowError(f"staged source SHA-256 mismatch: {destination}")
    final_stat = source.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(initial_stat, field) != getattr(final_stat, field) for field in fields):
        raise WorkflowError(f"source changed while staging: {source}")
    return {"source_sha256": actual, "staged_source_sha256": staged_sha,
            "source_bytes": initial_stat.st_size,
            "source_stat": {field: getattr(initial_stat, field) for field in fields}}


def _fraction(value: Any, field: str) -> Fraction:
    if not isinstance(value, str) or not value or value == "N/A":
        raise WorkflowError(f"missing/invalid {field}")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkflowError(f"missing/invalid {field}: {value!r}") from exc
    if result <= 0:
        raise WorkflowError(f"{field} must be positive")
    return result


def _sar(value: Any) -> str:
    if value in (None, "", "N/A", "0:1"):
        return "1:1"
    if not isinstance(value, str) or ":" not in value:
        raise WorkflowError(f"invalid sample_aspect_ratio: {value!r}")
    try:
        fraction = Fraction(value.replace(":", "/"))
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkflowError(f"invalid sample_aspect_ratio: {value!r}") from exc
    if fraction <= 0:
        raise WorkflowError(f"invalid sample_aspect_ratio: {value!r}")
    return f"{fraction.numerator}:{fraction.denominator}"


def probe_presentation(path: Path, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
                       pass_fds: tuple[int, ...] = ()) -> tuple[PresentationProbe, dict[str, Any]]:
    """Probe decoded display frames using exact best-effort presentation PTS.

    This deliberately does not count packets: originals may legitimately have
    multiple packets/slices per decoded frame.  The unchanged producer retains its
    stricter one-picture MP4-packet validation for the encoded HEVC it extracts.
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
           "-show_entries", "stream=codec_type,width,height,coded_width,coded_height,sample_aspect_ratio,time_base:frame=media_type,best_effort_timestamp",
           "-of", "json", str(path)]
    completed = run(cmd, check=True, capture_output=True, text=True, pass_fds=pass_fds)
    try:
        parsed = json.loads(completed.stdout)
        streams = [s for s in parsed.get("streams", []) if s.get("codec_type") == "video"]
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"ffprobe returned incomplete video geometry for {path}") from exc
    if width <= 0 or height <= 0:
        raise WorkflowError(f"ffprobe returned nonpositive decoded geometry for {path}")
    time_base = _fraction(stream.get("time_base"), "stream time_base")
    pts: list[int] = []
    for frame in parsed.get("frames", []):
        if frame.get("media_type") != "video":
            continue
        if "best_effort_timestamp" not in frame:
            raise WorkflowError(f"missing decoded-frame best-effort PTS: {path}")
        try:
            pts.append(int(frame["best_effort_timestamp"]))
        except (TypeError, ValueError) as exc:
            raise WorkflowError(f"invalid decoded-frame best-effort PTS: {path}") from exc
    if not pts:
        raise WorkflowError(f"no decoded video frames: {path}")
    if len(set(pts)) != len(pts):
        raise WorkflowError(f"duplicate decoded-frame PTS: {path}")
    presentation = tuple(Fraction(value) * time_base for value in sorted(pts))
    probe = PresentationProbe(width, height,
                              int(stream["coded_width"]) if stream.get("coded_width") is not None else None,
                              int(stream["coded_height"]) if stream.get("coded_height") is not None else None,
                              _sar(stream.get("sample_aspect_ratio")), time_base, presentation)
    record = {"command": cmd, "returncode": completed.returncode,
              "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
              "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest()}
    return probe, record


def compare_native_geometry_timing(source: PresentationProbe, encoded: PresentationProbe) -> dict[str, Any]:
    """Require equal decoded/display geometry and a bounded exact PTS timeline.

    Output timebase and global timestamp origin may change.  Every normalized output
    presentation time must remain within one output tick of the corresponding source
    time; this pointwise bound prevents cumulative drift and retains VFR gaps.
    """
    if (source.decoded_width, source.decoded_height) != (encoded.decoded_width, encoded.decoded_height):
        raise WorkflowError("decoded/display width or height changed during encode")
    if source.sample_aspect_ratio != encoded.sample_aspect_ratio:
        raise WorkflowError("sample aspect ratio changed during encode")
    if len(source.presentation_times) != len(encoded.presentation_times):
        raise WorkflowError("frame count changed during encode")
    tolerance = encoded.time_base
    source_zero, encoded_zero = source.presentation_times[0], encoded.presentation_times[0]
    errors = [abs((out - encoded_zero) - (original - source_zero))
              for original, out in zip(source.presentation_times, encoded.presentation_times)]
    if any(error > tolerance for error in errors):
        raise WorkflowError("relative presentation timeline changed beyond one output tick")
    return {
        "source": geometry_record(source), "encoded": geometry_record(encoded),
        "global_origin_offset_seconds": str(encoded_zero - source_zero),
        "output_tick_tolerance_seconds": str(tolerance),
        "max_relative_timeline_error_seconds": str(max(errors)),
        "comparison": "decoded display frames: exact Fraction(best_effort_PTS*time_base) in ascending presentation order; origin normalized",
    }


def geometry_record(probe: PresentationProbe) -> dict[str, Any]:
    return {"decoded_display_width": probe.decoded_width, "decoded_display_height": probe.decoded_height,
            "sample_aspect_ratio": probe.sample_aspect_ratio, "coded_width": probe.coded_width,
            "coded_height": probe.coded_height, "time_base": str(probe.time_base),
            "presentation_frame_count": len(probe.presentation_times)}


def _run_logged(cmd: Sequence[str], *, run: Callable[..., subprocess.CompletedProcess[str]], log_path: Path,
                build_root: Path, pass_fds: tuple[int, ...] = ()) -> dict[str, Any]:
    try:
        relative = log_path.relative_to(Path(build_root).absolute()).as_posix()
    except ValueError as exc:
        raise WorkflowError("command log path escapes build root") from exc
    safe_log = _safe_owned(build_root, relative)
    # Per-attempt names are immutable.  O_EXCL|O_NOFOLLOW prevents both an
    # accidental overwrite and a leaf-link redirect between validation/write.
    if safe_log.exists() or safe_log.is_symlink():
        raise WorkflowError(f"refusing to overwrite existing command log: {safe_log}")
    safe_log.parent.mkdir(parents=True, exist_ok=True)
    _safe_owned(build_root, safe_log.parent.relative_to(Path(build_root).absolute()).as_posix(), directory=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        # Open before the untrusted child/callback.  This FD remains attached to
        # the original directory entry if somebody later renames the log parent.
        fd = os.open(safe_log, flags, 0o600)
    except OSError as exc:
        raise WorkflowError(f"cannot safely create command log: {safe_log}") from exc
    try:
        completed = run(list(cmd), check=False, capture_output=True, text=True, pass_fds=pass_fds)
        content = "$ " + " ".join(cmd) + "\n\n[stdout]\n" + completed.stdout + "\n[stderr]\n" + completed.stderr
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        log_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    result = {"command": list(cmd), "returncode": completed.returncode,
              "log_relative_path": relative, "log_sha256": log_sha}
    if completed.returncode != 0:
        raise WorkflowError(f"external command failed ({completed.returncode}): {' '.join(cmd[:2])}")
    return result


def _fd_anchored_path(path: Path) -> bool:
    """True only for a Linux descriptor path intentionally passed to a child."""
    parts = Path(path).parts
    return len(parts) >= 5 and parts[:4] == ("/", "proc", "self", "fd") and parts[4].isdigit()


def validate_artifact(directory: Path) -> dict[str, Any]:
    """Validate producer schema 2.0 and all unpadded frame-axis agreements."""
    # /proc/self/fd/N is a deliberately held, O_NOFOLLOW directory descriptor.
    # Do not resolve it back through its symlinked procfs presentation.
    directory = Path(directory) if _fd_anchored_path(Path(directory)) else _reject_symlink_ancestors(Path(directory), field="artifact directory")
    if directory.is_symlink() or not directory.is_dir():
        raise WorkflowError(f"artifact directory must be a non-symlink directory: {directory}")
    required = [directory / name for name in ("features.npz", "metadata.json", "rows.csv", "frames.csv")]
    if any(path.is_symlink() for path in required):
        raise WorkflowError("extractor artifact required file is a symlink")
    if any(not path.is_file() for path in required):
        raise WorkflowError(f"extractor output missing required schema files: {directory}")
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != PRODUCER_SCHEMA_VERSION:
            raise WorkflowError("extractor did not write producer schema 2.0")
        n, t = int(metadata["rows"]), int(metadata["frames"])
        artifact_time_base = _fraction(metadata.get("time_base"), "producer metadata time_base")
        if n <= 0 or t <= 0:
            raise WorkflowError("extractor wrote empty rows or frames")
        with np.load(directory / "features.npz", allow_pickle=False) as features:
            missing = set(ROW_ARRAY_KEYS + FRAME_KEYS) - set(features.files)
            if missing:
                raise WorkflowError(f"features.npz missing arrays: {sorted(missing)}")
            arrays = {key: features[key].copy() for key in set(ROW_ARRAY_KEYS + FRAME_KEYS + ("total_payload_bytes",))}
        for key in ROW_ARRAY_KEYS:
            if arrays[key].shape != (n, t):
                raise WorkflowError(f"features.npz {key} has shape {arrays[key].shape}, expected {(n, t)}")
            if not np.issubdtype(arrays[key].dtype, np.number) or not np.isfinite(arrays[key]).all():
                raise WorkflowError(f"features.npz {key} must be finite numeric values")
        if arrays["total_payload_bytes"].shape != (t,) or not np.issubdtype(arrays["total_payload_bytes"].dtype, np.number) or not np.isfinite(arrays["total_payload_bytes"]).all():
            raise WorkflowError("features.npz total_payload_bytes must be finite presentation-frame values")
        for key in FRAME_KEYS:
            if arrays[key].shape != (t,):
                raise WorkflowError(f"features.npz {key} is not presentation-frame length")
        integer_arrays = ("row_bytes", "total_payload_bytes", "display_index", "decode_index", "pts", "slice_qp",
                          "temporal_id", "nal_type", "vcl_nal_bytes", "header_bytes", "payload_bytes", "cu_qp_delta_enabled")
        for key in integer_arrays:
            value = arrays[key]
            if not np.issubdtype(value.dtype, np.integer) or not np.isfinite(value).all():
                raise WorkflowError(f"features.npz {key} must contain finite integer values")
        if not np.issubdtype(arrays["pts_seconds"].dtype, np.number) or not np.isfinite(arrays["pts_seconds"]).all():
            raise WorkflowError("features.npz pts_seconds must contain finite numbers")
        if not np.array_equal(arrays["row_bytes"].sum(axis=0), arrays["total_payload_bytes"]):
            raise WorkflowError("features.npz row-byte sums disagree with total payload")
        if (arrays["row_bytes"] <= 0).any() or (arrays["total_payload_bytes"] <= 0).any():
            raise WorkflowError("features.npz row payloads must be positive")
        npz_pts = [int(value) for value in arrays["pts"]]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError(f"cannot validate extractor artifact: {directory}") from exc
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != n + 1 or rows[0] != [f"frame_{i}" for i in range(t)]:
        raise WorkflowError("rows.csv does not match schema-2.0 row/frame dimensions")
    try:
        csv_row_bytes = np.asarray([[int(cell) for cell in row] for row in rows[1:]], dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkflowError("rows.csv contains invalid non-integer row-byte values") from exc
    if csv_row_bytes.shape != (n, t) or (csv_row_bytes <= 0).any():
        raise WorkflowError("rows.csv contains nonpositive or incorrectly shaped row-byte values")
    if not np.array_equal(csv_row_bytes, arrays["row_bytes"]):
        raise WorkflowError("rows.csv row-byte cells do not exactly match features.npz")
    with (directory / "frames.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        frame_header = tuple(reader.fieldnames or ())
        frames = list(reader)
    if len(frames) != t or not frames or frame_header != FRAME_KEYS:
        raise WorkflowError("frames.csv header/count does not match producer frame metadata")
    integer_frame_fields = {"display_index", "decode_index", "pts", "slice_qp", "temporal_id", "nal_type",
                            "vcl_nal_bytes", "header_bytes", "payload_bytes", "cu_qp_delta_enabled"}
    try:
        for position, row in enumerate(frames):
            for key in FRAME_KEYS:
                expected = arrays[key][position]
                actual = row[key]
                if key in integer_frame_fields:
                    # int() alone accepts whitespace and 1.0; producer CSV values are canonical integers.
                    parsed = int(actual)
                    if str(parsed) != actual or parsed != int(expected):
                        raise WorkflowError(f"frames.csv {key} differs from features.npz at frame {position}")
                elif key == "pts_seconds":
                    parsed = float(actual)
                    if not np.isfinite(parsed) or parsed != float(expected):
                        raise WorkflowError(f"frames.csv {key} differs from features.npz at frame {position}")
                else:
                    if not isinstance(actual, str) or actual != str(expected):
                        raise WorkflowError(f"frames.csv {key} differs from features.npz at frame {position}")
        if [int(row["display_index"]) for row in frames] != list(range(t)):
            raise WorkflowError("frames.csv display indexes are not contiguous")
        if [int(row["pts"]) for row in frames] != npz_pts:
            raise WorkflowError("frames.csv PTS do not match features.npz")
    except (KeyError, ValueError, OverflowError) as exc:
        raise WorkflowError("frames.csv has invalid presentation metadata") from exc
    if len(set(npz_pts)) != t or any(later <= earlier for earlier, later in zip(npz_pts, npz_pts[1:])):
        raise WorkflowError("producer PTS are not unique ascending presentation timestamps")
    # Required leaves above are not the entire trust boundary: optional producer
    # files must not turn the post-validation inventory into a link traversal.
    hashes = _hash_tree(directory)
    return {"row_count": n, "frame_count": t, "artifact_files_sha256": hashes,
            "artifact_tree_sha256": hashlib.sha256(_json_bytes(hashes)).hexdigest(),
            "presentation_times_seconds": [str(Fraction(value) * artifact_time_base) for value in npz_pts],
            "producer_metadata": {key: metadata.get(key) for key in ("coded_width", "coded_height", "ctu_size", "time_base", "schema_version")}}


def _safe_build_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    """Resolve a declared build-relative path without accepting link traversal."""
    try:
        relative = portable_relative_path(relative, "build-relative path")
    except ContractError as exc:
        raise WorkflowError(str(exc)) from exc
    root = _reject_symlink_ancestors(Path(root), field="build root")
    if root.is_symlink() or not root.is_dir():
        raise WorkflowError(f"build root must be a non-symlink directory: {root}")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise WorkflowError(f"declared build path contains a symlink: {candidate}")
    if not (candidate.is_dir() if directory else candidate.is_file()):
        raise WorkflowError(f"declared build path is missing or wrong type: {relative}")
    return candidate


def _hash_tree(directory: Path) -> dict[str, str]:
    """Hash a complete artifact inventory without following any tree entry."""
    raw = Path(directory)
    directory = raw if _fd_anchored_path(raw) else _reject_symlink_ancestors(raw, field="artifact tree")
    if directory.is_symlink() or not directory.is_dir():
        raise WorkflowError(f"artifact tree must be a non-symlink directory: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        try:
            mode = os.lstat(path).st_mode
        except OSError as exc:
            raise WorkflowError(f"cannot safely inspect artifact entry: {path}") from exc
        if stat.S_ISLNK(mode):
            raise WorkflowError(f"artifact contains a symlink: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise WorkflowError(f"artifact contains a non-regular entry: {path}")
        result[path.relative_to(directory).as_posix()] = sha256_file(path)
    return result


def verify_native_build(index_dir: Path, source_manifest_dir: Path) -> dict[str, Any]:
    """Verify native ledger-bound encoded, artifact, and command-log bytes.

    ``load_extraction_index`` remains a lightweight schema/declared-checksum loader.
    Call this function before consuming native artifacts when content integrity is
    required.  The ledger intentionally does not hash the index, avoiding a cycle;
    the index instead binds the final ledger digest and its parent-directory base.
    """
    index_dir = Path(index_dir).absolute()
    # Resume-capable builds deliberately retain an old public index while a
    # retry is active.  It must not be consumed as if it described that active
    # generation.
    state_file = _state_path(index_dir.parent)
    if state_file.exists() or state_file.is_symlink():
        if state_file.is_symlink():
            raise WorkflowError("native extraction resume journal must not be a symlink")
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError("cannot read native extraction resume journal") from exc
        if state.get("format") != _STATE_FORMAT or state.get("version") != _STATE_VERSION or state.get("phase") != "finalized":
            raise WorkflowError("native extraction build is unfinished or has an invalid resume journal")
    index = load_extraction_index(index_dir, source_manifest_dir)
    binding = index["metadata"].get("native_workflow_ledger")
    if not isinstance(binding, Mapping) or set(binding) != {"path_base", "relative_path", "sha256"}:
        raise WorkflowError("index lacks a valid native workflow ledger binding")
    if binding["path_base"] != "parent_of_extraction_index_directory":
        raise WorkflowError("native ledger binding has unsupported path base")
    ledger_path = _safe_build_path(index_dir.parent, binding["relative_path"])
    if sha256_file(ledger_path) != binding["sha256"]:
        raise WorkflowError("bound native workflow ledger SHA-256 mismatch")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot read bound native workflow ledger") from exc
    if not isinstance(ledger, dict) or not isinstance(ledger.get("records"), list) or not isinstance(ledger.get("logs"), dict):
        raise WorkflowError("bound native workflow ledger has invalid structure")
    records = {entry.get("video_id"): entry for entry in ledger["records"] if isinstance(entry, dict)}
    index_ids = {row["video_id"] for row in index["records"]}
    if len(records) != len(ledger["records"]) or set(records) != index_ids:
        raise WorkflowError("bound native workflow ledger has duplicate, invalid, or mismatched video IDs")
    for relative, expected in ledger["logs"].items():
        path = _safe_build_path(index_dir.parent, relative)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise WorkflowError(f"command log SHA-256 mismatch: {relative}")

    def verify_stage_log(entry: Mapping[str, Any], stage: str, *, required: bool) -> None:
        record = entry.get(stage)
        if record is None:
            if required:
                raise WorkflowError(f"ledger lacks required {stage} log reference for successful video")
            return
        if not isinstance(record, Mapping):
            raise WorkflowError(f"ledger {stage} log record is invalid")
        relative, expected = record.get("log_relative_path"), record.get("log_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise WorkflowError(f"ledger {stage} log reference is incomplete")
        # A command record cannot be made self-consistent by simply clearing the
        # global log inventory: both declarations must name the same byte hash.
        if ledger["logs"].get(relative) != expected:
            raise WorkflowError(f"ledger {stage} log is absent from the log hash inventory")
        path = _safe_build_path(index_dir.parent, relative)
        if sha256_file(path) != expected:
            raise WorkflowError(f"ledger {stage} log SHA-256 mismatch")

    encoded_root = index_dir.parent / index["metadata"]["encoded_root_ref"]
    artifact_root = index_dir.parent / index["metadata"]["artifact_root_ref"]
    for row in index["records"]:
        entry = records.get(row["video_id"])
        if entry is None:
            raise WorkflowError(f"ledger lacks index video_id: {row['video_id']}")
        if row["status"] == "failed":
            # Failures can occur before encoding, while encoding, or while
            # extraction.  Verify only stage records that were completed/recorded.
            verify_stage_log(entry, "encoder", required=False)
            verify_stage_log(entry, "extractor", required=False)
            continue
        verify_stage_log(entry, "encoder", required=True)
        verify_stage_log(entry, "extractor", required=True)
        encoded = _safe_build_path(encoded_root, row["encoded_relative_path"])
        if sha256_file(encoded) != row["encoded_sha256"] or entry.get("encoded_sha256") != row["encoded_sha256"]:
            raise WorkflowError(f"encoded SHA-256 mismatch: {row['video_id']}")
        artifact_dir = _safe_build_path(artifact_root, row["artifact_relative_path"], directory=True)
        artifact = entry.get("artifact")
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("artifact_files_sha256"), Mapping):
            raise WorkflowError(f"ledger lacks artifact hashes: {row['video_id']}")
        actual = _hash_tree(artifact_dir)
        expected = dict(artifact["artifact_files_sha256"])
        if actual != expected or hashlib.sha256(_json_bytes(expected)).hexdigest() != artifact.get("artifact_tree_sha256"):
            raise WorkflowError(f"artifact SHA-256 inventory mismatch: {row['video_id']}")
    return {"verified": True, "record_count": len(index["records"]),
            "success_count": sum(row["status"] == "success" for row in index["records"]),
            "ledger_sha256": binding["sha256"]}


def _failure(video_id: str, code: str, error: Exception) -> dict[str, str]:
    message = " ".join(str(error).split())[:1000] or error.__class__.__name__
    return {"video_id": video_id, "status": "failed", "encoded_relative_path": "", "encoded_sha256": "",
            "artifact_relative_path": "", "producer_schema_version": PRODUCER_SCHEMA_VERSION,
            "row_count": "", "frame_count": "", "failure_code": code, "failure_message": message}


def _code(error: Exception) -> str:
    text = str(error).lower()
    for needle, code in (("sha-256", "source_hash_mismatch"), ("source changed", "source_mutated"),
                         ("geometry", "geometry_changed"), ("aspect", "geometry_changed"),
                         ("timeline", "timing_changed"), ("frame count", "timing_changed"),
                         ("external command", "subprocess_failed"), ("schema", "artifact_invalid")):
        if needle in text:
            return code
    return "processing_error"


def _preflight(source_root: Path, output_dir: Path, encoder_script: Path, *, allow_existing: bool = False) -> list[dict[str, Any]]:
    if output_dir.exists() and not allow_existing:
        raise WorkflowError(f"refusing to overwrite existing output directory: {output_dir}")
    # Derived media cannot live inside this checkout: it is too easy to commit or
    # accidentally reuse it as a source asset.  Resolve(strict=False) is sufficient
    # here because the leaf must be new and existing parent links resolve normally.
    repository_root = Path(__file__).resolve().parents[2]
    try:
        output_dir.absolute().resolve(strict=False).relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise WorkflowError(f"output directory must be outside the code repository: {output_dir}")
    if not encoder_script.is_file():
        raise WorkflowError(f"encoder script is missing: {encoder_script}")
    source_root = _reject_symlink_ancestors(Path(source_root), field="source root")
    if not source_root.is_dir() or source_root.is_symlink():
        raise WorkflowError(f"source root must be a non-symlink directory: {source_root}")
    missing = [name for name in ("ffmpeg", "ffprobe", "bash") if shutil.which(name) is None]
    if missing:
        raise WorkflowError(f"required executable(s) missing from PATH: {', '.join(missing)}")
    checks: list[dict[str, Any]] = []
    for cmd, required in ((["ffmpeg", "-hide_banner", "-encoders"], "libx265"),
                          (["ffmpeg", "-hide_banner", "-bsfs"], "trace_headers")):
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise WorkflowError(f"cannot verify ffmpeg {required} capability") from exc
        if required not in completed.stdout:
            raise WorkflowError(f"ffmpeg lacks required {required}")
        checks.append({"command": cmd, "returncode": completed.returncode,
                       "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})
    return checks


_STATE_DIR = ".native-wpp-state"
_STATE_FILE = "journal.json"
_STATE_FORMAT = "native_wpp_resume_state"
_STATE_VERSION = 3
_PROFILE_SETTINGS = {"gop": 16, "qp": 28, "bframes": 0, "codec": "libx265", "preset": "medium",
                     "pixel_format": "yuv420p", "fps_mode": "passthrough", "wpp": 1, "pools": 4,
                     "frame_threads": 1, "ctu": 64, "slices": 1, "aq_mode": 0, "cutree": 0,
                     "keyint": 16, "min_keyint": 16, "scenecut": 0, "open_gop": 0,
                     "input_policy": "native geometry/timing; no scaling, crop, FPS conversion, padding, or truncation"}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a file in an already-owned non-symlink directory."""
    _reject_symlink_ancestors(path.parent, field="atomic write parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path.parent, field="atomic write parent")
    if path.is_symlink():
        raise WorkflowError(f"refusing atomic replacement of symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _state_path(output_dir: Path) -> Path:
    root = _reject_symlink_ancestors(Path(output_dir), field="native extraction output")
    return _reject_symlink_ancestors(root / _STATE_DIR / _STATE_FILE, field="native extraction resume journal")


def _safe_owned(root: Path, relative: str, *, directory: bool | None = None, allow_missing: bool = True) -> Path:
    """Return an owned, simple descendant; never follow a link while cleaning it."""
    try:
        relative = portable_relative_path(relative, "owned build path")
    except ContractError as exc:
        raise WorkflowError(str(exc)) from exc
    root = _reject_symlink_ancestors(Path(root), field="owned build root")
    if root.is_symlink() or not root.is_dir():
        raise WorkflowError(f"owned build root must be a non-symlink directory: {root}")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise WorkflowError(f"owned build path contains a symlink: {candidate}")
    if not allow_missing and not candidate.exists():
        raise WorkflowError(f"owned build path is missing: {candidate}")
    if candidate.exists() and directory is not None and candidate.is_dir() != directory:
        raise WorkflowError(f"owned build path has wrong type: {candidate}")
    return candidate


def _remove_owned(root: Path, relative: str) -> None:
    path = _safe_owned(root, relative)
    if not path.exists():
        return
    if path.is_symlink():
        raise WorkflowError(f"refusing to remove symlink: {path}")
    if path.is_dir():
        # rmtree itself does not follow a directory symlink, but reject any nested
        # link explicitly so corruption cannot turn cleanup into a traversal primitive.
        for child in path.rglob("*"):
            if child.is_symlink():
                raise WorkflowError(f"refusing to remove tree containing symlink: {child}")
        shutil.rmtree(path)
    else:
        path.unlink()


def _open_dir_at(parent_fd: int, name: str, *, create: bool = False) -> int:
    """Open one private directory component without traversing a link."""
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise UnsafeOutputError(f"cannot safely open private output directory: {name}") from exc


def _open_output_dir(root_fd: int, relative: str, *, create: bool = False) -> int:
    """Open a simple output-relative directory by descriptor, optionally mkdirat."""
    try:
        parts = PurePosixPath(portable_relative_path(relative, "output directory")).parts
    except ContractError as exc:
        raise UnsafeOutputError(str(exc)) from exc
    current = os.dup(root_fd)
    try:
        for part in parts:
            next_fd = _open_dir_at(current, part, create=create)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _output_binding(root_fd: int, output_dir: Path, relative: str | None = None, held_fd: int | None = None) -> None:
    """Detect a pathname replacement while retaining the original FD as a safe sink."""
    logical = Path(output_dir) if relative is None else Path(output_dir) / relative
    try:
        stat = os.stat(logical, follow_symlinks=False)
        expected = os.fstat(root_fd if held_fd is None else held_fd)
    except OSError as exc:
        raise UnsafeOutputError("public output path disappeared or became unsafe") from exc
    if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino) or not os.path.isdir(logical):
        raise UnsafeOutputError("public output path was replaced while processing")


def _normalize_private_artifact_source(work_fd: int, logical_encoded_path: Path) -> None:
    """Replace a descriptor implementation path with the stable logical source."""
    artifact_fd = _open_dir_at(work_fd, "artifact")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            metadata_fd = os.open("metadata.json", flags, dir_fd=artifact_fd)
        except OSError as exc:
            raise WorkflowError("extractor metadata is missing or unsafe") from exc
        with os.fdopen(metadata_fd, "r+", encoding="utf-8") as handle:
            try:
                metadata = json.load(handle)
            except json.JSONDecodeError as exc:
                raise WorkflowError("extractor metadata is invalid") from exc
            if not isinstance(metadata, dict):
                raise WorkflowError("extractor metadata is invalid")
            # The unchanged producer records argv[1]; /proc/self/fd/N is an
            # implementation capability, not durable provenance.
            metadata["source"] = str(logical_encoded_path)
            handle.seek(0); handle.truncate()
            handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(artifact_fd)


def _publish_at(*, source_fd: int, source_name: str, destination_fd: int, destination_name: str,
                output_fd: int, output_dir: Path, destination_relative: str) -> None:
    """Atomically publish a fresh private child using only held directory FDs."""
    _output_binding(output_fd, output_dir)
    _output_binding(output_fd, output_dir, destination_relative, destination_fd)
    try:
        os.stat(destination_name, dir_fd=destination_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise UnsafeOutputError("refusing to overwrite a public output leaf")
    try:
        os.rename(source_name, destination_name, src_dir_fd=source_fd, dst_dir_fd=destination_fd)
    except OSError as exc:
        raise UnsafeOutputError("cannot atomically publish private output") from exc
    _output_binding(output_fd, output_dir)
    _output_binding(output_fd, output_dir, destination_relative, destination_fd)


@dataclass(frozen=True)
class _BuildLock:
    path: Path
    nonce: str
    device: int
    inode: int


def _acquire_build_lock(output_dir: Path) -> _BuildLock:
    parent = Path(output_dir).parent.absolute()
    _reject_symlink_ancestors(parent, field="output parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(parent, field="output parent")
    if not parent.is_dir():
        raise WorkflowError(f"output parent must be a non-symlink directory: {parent}")
    lock = _reject_symlink_ancestors(parent / f".{Path(output_dir).name}.native-wpp-build.lock", field="build lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise WorkflowError(f"native extraction build lock already exists (inspect/remove manually only after confirming no writer): {lock}") from exc
    nonce = uuid.uuid4().hex
    try:
        _atomic_json(lock / "owner.json", {"pid": os.getpid(), "hostname": socket.gethostname(),
                                             "started_utc": datetime.now(timezone.utc).isoformat(), "nonce": nonce})
        stat = lock.stat()
        return _BuildLock(lock, nonce, stat.st_dev, stat.st_ino)
    except Exception:
        shutil.rmtree(lock, ignore_errors=True)
        raise


def _release_build_lock(lock: _BuildLock) -> None:
    # Never delete a replacement lock.  A hard kill intentionally leaves ours.
    try:
        _reject_symlink_ancestors(lock.path, field="build lock")
        stat = lock.path.stat()
        if (stat.st_dev, stat.st_ino) != (lock.device, lock.inode):
            raise WorkflowError("build lock ownership changed; refusing to remove replacement lock")
        owner = lock.path / "owner.json"
        if owner.is_symlink():
            raise WorkflowError("build lock owner is a symlink; refusing to remove lock")
        value = json.loads(owner.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("nonce") != lock.nonce:
            raise WorkflowError("build lock ownership changed; refusing to remove replacement lock")
        owner.unlink()
        lock.path.rmdir()
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot safely release native extraction build lock") from exc


def _runtime_version(command: str) -> dict[str, str]:
    resolved = shutil.which(command) if os.path.sep not in command else command
    if not resolved:
        raise WorkflowError(f"required executable missing while fingerprinting: {command}")
    version_flag = "-version" if Path(resolved).name in ("ffmpeg", "ffprobe") else "--version"
    try:
        completed = subprocess.run([resolved, version_flag], check=False, capture_output=True, text=True)
    except OSError as exc:
        raise WorkflowError(f"cannot fingerprint executable: {command}") from exc
    text = completed.stdout + completed.stderr
    return {"path": str(Path(resolved).resolve()), "version_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "returncode": str(completed.returncode)}


def _fingerprint(source_manifest_dir: Path, source: Mapping[str, Any], encoder_script: Path,
                 extractor_script: Path, python_executable: str, runner_script: Path | None) -> dict[str, Any]:
    manifest_dir = Path(source_manifest_dir).resolve()
    runner = Path(runner_script) if runner_script is not None else Path(__file__).resolve().parents[1] / "build_native_wpp_extraction.py"
    helper = Path(__file__).resolve().with_name("manifest_contract.py")
    scripts = {"encoder": sha256_file(encoder_script), "extractor": sha256_file(extractor_script),
               "workflow": sha256_file(Path(__file__)), "manifest_contract": sha256_file(helper)}
    if runner.is_file():
        scripts["runner"] = sha256_file(runner)
    return {"format": "native_wpp_resume_fingerprint_v1",
            "source_manifest": {"directory": str(manifest_dir),
                                "records_sha256": source["metadata"]["records_sha256"],
                                "metadata_sha256": sha256_file(manifest_dir / "source_manifest.metadata.json"),
                                "ordered_records_sha256": hashlib.sha256(_json_bytes({"records": source["records"]})).hexdigest(),
                                "contract_version": source["metadata"]["contract_version"]},
            "input_ids": [item["video_id"] for item in source["records"]],
            "profile": {"name": PROFILE_NAME, "resolved_settings": _PROFILE_SETTINGS,
                        "encoder_invocation": "bash encode_wpp.sh INPUT_DIR (default arguments only)"},
            "scripts": scripts,
            "runtime": {"python": _runtime_version(python_executable), "ffmpeg": _runtime_version("ffmpeg"),
                        "ffprobe": _runtime_version("ffprobe"), "bash": _runtime_version("bash"),
                        "numpy": np.__version__}}


def _new_journal(source: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    items = {item["video_id"]: {"video_id": item["video_id"], "source_sha256": item["source_sha256"],
                                  "state": "pending", "attempts": []} for item in source["records"]}
    return {"format": _STATE_FORMAT, "version": _STATE_VERSION, "phase": "active",
            "fingerprint": dict(fingerprint), "fingerprint_sha256": hashlib.sha256(_json_bytes(fingerprint)).hexdigest(),
            "items": items, "finalization": None}


def _receipt_relative(digest: str, attempt_id: str) -> str:
    return f"{_STATE_DIR}/receipts/{digest}/{attempt_id}.json"


def _write_completion_receipt(output_dir: Path, journal: Mapping[str, Any], *, video_id: str, digest: str,
                              attempt: Mapping[str, Any], outcome: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """Create the immutable completion fact before the mutable journal points at it."""
    relative = _receipt_relative(digest, str(attempt["id"]))
    path = _safe_owned(output_dir, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_owned(output_dir, path.parent.relative_to(Path(output_dir).absolute()).as_posix(), directory=True)
    if path.exists() or path.is_symlink():
        raise WorkflowError("completion receipt path already exists")
    receipt = {"format": "native_wpp_attempt_receipt", "version": 1,
               "fingerprint_sha256": journal["fingerprint_sha256"], "video_id": video_id,
               "source_sha256": digest, "attempt_id": attempt["id"], "started_utc": attempt["started_utc"],
               "completed_utc": datetime.now(timezone.utc).isoformat(), "outcome": outcome,
               "payload": dict(payload), "payload_sha256": hashlib.sha256(_json_bytes(payload)).hexdigest()}
    encoded = _json_bytes(receipt)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise WorkflowError("cannot safely write completion receipt") from exc
    return {"relative_path": relative, "sha256": sha256_file(path)}


def _read_attempt_receipt(output_dir: Path, journal: Mapping[str, Any], *, video_id: str, digest: str,
                          attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Read one immutable attempt fact, including historical failed attempts."""
    reference, outcome = attempt.get("receipt"), attempt.get("outcome")
    if not isinstance(reference, Mapping) or set(reference) != {"relative_path", "sha256"}:
        raise WorkflowError("journal completion receipt reference is corrupt")
    expected_relative = _receipt_relative(digest, str(attempt.get("id", "")))
    if reference.get("relative_path") != expected_relative or not isinstance(reference.get("sha256"), str):
        raise WorkflowError("journal completion receipt path is corrupt")
    path = _safe_owned(output_dir, expected_relative, directory=False, allow_missing=False)
    if sha256_file(path) != reference["sha256"]:
        raise WorkflowError("completion receipt SHA-256 mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot read completion receipt") from exc
    expected = {"format": "native_wpp_attempt_receipt", "version": 1,
                "fingerprint_sha256": journal["fingerprint_sha256"], "video_id": video_id,
                "source_sha256": digest, "attempt_id": attempt["id"], "started_utc": attempt["started_utc"],
                "outcome": outcome}
    if (not isinstance(receipt, dict) or set(receipt) != set(expected) | {"completed_utc", "payload", "payload_sha256"} or
            not isinstance(receipt.get("completed_utc"), str) or not isinstance(receipt.get("payload"), dict) or
            receipt.get("payload_sha256") != hashlib.sha256(_json_bytes(receipt["payload"])).hexdigest() or
            any(receipt[key] != value for key, value in expected.items())):
        raise WorkflowError("completion receipt schema or provenance is corrupt")
    return receipt


def _validate_receipt(output_dir: Path, journal: Mapping[str, Any], *, video_id: str, digest: str,
                      attempt: Mapping[str, Any], outcome: str, reference: Any, payload: Mapping[str, Any]) -> None:
    if reference != attempt.get("receipt"):
        raise WorkflowError("terminal journal receipt is not bound to its completed attempt")
    receipt = _read_attempt_receipt(output_dir, journal, video_id=video_id, digest=digest, attempt=attempt)
    if outcome != attempt.get("outcome") or receipt["payload"] != dict(payload):
        raise WorkflowError("completion receipt does not match journal terminal record")


def _validate_receipt_inventory(output_dir: Path, journal: Mapping[str, Any]) -> None:
    """Receipt storage is an exact append-only inventory, never a loose cache."""
    expected: set[str] = set()
    for item in journal["items"].values():
        for attempt in item["attempts"]:
            if "outcome" in attempt:
                ref = attempt.get("receipt")
                if not isinstance(ref, Mapping) or not isinstance(ref.get("relative_path"), str):
                    raise WorkflowError("completed attempt lacks immutable receipt reference")
                expected.add(ref["relative_path"])
            elif item.get("state") == "running" and attempt is item["attempts"][-1]:
                # A receipt fsynced before its journal update is the only permitted
                # unreferenced file and is promoted by _recover_orphan_receipt.
                candidate = _receipt_relative(item["source_sha256"], attempt["id"])
                candidate_path = _safe_owned(output_dir, candidate)
                if candidate_path.exists():
                    expected.add(candidate)
    root = _safe_owned(output_dir, f"{_STATE_DIR}/receipts")
    actual: set[str] = set()
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise WorkflowError("completion receipt inventory is unsafe")
        for path in root.rglob("*"):
            if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
                raise WorkflowError("completion receipt inventory contains unsafe entry")
            if path.is_file():
                actual.add(path.relative_to(Path(output_dir)).as_posix())
    if actual != expected:
        raise WorkflowError("completion receipt inventory has missing, replayed, or unreferenced entries")


def _recover_orphan_receipt(output_dir: Path, journal: Mapping[str, Any], item: dict[str, Any]) -> bool:
    """Promote a receipt fsynced just before a crash, never a journal-only claim."""
    if item.get("state") != "running" or not item.get("attempts"):
        return False
    attempt = item["attempts"][-1]
    if not isinstance(attempt, dict) or "outcome" in attempt:
        return False
    digest, video_id = item["source_sha256"], item["video_id"]
    relative = _receipt_relative(digest, attempt.get("id", ""))
    path = _safe_owned(output_dir, relative)
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise WorkflowError("orphan completion receipt is unsafe")
    reference = {"relative_path": relative, "sha256": sha256_file(path)}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot read orphan completion receipt") from exc
    if not isinstance(receipt, dict) or receipt.get("outcome") not in ("success", "failed") or not isinstance(receipt.get("payload"), dict):
        raise WorkflowError("orphan completion receipt is corrupt")
    outcome, payload = receipt["outcome"], receipt["payload"]
    proposed = dict(item); proposed["attempts"] = [dict(a) for a in item["attempts"]]
    proposed["attempts"][-1]["outcome"] = outcome; proposed["attempts"][-1]["receipt"] = reference
    proposed["state"] = outcome; proposed["receipt"] = reference
    proposed["success" if outcome == "success" else "failure"] = payload
    _validate_journal_item(proposed, video_id=video_id, digest=digest, output_dir=output_dir, journal=journal)
    item.clear(); item.update(proposed)
    return True


def _validate_journal_item(item: Mapping[str, Any], *, video_id: str, digest: str, output_dir: Path, journal: Mapping[str, Any]) -> None:
    """Reject structurally invented terminal outcomes before any cleanup/retry."""
    state = item.get("state")
    allowed = {"video_id", "source_sha256", "state", "attempts"}
    if state == "success":
        allowed.update(("success", "receipt"))
    elif state == "failed":
        allowed.update(("failure", "receipt"))
    if set(item) != allowed or item.get("video_id") != video_id or item.get("source_sha256") != digest:
        raise WorkflowError("resume journal item identity or fields are corrupt")
    attempts = item.get("attempts")
    if state not in ("pending", "running", "success", "failed") or not isinstance(attempts, list):
        raise WorkflowError("resume journal item state is corrupt")
    seen: set[str] = set()
    for position, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) not in ({"id", "source_sha256", "started_utc"}, {"id", "source_sha256", "started_utc", "outcome", "receipt"}):
            raise WorkflowError("resume journal attempt receipt fields are corrupt")
        attempt_id = attempt.get("id")
        if not isinstance(attempt_id, str) or len(attempt_id) != 32 or not all(c in "0123456789abcdef" for c in attempt_id) or attempt_id in seen:
            raise WorkflowError("resume journal attempt ID is corrupt or duplicated")
        if attempt.get("source_sha256") != digest or not isinstance(attempt.get("started_utc"), str) or not attempt["started_utc"]:
            raise WorkflowError("resume journal attempt provenance is corrupt")
        if "outcome" in attempt and attempt["outcome"] not in ("success", "failed"):
            raise WorkflowError("resume journal attempt outcome is corrupt")
        if "outcome" in attempt:
            _read_attempt_receipt(output_dir, journal, video_id=video_id, digest=digest, attempt=attempt)
        if position < len(attempts) - 1 and "outcome" not in attempt:
            raise WorkflowError("resume journal has a nonterminal historical attempt")
        seen.add(attempt_id)
    if state == "pending" and attempts and "outcome" not in attempts[-1]:
        raise WorkflowError("pending journal item has an uncompleted attempt")
    if state == "running" and (not attempts or "outcome" in attempts[-1]):
        raise WorkflowError("running journal item lacks its active attempt")
    if state in ("success", "failed") and (not attempts or attempts[-1].get("outcome") != state):
        raise WorkflowError("terminal journal item does not match its latest attempt")
    if state == "success":
        success = item["success"]
        if not isinstance(success, dict) or set(success) != {"record", "ledger_entry"}:
            raise WorkflowError("successful resume journal item has corrupt completion fields")
        record, entry = success["record"], success["ledger_entry"]
        try:
            validate_extraction_index_records([record], [video_id])
        except ContractError as exc:
            raise WorkflowError("successful resume journal record is corrupt") from exc
        if not isinstance(entry, dict) or entry.get("video_id") != video_id or entry.get("attempt_id") != attempts[-1]["id"]:
            raise WorkflowError("successful resume journal ledger provenance is corrupt")
        artifact = entry.get("artifact")
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("artifact_files_sha256"), dict) or
                not artifact["artifact_files_sha256"] or not isinstance(artifact.get("artifact_tree_sha256"), str)):
            raise WorkflowError("successful resume journal artifact inventory is corrupt")
        _validate_receipt(output_dir, journal, video_id=video_id, digest=digest, attempt=attempts[-1], outcome="success",
                          reference=item["receipt"], payload=success)
        for stage in ("encoder", "extractor"):
            log = entry.get(stage)
            if not isinstance(log, dict) or set(log) != {"command", "returncode", "log_relative_path", "log_sha256"}:
                raise WorkflowError("successful resume journal command log provenance is corrupt")
            if not isinstance(log["log_relative_path"], str) or not isinstance(log["log_sha256"], str):
                raise WorkflowError("successful resume journal command log reference is corrupt")
    elif state == "failed":
        failure = item["failure"]
        if not isinstance(failure, dict) or set(failure) != {"code", "message", "stage", "ledger_entry"}:
            raise WorkflowError("failed resume journal failure provenance is corrupt")
        if (not isinstance(failure["code"], str) or not failure["code"] or not isinstance(failure["message"], str) or
                not failure["message"] or failure["stage"] not in {"cleanup", "source_verification", "source_probe", "encoder", "encoded_probe", "extractor", "artifact_validation"}):
            raise WorkflowError("failed resume journal failure details are corrupt")
        entry = failure["ledger_entry"]
        embedded = entry.get("failure") if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or entry.get("video_id") != video_id or entry.get("attempt_id") != attempts[-1]["id"] or
                not isinstance(embedded, dict) or embedded != {"code": failure["code"], "message": failure["message"], "stage": failure["stage"]}):
            raise WorkflowError("failed resume journal ledger provenance is corrupt")
        _validate_receipt(output_dir, journal, video_id=video_id, digest=digest, attempt=attempts[-1], outcome="failed",
                          reference=item["receipt"], payload=failure)


def _load_journal(output_dir: Path, source: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    state_file = _state_path(output_dir)
    try:
        if state_file.is_symlink():
            raise WorkflowError("resume journal must not be a symlink")
        journal = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot read native extraction resume journal") from exc
    if (not isinstance(journal, dict) or set(journal) not in ({"format", "version", "phase", "fingerprint", "fingerprint_sha256", "items", "finalization"},
                                                               {"format", "version", "phase", "fingerprint", "fingerprint_sha256", "items", "finalization", "preflight"}) or
            journal.get("format") != _STATE_FORMAT or journal.get("version") != _STATE_VERSION):
        raise WorkflowError("unsupported or corrupt native extraction resume journal")
    if "preflight" in journal and not isinstance(journal["preflight"], list):
        raise WorkflowError("resume journal preflight provenance is corrupt")
    if journal.get("phase") not in ("active", "finalizing", "finalized"):
        raise WorkflowError("resume journal has invalid phase")
    if journal.get("fingerprint") != fingerprint or journal.get("fingerprint_sha256") != hashlib.sha256(_json_bytes(fingerprint)).hexdigest():
        raise WorkflowError("resume fingerprint mismatch (manifest/profile/scripts/runtime changed)")
    items = journal.get("items")
    expected = {item["video_id"]: item["source_sha256"] for item in source["records"]}
    if not isinstance(items, dict) or set(items) != set(expected):
        raise WorkflowError("resume journal input IDs do not exactly match manifest")
    for video_id, digest in expected.items():
        item = items[video_id]
        if not isinstance(item, dict):
            raise WorkflowError("resume journal item is corrupt")
        _validate_journal_item(item, video_id=video_id, digest=digest, output_dir=output_dir, journal=journal)
    _validate_receipt_inventory(output_dir, journal)
    return journal


def _save_journal(output_dir: Path, journal: Mapping[str, Any]) -> None:
    state_root = _safe_owned(output_dir, _STATE_DIR, directory=True)
    _atomic_json(state_root / _STATE_FILE, journal)


def _rehash_sources(source: Mapping[str, Any], source_root: Path) -> None:
    for item in source["records"]:
        original = _safe_source_path(source_root, item["source_relative_path"])
        digest, stat = _stable_hash(original)
        if digest != item["source_sha256"] or stat.st_size != int(item["source_bytes"]):
            raise WorkflowError(f"source changed since manifest: {item['source_relative_path']}")


def _record_for(item: Mapping[str, str], artifact: Mapping[str, Any], encoded_sha: str) -> dict[str, str]:
    digest = item["source_sha256"]
    return {"video_id": item["video_id"], "status": "success", "encoded_relative_path": f"sha256/{digest}.mp4",
            "encoded_sha256": encoded_sha, "artifact_relative_path": f"sha256/{digest}",
            "producer_schema_version": PRODUCER_SCHEMA_VERSION, "row_count": str(artifact["row_count"]),
            "frame_count": str(artifact["frame_count"]), "failure_code": "", "failure_message": ""}


def _verify_journal_success(output_dir: Path, item: Mapping[str, Any]) -> None:
    success = item.get("success")
    if not isinstance(success, Mapping) or not isinstance(success.get("record"), Mapping) or not isinstance(success.get("ledger_entry"), Mapping):
        raise WorkflowError("successful resume journal item lacks committed record")
    record, entry = success["record"], success["ledger_entry"]
    digest = item["source_sha256"]
    if (record.get("video_id") != item["video_id"] or record.get("status") != "success" or
            record.get("encoded_relative_path") != f"sha256/{digest}.mp4" or record.get("artifact_relative_path") != f"sha256/{digest}"):
        raise WorkflowError("successful resume journal item has unsafe record identity")
    encoded = _safe_owned(output_dir / "encoded", f"sha256/{digest}.mp4", directory=False, allow_missing=False)
    if sha256_file(encoded) != record.get("encoded_sha256") or entry.get("encoded_sha256") != record.get("encoded_sha256"):
        raise WorkflowError(f"committed encoded output is missing or tampered: {item['video_id']}")
    artifact_dir = _safe_owned(output_dir / "artifacts", f"sha256/{digest}", directory=True, allow_missing=False)
    actual = _hash_tree(artifact_dir)
    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping) or actual != artifact.get("artifact_files_sha256"):
        raise WorkflowError(f"committed artifact is missing or tampered: {item['video_id']}")
    if validate_artifact(artifact_dir).get("artifact_tree_sha256") != artifact.get("artifact_tree_sha256"):
        raise WorkflowError(f"committed artifact validation changed: {item['video_id']}")
    for stage in ("encoder", "extractor"):
        log = entry.get(stage)
        if not isinstance(log, Mapping) or not isinstance(log.get("log_relative_path"), str) or not isinstance(log.get("log_sha256"), str):
            raise WorkflowError("successful resume journal item lacks command log")
        path = _safe_owned(output_dir, log["log_relative_path"], directory=False, allow_missing=False)
        if sha256_file(path) != log["log_sha256"]:
            raise WorkflowError(f"committed command log is missing or tampered: {item['video_id']}")


def _legacy_resume_noop(*, source_manifest_dir: Path, source_root: Path, output_dir: Path,
                        encoder_script: Path, extractor_script: Path) -> tuple[dict[str, Any], bool]:
    index = load_extraction_index(output_dir / "extraction-index", source_manifest_dir)
    profile, identity = index["metadata"]["processing_profile"], index["metadata"]["extractor_identity"]
    if profile.get("name") != PROFILE_NAME or profile.get("encoder_script_sha256") != sha256_file(encoder_script) or identity.get("extractor_script_sha256") != sha256_file(extractor_script):
        raise WorkflowError("legacy completed build is not compatible with current profile/scripts")
    _rehash_sources(load_source_manifest(source_manifest_dir), source_root)
    verify_native_build(output_dir / "extraction-index", source_manifest_dir)
    if any(row["status"] == "failed" for row in index["records"]):
        raise WorkflowError("legacy completed build has failures and cannot be retried; use a new output directory")
    return index, False


def _processing_profile(encoder_script: Path) -> dict[str, Any]:
    return {"name": PROFILE_NAME, "encoder_script_sha256": sha256_file(encoder_script),
            "encoder_invocation": "bash encode_wpp.sh INPUT_DIR (default arguments only)",
            "resolved_settings": _PROFILE_SETTINGS,
            "native_policy": "decoded/display dimensions and SAR equal; presentation timeline checked as exact Fractions with one output-tick pointwise tolerance; no FPS conversion"}


def _finalize_journal(*, output_dir: Path, source_manifest_dir: Path, source: Mapping[str, Any],
                      journal: dict[str, Any], processing_profile: Mapping[str, Any],
                      identity: Mapping[str, Any], test_hook: Callable[[str, Mapping[str, str]], None] | None) -> dict[str, Any]:
    """Publish a generation.  A crash leaves phase=finalizing, never a trusted stale index."""
    terminal = {"success", "failed"}
    if any(item["state"] not in terminal for item in journal["items"].values()):
        raise WorkflowError("cannot finalize extraction index while resume items are pending")
    if journal.get("phase") != "finalizing":
        journal["phase"] = "finalizing"
        journal["finalization"] = {"generation": uuid.uuid4().hex}
        _save_journal(output_dir, journal)
    finalization = journal.get("finalization")
    if not isinstance(finalization, dict) or not isinstance(finalization.get("generation"), str) or not finalization["generation"].isalnum():
        raise WorkflowError("resume journal finalization record is corrupt")
    generation = finalization["generation"]
    state_root = _safe_owned(output_dir, _STATE_DIR, directory=True)
    generation_rel = f"{_STATE_DIR}/finalizations/{generation}"
    generation_dir = _safe_owned(output_dir, generation_rel)
    generation_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = generation_dir / "build_ledger.json"
    records: list[dict[str, str]] = []
    ledger_rows: list[dict[str, Any]] = []
    for manifest_item in source["records"]:
        item = journal["items"][manifest_item["video_id"]]
        if item["state"] == "success":
            _verify_journal_success(output_dir, item)
            records.append(dict(item["success"]["record"]))
            ledger_rows.append(dict(item["success"]["ledger_entry"]))
        else:
            failure = item.get("failure")
            if not isinstance(failure, Mapping):
                raise WorkflowError("failed resume item lacks failure provenance")
            records.append(_failure(manifest_item["video_id"], str(failure.get("code", "processing_error")),
                                    WorkflowError(str(failure.get("message", "unknown failure")))))
            ledger_rows.append(dict(failure.get("ledger_entry", {"video_id": manifest_item["video_id"], "failure": dict(failure)})))
    metadata = {"workflow": "native_wpp_extraction_v2", "resume_state_format": _STATE_FORMAT,
                "source_manifest_dir": str(Path(source_manifest_dir).resolve()),
                "input_source_manifest_records_sha256": source["metadata"]["records_sha256"],
                "preflight": journal.get("preflight", []), "processing_profile": dict(processing_profile),
                "extractor_identity": dict(identity), "resume_fingerprint_sha256": journal["fingerprint_sha256"],
                "logs": {f"logs/{name}": digest for name, digest in _hash_tree(output_dir / "logs").items()},
                "records": ledger_rows}
    if finalization.get("ledger_sha256") is None:
        # No journal digest means no committed ledger, even if a crash (or an
        # untrusted writer) left bytes at the deterministic candidate path.
        # Recreate only that owned path; never adopt its hash as evidence.
        if ledger_path.exists():
            _remove_owned(output_dir, f"{generation_rel}/build_ledger.json")
        if test_hook:
            test_hook("before_final_ledger", {})
        _atomic_json(ledger_path, metadata)
        finalization["ledger_sha256"] = sha256_file(ledger_path)
        _save_journal(output_dir, journal)
        if test_hook:
            test_hook("after_final_ledger", {})
    elif not ledger_path.exists() or finalization["ledger_sha256"] != sha256_file(ledger_path):
        raise WorkflowError("finalization ledger is corrupt or does not match journal")
    binding = {"path_base": "parent_of_extraction_index_directory", "relative_path": "build_ledger.json",
               "sha256": finalization["ledger_sha256"]}
    candidate = generation_dir / "extraction-index"
    candidate_rel = f"{generation_rel}/extraction-index"
    public = output_dir / "extraction-index"
    public_already_published = False
    if finalization.get("candidate_index_metadata_sha") is None:
        # As with the ledger, a path not checkpointed in the journal is an
        # orphan, not a reusable publication candidate.
        if candidate.exists():
            _remove_owned(output_dir, candidate_rel)
        write_extraction_index(candidate, records, source_manifest_dir=source_manifest_dir,
                               encoded_root_ref="encoded", artifact_root_ref="artifacts",
                               processing_profile=processing_profile, extractor_identity=identity,
                               additional_metadata={"native_workflow_ledger": binding})
        finalization["candidate_index_metadata_sha"] = sha256_file(candidate / "extraction_index.metadata.json")
        _save_journal(output_dir, journal)
    else:
        try:
            if candidate.exists():
                existing = load_extraction_index(candidate, source_manifest_dir)
                metadata_path = candidate / "extraction_index.metadata.json"
            else:
                # Crash after candidate -> public rename, before final journal
                # checkpoint: the public directory itself is the durable candidate.
                existing = load_extraction_index(public, source_manifest_dir)
                metadata_path = public / "extraction_index.metadata.json"
                public_already_published = True
            if (existing["metadata"].get("native_workflow_ledger") != binding or
                    finalization["candidate_index_metadata_sha"] != sha256_file(metadata_path)):
                raise WorkflowError("finalization candidate index has wrong ledger binding")
        except (ContractError, OSError, ValueError) as exc:
            raise WorkflowError("finalization candidate index is corrupt") from exc
    # The journal remains finalizing until the matching public index is moved
    # into place, so a new ledger cannot be accepted with an old index after a
    # crash.  The immutable generation copy preserves the candidate evidence.
    _atomic_bytes(output_dir / "build_ledger.json", ledger_path.read_bytes())
    if not public_already_published:
        if public.exists():
            archive = _safe_owned(output_dir, f"{_STATE_DIR}/archives/index-{generation}")
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise WorkflowError("finalization archive collision")
            os.replace(public, archive)
        os.replace(candidate, public)
    if test_hook:
        test_hook("after_final_index", {})
    journal["phase"] = "finalized"
    finalization["public_index_sha256"] = sha256_file(public / "extraction_index.metadata.json")
    _save_journal(output_dir, journal)
    return load_extraction_index(public, source_manifest_dir)


def run_workflow(*, source_manifest_dir: Path, source_root: Path, output_dir: Path, encoder_script: Path,
                 extractor_script: Path, python_executable: str = sys.executable,
                 run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, resume: bool = False,
                 retry_failed: bool = False, runner_script: Path | None = None,
                 test_hook: Callable[[str, Mapping[str, str]], None] | None = None) -> tuple[dict[str, Any], bool]:
    """Build or safely resume one exact manifest.

    The only public completion marker is a ledger-bound extraction index.  State
    before that marker is private and intentionally rejected by the verifier.
    """
    source = load_source_manifest(source_manifest_dir)
    source_root, output_dir, encoder_script, extractor_script = (Path(source_root), Path(output_dir),
        Path(encoder_script), Path(extractor_script))
    if retry_failed and not resume:
        raise WorkflowError("--retry-failed requires --resume")
    if not extractor_script.is_file():
        raise WorkflowError(f"extractor script is missing: {extractor_script}")
    if resume and not output_dir.exists():
        raise WorkflowError("--resume requires an existing native extraction output directory")
    if not resume and output_dir.exists():
        raise WorkflowError(f"refusing to overwrite existing output directory: {output_dir}")
    lock = _acquire_build_lock(output_dir)
    try:
        # A v1 finalized build has no mutable state.  It is read-only compatible
        # only when it fully verifies against this exact manifest and scripts.
        if resume and not _state_path(output_dir).exists():
            return _legacy_resume_noop(source_manifest_dir=source_manifest_dir, source_root=source_root,
                                       output_dir=output_dir, encoder_script=encoder_script,
                                       extractor_script=extractor_script)
        checks = _preflight(source_root, output_dir, encoder_script, allow_existing=resume)
        fingerprint = _fingerprint(source_manifest_dir, source, encoder_script, extractor_script,
                                   python_executable, runner_script)
        profile = _processing_profile(encoder_script)
        identity = {"name": "native_wpp_workflow", "producer_schema_version": PRODUCER_SCHEMA_VERSION,
                    "extractor_script_sha256": sha256_file(extractor_script), "python_executable": python_executable}
        if resume:
            journal = _load_journal(output_dir, source, fingerprint)
            # Full originals rehash happens before examining reusable output or
            # deleting any stale attempt data.
            _rehash_sources(source, source_root)
            promoted = False
            for state_item in journal["items"].values():
                promoted = _recover_orphan_receipt(output_dir, journal, state_item) or promoted
            if promoted:
                _save_journal(output_dir, journal)
            for state_item in journal["items"].values():
                if state_item["state"] == "success":
                    _verify_journal_success(output_dir, state_item)
            if journal["phase"] == "finalized" and not retry_failed:
                index = load_extraction_index(output_dir / "extraction-index", source_manifest_dir)
                verify_native_build(output_dir / "extraction-index", source_manifest_dir)
                return index, any(row["status"] == "failed" for row in index["records"])
            if journal["phase"] == "finalized":
                if not any(item["state"] == "failed" for item in journal["items"].values()):
                    index = load_extraction_index(output_dir / "extraction-index", source_manifest_dir)
                    verify_native_build(output_dir / "extraction-index", source_manifest_dir)
                    return index, False
                journal["phase"] = "active"
            # An interrupted item cannot have committed success; re-run it from
            # original source and do not trust any orphaned output.
            changed = False
            for state_item in journal["items"].values():
                if state_item["state"] in ("running", "interrupted"):
                    state_item["state"] = "pending"; changed = True
                elif retry_failed and state_item["state"] == "failed":
                    # Keep every completed attempt's receipt reference in its
                    # historical attempt; only the mutable terminal projection is removed.
                    state_item["state"] = "pending"; state_item.pop("failure", None); state_item.pop("receipt", None); changed = True
            if changed or journal["phase"] == "active":
                _save_journal(output_dir, journal)
        else:
            output_dir.mkdir(parents=True)
            for name in ("encoded", "artifacts", "logs", ".work", _STATE_DIR):
                (output_dir / name).mkdir()
            journal = _new_journal(source, fingerprint)
            journal["preflight"] = checks
            _save_journal(output_dir, journal)

        output_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            _output_binding(output_fd, output_dir)
            for item in source["records"]:
                state_item = journal["items"][item["video_id"]]
                if state_item["state"] in ("success", "failed"):
                    continue
                digest, video_id, attempt_id = item["source_sha256"], item["video_id"], uuid.uuid4().hex
                attempt = {"id": attempt_id, "source_sha256": digest, "started_utc": datetime.now(timezone.utc).isoformat()}
                state_item["state"] = "running"; state_item["attempts"].append(attempt)
                _save_journal(output_dir, journal)
                work_rel = f".work/{digest}/{attempt_id}"
                encoded_rel, artifact_rel = f"encoded/sha256/{digest}.mp4", f"artifacts/sha256/{digest}"
                entry: dict[str, Any] = {"video_id": video_id, "source_relative_path": item["source_relative_path"], "attempt_id": attempt_id}
                stage = "cleanup"
                work_fd = input_fd = encoded_parent_fd = artifact_parent_fd = generated_parent_fd = None
                try:
                    _output_binding(output_fd, output_dir)
                    # Orphans are only removable at fixed manifest-SHA paths and are
                    # never reused as newly trusted bytes.
                    _remove_owned(output_dir, encoded_rel); _remove_owned(output_dir, artifact_rel)
                    work_fd = _open_output_dir(output_fd, work_rel, create=True)
                    input_fd = _open_dir_at(work_fd, "input", create=True)
                    encoded_parent_fd = _open_output_dir(output_fd, "encoded/sha256", create=True)
                    artifact_parent_fd = _open_output_dir(output_fd, "artifacts/sha256", create=True)
                    stage = "source_verification"
                    original = _safe_source_path(source_root, item["source_relative_path"])
                    staged = Path(f"/proc/self/fd/{input_fd}/{original.name}")
                    entry["source_verification"] = verified_stage_copy(original, digest, int(item["source_bytes"]), staged)
                    stage = "source_probe"
                    source_probe, entry["source_probe"] = probe_presentation(staged, run, pass_fds=(input_fd,))
                    _output_binding(output_fd, output_dir)
                    encode_log = output_dir / "logs" / f"{digest}.{attempt_id}.encode.log"
                    stage = "encoder"
                    pinned_input = Path(f"/proc/self/fd/{input_fd}")
                    entry["encoder"] = _run_logged(["bash", str(encoder_script), str(pinned_input)], run=run,
                                                     log_path=encode_log, build_root=output_dir, pass_fds=(input_fd,))
                    generated_dir_name = "hevc-wpp-gop-16-qp-28-b-0"
                    try:
                        generated_parent_fd = _open_dir_at(input_fd, generated_dir_name)
                    except UnsafeOutputError as exc:
                        try:
                            os.lstat(generated_dir_name, dir_fd=input_fd)
                        except FileNotFoundError:
                            raise WorkflowError("encoder completed without the expected fresh output (possible silent skip)") from exc
                        raise
                    generated = Path(f"/proc/self/fd/{generated_parent_fd}/{original.name}.mp4")
                    try:
                        generated_stat = os.stat(f"{original.name}.mp4", dir_fd=generated_parent_fd, follow_symlinks=False)
                    except OSError as exc:
                        raise WorkflowError("encoder completed without the expected fresh output (possible silent skip)") from exc
                    if not os.path.isfile(generated) or os.path.islink(generated):
                        raise WorkflowError("encoder output is not a fresh regular file")
                    if test_hook:
                        test_hook("after_encode", item)
                    stage = "encoded_probe"
                    encoded_probe, entry["encoded_probe"] = probe_presentation(generated, run, pass_fds=(generated_parent_fd,))
                    entry["native_geometry_timing"] = compare_native_geometry_timing(source_probe, encoded_probe)
                    _publish_at(source_fd=generated_parent_fd, source_name=f"{original.name}.mp4",
                                destination_fd=encoded_parent_fd, destination_name=f"{digest}.mp4",
                                output_fd=output_fd, output_dir=output_dir, destination_relative="encoded/sha256")
                    encoded_final = Path(f"/proc/self/fd/{encoded_parent_fd}/{digest}.mp4")
                    encoded_sha = sha256_file(encoded_final)
                    if test_hook:
                        test_hook("after_encoded_publish", item)
                    _output_binding(output_fd, output_dir)
                    extract_log = output_dir / "logs" / f"{digest}.{attempt_id}.extract.log"
                    stage = "extractor"
                    private_artifact = Path(f"/proc/self/fd/{work_fd}/artifact")
                    entry["extractor"] = _run_logged([python_executable, str(extractor_script), str(encoded_final), "--out", str(private_artifact), "--types", "all"],
                                                       run=run, log_path=extract_log, build_root=output_dir,
                                                       pass_fds=(encoded_parent_fd, work_fd))
                    stage = "artifact_validation"
                    _normalize_private_artifact_source(work_fd, output_dir / "encoded" / "sha256" / f"{digest}.mp4")
                    artifact = validate_artifact(private_artifact)
                    if tuple(Fraction(value) for value in artifact["presentation_times_seconds"]) != encoded_probe.presentation_times:
                        raise WorkflowError("producer artifact PTS/timing does not match encoded video")
                    _publish_at(source_fd=work_fd, source_name="artifact", destination_fd=artifact_parent_fd,
                                destination_name=digest, output_fd=output_fd, output_dir=output_dir,
                                destination_relative="artifacts/sha256")
                    if test_hook:
                        test_hook("after_artifact_validation", item)
                    entry["encoded_sha256"], entry["artifact"] = encoded_sha, artifact
                    success = {"record": _record_for(item, artifact, encoded_sha), "ledger_entry": entry}
                    attempt["outcome"] = "success"
                    receipt = _write_completion_receipt(output_dir, journal, video_id=video_id, digest=digest,
                                                        attempt=attempt, outcome="success", payload=success)
                    attempt["receipt"] = receipt
                    if test_hook:
                        test_hook("after_receipt_before_journal", item)
                    state_item["success"] = success; state_item["receipt"] = receipt; state_item["state"] = "success"
                    _save_journal(output_dir, journal)
                    if test_hook:
                        test_hook("after_success_commit", item)
                except UnsafeOutputError:
                    # An attacker replaced a public path.  Held FDs prevented an
                    # outside write; do not turn this into a per-video failure.
                    raise
                except Exception as exc:  # coverage continues; retry requires explicit opt-in
                    _output_binding(output_fd, output_dir)
                    _remove_owned(output_dir, encoded_rel); _remove_owned(output_dir, artifact_rel)
                    message, code = " ".join(str(exc).split())[:1000], _code(exc)
                    entry["failure"] = {"code": code, "message": message, "stage": stage}
                    failure = {"code": code, "message": message, "stage": stage, "ledger_entry": entry}
                    attempt["outcome"] = "failed"
                    receipt = _write_completion_receipt(output_dir, journal, video_id=video_id, digest=digest,
                                                        attempt=attempt, outcome="failed", payload=failure)
                    attempt["receipt"] = receipt
                    state_item["failure"] = failure; state_item["receipt"] = receipt; state_item["state"] = "failed"
                    _save_journal(output_dir, journal)
                finally:
                    for fd in (generated_parent_fd, artifact_parent_fd, encoded_parent_fd, input_fd, work_fd):
                        if fd is not None:
                            os.close(fd)
                    try:
                        _output_binding(output_fd, output_dir)
                    except UnsafeOutputError:
                        pass
                    else:
                        _remove_owned(output_dir, work_rel)
        finally:
            os.close(output_fd)

        index = _finalize_journal(output_dir=output_dir, source_manifest_dir=source_manifest_dir, source=source,
                                  journal=journal, processing_profile=profile, identity=identity, test_hook=test_hook)
        return index, any(record["status"] == "failed" for record in index["records"])
    finally:
        try:
            _release_build_lock(lock)
        except WorkflowError:
            # Preserve a processing/interruption exception over a release race;
            # otherwise surface the ownership warning to the caller.
            if sys.exc_info()[0] is None:
                raise
