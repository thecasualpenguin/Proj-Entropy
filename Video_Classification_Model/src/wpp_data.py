"""Strict, standalone data contract for native-row WPP classification.

This module deliberately does not use the legacy 3,000-packet CSV contract.
Manifest times are integer PTS ticks plus an exact rational time base; all window
assignment and duration weighting is performed with :class:`fractions.Fraction`.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

FEATURE_NAMES = ("delta_row_share", "row_share")
WINDOW_SECONDS = Fraction(2, 1)
LABELS = ("real", "aigc")
LABEL_TO_ID = {"real": 0, "aigc": 1}


@dataclass(frozen=True)
class WPPFrame:
    pts: int
    duration_pts: int
    rows: np.ndarray  # (native CTU rows, 2), including valid zero values


@dataclass(frozen=True)
class WPPVideo:
    video_id: str
    prompt_group: str
    label: int
    source: str
    time_base: Fraction
    clip_end_pts: int
    frames: tuple[WPPFrame, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class WPPWindow:
    video_id: str
    index: int
    frame_indices: tuple[int, ...]
    assigned_duration: Fraction
    nominal_start_pts: Fraction
    nominal_end_pts: Fraction
    crosses_nominal_boundary: bool

    @property
    def duration(self) -> Fraction:
        """Compatibility alias: aggregation always uses assigned native interval mass."""
        return self.assigned_duration


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt_text(value: str) -> str:
    """Stable prompt grouping fallback, intentionally conservative."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _label(value: Any, video_id: str) -> int:
    if value in (0, "0", "real", "REAL", "Real"):
        return 0
    if value in (1, "1", "aigc", "AIGC", "Aigc"):
        return 1
    raise ValueError(f"{video_id}: label must be real/0 or AIGC/1")


def _rows(value: Any, video_id: str, frame_index: int) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{video_id}: frame {frame_index} must have at least one native CTU row")
    parsed: list[list[float]] = []
    for row in value:
        if isinstance(row, Mapping):
            if tuple(row) != FEATURE_NAMES:
                raise ValueError(f"{video_id}: row feature keys must be exactly {FEATURE_NAMES}")
            item = [row[name] for name in FEATURE_NAMES]
        else:
            item = row
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{video_id}: every row must contain exactly {FEATURE_NAMES}")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in item):
            raise ValueError(f"{video_id}: features must be numeric")
        parsed.append([float(item[0]), float(item[1])])
    array = np.asarray(parsed, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{video_id}: features must be finite")
    return array


def _load_frames(item: Mapping[str, Any], manifest_path: Path) -> Any:
    if "frames" in item:
        return item["frames"]
    frame_path = item.get("frames_path")
    if not isinstance(frame_path, str) or not frame_path:
        raise ValueError(f"{item.get('video_id', '<unknown>')}: requires frames or frames_path")
    path = (manifest_path.parent / frame_path).resolve()
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    return content["frames"] if isinstance(content, Mapping) and "frames" in content else content


def _parse_complete(item: Mapping[str, Any], manifest_path: Path) -> WPPVideo:
    video_id = item.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("every manifest video requires a nonempty video_id")
    label = _label(item.get("label"), video_id)
    prompt = item.get("prompt_id")
    if prompt is None:
        text = item.get("prompt_text")
        if not isinstance(text, str) or not normalize_prompt_text(text):
            raise ValueError(f"{video_id}: requires prompt_id or nonempty prompt_text")
        prompt = normalize_prompt_text(text)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{video_id}: prompt group must be nonempty")
    source = item.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{video_id}: source is required")
    tb = item.get("time_base")
    if not isinstance(tb, Mapping):
        raise ValueError(f"{video_id}: time_base {{num, den}} is required")
    time_base = Fraction(_integer(tb.get("num"), "time_base.num", positive=True),
                         _integer(tb.get("den"), "time_base.den", positive=True))
    clip_end = _integer(item.get("clip_end_pts"), f"{video_id}.clip_end_pts", positive=True)
    adapter = item.get("adapter")
    if isinstance(adapter, Mapping) and adapter.get("kind") == "deepaction_npz_v1":
        feature_path = adapter.get("feature_path")
        if not isinstance(feature_path, str) or not Path(feature_path).is_file():
            raise ValueError(f"{video_id}: DeepAction adapter feature_path is missing")
        # Keep feature arrays lazy: training/normalization materialize one video at a time.
        return WPPVideo(video_id, prompt.strip(), label, source, time_base, clip_end, (), dict(item))
    raw_frames = _load_frames(item, manifest_path)
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"{video_id}: requires nonempty frames")
    frames: list[WPPFrame] = []
    previous = -1
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{video_id}: frame {index} must be an object")
        pts = _integer(frame.get("pts"), f"{video_id}.frames[{index}].pts")
        duration = _integer(frame.get("duration_pts"), f"{video_id}.frames[{index}].duration_pts", positive=True)
        if pts <= previous:
            raise ValueError(f"{video_id}: frame PTS must be strictly increasing")
        previous = pts
        frames.append(WPPFrame(pts, duration, _rows(frame.get("rows"), video_id, index)))
    if max(frame.pts + frame.duration_pts for frame in frames) != clip_end:
        raise ValueError(f"{video_id}: clip_end_pts must exactly equal the latest encoded frame end")
    return WPPVideo(video_id, prompt.strip(), label, source, time_base, clip_end, tuple(frames), dict(item))


def load_manifest(path: str | Path, *, expected_video_count: int | None = None,
                  explicit_failed_video_ids: Sequence[str] = (), approved_video_ids: Sequence[str] = (),
                  official_mapping: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[list[WPPVideo], dict[str, Any]]:
    """Load a WPP v1 JSON manifest and reject unapproved failures.

    The manifest is ``{"schema_version":"wpp-v1", "videos":[...]}``. Each
    item has ``status: complete|failed``. Failed entries are accepted only when
    their ids occur in the explicitly configured failure table.
    """
    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "wpp-v1" or not isinstance(raw.get("videos"), list):
        raise ValueError("WPP manifest must be a wpp-v1 object with a videos list")
    items = raw["videos"]
    if expected_video_count is not None and len(items) != expected_video_count:
        raise ValueError(f"manifest has {len(items)} inputs, expected {expected_video_count}")
    approved_failed = set(explicit_failed_video_ids)
    ids: set[str] = set(); videos: list[WPPVideo] = []; failures: list[str] = []; encoded_input_hashes: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("manifest videos must be objects")
        video_id = item.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in ids:
            raise ValueError("video_id values must be nonempty and unique")
        ids.add(video_id)
        status = item.get("status", "complete")
        if status == "complete":
            videos.append(_parse_complete(item, manifest_path))
            if isinstance(item.get("frames_path"), str):
                encoded_input_hashes[video_id] = sha256_file((manifest_path.parent / item["frames_path"]).resolve())
        elif status == "failed":
            failures.append(video_id)
            if video_id not in approved_failed:
                raise ValueError(f"failed input {video_id} is not in explicit_failed_video_ids")
        else:
            raise ValueError(f"{video_id}: status must be complete or failed")
    unknown = approved_failed - set(failures)
    if unknown:
        raise ValueError(f"explicit_failed_video_ids are not failed manifest inputs: {sorted(unknown)}")
    approved = set(approved_video_ids)
    if approved and ids != approved:
        raise ValueError("manifest input IDs differ from the approved input table")
    if official_mapping is not None:
        if set(official_mapping) != ids:
            raise ValueError("official mapping IDs differ from manifest input IDs")
        for video in videos:
            official = official_mapping[video.video_id]
            if not isinstance(official, Mapping) or official.get("source") != video.source or _label(official.get("label"), video.video_id) != video.label:
                raise ValueError(f"{video.video_id}: source or label differs from official mapping")
    if not videos:
        raise ValueError("manifest has no complete videos")
    assignments = raw.get("split_assignments")
    if assignments is not None and (not isinstance(assignments, Mapping) or set(assignments) != ids or set(assignments.values()) != {"train", "validation", "test"}):
        raise ValueError("internal split_assignments must cover every input and contain train/validation/test")
    return videos, {"manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                    "input_count": len(items), "complete_count": len(videos), "failed_video_ids": failures,
                    "encoded_input_sha256": encoded_input_hashes, "schema_version": "wpp-v1", "split_assignments": assignments,
                    "provenance": raw.get("provenance")}


def materialize_video(video: WPPVideo) -> WPPVideo:
    """Load one DeepAction NPZ only when needed, retaining no corpus-wide array cache."""
    if video.frames: return video
    adapter = video.metadata.get("adapter")
    if not isinstance(adapter, Mapping) or adapter.get("kind") != "deepaction_npz_v1":
        raise ValueError(f"{video.video_id}: video has no frames")
    feature_path = Path(adapter["feature_path"])
    expected_hash = adapter.get("features_fingerprint")
    if expected_hash and sha256_file(feature_path) != expected_hash: raise ValueError(f"{video.video_id}: NPZ hash differs from indexed artifact")
    with np.load(feature_path, allow_pickle=False) as payload:
        required = ("delta_row_share", "row_share", "pts")
        if any(key not in payload for key in required): raise ValueError(f"{video.video_id}: NPZ missing required feature/timing arrays")
        delta, share, pts = (np.asarray(payload[key]) for key in required)
    if delta.ndim != 2 or share.shape != delta.shape or pts.shape != (delta.shape[0],):
        raise ValueError(f"{video.video_id}: invalid NPZ frame/row layout")
    if not np.issubdtype(pts.dtype, np.integer) or len(pts) == 0 or np.any(np.diff(pts) <= 0):
        raise ValueError(f"{video.video_id}: NPZ PTS must be strictly increasing integers")
    sidecar = Path(adapter["timing_sidecar_path"])
    if not sidecar.is_file():
        if not adapter.get("allow_sidecar_generation", False): raise ValueError(f"{video.video_id}: required indexed packet timing sidecar is missing")
        command = [adapter["ffprobe_path"], "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base:packet=pts,duration", "-of", "json", adapter["encoded_path"]]
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout; timing = json.loads(raw)
        timing["dependency_sha256"] = {"encoded": sha256_file(adapter["encoded_path"]), "features": sha256_file(feature_path)}
        sidecar.parent.mkdir(parents=True, exist_ok=True); temporary = sidecar.with_suffix(sidecar.suffix + ".tmp"); temporary.write_text(json.dumps(timing, sort_keys=True)); os.replace(temporary, sidecar)
    if adapter.get("timing_sidecar_sha256") and sha256_file(sidecar) != adapter["timing_sidecar_sha256"]: raise ValueError(f"{video.video_id}: packet timing sidecar hash differs from index")
    timing = json.loads(sidecar.read_text())
    dependency = timing.get("dependency_sha256", {})
    if dependency.get("encoded") != sha256_file(adapter["encoded_path"]) or dependency.get("features") != sha256_file(feature_path):
        raise ValueError(f"{video.video_id}: packet timing dependency differs from index")
    streams = timing.get("streams", [])
    expected_tb = f"{video.time_base.numerator}/{video.time_base.denominator}"
    if len(streams) != 1 or streams[0].get("time_base") != expected_tb: raise ValueError(f"{video.video_id}: ffprobe stream time base differs from metadata")
    packets = timing.get("packets", [])
    packet_pts = np.asarray([p.get("pts") for p in packets], dtype=np.int64)
    durations = np.asarray([p.get("duration") for p in packets], dtype=np.int64)
    if len(packet_pts) != len(pts) or not np.array_equal(packet_pts, pts) or np.any(durations <= 0):
        raise ValueError(f"{video.video_id}: ffprobe packet PTS/durations do not align with NPZ presentation frames")
    if int(pts[-1] + durations[-1]) != video.clip_end_pts: raise ValueError(f"{video.video_id}: packet endpoint differs from indexed clip endpoint")
    values = np.stack((delta, share), axis=-1).astype(np.float32)
    if not np.isfinite(values).all(): raise ValueError(f"{video.video_id}: NPZ features must be finite")
    frames = tuple(WPPFrame(int(p), int(d), values[i]) for i, (p, d) in enumerate(zip(pts, durations)))
    return WPPVideo(video.video_id, video.prompt_group, video.label, video.source, video.time_base,
                    video.clip_end_pts, frames, video.metadata)


def make_windows(video: WPPVideo) -> tuple[WPPWindow, ...]:
    """Assign every frame once using exact two-second PTS windows from first PTS."""
    if not video.frames: raise ValueError("materialize native adapter videos before windowing")
    anchor = Fraction(video.frames[0].pts, 1)
    width = WINDOW_SECONDS / video.time_base  # exact PTS ticks, possibly fractional
    endpoint = Fraction(video.clip_end_pts, 1)
    buckets: dict[int, list[int]] = {}
    for index, frame in enumerate(video.frames):
        bucket = int((Fraction(frame.pts, 1) - anchor) // width)
        buckets.setdefault(bucket, []).append(index)
    result: list[WPPWindow] = []
    covered: set[int] = set()
    for index in sorted(buckets):
        start = anchor + index * width
        nominal_end = min(start + width, endpoint)
        if nominal_end <= start:
            raise ValueError(f"{video.video_id}: nonpositive elapsed window duration")
        frame_indices = tuple(buckets[index]); covered.update(frame_indices)
        # A native frame is never copied into a second window. Its whole encoded
        # presentation interval belongs to its start-PTS bucket, including a
        # VFR/terminal interval that crosses a two-second boundary. In normal
        # encoded streams these contiguous intervals sum exactly to clip duration.
        covered_duration = sum((Fraction(video.frames[i].duration_pts, 1) * video.time_base for i in frame_indices), Fraction())
        if covered_duration <= 0: raise ValueError(f"{video.video_id}: nonpositive assigned frame coverage")
        crosses = any(Fraction(video.frames[i].pts + video.frames[i].duration_pts, 1) > nominal_end for i in frame_indices)
        result.append(WPPWindow(video.video_id, index, frame_indices, covered_duration, start, nominal_end, crosses))
    if covered != set(range(len(video.frames))):
        raise AssertionError("every frame must belong to exactly one window")
    return tuple(result)


def windowing_record(video: WPPVideo, windows: Sequence[WPPWindow] | None = None) -> dict[str, Any]:
    """Serializable exact timing provenance for checkpoints/run input indexes."""
    if not video.frames: video = materialize_video(video)
    windows = tuple(windows or make_windows(video))
    total = sum((window.assigned_duration for window in windows), Fraction())
    return {"video_id": video.video_id, "clip_end_pts": video.clip_end_pts,
            "first_pts": video.frames[0].pts, "time_base": [video.time_base.numerator, video.time_base.denominator],
            "assigned_duration_total": str(total), "windows": [
                {"index": window.index, "nominal_start_pts": str(window.nominal_start_pts),
                 "nominal_end_pts": str(window.nominal_end_pts), "frame_indices": list(window.frame_indices),
                 "assigned_duration": str(window.assigned_duration), "crosses_nominal_boundary": window.crosses_nominal_boundary}
                for window in windows]}


class StreamingNormalizer:
    """Numerically stable train-only per-channel population z-score fit."""
    def __init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(2, dtype=np.float64)
        self.m2 = np.zeros(2, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("normalizer values must be finite nonempty (N,2) native rows")
        n = len(x); mean = x.mean(0); m2 = ((x - mean) ** 2).sum(0)
        if not self.count:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        total = self.count + n; delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total; self.count = total

    def fit_videos(self, videos: Iterable[WPPVideo]) -> dict[str, Any]:
        ids: list[str] = []
        for original in videos:
            video = materialize_video(original); ids.append(video.video_id)
            for frame in video.frames:
                self.update(frame.rows)
        if not self.count:
            raise ValueError("normalizer cannot fit empty training videos")
        std = np.sqrt(self.m2 / self.count)
        if not np.isfinite(std).all() or np.any(std == 0):
            raise ValueError("training features require nonzero finite standard deviation per channel")
        return {"feature_names": list(FEATURE_NAMES), "mean": self.mean.tolist(), "std": std.tolist(),
                "valid_native_row_count": self.count, "fit_split": "train", "fit_video_ids": ids}


def normalize_video(video: WPPVideo, stats: Mapping[str, Any]) -> WPPVideo:
    video = materialize_video(video)
    if tuple(stats.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("normalizer feature order differs from WPP feature contract")
    mean = np.asarray(stats["mean"], dtype=np.float32); std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
        raise ValueError("invalid normalizer statistics")
    frames = tuple(WPPFrame(f.pts, f.duration_pts, ((f.rows - mean) / std).astype(np.float32)) for f in video.frames)
    return WPPVideo(video.video_id, video.prompt_group, video.label, video.source, video.time_base,
                    video.clip_end_pts, frames, video.metadata)


def assigned_prompt_split(videos: Sequence[WPPVideo], assignments: Mapping[str, str]) -> dict[str, list[WPPVideo]]:
    """Use assignments made on the full pre-failure inventory and recheck prompt isolation."""
    result: dict[str, list[WPPVideo]] = {"train": [], "validation": [], "test": []}; seen: dict[str, str] = {}
    for video in videos:
        split = assignments.get(video.video_id)
        if split not in result: raise ValueError(f"{video.video_id}: missing saved prompt split assignment")
        previous = seen.setdefault(video.prompt_group, split)
        if previous != split: raise ValueError(f"prompt group {video.prompt_group!r} leaks across saved splits")
        result[split].append(video)
    for split in result: result[split].sort(key=lambda v: v.video_id)
    return result


def prompt_group_split(videos: Sequence[WPPVideo], seed: int = 42) -> dict[str, list[WPPVideo]]:
    """Deterministic 70/15/15 split at prompt-group level before window creation."""
    import random
    groups: dict[str, list[WPPVideo]] = {}
    for video in videos: groups.setdefault(video.prompt_group, []).append(video)
    # DeepAction variants intentionally share a prompt across real and AIGC.
    # Therefore stratifying individual labels would leak prompts; split groups directly.
    prompt_ids = sorted(groups)
    if len(prompt_ids) < 3: raise ValueError("need at least three prompt groups for 70/15/15 split")
    rng = random.Random(seed); rng.shuffle(prompt_ids)
    n = len(prompt_ids); n_train = round(.70 * n); n_val = round(.15 * n)
    n_train = min(max(1, n_train), n - 2); n_val = min(max(1, n_val), n - n_train - 1)
    output: dict[str, list[WPPVideo]] = {"train": [], "validation": [], "test": []}
    assignment = (("train", prompt_ids[:n_train]), ("validation", prompt_ids[n_train:n_train+n_val]), ("test", prompt_ids[n_train+n_val:]))
    for split, chosen in assignment:
        for prompt in chosen: output[split].extend(groups[prompt])
    for split in output: output[split].sort(key=lambda x: x.video_id)
    seen = [set(v.prompt_group for v in output[s]) for s in output]
    if seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]: raise AssertionError("prompt leakage")
    return output
