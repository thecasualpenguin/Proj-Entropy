"""Adapter from the audited DeepAction extractor artifacts to the internal WPP v1 index.

The adapter only reads corpus artifacts.  It creates a small, versioned index in
an output/run directory; NPZ feature arrays remain lazy and are never copied.
"""
from __future__ import annotations
import csv
import json
import subprocess
import os
import numpy as np
from fractions import Fraction
from pathlib import Path
from typing import Any

from wpp_data import normalize_prompt_text, sha256_file

ADAPTER_VERSION = "deepaction_npz_v1"
EXPECTED_SOURCE_COUNTS = {"Pexels": 100, "BDAnimateDiffLightning": 500, "CogVideoX5B": 500, "RunwayML": 505, "StableDiffusion": 500, "Veo": 100, "VideoPoet": 400}
REQUIRED_COLUMNS = {"video_id", "source", "label", "label_id", "original_path", "encoded_path", "feature_path", "status",
                    "prompt_id", "prompt_text", "prompt_identity", "encoded_time_base",
                    "encoded_presentation_duration_seconds", "encoded_first_pts_seconds", "features_fingerprint"}


def _fraction(value: str, name: str) -> Fraction:
    try: result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc: raise ValueError(f"invalid {name}: {value!r}") from exc
    if result < 0: raise ValueError(f"{name} must be nonnegative")
    return result


def _timing_sidecar(feature: Path, encoded: Path, time_base: Fraction, endpoint: int, ffprobe_path: str, destination: Path) -> str:
    """Generate and validate the packet-duration sidecar against NPZ PTS."""
    with np.load(feature, allow_pickle=False) as payload: pts = np.asarray(payload["pts"])
    completed = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base:packet=pts,duration", "-of", "json", str(encoded)], check=True, capture_output=True, text=True)
    raw = json.loads(completed.stdout); packets = raw.get("packets", []); streams = raw.get("streams", [])
    packet_pts = np.asarray([p.get("pts") for p in packets], dtype=np.int64); durations = np.asarray([p.get("duration") for p in packets], dtype=np.int64)
    if len(streams) != 1 or streams[0].get("time_base") != f"{time_base.numerator}/{time_base.denominator}" or not np.array_equal(packet_pts, pts) or np.any(durations <= 0) or int(packet_pts[-1] + durations[-1]) != endpoint:
        raise ValueError(f"{feature}: packet timing does not exactly align with NPZ/metadata")
    raw["dependency_sha256"] = {"encoded": sha256_file(encoded), "features": sha256_file(feature)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp"); temporary.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8"); os.replace(temporary, destination)
    return sha256_file(destination)


def _labels(row: dict[str, str]) -> int:
    source, label, label_id = row["source"], row["label"].strip().casefold(), row["label_id"]
    expected = 0 if source == "Pexels" else 1
    if label_id != str(expected) or label != ("real" if expected == 0 else "aigc"):
        raise ValueError(f"{row['video_id']}: source-derived official label disagrees with manifest")
    return expected


def _captions(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"id", "prompt"} <= set(rows[0]) or any(any(str(value).strip() for value in (row.get(None) or [])) for row in rows):
        raise ValueError("captions.csv must contain id,prompt only")
    captions = {row["id"].strip(): normalize_prompt_text(row["prompt"]) for row in rows}
    if len(captions) != len(rows) or any(not value for value in captions.values()): raise ValueError("captions.csv has duplicate/empty IDs or prompts")
    return captions


def _split_assignments(rows: list[dict[str, str]], seed: int = 42) -> dict[str, str]:
    """Assign all prompts before failures are filtered; variants stay together."""
    import random
    groups: dict[str, list[dict[str, str]]] = {}
    text_by_group: dict[str, str] = {}
    for row in rows:
        identity = row["prompt_identity"].strip() or normalize_prompt_text(row["prompt_text"])
        text = normalize_prompt_text(row["prompt_text"])
        if not identity or not text: raise ValueError(f"{row['video_id']}: missing prompt identity/text")
        if identity in text_by_group and text_by_group[identity] != text:
            raise ValueError(f"prompt identity {identity!r} maps to inconsistent caption text")
        text_by_group[identity] = text; groups.setdefault(identity, []).append(row)
    prompts = sorted(groups); rng = random.Random(seed); rng.shuffle(prompts)
    if len(prompts) < 3: raise ValueError("DeepAction inventory needs at least three prompt groups")
    n_train = round(.70 * len(prompts)); n_val = round(.15 * len(prompts))
    n_train = min(max(1, n_train), len(prompts) - 2); n_val = min(max(1, n_val), len(prompts) - n_train - 1)
    result: dict[str, str] = {}
    for split, selected in (("train", prompts[:n_train]), ("validation", prompts[n_train:n_train+n_val]), ("test", prompts[n_train+n_val:])):
        for prompt in selected:
            for row in groups[prompt]: result[row["video_id"]] = split
    return result


def build_deepaction_index(manifest_csv: str | Path, dataset_root: str | Path, output_json: str | Path,
                            expected_video_count: int = 2605, seed: int = 42, captions_csv: str | Path | None = None,
                            strict_hashes: bool = False, ffprobe_path: str = "/home/naoromi/miniforge3/envs/entropy/bin/ffprobe") -> dict[str, Any]:
    """Validate extractor/corpus provenance and create a lazy internal WPP v1 index."""
    manifest = Path(manifest_csv).expanduser().resolve(); root = Path(dataset_root).expanduser().resolve(); out = Path(output_json).expanduser().resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError(f"DeepAction manifest is missing columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))}")
        rows = list(reader)
    if len(rows) != expected_video_count: raise ValueError(f"DeepAction manifest has {len(rows)} entries, expected {expected_video_count}")
    if len({row["video_id"] for row in rows}) != len(rows): raise ValueError("DeepAction video_id values must be unique")
    from collections import Counter
    if Counter(row["source"] for row in rows) != EXPECTED_SOURCE_COUNTS: raise ValueError("DeepAction source inventory differs from approved 2,605 table")
    captions_path = Path(captions_csv).expanduser().resolve() if captions_csv else root / "captions.csv"
    if not captions_path.is_file(): raise ValueError(f"official captions.csv is required: {captions_path}")
    captions = _captions(captions_path)
    run_config_path = manifest.parent / "run_config.json"
    if not run_config_path.is_file(): raise ValueError("extractor run_config.json is required")
    run_config = json.loads(run_config_path.read_text())
    fingerprint = run_config.get("config_fingerprint")
    if run_config.get("runner_schema_version") != 5 or run_config.get("geometry_validation_policy") != "display_vs_coded_conformance_v1" or run_config.get("tools", {}).get("ffprobe") != ffprobe_path:
        raise ValueError("extractor run configuration/provenance differs from approved schema/tooling")
    if not isinstance(fingerprint, str) or any(row.get("config_fingerprint") != fingerprint for row in rows):
        raise ValueError("manifest config fingerprints differ from run_config.json")
    prompt_ids = {row["prompt_id"].strip() for row in rows}
    if len(prompt_ids) != 100 or set(captions) != prompt_ids: raise ValueError("official captions/prompt inventory must be exactly the approved 100 IDs")
    for row in rows:
        prompt_id = row["prompt_id"].strip()
        if prompt_id not in captions or captions[prompt_id] != normalize_prompt_text(row["prompt_text"]):
            raise ValueError(f"{row['video_id']}: manifest prompt differs from official captions.csv")
    manifest_inventory = {str(Path(row["original_path"]).resolve()) for row in rows}
    if any(not path.startswith(str(root) + "/") for path in manifest_inventory):
        raise ValueError("manifest original_path escapes the approved dataset root")
    missing = [path for path in manifest_inventory if not Path(path).is_file()]
    if missing: raise ValueError(f"approved corpus files missing from dataset root: {len(missing)}")
    assignments = _split_assignments(rows, seed)
    prompt_assignment = {row["prompt_identity"].strip() or normalize_prompt_text(row["prompt_text"]): assignments[row["video_id"]] for row in rows}
    from collections import Counter
    if Counter(prompt_assignment.values()) != Counter({"train": 70, "validation": 15, "test": 15}): raise ValueError("seed-42 prompt split must be exactly 70/15/15")
    videos: list[dict[str, Any]] = []; failed: list[dict[str, str]] = []
    for row in rows:
        label = _labels(row); status = row["status"].strip()
        if status not in ("complete", "failed"): raise ValueError(f"{row['video_id']}: unknown extractor status {status!r}")
        prompt = row["prompt_identity"].strip() or normalize_prompt_text(row["prompt_text"])
        width, height, fps = int(row.get("encoded_width") or 0), int(row.get("encoded_height") or 0), row.get("encoded_frame_rate", "")
        if status == "complete" and (width <= 0 or height <= 0 or _fraction(fps, "encoded_frame_rate") <= 0):
            raise ValueError(f"{row['video_id']}: complete artifact has malformed resolution/FPS metadata")
        base = {"video_id": row["video_id"], "status": status, "prompt_id": prompt, "prompt_text": row["prompt_text"],
                "label": "real" if label == 0 else "aigc", "source": str(Path(row["original_path"]).resolve()), "source_name": row["source"],
                "width": width or None, "height": height or None, "fps": fps or None,
                "extractor_row": {key: row[key] for key in row if key.endswith("fingerprint") or key in ("source_relative_path", "generator", "action")}}
        if status == "failed":
            base["failure"] = {"error": row.get("error", ""), "feature_path": row.get("feature_path", "")}; failed.append({"video_id": row["video_id"], "error": row.get("error", "")})
            videos.append(base); continue
        feature_root = Path(row["feature_path"]).resolve(); feature = feature_root / "features.npz"; metadata = feature_root / "metadata.json"; rows_csv = feature_root / "rows.csv"; encoded = Path(row["encoded_path"]).resolve(); original = Path(row["original_path"]).resolve()
        if not feature.is_file() or not metadata.is_file() or not rows_csv.is_file() or not encoded.is_file() or not original.is_file(): raise ValueError(f"{row['video_id']}: complete artifact is missing a required dependency")
        if strict_hashes:
            for path, column in ((original, "source_fingerprint"), (encoded, "encoded_fingerprint"), (feature, "features_fingerprint"), (metadata, "metadata_fingerprint"), (rows_csv, "rows_csv_fingerprint")):
                if sha256_file(path) != row.get(column, ""): raise ValueError(f"{row['video_id']}: {column} mismatch")
        meta_json = json.loads(metadata.read_text())
        deep = meta_json.get("deepaction_wpp", {})
        if deep.get("video_id") != row["video_id"] or deep.get("original_source") != str(original) or deep.get("encoded_fingerprint") != row.get("encoded_fingerprint") or deep.get("config_fingerprint") != row.get("config_fingerprint"):
            raise ValueError(f"{row['video_id']}: metadata provenance/config mismatch")
        time_base = _fraction(row["encoded_time_base"], "encoded_time_base")
        first_seconds = _fraction(row["encoded_first_pts_seconds"], "encoded_first_pts_seconds")
        duration_seconds = _fraction(row["encoded_presentation_duration_seconds"], "encoded_presentation_duration_seconds")
        first_pts = first_seconds / time_base; endpoint = first_pts + duration_seconds / time_base
        if first_pts.denominator != 1 or endpoint.denominator != 1: raise ValueError(f"{row['video_id']}: timing does not align to encoded PTS ticks")
        sidecar_path = out.parent / "timing_sidecars" / f"{row['video_id']}.json"
        sidecar_hash = _timing_sidecar(feature, encoded, time_base, int(endpoint), ffprobe_path, sidecar_path) if strict_hashes else None
        base.update({"time_base": {"num": time_base.numerator, "den": time_base.denominator}, "clip_end_pts": int(endpoint),
                     "adapter": {"kind": ADAPTER_VERSION, "feature_path": str(feature), "metadata_path": str(metadata), "encoded_path": str(encoded),
                                 "timing_sidecar_path": str(sidecar_path), "timing_sidecar_sha256": sidecar_hash, "allow_sidecar_generation": not strict_hashes, "ffprobe_path": ffprobe_path,
                                 "features_fingerprint": row["features_fingerprint"], "metadata_fingerprint": row.get("metadata_fingerprint", "")}})
        videos.append(base)
    index = {"schema_version": "wpp-v1", "adapter_version": ADAPTER_VERSION, "videos": videos,
             "split_assignments": assignments, "provenance": {"extractor_manifest": str(manifest), "extractor_manifest_sha256": sha256_file(manifest),
             "dataset_root": str(root), "captions_csv": str(captions_path), "captions_sha256": sha256_file(captions_path), "run_config": str(run_config_path), "run_config_sha256": sha256_file(run_config_path), "config_fingerprint": fingerprint, "expected_video_count": expected_video_count, "seed": seed,
             "complete_count": len(rows) - len(failed), "failed": failed}}
    if strict_hashes:
        for video in videos:
            if video["status"] != "complete": continue
            adapter = video["adapter"]; sidecar = Path(adapter["timing_sidecar_path"])
            if not sidecar.is_file() or sha256_file(sidecar) != adapter["timing_sidecar_sha256"]: raise ValueError(f"{video['video_id']}: durable timing sidecar sweep failed")
            dependency = json.loads(sidecar.read_text()).get("dependency_sha256", {})
            if dependency.get("encoded") != sha256_file(adapter["encoded_path"]) or dependency.get("features") != sha256_file(adapter["feature_path"]):
                raise ValueError(f"{video['video_id']}: durable timing dependency sweep failed")
    out.parent.mkdir(parents=True, exist_ok=True); temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, out)
    return index
