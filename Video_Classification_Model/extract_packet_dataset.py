#!/usr/bin/env python3
"""Build group-aware packet-size CSV splits from a class/group video tree."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VECTOR_LENGTH = 3000
SPLITS = ("train", "validation", "test")
VIDEO_EXTENSIONS = frozenset({
    ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".mts", ".m2ts", ".ts", ".webm", ".wmv",
})


class ExtractionError(RuntimeError):
    """An input or ffprobe error that makes a dataset unsafe to create."""


@dataclass(frozen=True)
class Video:
    path: Path
    source: str
    class_label: str
    group: str


def discover_videos(input_dir: Path) -> list[Video]:
    """Discover all supported videos; directory names are literal labels/groups."""
    videos: list[Video] = []
    for path in sorted(input_dir.rglob("*"), key=lambda p: (p.as_posix().casefold(), p.as_posix())):
        if not path.is_file() or path.suffix.casefold() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(input_dir)
        if len(relative.parts) < 3:
            raise ExtractionError(
                f"video lacks required class/group hierarchy: {relative} "
                "(expected input/class/group/.../video)"
            )
        videos.append(Video(path, relative.as_posix(), relative.parts[0], relative.parts[1]))
    if not videos:
        raise ExtractionError(f"no supported videos found under {input_dir}")
    return videos


def split_counts(group_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < len(SPLITS):
        raise ExtractionError(f"each class needs at least {len(SPLITS)} groups; found {group_count}")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ExtractionError("split ratios must be positive and sum to 1")
    raw = [group_count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))[: group_count - sum(counts)]:
        counts[index] += 1
    # A positive ratio may still round to zero. Reserve one group for every split,
    # moving one from the largest donor deterministically.
    for index, count in enumerate(counts):
        if count:
            continue
        donor = max((i for i in range(3) if counts[i] > 1), key=lambda i: (counts[i], -i), default=None)
        if donor is None:
            raise ExtractionError("ratios cannot represent every split for this class")
        counts[donor] -= 1
        counts[index] = 1
    return tuple(counts)  # type: ignore[return-value]


def assign_splits(videos: Iterable[Video], ratios: tuple[float, float, float], seed: int) -> dict[Video, str]:
    by_class: dict[str, dict[str, list[Video]]] = {}
    for video in videos:
        by_class.setdefault(video.class_label, {}).setdefault(video.group, []).append(video)
    assigned: dict[Video, str] = {}
    # A per-class derived seed means adding another class cannot perturb existing classes.
    for class_label in sorted(by_class):
        groups = sorted(by_class[class_label])
        counts = split_counts(len(groups), ratios)
        rng = random.Random(f"{seed}:{class_label}")
        rng.shuffle(groups)
        start = 0
        for split, count in zip(SPLITS, counts):
            for group in groups[start:start + count]:
                for video in by_class[class_label][group]:
                    assigned[video] = split
            start += count
    return assigned


def run_ffprobe(arguments: list[str], video: Path) -> str:
    command = ["ffprobe", "-v", "error", *arguments, str(video)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ExtractionError("ffprobe was not found on PATH; install FFmpeg") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise ExtractionError(f"ffprobe failed for {video}: {detail}")
    return completed.stdout


def packet_sizes_bits(video: Path) -> list[int]:
    """Read the default video stream's packets, falling back to the first video stream."""
    raw_streams = run_ffprobe(["-show_streams", "-of", "json"], video)
    try:
        streams = json.loads(raw_streams)["streams"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"ffprobe returned invalid stream metadata for {video}") from exc
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ExtractionError(f"no video stream found in {video}")
    selected_index = next(
        (index for index, stream in enumerate(video_streams)
         if stream.get("disposition", {}).get("default") == 1),
        0,
    )
    raw_packets = run_ffprobe([
        "-select_streams", f"v:{selected_index}", "-show_entries", "packet=size", "-of", "csv=p=0",
    ], video)
    sizes: list[int] = []
    for line in raw_packets.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            byte_count = int(value)
        except ValueError as exc:
            raise ExtractionError(f"ffprobe returned invalid packet size {value!r} for {video}") from exc
        if byte_count < 0:
            raise ExtractionError(f"ffprobe returned negative packet size for {video}")
        sizes.append(byte_count * 8)
    return sizes


def output_paths(root: Path) -> list[Path]:
    return [root / "manifest.csv", *(root / split for split in SPLITS)]


def check_collisions(output_dir: Path, overwrite: bool) -> None:
    if (output_dir.exists() or output_dir.is_symlink()) and not output_dir.is_dir():
        raise ExtractionError(f"output path exists but is not a directory: {output_dir}")
    collisions = [path for path in output_paths(output_dir) if path.exists() or path.is_symlink()]
    if collisions and not overwrite:
        joined = ", ".join(str(path) for path in collisions)
        raise ExtractionError(f"managed output already exists ({joined}); pass --overwrite to replace it")


def write_outputs(stage: Path, records: list[tuple[Video, str, list[int]]]) -> None:
    headers = [f"size{i}" for i in range(VECTOR_LENGTH)]
    by_split = {split: [] for split in SPLITS}
    for record in records:
        by_split[record[1]].append(record)
    with (stage / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "class_label", "group", "original_packet_count", "split"])
        for video, split, packets in records:
            writer.writerow([video.source, video.class_label, video.group, len(packets), split])
    for split in SPLITS:
        folder = stage / split
        folder.mkdir()
        with (folder / f"{split}-data.csv").open("w", newline="", encoding="utf-8") as data_file, \
             (folder / f"{split}-labels.csv").open("w", newline="", encoding="utf-8") as label_file:
            data_writer, label_writer = csv.writer(data_file), csv.writer(label_file)
            data_writer.writerow(headers); label_writer.writerow(["class_label"])
            for video, _, packets in by_split[split]:
                data_writer.writerow((packets[:VECTOR_LENGTH] + [0] * VECTOR_LENGTH)[:VECTOR_LENGTH])
                label_writer.writerow([video.class_label])


def replace_outputs(stage: Path, output_dir: Path, overwrite: bool) -> None:
    """Install staged files, retaining old managed outputs until all installs succeed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = stage / ".previous-managed-output"
    backup.mkdir()
    moved: list[Path] = []
    installed: list[Path] = []
    try:
        for target in output_paths(output_dir):
            if target.exists() or target.is_symlink():
                if not overwrite:  # Defensive: checked before expensive work as well.
                    raise ExtractionError(f"managed output already exists: {target}")
                os.replace(target, backup / target.name)
                moved.append(target)
        for target in output_paths(output_dir):
            os.replace(stage / target.name, target)
            installed.append(target)
    except Exception:
        for target in installed:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
        for target in moved:
            os.replace(backup / target.name, target)
        raise


def build_dataset(input_dir: Path, output_dir: Path, ratios: tuple[float, float, float], seed: int, overwrite: bool) -> int:
    if not input_dir.is_dir():
        raise ExtractionError(f"input directory does not exist or is not a directory: {input_dir}")
    check_collisions(output_dir, overwrite)
    videos = discover_videos(input_dir)
    assignments = assign_splits(videos, ratios, seed)
    records = [(video, assignments[video], packet_sizes_bits(video.path)) for video in videos]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        write_outputs(stage, records)
        replace_outputs(stage, output_dir, overwrite)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return len(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract 3,000 packet-size vectors into group-safe CSV splits.")
    parser.add_argument("--input-dir", required=True, type=Path, help="root containing class/group/.../video files")
    parser.add_argument("--output-dir", type=Path, default=here / "extracted_frames",
                        help="output directory (default: Video_Classification_Model/extracted_frames)")
    parser.add_argument("--train-ratio", type=float, default=.68)
    parser.add_argument("--validation-ratio", type=float, default=.16)
    parser.add_argument("--test-ratio", type=float, default=.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="replace only this tool's managed output files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = build_dataset(args.input_dir.resolve(), args.output_dir.resolve(),
                              (args.train_ratio, args.validation_ratio, args.test_ratio), args.seed, args.overwrite)
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} videos to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
