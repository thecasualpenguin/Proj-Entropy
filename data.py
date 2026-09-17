"""Convert CASIA-B PNG sequences into one HEVC MP4 per subject/type/view."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from tqdm import tqdm


def dataset_root() -> Path:
    return Path(
        os.environ.get("SANDBOX_ROOT", Path(__file__).resolve().parents[2])
    ).expanduser() / "datasets"


def discover_sequences(source: Path):
    """Yield sequences in deterministic order, sorting by numeric frame index."""
    view_dirs = sorted(tqdm(
        source.glob("*/*/*"), desc="Finding sequence directories", unit="dir",
    ))
    for view_dir in tqdm(view_dirs, desc="Discovering sequences", unit="sequence"):
        if not view_dir.is_dir():
            continue
        frames = [
            p for p in view_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
        ]
        if not frames:
            raise ValueError(f"No PNG frames in {view_dir}")
        subject, sequence, view = view_dir.relative_to(source).parts
        pattern = re.compile(
            rf"{re.escape(subject + '-' + sequence + '-' + view)}-(\d+)\.png",
            re.IGNORECASE,
        )
        indexed = {}
        for frame in frames:
            match = pattern.fullmatch(frame.name)
            if match is None:
                raise ValueError(f"Unexpected CASIA-B frame name: {frame}")
            index = int(match[1])
            if index in indexed:
                raise ValueError(f"Duplicate frame index {index} in {view_dir}")
            indexed[index] = frame
        yield subject, sequence, view, [indexed[i] for i in sorted(indexed)]


def encode_video(
    frames: list[Path], output: Path, *, ffmpeg: str, fps: float, bitrate_mbps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Sequential links preserve every available frame even when numbering has gaps.
    # Encode beside the destination so replacement is atomic after success.
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=output.parent) as tmp:
        temporary = Path(tmp)
        for index, frame in enumerate(frames):
            (temporary / f"{index:08d}.png").symlink_to(frame.resolve())
        video = temporary / "video.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-xerror", "-framerate", str(fps), "-start_number", "0",
            "-i", str(temporary / "%08d.png"), "-frames:v", str(len(frames)),
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "hevc_videotoolbox", "-allow_sw", "0",
            "-b:v", str(round(bitrate_mbps * 1_000_000)),
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
            "-movflags", "+faststart", str(video),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Encoding failed for {output}:\n{result.stderr.strip()}")
        video.replace(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=dataset_root() / "casia-b")
    parser.add_argument("--output", type=Path, default=dataset_root() / "casia-b-video")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument(
        "--bitrate-mbps", type=float, default=4,
        help="Target hardware encoder bitrate in Mbps (default: 4)",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="FFmpeg executable with hevc_videotoolbox",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing videos")
    parser.add_argument("--dry-run", action="store_true", help="List conversions without writing files")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.bitrate_mbps) or args.bitrate_mbps < 0.000001:
        parser.error("--bitrate-mbps must be finite and at least 0.000001")
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Source directory does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate, non-nested directories")
    try:
        sequences = list(discover_sequences(source))
        if not sequences:
            raise ValueError(f"No subject/type/view sequences found in {source}")
        if not args.dry_run:
            ffmpeg = shutil.which(args.ffmpeg)
            if ffmpeg is None:
                raise ValueError("FFmpeg is required; install a macOS build with hevc_videotoolbox or set --ffmpeg")
            encoders = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                check=True,
            )
            if "hevc_videotoolbox" not in encoders.stdout:
                raise ValueError(f"{ffmpeg} does not include the hevc_videotoolbox hardware H.265 encoder")
        converted = skipped = 0
        for index, (subject, sequence, view, frames) in enumerate(sequences, 1):
            video = output / subject / sequence / f"{view}.mp4"
            if video.exists() and not args.overwrite:
                skipped += 1
                print(f"[{index}/{len(sequences)}] Skip existing {video}")
                continue
            print(f"[{index}/{len(sequences)}] {subject}/{sequence}/{view}: "
                  f"{len(frames)} frames -> {video}", flush=True)
            if args.dry_run:
                continue
            encode_video(frames, video, ffmpeg=ffmpeg, fps=args.fps,
                         bitrate_mbps=args.bitrate_mbps)
            metadata = {
                "subject": subject, "sequence": sequence, "view": view,
                "source": str(source / subject / sequence / view),
                "frames": [frame.name for frame in frames], "frame_count": len(frames),
                "fps": args.fps, "duration_seconds": len(frames) / args.fps,
                "codec": "hevc", "encoder": "hevc_videotoolbox",
                "hardware_required": True, "bitrate_mbps": args.bitrate_mbps,
            }
            video.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            converted += 1
        print(f"{'Planned' if args.dry_run else 'Finished'}: {len(sequences)} sequences, "
              f"{converted} converted, {skipped} skipped.")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
