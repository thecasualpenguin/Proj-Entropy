#!/usr/bin/env python3
"""Build or resume the approved full train/test WPP class-attribution cache.

The cache is resumable.  It always declares the complete train/test expected
set, including when --max-new-videos is used for an approved pilot.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

# Direct execution sets sys.path[0] to scripts/, not the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wpp_class_attribution import build_class_attribution_cache
from src.wpp_explain_types import DataRoots, ProgressEvent
from src.wpp_local_paths import local_config_path

DEFAULT_TRAINING_ROOT = Path("/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-training")
DEFAULT_PACKED_BUNDLE = Path("/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-packed")
DEFAULT_SOURCE_MANIFEST = Path("/mnt/d/projects/proj-entropy/datasets/deepaction_v1_manifest_seed42_quarantine_v1")
DEFAULT_EXTRACTION_INDEX = Path("/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1/extraction-index")
DEFAULT_SOURCE_VIDEOS = Path("/mnt/d/projects/proj-entropy/datasets/deepaction_v1")


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT,
                        help="Native training run containing checkpoints/best.pt.")
    parser.add_argument("--packed-bundle", type=Path, default=DEFAULT_PACKED_BUNDLE,
                        help="Approved packed native-WPP bundle directory.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST,
                        help="Approved source-manifest directory.")
    parser.add_argument("--extraction-index", type=Path, default=DEFAULT_EXTRACTION_INDEX,
                        help="Approved native extraction-index directory.")
    parser.add_argument("--source-videos", type=Path, default=DEFAULT_SOURCE_VIDEOS,
                        help="Root containing the source videos referenced by the manifest.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory; defaults to TRAINING_ROOT/class_attributions.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:N], or mps (default: auto).")
    parser.add_argument("--ig-steps", type=int, default=32,
                        help="Integrated-gradients steps; must be at least 2 (default: 32).")
    parser.add_argument("--attention-chunk-queries", type=int, default=128,
                        help="Attention query chunk size (default: 128).")
    parser.add_argument("--max-new-videos", type=_nonnegative_int, default=None,
                        help="Approved pilot limit only; preserves the full train/test expected set for resume.")
    return parser.parse_args(argv)


def _resolved(path: Path | None) -> str | None:
    # Translate approved /mnt/<drive>/ defaults before Windows Path.resolve()
    # would anchor them to the current drive's \\mnt directory.
    return None if path is None else str(local_config_path(path).resolve())


def _startup_config(args: argparse.Namespace) -> dict[str, object]:
    training_root = local_config_path(args.training_root).resolve()
    cache_dir = local_config_path(args.cache_dir).resolve() if args.cache_dir is not None else training_root / "class_attributions"
    return {
        "training_root": str(training_root),
        "packed_bundle": _resolved(args.packed_bundle),
        "source_manifest": _resolved(args.source_manifest),
        "extraction_index": _resolved(args.extraction_index),
        "source_videos": _resolved(args.source_videos),
        "cache_dir": str(cache_dir),
        "splits": ["train", "test"],
        "checkpoint": "best.pt",
        "device": args.device,
        "ig_steps": args.ig_steps,
        "attention_chunk_queries": args.attention_chunk_queries,
        "max_new_videos": args.max_new_videos,
    }


def _progress(started_at: float):
    def emit(event: ProgressEvent) -> None:
        counts = ""
        if event.completed is not None or event.total is not None:
            counts = f" completed={event.completed if event.completed is not None else '?'}"
            counts += f" expected={event.total if event.total is not None else '?'}"
        elapsed = time.monotonic() - started_at
        print(f"progress stage={event.stage}{counts} elapsed_s={elapsed:.1f} message={event.message}", flush=True)
    return emit


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _startup_config(args)
    print("starting WPP class-attribution cache build", flush=True)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    cancel_requested = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    started_at = time.monotonic()

    def request_cancel(_signum: int, _frame: object) -> None:
        if not cancel_requested.is_set():
            print("cancellation requested; finishing the current operation and preserving resumable cache state", flush=True)
        cancel_requested.set()

    signal.signal(signal.SIGINT, request_cancel)
    try:
        result = build_class_attribution_cache(
            config["training_root"],
            roots=DataRoots(
                packed_bundle=config["packed_bundle"],
                source_manifest=config["source_manifest"],
                index=config["extraction_index"],
                source_video_root=config["source_videos"],
            ),
            cache_dir=config["cache_dir"],
            splits=("train", "test"),
            ig_steps=args.ig_steps,
            attention_chunk_queries=args.attention_chunk_queries,
            device=args.device,
            progress=_progress(started_at),
            should_cancel=cancel_requested.is_set,
            max_new_videos=args.max_new_videos,
        )
    except KeyboardInterrupt:
        # Covers interruption before the backend can poll should_cancel.
        print("interrupted; rerun this command to resume the existing cache", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        # The backend validates provenance, cache ownership, and model inputs;
        # render every operational failure without a terminal traceback.
        print(f"cache build error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    print("cache build result", flush=True)
    print(json.dumps(asdict(result), default=str, indent=2, sort_keys=True), flush=True)
    if not result.complete:
        print("cache is incomplete but resumable; rerun this command to continue", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
