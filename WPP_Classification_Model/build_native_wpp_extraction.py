#!/usr/bin/env python3
"""Build a native-geometry/timing WPP extraction index from one source manifest.

The runner is deliberately partition-agnostic and processes every record in the
supplied manifest.  It is not a model packer and never pads/truncates frame axes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.manifest_contract import ContractError
from src.native_wpp_workflow import WorkflowError, run_workflow

DEFAULT_PRODUCER = Path("/mnt/d/projects/proj-entropy/entropy-cabac-wpp/entropy-cabac-wpp")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Published source-manifest directory; all of its records are processed.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Non-symlink root containing source_relative_path files.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New derived build directory, or an existing owned build with --resume.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted stateful build in --output-dir; finalized compatible builds are verified no-ops.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="With --resume only, retry terminal per-video failures; default preserves them.")
    parser.add_argument("--producer-dir", type=Path, default=DEFAULT_PRODUCER,
                        help="Directory containing unchanged encode_wpp.sh and extract_wpp.py.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index, had_failures = run_workflow(
            source_manifest_dir=args.source_manifest,
            source_root=args.source_root,
            output_dir=args.output_dir,
            encoder_script=args.producer_dir / "encode_wpp.sh",
            extractor_script=args.producer_dir / "extract_wpp.py",
            resume=args.resume,
            retry_failed=args.retry_failed,
            runner_script=Path(__file__),
        )
        print(json.dumps({"output_dir": str(args.output_dir), "record_count": index["metadata"]["record_count"],
                          "failed": had_failures}, indent=2, sort_keys=True))
        return 1 if had_failures else 0
    except (WorkflowError, ContractError, OSError, ValueError) as exc:
        print(f"workflow error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
