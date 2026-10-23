#!/usr/bin/env python3
"""Pack a complete verified native WPP build into a bounded video-aware CSV bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.wpp_dataset import WPPDatasetError, pack_wpp_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Exact complete approved source-manifest directory.")
    parser.add_argument("--index", type=Path, required=True,
                        help="Exact complete native extraction-index directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh output directory, outside this checkout and native inputs.")
    args = parser.parse_args(argv)
    try:
        metadata = pack_wpp_dataset(args.source_manifest, args.index, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir), "partition_counts": metadata["partition_counts"],
                          "class_mapping": metadata["class_mapping"]}, indent=2, sort_keys=True))
        return 0
    except (WPPDatasetError, OSError, ValueError) as exc:
        print(f"WPP pack error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
