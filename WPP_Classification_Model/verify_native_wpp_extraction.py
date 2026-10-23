#!/usr/bin/env python3
"""Verify ledger-bound native WPP encoded/artifact/log bytes for an extraction index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.manifest_contract import ContractError
from src.native_wpp_workflow import WorkflowError, verify_native_build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Native build extraction-index directory.")
    parser.add_argument("--source-manifest", type=Path, required=True, help="Exact source manifest used by the index.")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify_native_build(args.index, args.source_manifest), indent=2, sort_keys=True))
        return 0
    except (WorkflowError, ContractError, OSError, ValueError) as exc:
        print(f"integrity error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
