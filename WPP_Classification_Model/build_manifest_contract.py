#!/usr/bin/env python3
"""Inspect or materialize approved upstream source-manifest contracts.

This tool never encodes/extracts media and never invokes the model pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.deepaction_manifest import (
    DUPLICATE_POLICIES,
    EXCLUSION_REPORT_FILE,
    build_deepaction_manifest_inputs,
    inspect_deepaction_audit,
)
from src.manifest_contract import ContractError, sha256_file, write_source_manifest

DEFAULT_ROOT = Path("/mnt/d/projects/proj-entropy/datasets/deepaction_v1")
DEFAULT_AUDIT = Path("/mnt/d/projects/proj-entropy/datasets/deepaction_v1_wpp_512_15fps_splits_seed42/split_audit_manifest.csv")


def _duplicate_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--duplicate-policy", choices=sorted(DUPLICATE_POLICIES), default="error",
        help=("duplicate original-content handling. Default 'error' preserves generic strict uniqueness; "
              "DeepAction-only 'quarantine-all' excludes every occurrence in each repeated audit SHA-256 group."),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect-deepaction", help="cheap audit/source inventory validation; does not hash file contents")
    inspect.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    inspect.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    inspect.add_argument("--no-inventory", action="store_true", help="skip recursive .mp4 inventory comparison")
    _duplicate_policy_argument(inspect)
    build = subparsers.add_parser("build-deepaction", help="hash originals, verify audit, then atomically write a source manifest")
    build.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--dataset-id", default="deepaction_v1")
    build.add_argument("--source-root-ref", default="deepaction_v1")
    build.add_argument("--verify-source-hashes", action="store_true", help="required: read every original (including exclusions) and compare it to audit SHA-256")
    _duplicate_policy_argument(build)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "inspect-deepaction":
            print(json.dumps(inspect_deepaction_audit(
                args.source_root, args.audit, check_inventory=not args.no_inventory,
                duplicate_policy=args.duplicate_policy,
            ), indent=2, sort_keys=True))
            return 0
        if not args.verify_source_hashes:
            raise ContractError("refusing to materialize: --verify-source-hashes is required")
        inputs = build_deepaction_manifest_inputs(
            args.source_root, args.audit, verify_source_hashes=True, duplicate_policy=args.duplicate_policy,
        )
        stats = inputs["stats"]
        result = write_source_manifest(
            args.output_dir, inputs["records"], dataset_id=args.dataset_id, source_root_ref=args.source_root_ref,
            bundle_files={EXCLUSION_REPORT_FILE: inputs["exclusion_report_bytes"]},
            provenance={
                "adapter": "deepaction_v1", "legacy_split_audit_path": str(args.audit.resolve()),
                "legacy_split_audit_sha256": sha256_file(args.audit),
                "legacy_split_names": {"val": "validation"},
                "legacy_video_id": "stored only as provenance; never used as canonical identity",
                "original_source_hash_verification": {
                    "verified": True, "verified_file_count": stats["audit_rows"],
                    "scope": "every audited original source, retained and excluded",
                },
                "duplicate_quarantine": {
                    "policy": args.duplicate_policy,
                    "inventory_rows": stats["audit_rows"], "retained_rows": stats["retained_rows"],
                    "excluded_rows": stats["excluded_rows"], "duplicate_groups": stats["duplicate_groups"],
                    "report_file": EXCLUSION_REPORT_FILE,
                    "report_sha256": inputs["exclusion_report_sha256"],
                    "report_row_count": len(inputs["exclusion_rows"]),
                    "accounting": "inventory_rows = retained_rows + excluded_rows",
                },
                "retained_partition_counts": stats["retained_split_counts"],
                "retained_prompt_group_count": stats["retained_group_count"],
            },
        )
        print(json.dumps({"output_dir": str(args.output_dir), "build": stats, **result["metadata"]}, indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
