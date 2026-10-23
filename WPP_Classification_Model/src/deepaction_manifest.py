"""DeepAction v1 adapter for the generic upstream source-manifest contract.

The normal source-manifest contract deliberately rejects duplicate original bytes.  The
only exception implemented here is an explicit, dataset-specific, auditable policy
which quarantines *every* member of a duplicate group before handing retained rows to
the generic contract.
"""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .manifest_contract import ContractError, canonical_video_id, portable_relative_path, sha256_file, validate_source_records

DEEPACTION_LABELS = (
    "BDAnimateDiffLightning", "CogVideoX5B", "Pexels", "RunwayML",
    "StableDiffusion", "Veo", "VideoPoet",
)
AUDIT_FIELDS = (
    "video_id", "split", "prompt_id", "source", "packaged_features_path",
    "packaged_features_sha256", "source_features_path", "source_features_sha256",
    "encoded_path", "encoded_sha256", "original_source_path", "original_source_sha256",
    "packet_time_base", "packet_count",
)
EXCLUSION_REPORT_FILE = "excluded_duplicate_sources.csv"
EXCLUSION_FIELDS = (
    "canonical_video_id", "legacy_video_id", "original_source_sha256", "source_relative_path",
    "label", "prompt_id", "group_id", "original_partition", "partition",
    "exclusion_reason", "duplicate_group_id", "duplicate_group_count",
)
DUPLICATE_POLICIES = frozenset(("error", "quarantine-all"))
_SPLIT_NORMALIZATION = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def _audit_rows(audit_path: Path) -> list[dict[str, str]]:
    try:
        with audit_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != AUDIT_FIELDS:
                raise ContractError("DeepAction split audit header differs from the approved legacy audit schema")
            rows = list(reader)
    except OSError as exc:
        raise ContractError(f"cannot read DeepAction audit: {exc}") from exc
    if not rows:
        raise ContractError("DeepAction audit is empty")
    return rows


def _path_fields(root: Path, audit_row: dict[str, str]) -> tuple[str, str, str, Path]:
    """Validate audit path components before resolution so links cannot be hidden."""
    raw_path = Path(audit_row["original_source_path"])
    if not raw_path.is_absolute():
        raise ContractError("DeepAction audit original_source_path must be absolute")
    try:
        raw_relative = raw_path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"audit source path is outside source root: {raw_path}") from exc
    if any(part in ("", ".", "..") for part in raw_relative.parts):
        raise ContractError(f"audit source path is not a canonical child of source root: {raw_path}")
    candidate = root
    for component in raw_relative.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ContractError(f"audit source path contains a symlink component: {candidate}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"audit source path is outside source root: {raw_path}") from exc
    parts = relative.parts
    if len(parts) != 3 or raw_path.suffix.lower() != ".mp4":
        raise ContractError(f"DeepAction source path must be <label>/<prompt_id>/<variant>.mp4: {relative}")
    label, prompt_id, _variant = parts
    if label != audit_row["source"]:
        raise ContractError(f"audit label/path disagreement at {relative}")
    if prompt_id != audit_row["prompt_id"]:
        raise ContractError(f"audit prompt/path disagreement at {relative}")
    if not prompt_id.isdecimal() or (len(prompt_id) > 1 and prompt_id.startswith("0")):
        raise ContractError(f"prompt_id must be a canonical non-negative integer: {prompt_id}")
    return portable_relative_path(relative.as_posix(), "source_relative_path"), label, prompt_id, resolved


def _validate_duplicate_policy(duplicate_policy: str) -> None:
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ContractError(f"unsupported DeepAction duplicate policy: {duplicate_policy!r}")


def _collect_deepaction_rows(source_root: Path, audit_path: Path, *, check_inventory: bool) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate every audit/source relationship without reading video bytes."""
    root = Path(source_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"DeepAction source root must be a non-symlink directory: {root}")
    audit = _audit_rows(Path(audit_path).resolve())
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_paths: set[str] = set()
    prompt_partitions: dict[str, set[str]] = defaultdict(set)
    entries: list[dict[str, Any]] = []
    source_bytes = 0
    for row in audit:
        source_relative_path, label, prompt_id, path = _path_fields(root, row)
        if label not in DEEPACTION_LABELS:
            raise ContractError(f"unapproved DeepAction label: {label}")
        if row["split"] not in _SPLIT_NORMALIZATION:
            raise ContractError(f"unsupported DeepAction legacy split: {row['split']}")
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ContractError(f"audit source must be a non-empty non-symlink regular file: {source_relative_path}")
        source_hash = row["original_source_sha256"]
        if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
            raise ContractError(f"invalid original_source_sha256 for {source_relative_path}")
        if source_relative_path in seen_paths:
            raise ContractError(f"duplicate audit original source path: {source_relative_path}")
        seen_paths.add(source_relative_path)
        normalized_partition = _SPLIT_NORMALIZATION[row["split"]]
        entry = {
            "audit": row, "path": path, "source_relative_path": source_relative_path,
            "label": label, "prompt_id": prompt_id, "partition": normalized_partition,
            "source_sha256": source_hash, "source_bytes": path.stat().st_size,
            "group_id": f"dataset:prompt:{prompt_id}",
        }
        entries.append(entry)
        by_hash[source_hash].append(entry)
        prompt_partitions[prompt_id].add(normalized_partition)
        source_bytes += entry["source_bytes"]
    labels = {entry["label"] for entry in entries}
    if labels != set(DEEPACTION_LABELS):
        raise ContractError(f"DeepAction labels must be exactly {list(DEEPACTION_LABELS)}, got {sorted(labels)}")
    bad_prompts = {prompt: sorted(values) for prompt, values in prompt_partitions.items() if len(values) != 1}
    if bad_prompts:
        raise ContractError(f"prompt groups span partitions: {bad_prompts}")
    inventory_count: int | None = None
    if check_inventory:
        inventory: set[str] = set()
        for label in DEEPACTION_LABELS:
            label_root = root / label
            if not label_root.is_dir():
                raise ContractError(f"DeepAction label directory missing: {label_root}")
            for path in label_root.rglob("*"):
                if path.is_symlink():
                    raise ContractError(f"symlink not allowed in DeepAction source tree: {path}")
                if path.is_file() and path.suffix.lower() == ".mp4":
                    inventory.add(portable_relative_path(path.relative_to(root).as_posix(), "source_relative_path"))
        if inventory != seen_paths:
            raise ContractError(f"audit/inventory mismatch: unlisted={len(inventory - seen_paths)}, missing={len(seen_paths - inventory)}")
        inventory_count = len(inventory)
    duplicates = {digest: members for digest, members in by_hash.items() if len(members) > 1}
    stats = {
        "audit_rows": len(entries), "unique_prompts": len(prompt_partitions), "source_bytes": source_bytes,
        "inventory_count": inventory_count,
        "split_counts": dict(Counter(entry["partition"] for entry in entries)),
        "label_counts": dict(Counter(entry["label"] for entry in entries)),
        "duplicate_groups": len(duplicates),
        "duplicate_rows": sum(len(members) for members in duplicates.values()),
    }
    return entries, duplicates, stats


def _partition_duplicate_rows(entries: list[dict[str, Any]], duplicates: dict[str, list[dict[str, Any]]], duplicate_policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_duplicate_policy(duplicate_policy)
    if duplicates and duplicate_policy == "error":
        digest, members = next(iter(duplicates.items()))
        paths = [entry["source_relative_path"] for entry in members]
        raise ContractError(f"duplicate original content/canonical ID collision ({len(duplicates)} hashes); example {digest}: {paths}. "
                            "Use the explicit DeepAction-only --duplicate-policy quarantine-all to exclude every member.")
    excluded_hashes = set(duplicates) if duplicate_policy == "quarantine-all" else set()
    retained = [entry for entry in entries if entry["source_sha256"] not in excluded_hashes]
    excluded = [entry for entry in entries if entry["source_sha256"] in excluded_hashes]
    # This is intentionally an accounting invariant, not a keep-one deduplication.
    if len(retained) + len(excluded) != len(entries):
        raise ContractError("internal DeepAction inventory accounting failure")
    return retained, excluded


def inspect_deepaction_audit(source_root: Path, audit_path: Path, *, check_inventory: bool = True,
                             duplicate_policy: str = "error") -> dict[str, Any]:
    """Cheap audit/path/stat validation; intentionally does not read/hash video contents."""
    # Preserve strict collision reporting even when an unrelated inventory issue is
    # also present: collision detection needs no directory walk or content hashing.
    if duplicate_policy == "error":
        entries, duplicates, _ = _collect_deepaction_rows(source_root, audit_path, check_inventory=False)
        _partition_duplicate_rows(entries, duplicates, duplicate_policy)
    entries, duplicates, stats = _collect_deepaction_rows(source_root, audit_path, check_inventory=check_inventory)
    retained, excluded = _partition_duplicate_rows(entries, duplicates, duplicate_policy)
    stats.update({
        "duplicate_policy": duplicate_policy,
        "retained_rows": len(retained), "excluded_rows": len(excluded),
        "retained_split_counts": dict(Counter(entry["partition"] for entry in retained)),
        "quarantine_group_ids": [canonical_video_id(digest) for digest in sorted(duplicates)],
    })
    return stats


def _records(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "video_id": canonical_video_id(entry["source_sha256"]), "source_sha256": entry["source_sha256"],
        "source_relative_path": entry["source_relative_path"], "source_bytes": entry["source_bytes"],
        "label": entry["label"], "group_id": entry["group_id"], "partition": entry["partition"],
        "prompt_id": entry["prompt_id"], "variant": Path(entry["source_relative_path"]).name,
        "legacy_video_id": entry["audit"]["video_id"],
    } for entry in entries]


def exclusion_report_rows(excluded: list[dict[str, Any]], duplicates: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    """Create a complete, deterministic report for every quarantined occurrence."""
    report: list[dict[str, str]] = []
    for entry in excluded:
        digest = entry["source_sha256"]
        report.append({
            "canonical_video_id": canonical_video_id(digest), "legacy_video_id": entry["audit"]["video_id"],
            "original_source_sha256": digest, "source_relative_path": entry["source_relative_path"],
            "label": entry["label"], "prompt_id": entry["prompt_id"], "group_id": entry["group_id"],
            "original_partition": entry["audit"]["split"], "partition": entry["partition"],
            "exclusion_reason": "duplicate_original_source_sha256_quarantine_all",
            "duplicate_group_id": canonical_video_id(digest), "duplicate_group_count": str(len(duplicates[digest])),
        })
    return sorted(report, key=lambda row: row["source_relative_path"])


def exclusion_report_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EXCLUSION_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_deepaction_manifest_inputs(source_root: Path, audit_path: Path, *, verify_source_hashes: bool,
                                      duplicate_policy: str = "error") -> dict[str, Any]:
    """Return validated retained records plus a full duplicate-quarantine report.

    Hashing, when requested, covers retained and excluded originals.  Therefore a
    stale audit cannot cause changed content to be quarantined or published.
    """
    entries, duplicates, stats = _collect_deepaction_rows(source_root, audit_path, check_inventory=True)
    retained, excluded = _partition_duplicate_rows(entries, duplicates, duplicate_policy)
    if verify_source_hashes:
        actual_hashes: dict[str, str] = {}
        for entry in entries:  # every audited original, including excluded sources
            actual = sha256_file(entry["path"])
            actual_hashes[entry["source_relative_path"]] = actual
            if actual != entry["source_sha256"]:
                raise ContractError(f"original source hash mismatch: {entry['source_relative_path']}")
        actual_groups: dict[str, list[str]] = defaultdict(list)
        for entry in entries:
            actual_groups[actual_hashes[entry["source_relative_path"]]].append(entry["source_relative_path"])
        actual_duplicates = {digest: sorted(paths) for digest, paths in actual_groups.items() if len(paths) > 1}
        audited_duplicates = {digest: sorted(member["source_relative_path"] for member in members) for digest, members in duplicates.items()}
        if actual_duplicates != audited_duplicates:
            raise ContractError("actual original hashes reveal duplicate groups not represented by the audit; refusing policy assumption")
    records = _records(retained)
    # Keep generic strict source SHA/ID uniqueness in force before publication.
    normalized_records = validate_source_records(records)
    report_rows = exclusion_report_rows(excluded, duplicates)
    report_bytes = exclusion_report_bytes(report_rows)
    if len(report_rows) != len(excluded):
        raise ContractError("DeepAction exclusion report does not account for every excluded source")
    stats.update({
        "duplicate_policy": duplicate_policy, "retained_rows": len(normalized_records),
        "excluded_rows": len(report_rows), "retained_split_counts": dict(Counter(row["partition"] for row in normalized_records)),
        "retained_group_count": len({row["group_id"] for row in normalized_records}),
        "verified_source_hashes": verify_source_hashes,
    })
    return {
        "records": normalized_records, "exclusion_rows": report_rows, "exclusion_report_bytes": report_bytes,
        "exclusion_report_sha256": hashlib.sha256(report_bytes).hexdigest(), "stats": stats,
    }


def build_deepaction_records(source_root: Path, audit_path: Path, *, verify_source_hashes: bool,
                              duplicate_policy: str = "error") -> list[dict[str, Any]]:
    """Compatibility helper returning only retained source-manifest records."""
    return build_deepaction_manifest_inputs(
        source_root, audit_path, verify_source_hashes=verify_source_hashes, duplicate_policy=duplicate_policy,
    )["records"]
