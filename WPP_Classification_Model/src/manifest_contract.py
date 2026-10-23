"""Versioned source-manifest and extraction-index contracts.

These contracts deliberately describe originals and future extraction outputs separately.
They do not inspect encoded media or artifact contents.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

SOURCE_CONTRACT_NAME = "approved_upstream_source_manifest"
EXTRACTION_CONTRACT_NAME = "extraction_index"
CONTRACT_VERSION = "1.0"
SOURCE_RECORDS_FILE = "source_manifest.csv"
SOURCE_METADATA_FILE = "source_manifest.metadata.json"
INDEX_RECORDS_FILE = "extraction_index.csv"
INDEX_METADATA_FILE = "extraction_index.metadata.json"

_SOURCE_IDENTITY = "video_id is sha256:<full original source SHA-256>"
_SOURCE_ORDERING = {
    "records": "lexicographic source_relative_path",
    "paths": "safe POSIX paths relative to source_root_ref",
}
_SOURCE_FUTURE_GUIDANCE = "Use original native geometry and timing; this manifest defines no encoding profile."

SOURCE_FIELDS = (
    "video_id", "source_sha256", "source_relative_path", "source_bytes", "label",
    "group_id", "partition", "prompt_id", "variant", "legacy_video_id",
)
INDEX_FIELDS = (
    "video_id", "status", "encoded_relative_path", "encoded_sha256",
    "artifact_relative_path", "producer_schema_version", "row_count", "frame_count",
    "failure_code", "failure_message",
)
_PARTITIONS = frozenset(("train", "validation", "test"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when contract input is incomplete, unsafe, or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _text(value: Any, field: str, *, optional: bool = False) -> str:
    if value is None and optional:
        return ""
    if not isinstance(value, str):
        raise ContractError(f"{field} must be a string")
    value = value.strip()
    if not value and not optional:
        raise ContractError(f"{field} cannot be empty")
    return value


def _positive_or_zero(value: Any, field: str, *, positive: bool = False) -> str:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an integer") from exc
    if str(value).strip() != str(number) or number < (1 if positive else 0):
        raise ContractError(f"{field} must be {'positive' if positive else 'nonnegative'}")
    return str(number)


def portable_relative_path(value: Any, field: str = "relative_path") -> str:
    value = _text(value, field)
    if "\\" in value:
        raise ContractError(f"{field} must use portable POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError(f"{field} must be a safe, non-empty relative path")
    return path.as_posix()


def _optional_path(value: Any, field: str) -> str:
    if value is None or value == "":
        return ""
    return portable_relative_path(value, field)


def canonical_video_id(source_sha256: str) -> str:
    return "sha256:" + _sha(source_sha256, "source_sha256")


def validate_source_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Normalize and validate records without making dataset-specific assumptions."""
    normalized: list[dict[str, str]] = []
    seen_ids: dict[str, str] = {}
    seen_paths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    allowed = set(SOURCE_FIELDS)
    for position, raw in enumerate(records, 1):
        if not isinstance(raw, Mapping):
            raise ContractError(f"source record {position} must be a mapping")
        extra = set(raw) - allowed
        missing = set(SOURCE_FIELDS[:5]) - set(raw)
        if extra or missing:
            raise ContractError(f"source record {position} has unsupported or missing fields: "
                                f"extra={sorted(extra)}, missing={sorted(missing)}")
        digest = _sha(raw["source_sha256"], "source_sha256")
        video_id = _text(raw["video_id"], "video_id")
        if video_id != canonical_video_id(digest):
            raise ContractError("video_id must be exactly sha256:<source_sha256>")
        path = portable_relative_path(raw["source_relative_path"], "source_relative_path")
        row = {
            "video_id": video_id, "source_sha256": digest, "source_relative_path": path,
            "source_bytes": _positive_or_zero(raw["source_bytes"], "source_bytes"),
            "label": _text(raw["label"], "label"),
            "group_id": _text(raw.get("group_id"), "group_id", optional=True),
            "partition": _text(raw.get("partition"), "partition", optional=True),
            "prompt_id": _text(raw.get("prompt_id"), "prompt_id", optional=True),
            "variant": _text(raw.get("variant"), "variant", optional=True),
            "legacy_video_id": _text(raw.get("legacy_video_id"), "legacy_video_id", optional=True),
        }
        if row["partition"] and row["partition"] not in _PARTITIONS:
            raise ContractError("partition must be train, validation, or test when supplied")
        if video_id in seen_ids:
            raise ContractError(f"duplicate/canonical video_id collision {video_id}: "
                                f"{seen_ids[video_id]} and {path}")
        if digest in seen_hashes:
            raise ContractError(f"duplicate original content/canonical ID collision {digest}: "
                                f"{seen_hashes[digest]} and {path}")
        if path in seen_paths:
            raise ContractError(f"duplicate source_relative_path: {path}")
        seen_ids[video_id] = path
        seen_hashes[digest] = path
        seen_paths.add(path)
        normalized.append(row)
    if not normalized:
        raise ContractError("source manifest cannot be empty")
    return sorted(normalized, key=lambda row: row["source_relative_path"])


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _publish_directory(destination: Path, writer: Callable[[Path], None]) -> None:
    """Publish under an exclusive, per-destination cooperative-builder lock.

    The lock prevents two contract writers using this function from racing into an
    overwrite.  A stale lock fails closed and must be removed by an operator; it
    is never stolen.  This is not a lock against uncooperative external writers.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.publish.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ContractError(f"publication lock already exists: {lock}") from exc
    staging: Path | None = None
    try:
        if destination.exists():
            raise ContractError(f"refusing to overwrite existing output: {destination}")
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        writer(staging)
        # Cooperating writers cannot create destination while this lock is held;
        # retain this check for an independently-created destination before rename.
        if destination.exists():
            raise ContractError(f"refusing to overwrite existing output: {destination}")
        os.replace(staging, destination)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock.rmdir()


def write_source_manifest(output_dir: Path, records: Iterable[Mapping[str, Any]], *, dataset_id: str,
                          source_root_ref: str, provenance: Mapping[str, Any] | None = None,
                          bundle_files: Mapping[str, bytes] | None = None) -> dict[str, Any]:
    """Validate then atomically publish one source manifest directory.

    ``bundle_files`` is for small, immutable companion reports.  It is deliberately
    byte-oriented: dataset adapters own their report schema, while the generic
    contract keeps its strict source-record uniqueness rules unchanged.
    """
    rows = validate_source_records(records)
    dataset_id = _text(dataset_id, "dataset_id")
    source_root_ref = portable_relative_path(source_root_ref, "source_root_ref")
    provenance = dict(provenance or {})
    validated_bundle_files: dict[str, bytes] = {}
    for name, content in dict(bundle_files or {}).items():
        safe_name = portable_relative_path(name, "bundle file path")
        if safe_name in (SOURCE_RECORDS_FILE, SOURCE_METADATA_FILE):
            raise ContractError(f"bundle file path conflicts with contract file: {safe_name}")
        if not isinstance(content, bytes):
            raise ContractError(f"bundle file content must be bytes: {safe_name}")
        validated_bundle_files[safe_name] = content
    def write(staging: Path) -> None:
        csv_hash = _write_csv(staging / SOURCE_RECORDS_FILE, SOURCE_FIELDS, rows)
        for name, content in validated_bundle_files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        metadata = {
            "contract_name": SOURCE_CONTRACT_NAME, "contract_version": CONTRACT_VERSION,
            "dataset_id": dataset_id, "source_root_ref": source_root_ref,
            "records_file": SOURCE_RECORDS_FILE, "records_sha256": csv_hash,
            "record_count": len(rows),
            "identity": _SOURCE_IDENTITY,
            "ordering": _SOURCE_ORDERING,
            "future_processing_guidance": _SOURCE_FUTURE_GUIDANCE,
            "provenance": provenance,
        }
        (staging / SOURCE_METADATA_FILE).write_bytes(_json_bytes(metadata))
    _publish_directory(Path(output_dir), write)
    return load_source_manifest(Path(output_dir))


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ContractError(f"{path.name} header must be exactly {list(fields)}")
            return list(reader)
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("metadata must be a JSON object")
    return value


def load_source_manifest(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    metadata = _read_metadata(directory / SOURCE_METADATA_FILE)
    if metadata.get("contract_name") != SOURCE_CONTRACT_NAME or metadata.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("unsupported source-manifest contract name or version")
    raw_rows = _read_csv(directory / SOURCE_RECORDS_FILE, SOURCE_FIELDS)
    rows = validate_source_records(raw_rows)
    if raw_rows != rows:
        raise ContractError("source manifest records are not in required deterministic path order")
    if metadata.get("records_file") != SOURCE_RECORDS_FILE or metadata.get("record_count") != len(rows):
        raise ContractError("source metadata records file or count does not match")
    if metadata.get("records_sha256") != sha256_file(directory / SOURCE_RECORDS_FILE):
        raise ContractError("source manifest CSV SHA-256 does not match metadata")
    _text(metadata.get("dataset_id"), "dataset_id")
    portable_relative_path(metadata.get("source_root_ref"), "source_root_ref")
    if metadata.get("identity") != _SOURCE_IDENTITY:
        raise ContractError("source metadata identity declaration is missing or unsupported")
    if metadata.get("ordering") != _SOURCE_ORDERING:
        raise ContractError("source metadata ordering declaration is missing or unsupported")
    if metadata.get("future_processing_guidance") != _SOURCE_FUTURE_GUIDANCE:
        raise ContractError("source metadata future processing guidance is missing or unsupported")
    if not isinstance(metadata.get("provenance"), dict):
        raise ContractError("source metadata provenance must be an object")
    return {"metadata": metadata, "records": rows}


def _non_symlink_source_path(source_root: Path, relative: str, *, field: str) -> Path:
    """Reject symlink components before resolving, then retain a resolved-root check."""
    raw_root = Path(source_root).absolute()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ContractError(f"{field} source root must be a non-symlink directory: {raw_root}")
    candidate = raw_root
    for component in PurePosixPath(relative).parts:
        candidate /= component
        if candidate.is_symlink():
            raise ContractError(f"{field} contains a symlink component: {candidate}")
    root = raw_root.resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{field} escapes source root: {relative}") from exc
    return resolved


def records_from_discovery(source_root: Path, discovery: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """General builder helper; a caller supplies dataset-specific label/group discovery."""
    records: list[dict[str, Any]] = []
    for item in discovery:
        relative = portable_relative_path(item.get("source_relative_path"), "source_relative_path")
        path = _non_symlink_source_path(source_root, relative, field="discovered path")
        if not path.is_file():
            raise ContractError(f"discovered source must be a non-symlink file: {relative}")
        digest = sha256_file(path)
        records.append({**item, "source_relative_path": relative, "source_sha256": digest,
                        "video_id": canonical_video_id(digest), "source_bytes": path.stat().st_size})
    return records


def validate_extraction_index_records(records: Iterable[Mapping[str, Any]], source_video_ids: Iterable[str]) -> list[dict[str, str]]:
    source_ids = set(source_video_ids)
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed = set(INDEX_FIELDS)
    for position, raw in enumerate(records, 1):
        if not isinstance(raw, Mapping):
            raise ContractError(f"index record {position} must be a mapping")
        extra, missing = set(raw) - allowed, set(INDEX_FIELDS[:2]) - set(raw)
        if extra or missing:
            raise ContractError(f"index record {position} has unsupported or missing fields: extra={sorted(extra)}, missing={sorted(missing)}")
        video_id = _text(raw["video_id"], "video_id")
        if video_id not in source_ids:
            raise ContractError(f"extraction index references unknown source video_id: {video_id}")
        if video_id in seen:
            raise ContractError(f"duplicate extraction-index video_id: {video_id}")
        status = _text(raw["status"], "status")
        if status not in ("success", "failed"):
            raise ContractError("extraction status must be success or failed")
        row = {field: _text(raw.get(field), field, optional=True) for field in INDEX_FIELDS}
        row["video_id"], row["status"] = video_id, status
        schema = row["producer_schema_version"]
        if schema != "2.0":
            raise ContractError("producer_schema_version must be exactly 2.0")
        if status == "success":
            row["encoded_relative_path"] = portable_relative_path(row["encoded_relative_path"], "encoded_relative_path")
            row["artifact_relative_path"] = portable_relative_path(row["artifact_relative_path"], "artifact_relative_path")
            row["encoded_sha256"] = _sha(row["encoded_sha256"], "encoded_sha256")
            row["row_count"] = _positive_or_zero(row["row_count"], "row_count", positive=True)
            row["frame_count"] = _positive_or_zero(row["frame_count"], "frame_count", positive=True)
            if row["failure_code"] or row["failure_message"]:
                raise ContractError("successful extraction records cannot contain failure fields")
        else:
            if not row["failure_code"] or not row["failure_message"]:
                raise ContractError("failed extraction records require failure_code and failure_message")
            if any(row[field] for field in ("encoded_relative_path", "encoded_sha256", "artifact_relative_path", "row_count", "frame_count")):
                raise ContractError("failed extraction records must not fabricate output fields")
        seen.add(video_id); normalized.append(row)
    if seen != source_ids:
        missing = sorted(source_ids - seen)
        raise ContractError(f"extraction index must cover every source manifest ID; missing {len(missing)}")
    return sorted(normalized, key=lambda row: row["video_id"])


def write_extraction_index(output_dir: Path, records: Iterable[Mapping[str, Any]], *, source_manifest_dir: Path,
                           encoded_root_ref: str, artifact_root_ref: str, processing_profile: Mapping[str, Any],
                           extractor_identity: Mapping[str, Any],
                           additional_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and atomically publish an index. No media/artifact content is inspected."""
    source = load_source_manifest(source_manifest_dir)
    rows = validate_extraction_index_records(records, (row["video_id"] for row in source["records"]))
    for name, value in (("processing_profile", processing_profile), ("extractor_identity", extractor_identity)):
        if not isinstance(value, Mapping) or not value:
            raise ContractError(f"{name} must be a non-empty metadata object")
    extra = dict(additional_metadata or {})
    reserved = {"contract_name", "contract_version", "records_file", "records_sha256", "record_count",
                "input_source_manifest_records_sha256", "input_source_manifest_contract_version",
                "encoded_root_ref", "artifact_root_ref", "processing_profile", "extractor_identity",
                "validation_scope"}
    if set(extra) & reserved:
        raise ContractError(f"additional extraction-index metadata conflicts with contract fields: {sorted(set(extra) & reserved)}")
    if any(not isinstance(name, str) or not name for name in extra):
        raise ContractError("additional extraction-index metadata keys must be non-empty strings")
    def write(staging: Path) -> None:
        csv_hash = _write_csv(staging / INDEX_RECORDS_FILE, INDEX_FIELDS, rows)
        metadata = {
            "contract_name": EXTRACTION_CONTRACT_NAME, "contract_version": CONTRACT_VERSION,
            "records_file": INDEX_RECORDS_FILE, "records_sha256": csv_hash, "record_count": len(rows),
            "input_source_manifest_records_sha256": source["metadata"]["records_sha256"],
            "input_source_manifest_contract_version": CONTRACT_VERSION,
            "encoded_root_ref": portable_relative_path(encoded_root_ref, "encoded_root_ref"),
            "artifact_root_ref": portable_relative_path(artifact_root_ref, "artifact_root_ref"),
            "processing_profile": dict(processing_profile), "extractor_identity": dict(extractor_identity),
            "validation_scope": "schema, paths, IDs, metadata, and declared checksums only; artifact-content verification is future work.",
            **extra,
        }
        (staging / INDEX_METADATA_FILE).write_bytes(_json_bytes(metadata))
    _publish_directory(Path(output_dir), write)
    return load_extraction_index(Path(output_dir), source_manifest_dir)


def load_extraction_index(directory: Path, source_manifest_dir: Path) -> dict[str, Any]:
    directory = Path(directory)
    source = load_source_manifest(source_manifest_dir)
    metadata = _read_metadata(directory / INDEX_METADATA_FILE)
    if metadata.get("contract_name") != EXTRACTION_CONTRACT_NAME or metadata.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("unsupported extraction-index contract name or version")
    raw_rows = _read_csv(directory / INDEX_RECORDS_FILE, INDEX_FIELDS)
    rows = validate_extraction_index_records(raw_rows, (row["video_id"] for row in source["records"]))
    if raw_rows != rows:
        raise ContractError("extraction-index records are not in required deterministic video_id order")
    if metadata.get("records_file") != INDEX_RECORDS_FILE or metadata.get("record_count") != len(rows):
        raise ContractError("extraction-index metadata records file or count does not match")
    if metadata.get("records_sha256") != sha256_file(directory / INDEX_RECORDS_FILE):
        raise ContractError("extraction-index CSV SHA-256 does not match metadata")
    if metadata.get("input_source_manifest_records_sha256") != source["metadata"]["records_sha256"]:
        raise ContractError("extraction index does not reference this exact source manifest")
    portable_relative_path(metadata.get("encoded_root_ref"), "encoded_root_ref")
    portable_relative_path(metadata.get("artifact_root_ref"), "artifact_root_ref")
    if (not isinstance(metadata.get("processing_profile"), dict) or not metadata["processing_profile"]
            or not isinstance(metadata.get("extractor_identity"), dict) or not metadata["extractor_identity"]):
        raise ContractError("extraction-index processing profile and extractor identity must be non-empty objects")
    return {"metadata": metadata, "records": rows}
