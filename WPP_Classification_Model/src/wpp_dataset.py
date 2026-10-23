"""Bounded, video-aware WPP classification data bundle.

This module intentionally has no model, training, normalization, encoding, or split
logic.  It packs the already verified native producer ``row_bytes`` arrays as one
variable-row video example at a time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest_contract import (
    ContractError, INDEX_METADATA_FILE, INDEX_RECORDS_FILE, SOURCE_METADATA_FILE,
    SOURCE_RECORDS_FILE, _publish_directory, canonical_video_id, load_extraction_index,
    load_source_manifest, portable_relative_path, sha256_file,
)
from .native_wpp_workflow import WorkflowError, verify_native_build

TIME_STEPS = 3000
PARTITIONS = ("train", "validation", "test")
DATA_COLUMNS = tuple(f"size{i}" for i in range(TIME_STEPS))
LABEL_FIELDS = (
    "video_id", "source_sha256", "source_relative_path", "label", "group_id", "partition",
    "stacked_row_start", "stacked_row_count", "R_v", "original_frame_count", "used_frame_count",
    "time_truncated", "artifact_relpath",
)
BUNDLE_SCHEMA = "bounded_wpp_classification_dataset"
BUNDLE_VERSION = "1.0"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class WPPDatasetError(ValueError):
    """The immutable WPP bundle or a native input failed its contract."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _safe_existing_dir(root: Path, relative: str) -> Path:
    """Resolve a contract-relative existing directory without link traversal."""
    try:
        relative = portable_relative_path(relative, "artifact_relative_path")
    except ContractError as exc:
        raise WPPDatasetError(str(exc)) from exc
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise WPPDatasetError(f"artifact root must be a non-symlink directory: {root}")
    candidate = root
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise WPPDatasetError(f"artifact path contains a symlink: {candidate}")
    if not candidate.is_dir():
        raise WPPDatasetError(f"artifact directory is missing: {relative}")
    return candidate


def _safe_output(destination: Path, source_manifest: Path, index: Path) -> Path:
    destination = Path(destination).expanduser().absolute()
    if destination.name in ("", ".", ".."):
        raise WPPDatasetError("output directory is an unsafe filesystem root")
    ancestor = destination.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise WPPDatasetError(f"output directory has a symlinked parent: {ancestor}")
    if destination.exists():
        raise WPPDatasetError(f"refusing to overwrite existing output: {destination}")
    # Derived bundles should not be accidentally made part of the checkout or an
    # input contract/build tree.  The publication helper creates a fresh staging dir.
    checkout = Path(__file__).resolve().parents[2]
    for protected in (checkout, Path(source_manifest).absolute(), Path(index).absolute()):
        try:
            destination.resolve(strict=False).relative_to(protected.resolve())
        except ValueError:
            continue
        raise WPPDatasetError(f"output directory must not be inside protected path: {protected}")
    return destination


def _reject_symlink_components(path: Path, *, directory: bool | None = None) -> Path:
    """Return an absolute existing path only when no supplied component is a link."""
    absolute = Path(path).expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise WPPDatasetError(f"path contains a symlink component: {current}")
    if directory is True and not absolute.is_dir():
        raise WPPDatasetError(f"expected directory: {absolute}")
    if directory is False and not absolute.is_file():
        raise WPPDatasetError(f"expected regular file: {absolute}")
    return absolute


def _bundle_file(root: Path, name: str) -> Path:
    if "/" in name or "\\" in name:
        raise WPPDatasetError("bundle file name must not contain a path separator")
    return _reject_symlink_components(root / name, directory=False)


def _strict_contract_inputs(source_manifest: str | Path, index: str | Path) -> tuple[Path, Path]:
    """Preflight every contract path before loading or publishing derived output."""
    source_dir = _reject_symlink_components(Path(source_manifest), directory=True)
    index_dir = _reject_symlink_components(Path(index), directory=True)
    for root, names in ((source_dir, (SOURCE_RECORDS_FILE, SOURCE_METADATA_FILE)),
                        (index_dir, (INDEX_RECORDS_FILE, INDEX_METADATA_FILE))):
        for name in names:
            _bundle_file(root, name)
    return source_dir, index_dir


def _load_row_bytes(artifact_dir: Path, index_row: Mapping[str, str]) -> tuple[np.ndarray, int, int]:
    """Read only producer raw bytes and validate exact native dimensions."""
    try:
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "2.0":
            raise WPPDatasetError("artifact producer schema must be 2.0")
        rows, frames = int(metadata["rows"]), int(metadata["frames"])
        with np.load(artifact_dir / "features.npz", allow_pickle=False) as archive:
            if "row_bytes" not in archive.files or "display_index" not in archive.files:
                raise WPPDatasetError("artifact features.npz lacks row_bytes or display_index")
            raw = archive["row_bytes"]
            display_index = archive["display_index"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, WPPDatasetError):
            raise
        raise WPPDatasetError(f"cannot read native row_bytes artifact: {artifact_dir}") from exc
    if rows <= 0 or frames <= 0 or raw.shape != (rows, frames):
        raise WPPDatasetError("artifact row_bytes dimensions do not match metadata")
    if display_index.shape != (frames,) or display_index.dtype.kind not in "iu" or not np.array_equal(display_index, np.arange(frames, dtype=display_index.dtype)):
        raise WPPDatasetError("artifact frame axis is not contiguous producer display order")
    if rows != int(index_row["row_count"]) or frames != int(index_row["frame_count"]):
        raise WPPDatasetError("artifact dimensions do not match extraction index")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b" or not np.isfinite(raw).all() or (raw < 0).any():
        raise WPPDatasetError("row_bytes must be finite nonnegative integer raw byte counts")
    # CSV representation retains exact decimal integers; float32 loader values must
    # be representable as finite floats, even though they intentionally are not normalized.
    if raw.size and int(raw.max()) > int(np.finfo(np.float32).max):
        raise WPPDatasetError("row_bytes exceeds finite float32 loader range")
    return raw, rows, frames


def _csv_hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.csv"))}


def pack_wpp_dataset(source_manifest: str | Path, index: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Atomically pack a complete verified native build into fixed-time CSV splits.

    ``verify_native_build`` is deliberately called before any artifact is opened;
    the lightweight extraction-index loader alone is never treated as content proof.
    """
    # Validate every parent/leaf path before even creating an output lock/staging
    # directory. A successful publication must never be followed by a routine
    # strict-loader failure caused by unsafe parent contract paths.
    source_dir, index_dir = _strict_contract_inputs(source_manifest, index)
    destination = _safe_output(Path(output_dir), source_dir, index_dir)
    try:
        source = load_source_manifest(source_dir)
        extraction = load_extraction_index(index_dir, source_dir)
        native = verify_native_build(index_dir, source_dir)
    except (ContractError, WorkflowError, OSError, ValueError) as exc:
        raise WPPDatasetError(f"native input verification failed: {exc}") from exc
    failed = [row["video_id"] for row in extraction["records"] if row["status"] != "success"]
    if failed:
        raise WPPDatasetError(f"extraction index contains failed records ({len(failed)}); refusing partial dataset")
    index_by_id = {row["video_id"]: row for row in extraction["records"]}
    group_partition: dict[str, str] = {}
    for item in source["records"]:
        if item["partition"] not in PARTITIONS:
            raise WPPDatasetError(f"source record lacks approved partition: {item['video_id']}")
        group = item["group_id"]
        if group and group in group_partition and group_partition[group] != item["partition"]:
            raise WPPDatasetError(f"nonempty group_id spans partitions: {group}")
        if group:
            group_partition[group] = item["partition"]
    train_labels = sorted({row["label"] for row in source["records"] if row["partition"] == "train"})
    if not train_labels:
        raise WPPDatasetError("at least one training example is required for class mapping")
    class_mapping = {label: position for position, label in enumerate(train_labels)}
    for item in source["records"]:
        if item["partition"] != "train" and item["label"] not in class_mapping:
            raise WPPDatasetError(f"{item['partition']} label absent from training labels: {item['label']!r}")

    published_metadata: dict[str, Any] | None = None

    def write(staging: Path) -> None:
        nonlocal published_metadata
        writers: dict[str, tuple[Any, csv.DictWriter, csv.writer, int, int]] = {}
        try:
            for partition in PARTITIONS:
                data_handle = (staging / f"{partition}-data.csv").open("w", newline="", encoding="utf-8")
                labels_handle = (staging / f"{partition}-label-manifest.csv").open("w", newline="", encoding="utf-8")
                data_writer = csv.writer(data_handle, lineterminator="\n")
                label_writer = csv.DictWriter(labels_handle, fieldnames=LABEL_FIELDS, extrasaction="raise", lineterminator="\n")
                data_writer.writerow(DATA_COLUMNS); label_writer.writeheader()
                writers[partition] = (data_handle, labels_handle, data_writer, label_writer, 0, 0)
            # Source order is contract order; do not sort by split or label.
            artifact_root = index_dir.parent / extraction["metadata"]["artifact_root_ref"]
            for source_row in source["records"]:
                partition = source_row["partition"]
                index_row = index_by_id[source_row["video_id"]]
                artifact = _safe_existing_dir(artifact_root, index_row["artifact_relative_path"])
                raw, row_count, frame_count = _load_row_bytes(artifact, index_row)
                data_handle, label_handle, data_writer, label_writer, offset, video_count = writers[partition]
                used = min(frame_count, TIME_STEPS)
                # Do not resize/reorder producer axes: axis 0 is WPP top-to-bottom,
                # axis 1 is presentation/display order.  Only fixed right padding/truncation.
                padded = np.zeros((row_count, TIME_STEPS), dtype=raw.dtype)
                padded[:, :used] = raw[:, :used]
                for row in padded:
                    data_writer.writerow([int(value) for value in row])
                label_writer.writerow({
                    "video_id": source_row["video_id"], "source_sha256": source_row["source_sha256"],
                    "source_relative_path": source_row["source_relative_path"], "label": source_row["label"],
                    "group_id": source_row["group_id"], "partition": partition,
                    "stacked_row_start": offset, "stacked_row_count": row_count, "R_v": row_count,
                    "original_frame_count": frame_count, "used_frame_count": used,
                    "time_truncated": "true" if frame_count > TIME_STEPS else "false",
                    "artifact_relpath": index_row["artifact_relative_path"],
                })
                writers[partition] = (data_handle, label_handle, data_writer, label_writer,
                                      offset + row_count, video_count + 1)
        finally:
            for data_handle, label_handle, *_ in writers.values():
                data_handle.close(); label_handle.close()
        checksums = _csv_hashes(staging)
        partitions = {name: {"video_count": values[5], "stacked_row_count": values[4],
                             "data_file": f"{name}-data.csv", "label_manifest_file": f"{name}-label-manifest.csv"}
                      for name, values in writers.items()}
        metadata = {
            "schema_name": BUNDLE_SCHEMA, "schema_version": BUNDLE_VERSION,
            "time_steps": TIME_STEPS,
            "feature": {"name": "row_bytes", "dtype": "raw_nonnegative_integer_byte_count_csv; loader=float32",
                        "axis_semantics": "per-video R_v WPP rows in top-to-bottom order x first 3000 presentation frames",
                        "padding": "right zero padding only when original_frame_count < 3000; zeros are valid observed byte counts"},
            "source_manifest": {"records_sha256": source["metadata"]["records_sha256"],
                                "metadata_sha256": sha256_file(source_dir / "source_manifest.metadata.json")},
            "extraction_index": {"records_sha256": extraction["metadata"]["records_sha256"],
                                 "metadata_sha256": sha256_file(index_dir / "extraction_index.metadata.json"),
                                 "native_ledger_sha256": native["ledger_sha256"]},
            "class_mapping": class_mapping, "partition_counts": partitions,
            "output_checksums": checksums,
            "label_manifest": {"row_unit": "one video", "stacked_row_count_field": "number of contiguous numeric rows; no redundant alias is emitted"},
            "resource_behavior": "packing streams one video and writes CSV rows; loading indexes byte offsets and reads only requested video blocks",
        }
        (staging / "datasetmetadata.json").write_bytes(_json_bytes(metadata))
        # Reuse the strict loader's global label/provenance validation while still
        # in staging. This makes malformed generated metadata fail before publish.
        published_metadata = WPPDatasetBundle(staging, source_manifest=source_dir, index=index_dir,
                                              verify_checksums=True).metadata
    try:
        _publish_directory(destination, write)
    except (ContractError, OSError, ValueError) as exc:
        raise WPPDatasetError(str(exc)) from exc
    if published_metadata is None:  # defensive: writer always assigns before atomic rename
        raise WPPDatasetError("packer did not validate staged metadata")
    return published_metadata


def _integer(text: str, field: str, *, positive: bool = False) -> int:
    if not isinstance(text, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", text):
        raise WPPDatasetError(f"{field} must be a canonical nonnegative integer")
    value = int(text)
    if positive and value <= 0:
        raise WPPDatasetError(f"{field} must be positive")
    return value


def _load_metadata(directory: Path, verify_checksums: bool) -> dict[str, Any]:
    directory = _reject_symlink_components(directory, directory=True)
    try:
        metadata = json.loads(_bundle_file(directory, "datasetmetadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WPPDatasetError("cannot read datasetmetadata.json") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_name") != BUNDLE_SCHEMA or metadata.get("schema_version") != BUNDLE_VERSION:
        raise WPPDatasetError("unsupported WPP dataset metadata schema")
    if metadata.get("time_steps") != TIME_STEPS or not isinstance(metadata.get("feature"), dict):
        raise WPPDatasetError("invalid fixed-time feature metadata")
    checksums = metadata.get("output_checksums")
    expected = {f"{split}-{kind}.csv" for split in PARTITIONS for kind in ("data", "label-manifest")}
    if not isinstance(checksums, dict) or set(checksums) != expected:
        raise WPPDatasetError("metadata must checksum exactly the six split CSV files")
    for name, digest in checksums.items():
        path = _bundle_file(directory, name)
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise WPPDatasetError(f"invalid dataset CSV checksum declaration: {name}")
        if verify_checksums and sha256_file(path) != digest:
            raise WPPDatasetError(f"dataset CSV checksum mismatch: {name}")
    return metadata


@dataclass(frozen=True)
class _VideoRecord:
    video_id: str
    source_sha256: str
    source_relative_path: str
    label: str
    group_id: str
    partition: str
    start: int
    rows: int
    original_frames: int
    used_frames: int
    truncated: bool
    artifact_relpath: str


def _read_label_manifest(root: Path, partition: str) -> tuple[_VideoRecord, ...]:
    path = _bundle_file(root, f"{partition}-label-manifest.csv")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != LABEL_FIELDS:
                raise WPPDatasetError("label manifest header is not the exact bundle schema")
            rows = list(reader)
    except OSError as exc:
        raise WPPDatasetError(f"cannot read label manifest: {path}") from exc
    records: list[_VideoRecord] = []
    cursor = 0
    for row in rows:
        if set(row) != set(LABEL_FIELDS) or any(value is None for value in row.values()):
            raise WPPDatasetError("label manifest row has missing/extra fields")
        digest = row["source_sha256"]
        if not _SHA.fullmatch(digest) or row["video_id"] != canonical_video_id(digest):
            raise WPPDatasetError("label manifest source identity is invalid")
        try:
            source_path = portable_relative_path(row["source_relative_path"], "source_relative_path")
            artifact = portable_relative_path(row["artifact_relpath"], "artifact_relpath")
        except ContractError as exc:
            raise WPPDatasetError(str(exc)) from exc
        if row["partition"] != partition or not row["label"]:
            raise WPPDatasetError("label manifest partition or label is invalid")
        start, count, rv = (_integer(row["stacked_row_start"], "stacked_row_start"),
                            _integer(row["stacked_row_count"], "stacked_row_count", positive=True),
                            _integer(row["R_v"], "R_v", positive=True))
        original, used = (_integer(row["original_frame_count"], "original_frame_count", positive=True),
                          _integer(row["used_frame_count"], "used_frame_count", positive=True))
        if (start != cursor or count != rv or used != min(original, TIME_STEPS)
                or row["time_truncated"] not in ("true", "false")
                or (row["time_truncated"] == "true") != (original > TIME_STEPS)):
            raise WPPDatasetError("label manifest block dimensions/truncation declaration is invalid")
        records.append(_VideoRecord(row["video_id"], digest, source_path, row["label"], row["group_id"], partition,
                                    start, count, original, used, row["time_truncated"] == "true", artifact))
        cursor += count
    return tuple(records)


def _trusted_parents(source_manifest: Path, index: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dir, index_dir = _strict_contract_inputs(source_manifest, index)
    try:
        return load_source_manifest(source_dir), load_extraction_index(index_dir, source_dir)
    except (ContractError, OSError, ValueError) as exc:
        raise WPPDatasetError(f"trusted source/index contract is invalid: {exc}") from exc


class WPPDatasetBundle:
    """One strict, reusable validation of a packed bundle and its trusted parents.

    Parent directories are explicit API inputs: the bundle's self-declared digests
    are compared to those live contracts rather than treated as authentication.
    Numeric rows remain lazy; only label manifests and requested split CSV are read.
    """
    def __init__(self, directory: str | Path, *, source_manifest: str | Path, index: str | Path,
                 verify_checksums: bool = True):
        self.directory = _reject_symlink_components(Path(directory), directory=True)
        self.source_manifest, self.index = _strict_contract_inputs(source_manifest, index)
        self.metadata = _load_metadata(self.directory, verify_checksums)
        self.source, self.extraction = _trusted_parents(self.source_manifest, self.index)
        self.records_by_partition = {part: _read_label_manifest(self.directory, part) for part in PARTITIONS}
        self._validate_global_contract()

    def _validate_global_contract(self) -> None:
        source_meta = {"records_sha256": self.source["metadata"]["records_sha256"],
                       "metadata_sha256": sha256_file(_bundle_file(self.source_manifest, "source_manifest.metadata.json"))}
        index_meta = {"records_sha256": self.extraction["metadata"]["records_sha256"],
                      "metadata_sha256": sha256_file(_bundle_file(self.index, "extraction_index.metadata.json"))}
        binding = self.extraction["metadata"].get("native_workflow_ledger")
        if not isinstance(binding, Mapping) or not isinstance(binding.get("sha256"), str) or not _SHA.fullmatch(binding["sha256"]):
            raise WPPDatasetError("trusted extraction index lacks a valid native ledger binding")
        index_meta["native_ledger_sha256"] = binding["sha256"]
        if self.metadata.get("source_manifest") != source_meta or self.metadata.get("extraction_index") != index_meta:
            raise WPPDatasetError("bundle provenance does not bind the supplied trusted source manifest/index")
        counts = self.metadata.get("partition_counts")
        if not isinstance(counts, dict) or set(counts) != set(PARTITIONS):
            raise WPPDatasetError("metadata partition_counts must cover exactly train, validation, and test")
        by_id = {row["video_id"]: row for row in self.extraction["records"]}
        source_by_partition = {part: [row for row in self.source["records"] if row["partition"] == part] for part in PARTITIONS}
        if sum(len(rows) for rows in source_by_partition.values()) != len(self.source["records"]):
            raise WPPDatasetError("trusted source manifest has records without approved partitions")
        seen_ids: set[str] = set(); seen_sha: set[str] = set(); seen_paths: set[str] = set(); group_partition: dict[str, str] = {}
        all_records: list[_VideoRecord] = []
        for part in PARTITIONS:
            records, sources = self.records_by_partition[part], source_by_partition[part]
            declared = counts[part]
            expected_rows = sum(record.rows for record in records)
            if (not isinstance(declared, dict) or declared.get("data_file") != f"{part}-data.csv"
                    or declared.get("label_manifest_file") != f"{part}-label-manifest.csv"
                    or declared.get("video_count") != len(records) or declared.get("stacked_row_count") != expected_rows):
                raise WPPDatasetError("metadata partition counts/files do not match global label manifests")
            if len(records) != len(sources):
                raise WPPDatasetError("label manifests do not provide exact trusted source coverage")
            for record, source in zip(records, sources):
                index = by_id.get(record.video_id)
                if index is None or index["status"] != "success":
                    raise WPPDatasetError("label manifest references unavailable trusted extraction")
                if (record.video_id != source["video_id"] or record.source_sha256 != source["source_sha256"]
                        or record.source_relative_path != source["source_relative_path"] or record.label != source["label"]
                        or record.group_id != source["group_id"] or record.partition != source["partition"]
                        or record.rows != int(index["row_count"]) or record.original_frames != int(index["frame_count"])
                        or record.artifact_relpath != index["artifact_relative_path"]):
                    raise WPPDatasetError("label manifest differs from supplied trusted source/index provenance")
                if record.video_id in seen_ids or record.source_sha256 in seen_sha or record.source_relative_path in seen_paths:
                    raise WPPDatasetError("duplicate video/source identity across bundle partitions")
                seen_ids.add(record.video_id); seen_sha.add(record.source_sha256); seen_paths.add(record.source_relative_path)
                if record.group_id:
                    existing = group_partition.setdefault(record.group_id, part)
                    if existing != part:
                        raise WPPDatasetError("nonempty group_id spans bundle partitions")
                all_records.append(record)
        if seen_ids != set(by_id) or len(all_records) != len(self.source["records"]):
            raise WPPDatasetError("bundle source coverage is incomplete or has extra videos")
        training_labels = sorted({record.label for record in self.records_by_partition["train"]})
        expected_mapping = {label: position for position, label in enumerate(training_labels)}
        if not expected_mapping or self.metadata.get("class_mapping") != expected_mapping:
            raise WPPDatasetError("class_mapping must exactly equal sorted unique trusted training labels")
        heldout = {record.label for part in ("validation", "test") for record in self.records_by_partition[part]}
        if not heldout.issubset(expected_mapping):
            raise WPPDatasetError("held-out label absent from trusted training class mapping")

    def partition_view(self, partition: str) -> "WPPVideoDataset":
        return WPPVideoDataset(self, partition)


def load_wpp_metadata(directory: str | Path, *, source_manifest: str | Path, index: str | Path,
                      verify_checksums: bool = True) -> dict[str, Any]:
    """Strictly validate bundle ancestry/label manifests without reading numeric rows."""
    return WPPDatasetBundle(directory, source_manifest=source_manifest, index=index,
                            verify_checksums=verify_checksums).metadata


class WPPVideoDataset:
    """NumPy video view with explicit trusted parent provenance.

    Pass a ``WPPDatasetBundle`` to reuse its one-time six-file checksum and global
    label/provenance validation when constructing multiple split views.  Otherwise
    ``source_manifest`` and ``index`` are required; no self-declared-digest fallback
    exists. Requested numeric data is scanned once and accessed by byte offset.
    """
    def __init__(self, directory: str | Path | WPPDatasetBundle, partition: str, *,
                 source_manifest: str | Path | None = None, index: str | Path | None = None,
                 verify_checksums: bool = True):
        if partition not in PARTITIONS:
            raise WPPDatasetError(f"partition must be one of {PARTITIONS}")
        if isinstance(directory, WPPDatasetBundle):
            if source_manifest is not None or index is not None:
                raise WPPDatasetError("do not provide parent paths when using a WPPDatasetBundle")
            self.bundle = directory
        else:
            if source_manifest is None or index is None:
                raise WPPDatasetError("source_manifest and index are required trusted parent inputs")
            self.bundle = WPPDatasetBundle(directory, source_manifest=source_manifest, index=index,
                                           verify_checksums=verify_checksums)
        self.directory, self.partition = self.bundle.directory, partition
        self.metadata = self.bundle.metadata; self.class_mapping = dict(self.metadata["class_mapping"])
        self.records = self.bundle.records_by_partition[partition]
        self.data_path = _bundle_file(self.directory, f"{partition}-data.csv")
        self.offsets = self._scan_data()
        expected_rows = sum(record.rows for record in self.records)
        if len(self.offsets) != expected_rows:
            raise WPPDatasetError(f"numeric row count does not equal label block total: {len(self.offsets)} != {expected_rows}")

    def _scan_data(self) -> tuple[int, ...]:
        with self.data_path.open("rb") as handle:
            header = handle.readline().decode("utf-8").rstrip("\r\n").split(",")
            if tuple(header) != DATA_COLUMNS:
                raise WPPDatasetError("data CSV header must be exactly size0 through size2999")
            offsets: list[int] = []
            while True:
                position = handle.tell(); line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise WPPDatasetError("data CSV has unterminated row")
                try:
                    cells = line.decode("utf-8").rstrip("\r\n").split(",")
                except UnicodeDecodeError as exc:
                    raise WPPDatasetError("data CSV is not UTF-8") from exc
                if len(cells) != TIME_STEPS:
                    raise WPPDatasetError("data CSV row has wrong fixed frame width")
                if any(not re.fullmatch(r"(?:0|[1-9][0-9]*)", cell) for cell in cells):
                    raise WPPDatasetError("data CSV values must be finite nonnegative integer byte counts")
                if any(int(cell) > int(np.finfo(np.float32).max) for cell in cells):
                    raise WPPDatasetError("data CSV value exceeds finite float32 range")
                offsets.append(position)
        return tuple(offsets)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        values = np.empty((record.rows, TIME_STEPS), dtype=np.float32)
        with self.data_path.open("rb") as handle:
            for local_row, offset in enumerate(self.offsets[record.start:record.start + record.rows]):
                handle.seek(offset)
                line = handle.readline().decode("utf-8").rstrip("\r\n")
                parsed = np.fromstring(line, sep=",", dtype=np.float32)
                if parsed.shape != (TIME_STEPS,):
                    raise WPPDatasetError("numeric CSV changed after validation")
                values[local_row] = parsed
        if record.used_frames < TIME_STEPS and np.any(values[:, record.used_frames:] != 0):
            raise WPPDatasetError("right-padding cells are nonzero")
        return {"values": values, "used_time_length": record.used_frames, "row_count": record.rows,
                "label": record.label, "class_id": self.class_mapping[record.label], "video_id": record.video_id,
                "source_sha256": record.source_sha256, "source_relative_path": record.source_relative_path,
                "group_id": record.group_id, "partition": record.partition,
                "original_frame_count": record.original_frames, "time_truncated": record.truncated,
                "artifact_relpath": record.artifact_relpath}


def collate_wpp_videos(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate NumPy video items, padding only R to this batch's maximum.

    ``valid_mask`` is the outer product of row/time masks; invalid cells are zero.
    No zero-valued observed cell is used to infer either mask.
    """
    if not items:
        raise WPPDatasetError("cannot collate an empty batch")
    max_rows = max(int(item["row_count"]) for item in items)
    batch = len(items)
    values = np.zeros((batch, max_rows, TIME_STEPS), dtype=np.float32)
    row_mask = np.zeros((batch, max_rows), dtype=bool)
    time_mask = np.zeros((batch, TIME_STEPS), dtype=bool)
    for position, item in enumerate(items):
        x, rows, used = np.asarray(item["values"], dtype=np.float32), int(item["row_count"]), int(item["used_time_length"])
        if x.shape != (rows, TIME_STEPS) or rows <= 0 or not (0 < used <= TIME_STEPS):
            raise WPPDatasetError("invalid item passed to WPP collator")
        if not np.isfinite(x).all() or (x < 0).any() or (used < TIME_STEPS and np.any(x[:, used:] != 0)):
            raise WPPDatasetError("invalid raw values passed to WPP collator")
        values[position, :rows] = x; row_mask[position, :rows] = True; time_mask[position, :used] = True
    valid_mask = row_mask[:, :, None] & time_mask[:, None, :]
    values[~valid_mask] = 0
    return {"values": values, "row_mask": row_mask, "time_mask": time_mask, "valid_mask": valid_mask,
            "labels": np.asarray([int(item["class_id"]) for item in items], dtype=np.int64),
            "video_ids": [str(item["video_id"]) for item in items],
            "used_time_lengths": np.asarray([int(item["used_time_length"]) for item in items], dtype=np.int64),
            "row_counts": np.asarray([int(item["row_count"]) for item in items], dtype=np.int64)}
