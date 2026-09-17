"""Strict loading, hashing, splitting, and preprocessing for supervised packet data."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = tuple(f"size{i}" for i in range(3000))


@dataclass(frozen=True)
class PairData:
    values: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray
    row_hashes: Tuple[str, ...]
    data_path: Path
    label_path: Path
    data_checksum: str
    label_checksum: str
    lengths: np.ndarray | None = None

    def subset(self, indices: Sequence[int]) -> "PairData":
        idx = np.asarray(indices, dtype=np.int64)
        return PairData(self.values[idx], self.labels[idx], self.source_indices[idx],
                        tuple(self.row_hashes[i] for i in idx), self.data_path, self.label_path,
                        self.data_checksum, self.label_checksum, None if self.lengths is None else self.lengths[idx])


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_label(value: object) -> str:
    if pd.isna(value):
        raise ValueError("labels must not contain missing values")
    label = str(value).strip().casefold()
    if not label:
        raise ValueError("labels must not be empty after trimming whitespace")
    return label


def row_sha256(row: np.ndarray) -> str:
    values = np.asarray(row, dtype=np.float64)
    values = np.where(values == 0, 0.0, values).astype(">f8", copy=False)
    return sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def infer_lengths(values: np.ndarray) -> np.ndarray:
    """Return contiguous nonzero-prefix lengths, rejecting holes and empty rows."""
    x = np.asarray(values)
    nonzero = x != 0
    lengths = nonzero.sum(axis=1).astype(np.int64)
    if np.any(lengths == 0): raise ValueError("length-aware sequences must contain at least one nonzero packet")
    expected = np.arange(x.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(nonzero, expected): raise ValueError("length-aware sequences require a contiguous nonzero prefix (zero followed by nonzero found)")
    return lengths


def load_pair(data_csv: str | Path, label_csv: str | Path, length_aware: bool = False) -> PairData:
    """Load one pair without deduplicating it; splitting decides duplicate policy."""
    data_path, label_path = Path(data_csv).expanduser().resolve(), Path(label_csv).expanduser().resolve()
    frame = pd.read_csv(data_path)
    if tuple(frame.columns) != FEATURE_COLUMNS:
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))[:10]
        unexpected = sorted(set(frame.columns) - set(FEATURE_COLUMNS))[:10]
        raise ValueError("data columns must be exactly size0 through size2999, in order; "
                         f"missing={missing}, unexpected={unexpected}")
    bad_types = [c for c in frame if not is_numeric_dtype(frame[c]) or is_bool_dtype(frame[c])]
    if bad_types:
        raise ValueError(f"all packet sizes must be numeric; invalid columns: {bad_types[:10]}")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all(): raise ValueError("packet sizes must be finite and missing values are forbidden")
    if (values < 0).any(): raise ValueError("packet sizes must be nonnegative")
    lengths = infer_lengths(values) if length_aware else None
    if not length_aware:
        zero = np.flatnonzero(values.std(axis=1, ddof=0) == 0)
        if len(zero): raise ValueError(f"zero-standard-deviation sequence at data row {int(zero[0])}")
    labels_frame = pd.read_csv(label_path)
    if labels_frame.shape[1] != 1: raise ValueError("label CSV must contain exactly one column")
    if len(labels_frame) != len(values):
        raise ValueError(f"data and label row counts differ: {len(values)} != {len(labels_frame)}")
    labels = np.asarray([canonicalize_label(v) for v in labels_frame.iloc[:, 0]], dtype=object)
    hashes = tuple(row_sha256(row) for row in values)
    return PairData(values, labels, np.arange(len(values), dtype=np.int64), hashes,
                    data_path, label_path, file_sha256(data_path), file_sha256(label_path), lengths)


def normalize_length_aware(values: np.ndarray, lengths: np.ndarray, mean: float | None = None, std: float | None = None) -> tuple[np.ndarray, float, float]:
    """Fit/apply a scalar z-score over real prefix values only and retain zero padding."""
    x = np.asarray(values, dtype=np.float64); lengths = np.asarray(lengths, dtype=np.int64)
    valid = np.arange(x.shape[1])[None, :] < lengths[:, None]
    real = x[valid]
    if mean is None: mean = float(real.mean())
    if std is None: std = float(real.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0: raise ValueError("length-aware training values require nonzero finite global standard deviation")
    out = ((x - mean) / std).astype(np.float32); out[~valid] = 0
    return out, float(mean), float(std)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Independently z-score rows with population standard deviation, as float32."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3000: raise ValueError("expected a two-dimensional array with 3000 columns")
    mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True, ddof=0)
    if not np.isfinite(x).all() or (std == 0).any(): raise ValueError("cannot normalize non-finite or constant sequences")
    return ((x - mean) / std).astype(np.float32)


def _detail(pair: PairData) -> Dict[str, object]:
    return {"source_indices": pair.source_indices.tolist(), "row_hashes": list(pair.row_hashes),
            "row_count": len(pair.labels)}


def automatic_split(source: PairData, seed: int = 42) -> tuple[Dict[str, PairData], Dict[str, Dict[str, object]]]:
    """Deduplicate then perform deterministic stratified 70/15/15 splitting."""
    first: Dict[str, Tuple[str, int]] = {}; keep = []
    for pos, (digest, label) in enumerate(zip(source.row_hashes, source.labels)):
        if digest not in first:
            first[digest] = (str(label), pos); keep.append(pos)
        elif first[digest][0] != label:
            raise ValueError(f"duplicate sequence has conflicting labels at source rows "
                             f"{first[digest][1]} ({first[digest][0]!r}) and {pos} ({label!r})")
    unique = source.subset(keep); positions = np.arange(len(unique.labels))
    try:
        train_idx, remainder_idx = train_test_split(positions, test_size=0.30, random_state=seed,
                                                     shuffle=True, stratify=unique.labels)
        val_idx, test_idx = train_test_split(remainder_idx, test_size=0.50, random_state=seed,
                                             shuffle=True, stratify=unique.labels[remainder_idx])
    except ValueError as exc:
        raise ValueError(f"cannot create class-stratified 70/15/15 split: {exc}") from exc
    pieces = {"train": unique.subset(np.sort(train_idx)), "val": unique.subset(np.sort(val_idx)),
              "test": unique.subset(np.sort(test_idx))}
    sets = [set(p.row_hashes) for p in pieces.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("a sequence appears in more than one automatic split")
    return pieces, {name: _detail(pair) for name, pair in pieces.items()}


def manual_split(train: PairData, val: PairData, test: PairData) -> tuple[Dict[str, PairData], Dict[str, Dict[str, object]]]:
    """Validate independent manual pairs for hash overlap and unseen labels."""
    pieces = {"train": train, "val": val, "test": test}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(pieces[left].row_hashes) & set(pieces[right].row_hashes)
        if overlap: raise ValueError(f"sequence overlap detected between {left} and {right}: {len(overlap)} row hash(es)")
    known = set(train.labels)
    for name in ("val", "test"):
        unseen = sorted(set(pieces[name].labels) - known)
        if unseen: raise ValueError(f"{name} labels absent from training split: {unseen}")
    return pieces, {name: _detail(pair) for name, pair in pieces.items()}
