"""Video-format classification reports for the standalone native WPP runner.

This module deliberately depends only on the native bundle/model contracts.  It
contains no legacy model or loader imports.  The public writer mirrors the
established fourteen files in an ``evaluations`` directory while retaining
native provenance in a separate file owned by the caller.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .classification_metrics import (CLASSIFICATION_METRICS, class_coverage,
                                     classification_scores, per_class_scores,
                                     zero_division_details)

EPOCH_HEADER = (
    "epoch", "train_loss", "validation_loss",
    "train_accuracy", "train_macro_precision", "train_weighted_precision",
    "train_macro_recall", "train_weighted_recall", "train_macro_f1", "train_weighted_f1",
    "validation_accuracy", "validation_macro_precision", "validation_weighted_precision",
    "validation_macro_recall", "validation_weighted_recall", "validation_macro_f1", "validation_weighted_f1",
    "learning_rate", "duration_seconds",
)
PER_CLASS_HEADER = ("class", "precision", "recall", "f1", "sample_count")
REPORT_FILES = frozenset((
    "accuracy_curves.png", "f1_curves.png", "loss_curves.png", "per_epoch_metrics.csv",
    "per_epoch_metrics.json", "precision_curves.png", "recall_curves.png", "summarized_metrics.json",
    "test_confusion_matrix_normalized.csv", "test_confusion_matrix_normalized.png",
    "test_confusion_matrix_raw.csv", "test_confusion_matrix_raw.png",
    "test_per_class_metrics.csv", "test_per_class_metrics.json",
))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _atomic_csv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(temporary, dpi=140, format="png")
    plt.close(figure)
    os.replace(temporary, path)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number_or_none(value: Any) -> float | None:
    """Keep explicit unknowns null, but reject computed NaN/Inf values."""
    if value is None:
        return None
    if not _finite_number(value):
        raise ValueError("report values must be finite or explicitly null when historically unavailable")
    return float(value)


def flatten_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert native history records to the exact public report schema.

    Older native runs hold validation metrics as a nested mapping and lack the
    online train metrics.  Missing values are represented by JSON null / blank
    CSV fields rather than invented scores.
    """
    flattened: list[dict[str, Any]] = []
    for source in history:
        epoch = source.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("history epoch must be a positive integer")
        validation = source.get("validation_metrics", {})
        if validation is None:
            validation = {}
        if not isinstance(validation, Mapping):
            raise ValueError("history validation_metrics must be a mapping")
        row: dict[str, Any] = {key: None for key in EPOCH_HEADER}
        row["epoch"] = epoch
        row["train_loss"] = _number_or_none(source.get("train_loss", source.get("train_cross_entropy")))
        row["validation_loss"] = _number_or_none(source.get("validation_loss", source.get("validation_cross_entropy")))
        for metric in CLASSIFICATION_METRICS:
            row[f"train_{metric}"] = _number_or_none(source.get(f"train_{metric}"))
            row[f"validation_{metric}"] = _number_or_none(source.get(f"validation_{metric}", validation.get(metric)))
        row["learning_rate"] = _number_or_none(source.get("learning_rate"))
        row["duration_seconds"] = _number_or_none(source.get("duration_seconds"))
        flattened.append(row)
    return flattened


def best_observed(history: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Independently maximize each classification metric; first epoch wins ties."""
    result: dict[str, dict[str, dict[str, float | int]]] = {"train": {}, "validation": {}}
    for split in result:
        for metric in CLASSIFICATION_METRICS:
            key = f"{split}_{metric}"
            for row in history:
                value, epoch = row.get(key), row.get("epoch")
                if not _finite_number(value) or isinstance(epoch, bool) or not isinstance(epoch, int):
                    continue
                candidate = float(value)
                saved = result[split].get(metric)
                if saved is None or candidate > saved["value"] or (candidate == saved["value"] and epoch < saved["epoch"]):
                    result[split][metric] = {"value": candidate, "epoch": epoch}
    return result


def report_evaluation(model_result: Mapping[str, Any], *, checkpoint: str, data_path: str,
                      data_checksum: str, label_path: str, label_checksum: str,
                      parameter_count: int, evaluation_kind: str = "original_test",
                      duration_seconds: float | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Create the canonical summary and matrices from native inference output."""
    class_names = model_result.get("class_names")
    truths = model_result.get("_truths")
    predictions = model_result.get("_predictions")
    if (not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names)
            or not isinstance(truths, list) or not isinstance(predictions, list) or len(truths) != len(predictions) or not truths):
        raise ValueError("reporting requires nonempty retained native truths and predictions")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(class_names)
           for value in [*truths, *predictions]):
        raise ValueError("reporting labels are outside configured class order")
    raw = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, prediction in zip(truths, predictions):
        raw[truth, prediction] += 1
    normalized = np.divide(raw, raw.sum(axis=1, keepdims=True), out=np.zeros_like(raw, dtype=float),
                           where=raw.sum(axis=1, keepdims=True) != 0)
    values = classification_scores(truths, predictions)
    duration = _number_or_none(duration_seconds)
    coverage = class_coverage([class_names[index] for index in truths], class_names)
    cross_entropy = model_result.get("cross_entropy")
    if not _finite_number(cross_entropy):
        raise ValueError("reporting requires finite cross_entropy")
    summary: dict[str, Any] = {
        "accuracy": values["accuracy"], "best_observed_per_epoch_metrics": {}, "checkpoint": str(checkpoint),
        "class_coverage": coverage, "created_at": _now(), "data_checksum_sha256": data_checksum,
        "data_path": str(data_path), "evaluation_duration_seconds": duration,
        "evaluation_kind": evaluation_kind, "label_checksum_sha256": label_checksum,
        "label_path": str(label_path), "macro_f1": values["macro_f1"], "macro_precision": values["macro_precision"],
        "macro_recall": values["macro_recall"], "parameter_count": int(parameter_count), "sample_count": len(truths),
        "samples_per_second": (len(truths) / duration if duration and duration > 0 else None),
        "test_loss": float(cross_entropy), "weighted_f1": values["weighted_f1"],
        "weighted_precision": values["weighted_precision"], "weighted_recall": values["weighted_recall"],
        "zero_division_handling": zero_division_details(truths, predictions, class_names),
    }
    per_class = per_class_scores(truths, predictions, class_names)
    return summary, per_class, raw, normalized


def _plot_curves(path: Path, history: Sequence[Mapping[str, Any]], keys: Sequence[str], ylabel: str, title: str) -> None:
    # Match the reference default 6.4x4.8 inch figure at 140 dpi (896x672).
    figure, axes = plt.subplots(figsize=(6.4, 4.8))
    for key in keys:
        points = [(row["epoch"], float(row[key])) for row in history if _finite_number(row.get(key))]
        if points:
            axes.plot(*zip(*points), label=key)
    axes.set(xlabel="epoch", ylabel=ylabel, title=title)
    if axes.lines:
        axes.legend()
    axes.grid(alpha=.25)
    figure.tight_layout()
    _atomic_figure(path, figure)


def _write_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    rows = []
    for name, row in zip(class_names, matrix):
        rows.append({"true\\pred": name, **{label: value.item() if isinstance(value, np.generic) else value
                                               for label, value in zip(class_names, row)}})
    _atomic_csv(path, ("true\\pred", *class_names), rows)


def _plot_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str], title: str, fmt: str) -> None:
    side = max(6.72, min(14.0, len(class_names) * .9))
    figure, axes = plt.subplots(figsize=(side, side))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    axes.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names,
             yticklabels=class_names, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    if len(class_names) <= 20:
        threshold = float(matrix.max()) / 2 if matrix.size else 0.0
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                axes.text(column, row, format(matrix[row, column], fmt), ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black", fontsize=7)
    figure.tight_layout()
    _atomic_figure(path, figure)


def write_evaluations(evaluations: str | Path, *, history: Sequence[Mapping[str, Any]],
                      summary: Mapping[str, Any], per_class: Sequence[Mapping[str, Any]], raw: np.ndarray,
                      normalized: np.ndarray, class_names: Sequence[str]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Atomically publish exactly fourteen files into a fresh evaluations directory."""
    destination = Path(evaluations).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("compatibility evaluations directory must be fresh and is never overwritten")
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError("compatibility evaluations parent must be an owned directory")
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"compatibility evaluations parent contains a symlink: {current}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    try:
        records = flatten_history(history)
        maxima = best_observed(records)
        summary_value = dict(summary)
        summary_value["best_observed_per_epoch_metrics"] = maxima
        # Reject accidental schema drift rather than silently publishing extra fields.
        expected_summary = {
            "accuracy", "best_observed_per_epoch_metrics", "checkpoint", "class_coverage", "created_at",
            "data_checksum_sha256", "data_path", "evaluation_duration_seconds", "evaluation_kind",
            "label_checksum_sha256", "label_path", "macro_f1", "macro_precision", "macro_recall",
            "parameter_count", "sample_count", "samples_per_second", "test_loss", "weighted_f1",
            "weighted_precision", "weighted_recall", "zero_division_handling",
        }
        if set(summary_value) != expected_summary:
            raise ValueError("evaluation summary schema is invalid")
        atomic_json(stage / "per_epoch_metrics.json", records)
        _atomic_csv(stage / "per_epoch_metrics.csv", EPOCH_HEADER, records)
        _plot_curves(stage / "loss_curves.png", records, ("train_loss", "validation_loss"), "loss", "Training and validation loss")
        _plot_curves(stage / "accuracy_curves.png", records, ("train_accuracy", "validation_accuracy"), "accuracy", "Accuracy")
        _plot_curves(stage / "precision_curves.png", records,
                     ("train_macro_precision", "train_weighted_precision", "validation_macro_precision", "validation_weighted_precision"),
                     "precision", "Precision")
        _plot_curves(stage / "recall_curves.png", records,
                     ("train_macro_recall", "train_weighted_recall", "validation_macro_recall", "validation_weighted_recall"),
                     "recall", "Recall")
        _plot_curves(stage / "f1_curves.png", records,
                     ("train_macro_f1", "train_weighted_f1", "validation_macro_f1", "validation_weighted_f1"), "F1", "F1")
        atomic_json(stage / "summarized_metrics.json", summary_value)
        per_class_rows = [dict(row) for row in per_class]
        if any(tuple(row) != PER_CLASS_HEADER for row in per_class_rows):
            raise ValueError("per-class report schema is invalid")
        atomic_json(stage / "test_per_class_metrics.json", per_class_rows)
        _atomic_csv(stage / "test_per_class_metrics.csv", PER_CLASS_HEADER, per_class_rows)
        _write_matrix(stage / "test_confusion_matrix_raw.csv", raw, class_names)
        _write_matrix(stage / "test_confusion_matrix_normalized.csv", normalized, class_names)
        _plot_matrix(stage / "test_confusion_matrix_raw.png", raw, class_names, "Confusion matrix", "d")
        _plot_matrix(stage / "test_confusion_matrix_normalized.png", normalized, class_names,
                     "Row-normalized confusion matrix", ".2f")
        published = {path.name for path in stage.iterdir() if path.is_file()}
        if published != REPORT_FILES:
            raise ValueError("evaluation report publication is incomplete")
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return maxima


def timed_evaluation(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    return callable_(), time.perf_counter() - started
