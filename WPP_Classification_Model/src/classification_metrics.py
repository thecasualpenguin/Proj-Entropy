"""Shared classification metric definitions and class-coverage reporting.

Aggregate precision, recall, and F1 use only labels present in the evaluated
split.  Per-class reports always retain the complete configured label universe.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

CLASSIFICATION_METRICS = (
    "accuracy",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "weighted_recall",
    "macro_f1",
    "weighted_f1",
)


def _present_ids(truth: Sequence[int]) -> list[int]:
    return sorted({int(value) for value in truth})


def classification_scores(truth: Sequence[int], predicted: Sequence[int]) -> dict[str, float]:
    """Return the agreed seven metrics for one split.

    Macro and weighted averages are calculated over classes represented in
    ``truth``.  Predictions to configured-but-absent classes remain errors:
    they contribute false negatives to the actual, present class.
    """
    present = _present_ids(truth)
    if not present:
        raise ValueError("classification metrics require at least one ground-truth label")
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=present, average="macro", zero_division=0)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=present, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(macro_precision),
        "weighted_precision": float(weighted_precision),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def per_class_scores(truth: Sequence[int], predicted: Sequence[int], class_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return one report row per configured class, including absent classes."""
    ids = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=ids, zero_division=0)
    return [{"class": class_names[index], "precision": float(precision[index]),
             "recall": float(recall[index]), "f1": float(f1[index]),
             "sample_count": int(support[index])} for index in ids]


def class_coverage(labels: Sequence[str], configured_classes: Sequence[str]) -> dict[str, Any]:
    """Describe the observed ground-truth class coverage of a split."""
    configured = list(configured_classes)
    observed = Counter(str(label) for label in labels)
    present = [name for name in configured if observed.get(name, 0)]
    absent = [name for name in configured if not observed.get(name, 0)]
    unexpected = sorted(set(observed) - set(configured))
    return {"configured_classes": configured, "present_classes": present,
            "absent_classes": absent, "class_distribution": dict(sorted(observed.items())),
            "unexpected_classes": unexpected}


def zero_division_details(truth: Sequence[int], predicted: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    """Report configured classes that received no predictions."""
    predicted_counts = Counter(int(value) for value in predicted)
    no_predictions = [class_names[index] for index in range(len(class_names)) if not predicted_counts[index]]
    return {"value_used": 0, "classes_with_no_predictions": no_predictions,
            "occurred": bool(no_predictions)}


def coverage_warnings(split_coverages: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return warning text for missing coverage without changing any split."""
    warnings: list[str] = []
    train_classes = set(split_coverages["train"]["present_classes"])
    for split in ("validation", "test"):
        coverage = split_coverages[split]
        if coverage["absent_classes"]:
            warnings.append(f"{split} split lacks configured classes: {coverage['absent_classes']}")
        missing_from_training = sorted(set(coverage["present_classes"]) - train_classes)
        unseen_in_configured_training = coverage.get("unexpected_classes", [])
        if missing_from_training or unseen_in_configured_training:
            warnings.append(f"{split} split has classes missing from training: "
                            f"{missing_from_training or unseen_in_configured_training}")
    return warnings
