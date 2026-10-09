"""Video-level metrics for the standalone binary WPP workflow."""
from __future__ import annotations
from typing import Any, Sequence
from collections import Counter
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def _safe_auc(y: np.ndarray, score: np.ndarray, fn) -> float | None:
    return float(fn(y, score)) if len(np.unique(y)) == 2 else None


def video_metrics(labels: Sequence[int], probabilities: Sequence[Sequence[float]], sources: Sequence[dict] = ()) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); p = np.asarray(probabilities, dtype=float); pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y, pred, labels=sorted(set(y)), average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(y, pred, labels=sorted(set(y)), average="weighted", zero_division=0)
    result: dict[str, Any] = {
        "sample_count": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(macro_p), "weighted_precision": float(weighted_p), "macro_recall": float(macro_r), "weighted_recall": float(weighted_r),
        "macro_f1": float(macro_f), "weighted_f1": float(weighted_f), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]),
                               "f1": float(f1[index]), "support": int(support[index]),
                               "roc_auc": _safe_auc((y == index).astype(int), p[:, index], roc_auc_score),
                               "pr_auc": _safe_auc((y == index).astype(int), p[:, index], average_precision_score)}
                      for index, name in enumerate(("real", "aigc"))},
    }
    if sources:
        required = ("video_id", "source", "width", "height", "fps")
        malformed = [entry.get("video_id", "<unknown>") for entry in sources
                     if any(entry.get(key) in (None, "") for key in required)]
        if malformed: raise ValueError(f"malformed source/resolution/fps metadata: {malformed[:10]}")
        source_counts = Counter(str(entry["source"]) for entry in sources)
        resolution_counts = Counter(f"{entry['width']}x{entry['height']}" for entry in sources)
        fps_counts = Counter(str(entry["fps"]) for entry in sources)
        outcomes = {}
        for name, indices in {name: [i for i, entry in enumerate(sources) if str(entry['source']) == name] for name in source_counts}.items():
            truth, predicted, positive = y[indices], pred[indices], p[indices, 1]
            outcomes[name] = {"samples": len(indices), "accuracy": float((truth == predicted).mean()),
                              "predicted_aigc_rate": float((predicted == 1).mean()), "mean_aigc_probability": float(positive.mean()),
                              "roc_auc_aigc": _safe_auc(truth, positive, roc_auc_score), "pr_auc_aigc": _safe_auc(truth, positive, average_precision_score)}
        result["source_resolution_fps"] = {"source_counts": dict(sorted(source_counts.items())),
            "resolution_counts": dict(sorted(resolution_counts.items())), "fps_counts": dict(sorted(fps_counts.items())), "source_outcomes": outcomes,
            "per_video": [{key: entry[key] for key in required} for entry in sources]}
    return result
