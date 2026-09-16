#!/usr/bin/env python3
"""Frozen linear probes for residual blocks in a modern supervised run.

The probe consumes a ``supervised.py`` format-version-1 checkpoint and its
recorded split manifest.  It never trains or changes the encoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")  # Probe artifacts must be generated without a display.
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from model_resnet_tsc import ResNet
from supervised import choose_device
from supervised_data import PairData, load_pair, normalize_rows

PREPROCESSING = {"method": "per_sequence_zscore", "sequence_length": 3000,
                 "standard_deviation": "population", "epsilon": None}
BLOCKS = (1, 2, 3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state", "model_config", "preprocessing_config", "label_mapping", "split_information"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is not a modern supervised checkpoint; missing fields: {sorted(missing)}")
    if checkpoint.get("format_version") != 1:
        raise ValueError("checkpoint format_version must be 1")
    mapping = checkpoint["label_mapping"]
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("checkpoint label mapping must be a non-empty mapping")
    ids = list(mapping.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in ids) or sorted(ids) != list(range(len(ids))):
        raise ValueError("checkpoint label mapping IDs must be unique contiguous integers 0..K-1")
    if checkpoint["preprocessing_config"] != PREPROCESSING:
        raise ValueError("checkpoint preprocessing is incompatible with the frozen probe")
    return checkpoint


def _checked_pair(meta: Mapping[str, Any], data_path: Optional[str] = None,
                  label_path: Optional[str] = None) -> PairData:
    """Load a recorded source (or byte-identical relocation) and check hashes."""
    if (data_path is None) != (label_path is None):
        raise ValueError("--data-path and --label-path must be supplied together")
    pair = load_pair(data_path or meta["data_path"], label_path or meta["label_path"])
    if pair.data_checksum != meta["data_checksum_sha256"] or pair.label_checksum != meta["label_checksum_sha256"]:
        raise ValueError("data/label SHA-256 does not match the source recorded by training")
    return pair


def load_recorded_splits(checkpoint: Mapping[str, Any], data_path: Optional[str] = None,
                         label_path: Optional[str] = None) -> tuple[dict[str, PairData], dict[str, int]]:
    """Recreate precisely the checkpoint's splits, including row/index checks.

    Relocation overrides apply to automatic-split runs, whose one original
    source can be moved as a byte-identical data/label pair.  Manual runs have
    three independent sources and deliberately require their recorded paths.
    """
    manifest = checkpoint["split_information"]
    mapping = {str(key): int(value) for key, value in checkpoint["label_mapping"].items()}
    mode = manifest.get("mode")
    if mode == "auto":
        source = _checked_pair(manifest["sources"]["source"], data_path, label_path)
        pieces = {}
        for name in ("train", "val", "test"):
            detail = manifest["splits"][name]
            indices = np.asarray(detail["source_indices"], dtype=np.int64)
            if (indices < 0).any() or (indices >= len(source.labels)).any():
                raise ValueError(f"saved {name} source indices are invalid")
            piece = source.subset(indices)
            if list(piece.source_indices) != list(detail["source_indices"]) or list(piece.row_hashes) != detail["row_hashes"]:
                raise ValueError(f"saved {name} row hashes or source indices no longer match split manifest")
            pieces[name] = piece
    elif mode == "manual":
        if data_path or label_path:
            raise ValueError("relocation overrides are supported only for automatic-split runs")
        pieces = {}
        for name in ("train", "val", "test"):
            piece = _checked_pair(manifest["sources"][name])
            detail = manifest["splits"][name]
            if list(piece.source_indices) != list(detail["source_indices"]) or list(piece.row_hashes) != detail["row_hashes"]:
                raise ValueError(f"saved {name} row hashes or source indices no longer match split manifest")
            pieces[name] = piece
    else:
        raise ValueError(f"unsupported split manifest mode: {mode!r}")
    _validate_split_integrity(pieces, manifest, source if mode == "auto" else None)
    return pieces, mapping


def _validate_split_integrity(pieces: Mapping[str, PairData], manifest: Mapping[str, Any],
                              auto_source: Optional[PairData] = None) -> None:
    """Reject overlap/tampering before any features can be extracted."""
    names = ("train", "val", "test")
    for name in names:
        detail = manifest["splits"][name]
        if len(detail["source_indices"]) != len(detail["row_hashes"]) or len(detail["row_hashes"]) != len(pieces[name].labels):
            raise ValueError(f"saved {name} split count is inconsistent with its indices or row hashes")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(pieces[left].row_hashes) & set(pieces[right].row_hashes):
            raise ValueError(f"saved split row hashes overlap between {left} and {right}")
    if manifest["mode"] == "auto":
        indices = {name: list(manifest["splits"][name]["source_indices"]) for name in names}
        if any(len(set(values)) != len(values) for values in indices.values()):
            raise ValueError("saved split contains duplicate source indices")
        if any(set(indices[left]) & set(indices[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("saved source indices overlap between splits")
        union = set().union(*map(set, indices.values()))
        expected = manifest["sources"]["source"].get("row_count")
        if expected is not None and (len(union) != sum(map(len, indices.values())) or len(union) > expected):
            raise ValueError("saved split index union is inconsistent with source row count")
        if auto_source is None:
            raise AssertionError("automatic split validation requires its loaded source")
        # Match automatic_split exactly: retain the first row for each hash and
        # reject a later duplicate with a conflicting label.
        first: dict[str, tuple[str, int]] = {}
        for digest, label, index in zip(auto_source.row_hashes, auto_source.labels, auto_source.source_indices):
            label = str(label)
            if digest not in first:
                first[digest] = (label, int(index))
            elif first[digest][0] != label:
                raise ValueError("automatic source has duplicate rows with conflicting labels")
        expected_pairs = {(index, digest) for digest, (_, index) in first.items()}
        saved_pairs = {(int(index), digest) for name in names
                       for index, digest in zip(manifest["splits"][name]["source_indices"],
                                                manifest["splits"][name]["row_hashes"])}
        if saved_pairs != expected_pairs:
            raise ValueError("saved automatic split union is incomplete or differs from first-occurrence source rows")
    else:
        expected_total = sum(manifest["sources"][name].get("row_count", len(pieces[name].labels)) for name in names)
        if sum(len(pieces[name].labels) for name in names) != expected_total:
            raise ValueError("saved manual split total is inconsistent with source row counts")


def reconstruct_encoder(checkpoint: Mapping[str, Any], device: torch.device) -> ResNet:
    mapping = checkpoint["label_mapping"]
    model = ResNet((1, 3000), len(mapping), checkpoint["model_config"]["initial_feature_maps"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def extract_block_features(model: ResNet, pair: PairData, device: torch.device,
                           batch_size: int) -> dict[int, np.ndarray]:
    """Return temporal means from residual blocks 1/2/3 (block 3 is final GAP)."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    values = normalize_rows(pair.values)[:, None, :]
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=False)
    extracted: dict[int, list[np.ndarray]] = {block: [] for block in BLOCKS}
    model.eval()
    with torch.inference_mode():
        for (xb,) in loader:
            out1 = model.layer1(xb.to(device))
            out2 = model.layer2(out1)
            out3 = model.layer3(out2)
            for block, output in ((1, out1), (2, out2), (3, out3)):
                extracted[block].append(output.mean(dim=2).cpu().numpy())
    return {block: np.concatenate(parts, axis=0) for block, parts in extracted.items()}


def _labels(pair: PairData, mapping: Mapping[str, int]) -> np.ndarray:
    unknown = sorted(set(pair.labels) - set(mapping))
    if unknown:
        raise ValueError(f"labels not present in checkpoint mapping: {unknown}")
    return np.asarray([mapping[str(label)] for label in pair.labels], dtype=np.int64)


def _fit_probe(x: np.ndarray, y: np.ndarray, c: float, max_iter: int) -> tuple[StandardScaler, LogisticRegression, list[str]]:
    scaler = StandardScaler().fit(x)
    probe = LogisticRegression(C=c, solver="lbfgs", multi_class="multinomial", class_weight=None,
                               max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe.fit(scaler.transform(x), y)
    messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return scaler, probe, messages


def _full_probabilities(probe: LogisticRegression, probabilities: np.ndarray, class_count: int) -> np.ndarray:
    """Align predict_proba columns with the checkpoint's complete label IDs."""
    full = np.zeros((len(probabilities), class_count), dtype=float)
    classes = np.asarray(probe.classes_, dtype=int)
    if (classes < 0).any() or (classes >= class_count).any() or len(classes) != probabilities.shape[1]:
        raise ValueError("probe predict_proba classes are incompatible with checkpoint labels")
    full[:, classes] = probabilities
    return full


def _classification_values(y: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray,
                           class_count: int) -> dict[str, float]:
    """Selection/test scalar metrics using the full checkpoint label universe."""
    ids = list(range(class_count))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=ids, average="macro", zero_division=0)),
            "multinomial_log_loss": float(log_loss(y, probabilities, labels=ids))}


def _metrics(y: np.ndarray, predicted: np.ndarray, class_names: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    ids = list(range(len(class_names)))
    raw = confusion_matrix(y, predicted, labels=ids)
    normalized = np.divide(raw, raw.sum(axis=1, keepdims=True), out=np.zeros_like(raw, dtype=float),
                           where=raw.sum(axis=1, keepdims=True) != 0)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=ids, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y, predicted, labels=ids, average="macro", zero_division=0)
    metrics = {"sample_count": int(len(y)), "accuracy": float(accuracy_score(y, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
               "macro_precision": float(macro_precision), "macro_recall": float(macro_recall),
               "macro_f1": float(macro_f1)}
    per_class = [{"class": class_names[index], "precision": float(precision[index]),
                  "recall": float(recall[index]), "f1": float(f1[index]),
                  "sample_count": int(support[index])} for index in ids]
    return metrics, per_class, raw, normalized


def _reference_provenance(checkpoint_path: Path, checkpoint: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    """Validate an original-test artifact, including a relocated parent run."""
    saved_checkpoint = raw.get("checkpoint")
    if raw.get("evaluation_kind") != "original_test" or not isinstance(saved_checkpoint, str):
        return None, "requires evaluation_kind=original_test and a checkpoint path"
    recorded = Path(saved_checkpoint).expanduser()
    current_run = checkpoint_path.parent.parent
    try:
        exact = recorded.resolve() == checkpoint_path
    except OSError:
        exact = False
    current_relative = checkpoint_path.relative_to(current_run)
    relocated = (recorded.parent.parent.name == current_run.name
                 and Path(recorded.parent.name, recorded.name) == current_relative)
    if not exact and not relocated:
        return None, "saved checkpoint is neither the exact path nor the same run-relative checkpoint"

    # A relocated artifact has no authoritative checkpoint digest in the legacy
    # schema.  Therefore compare every independently recorded identity field
    # that it does contain, rather than trusting an old absolute path.
    details: dict[str, Any] = {"recorded_checkpoint": str(recorded), "current_checkpoint": str(checkpoint_path),
                               "run_name": current_run.name, "checkpoint_relative_path": str(current_relative),
                               "current_checkpoint_epoch": checkpoint.get("completed_epoch"), "checked_fields": {}}
    for field in ("checkpoint_epoch", "checkpoint_completed_epoch"):
        if field in raw:
            expected, observed = checkpoint.get("completed_epoch"), raw[field]
            details["checked_fields"][field] = {"expected": expected, "observed": observed}
            if isinstance(observed, bool) or not isinstance(observed, (int, np.integer)) or expected != observed:
                return None, f"saved {field} does not match checkpoint epoch"
    config, mapping = checkpoint.get("model_config", {}), checkpoint.get("label_mapping", {})
    expected_count = sum(parameter.numel() for parameter in ResNet(
        (1, 3000), len(mapping), config["initial_feature_maps"]).parameters())
    # The original evaluator records this as parameter_count.  It is a useful
    # structural check when accepting a relocated legacy artifact.
    for field in ("parameter_count", "checkpoint_parameter_count"):
        if field in raw:
            observed = raw[field]
            details["checked_fields"][field] = {"expected": expected_count, "observed": observed}
            if isinstance(observed, bool) or not isinstance(observed, (int, np.integer)) or observed != expected_count:
                return None, f"saved {field} does not match checkpoint"
    manifest = checkpoint.get("split_information", {})
    source = (manifest.get("sources", {}).get("source") if manifest.get("mode") == "auto"
              else manifest.get("sources", {}).get("test", {}))
    split = manifest.get("splits", {}).get("test", {})
    for field, expected in (("sample_count", split.get("row_count")),
                            ("data_checksum_sha256", source.get("data_checksum_sha256") if isinstance(source, Mapping) else None),
                            ("label_checksum_sha256", source.get("label_checksum_sha256") if isinstance(source, Mapping) else None)):
        if field in raw and expected is not None:
            details["checked_fields"][field] = {"expected": expected, "observed": raw[field]}
            if raw[field] != expected:
                return None, f"saved {field} does not match checkpoint split provenance"
    return {"method": "exact_path" if exact else "relocated_run_relative", "details": details}, ""


def _load_original_test_reference(checkpoint_path: Path, checkpoint: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Safely load comparable final-test metrics without affecting selection."""
    if checkpoint is None:
        checkpoint = _load_checkpoint(checkpoint_path, torch.device("cpu"))
    path = checkpoint_path.parent.parent / "test_metrics.json"
    result: dict[str, Any] = {"status": "unavailable", "source": str(path), "values": {}, "omitted_metrics": []}
    if not path.is_file():
        result["reason"] = "parent run has no test_metrics.json"
        return result
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.update(status="malformed", reason=f"could not parse test_metrics.json: {exc}")
        return result
    if not isinstance(raw, Mapping):
        result.update(status="malformed", reason="test_metrics.json must contain an object")
        return result
    validation, reason = _reference_provenance(checkpoint_path, checkpoint, raw)
    if validation is None:
        result.update(status="invalid_provenance", reason=reason)
        return result
    key_options = {"accuracy": ("accuracy",), "balanced_accuracy": ("balanced_accuracy",),
                   "macro_f1": ("macro_f1",), "multinomial_log_loss": ("multinomial_log_loss", "log_loss", "test_loss")}
    values, source_keys, omitted = {}, {}, []
    for metric, keys in key_options.items():
        value = next((raw[key] for key in keys if key in raw), None)
        source_key = next((key for key in keys if key in raw), None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            omitted.append(metric)
        else:
            values[metric], source_keys[metric] = float(value), source_key
    result.update(status="available", values=values, source_keys=source_keys, omitted_metrics=omitted,
                  validation=validation)
    return result


def _plot_selection_curves(out: Path, rows: Sequence[Mapping[str, Any]], metric: str, filename: str,
                           reference: Mapping[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for block in BLOCKS:
        block_rows = sorted((row for row in rows if row["block"] == block), key=lambda row: row["C"])
        for split, style in (("train", "-o"), ("validation", "--s")):
            ax.plot([row["C"] for row in block_rows], [row[f"{split}_{metric}"] for row in block_rows], style,
                    label=f"Block {block} {split}")
    value = reference.get("values", {}).get(metric)
    if value is not None:
        ax.axhline(value, color="black", linestyle="--", linewidth=1.25,
                   label=f"Original ResNet test {reference.get('source_keys', {}).get(metric, metric)}")
    ax.set_xscale("log"); ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel(metric.replace("_", " ")); ax.set_title("Regularization-selection curve")
    ax.legend(fontsize="small"); fig.tight_layout(); fig.savefig(out / filename, dpi=140); plt.close(fig)


def _plot_final_test_metrics(out: Path, metrics: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]) -> None:
    blocks = [row["block"] for row in metrics]
    fig, (scores, losses) = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        scores.plot(blocks, [row[metric] for row in metrics], "o-", label=metric.replace("_", " "))
        value = reference.get("values", {}).get(metric)
        if value is not None:
            scores.axhline(value, linestyle="--", linewidth=1, label=f"Original ResNet {metric.replace('_', ' ')}")
    losses.plot(blocks, [row["multinomial_log_loss"] for row in metrics], "o-", color="tab:red", label="probe log loss")
    value = reference.get("values", {}).get("multinomial_log_loss")
    if value is not None:
        losses.axhline(value, color="black", linestyle="--", linewidth=1.25,
                       label="Original ResNet test loss")
    scores.set(title="Final held-out test scores", xlabel="Residual block", ylabel="score", xticks=blocks)
    losses.set(title="Final held-out multinomial log-loss", xlabel="Residual block", ylabel="log loss", xticks=blocks)
    scores.legend(fontsize="small"); losses.legend(fontsize="small"); fig.tight_layout()
    fig.savefig(out / "final_test_metrics_by_block.png", dpi=140); plt.close(fig)


def _write_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def run_linear_probe(checkpoint_path: Path, *, data_path: Optional[str] = None,
                     label_path: Optional[str] = None, device_spec: str = "auto",
                     batch_size: int = 64, c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
                     max_iter: int = 1000, output_root: Optional[Path] = None) -> Path:
    """Select C on validation macro-F1, refit on train+validation, test once."""
    if not c_values or any(not np.isfinite(float(c)) or float(c) <= 0 for c in c_values):
        raise ValueError("C values must be non-empty, finite, and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    device = choose_device(device_spec)
    print(f"Resolved device: {device}", flush=True)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    pieces, mapping = load_recorded_splits(checkpoint, data_path, label_path)
    model = reconstruct_encoder(checkpoint, device)
    features = {name: extract_block_features(model, pair, device, batch_size) for name, pair in pieces.items()}
    # Keep the test labels out of hyperparameter selection entirely.
    targets = {name: _labels(pieces[name], mapping) for name in ("train", "val")}
    class_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (output_root or checkpoint_path.parent.parent / "linear_probes") / stamp
    out.mkdir(parents=True, exist_ok=False)
    reference = _load_original_test_reference(checkpoint_path, checkpoint)
    _json(out / "metadata.json", {"created_at": _now(), "checkpoint": str(checkpoint_path),
          "device": str(device), "sklearn_version": sklearn.__version__, "solver": "lbfgs",
          "multi_class": "multinomial", "class_weight": None, "blocks": list(BLOCKS),
          "pooling": "mean over time; block 3 equals the encoder final global average pool", "C_values": [float(c) for c in c_values],
          "split_mode": checkpoint["split_information"]["mode"], "data_path_override": data_path, "label_path_override": label_path,
          "original_resnet_test_reference": reference})
    selection_rows: list[dict[str, Any]] = []
    selected: dict[int, tuple[float, float]] = {}
    for block in BLOCKS:
        best: Optional[tuple[float, float]] = None  # macro-f1, C
        for c in sorted(set(float(value) for value in c_values)):
            scaler, probe, convergence = _fit_probe(features["train"][block], targets["train"], c, max_iter)
            train_scaled = scaler.transform(features["train"][block])
            validation_scaled = scaler.transform(features["val"][block])
            train_prediction = probe.predict(train_scaled)
            validation_prediction = probe.predict(validation_scaled)
            train_probabilities = _full_probabilities(probe, probe.predict_proba(train_scaled), len(mapping))
            validation_probabilities = _full_probabilities(probe, probe.predict_proba(validation_scaled), len(mapping))
            train_values = _classification_values(targets["train"], train_prediction, train_probabilities, len(mapping))
            validation_values = _classification_values(targets["val"], validation_prediction, validation_probabilities, len(mapping))
            score = validation_values["macro_f1"]
            selection_rows.append({"block": block, "C": c,
                                   **{f"train_{key}": value for key, value in train_values.items()},
                                   **{f"validation_{key}": value for key, value in validation_values.items()},
                                   "convergence_warning": bool(convergence), "warning": " | ".join(convergence)})
            if convergence:
                print(f"WARNING: block {block}, C={c} did not converge: {' | '.join(convergence)}", file=sys.stderr)
            if best is None or score > best[0] or (score == best[0] and c < best[1]):
                best = (score, c)
        assert best is not None
        selected[block] = best

    # Only after every block's C is fixed do we read test labels and test once.
    test_targets = _labels(pieces["test"], mapping)
    for block in BLOCKS:
        validation_score, selected_c = selected[block]
        train_val = np.concatenate((features["train"][block], features["val"][block]))
        target_train_val = np.concatenate((targets["train"], targets["val"]))
        scaler, probe, convergence = _fit_probe(train_val, target_train_val, selected_c, max_iter)
        if convergence:
            print(f"WARNING: final block {block}, C={selected_c} did not converge: {' | '.join(convergence)}", file=sys.stderr)
        test_scaled = scaler.transform(features["test"][block])
        prediction = probe.predict(test_scaled)
        probabilities = _full_probabilities(probe, probe.predict_proba(test_scaled), len(mapping))
        metrics, per_class, raw, normalized = _metrics(test_targets, prediction, class_names)
        metrics["multinomial_log_loss"] = _classification_values(test_targets, prediction, probabilities, len(mapping))["multinomial_log_loss"]
        metrics.update({"block": block, "selected_C": selected_c, "validation_macro_f1": validation_score,
                        "final_convergence_warning": bool(convergence), "final_warning": " | ".join(convergence)})
        block_out = out / f"block{block}"; block_out.mkdir()
        _json(block_out / "final_metrics.json", metrics); _json(block_out / "per_class_metrics.json", per_class)
        with (block_out / "per_class_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_class[0])); writer.writeheader(); writer.writerows(per_class)
        _write_matrix(block_out / "confusion_matrix_raw.csv", raw, class_names)
        _write_matrix(block_out / "confusion_matrix_normalized.csv", normalized, class_names)
        joblib.dump(scaler, block_out / "scaler.joblib")
        joblib.dump(probe, block_out / "probe.joblib")
    with (out / "selection_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0])); writer.writeheader(); writer.writerows(selection_rows)
    _json(out / "selection_metrics.json", selection_rows)
    _plot_selection_curves(out, selection_rows, "accuracy", "selection_accuracy_vs_C.png", reference)
    _plot_selection_curves(out, selection_rows, "macro_f1", "selection_macro_f1_vs_C.png", reference)
    _plot_selection_curves(out, selection_rows, "multinomial_log_loss", "selection_log_loss_vs_C.png", reference)
    final_metrics = [json.loads((out / f"block{block}" / "final_metrics.json").read_text(encoding="utf-8"))
                     for block in BLOCKS]
    _plot_final_test_metrics(out, final_metrics, reference)
    return out


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--data-path", help="Byte-identical relocated automatic-split data CSV")
    result.add_argument("--label-path", help="Byte-identical relocated automatic-split label CSV")
    result.add_argument("--device", default="auto")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--c", dest="c_values", type=float, nargs="+", default=[.01, .1, 1., 10.])
    result.add_argument("--max-iter", type=int, default=1000)
    result.add_argument("--output-root", type=Path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        out = run_linear_probe(args.checkpoint, data_path=args.data_path, label_path=args.label_path,
                               device_spec=args.device, batch_size=args.batch_size, c_values=args.c_values,
                               max_iter=args.max_iter, output_root=args.output_root)
        print(f"Linear-probe artifacts saved to {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
