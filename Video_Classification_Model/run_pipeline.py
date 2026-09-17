#!/usr/bin/env python3
"""Reproducible supervised PyTorch training and evaluation for packet sequences."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             precision_score, recall_score, f1_score)
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from model_resnet_tsc import ResNet
from supervised_data import (PairData, automatic_split, load_pair,
                             manual_split, normalize_rows)

PREPROCESSING = {
    "method": "per_sequence_zscore", "sequence_length": 3000,
    "standard_deviation": "population", "epsilon": None,
}
STATES = ("initializing", "running", "paused", "early_stopped", "complete", "failed")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("configuration root must be a mapping")
    return obj


def validate_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    default = read_yaml(HERE / "configs" / "default.yaml")
    allowed_top = set(default)
    unknown = set(raw) - allowed_top
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    cfg = copy.deepcopy(default)
    for section, values in raw.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"configuration section {section!r} must be a mapping")
        unknown_keys = set(values) - set(cfg[section])
        if unknown_keys:
            raise ValueError(f"unknown keys in {section}: {sorted(unknown_keys)}")
        for key, value in values.items():
            if isinstance(value, Mapping) and isinstance(cfg[section].get(key), Mapping):
                sub_unknown = set(value) - set(cfg[section][key])
                if sub_unknown:
                    raise ValueError(f"unknown keys in {section}.{key}: {sorted(sub_unknown)}")
                cfg[section][key].update(value)
            else:
                cfg[section][key] = value
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    if d["mode"] not in ("auto", "manual"):
        raise ValueError("data.mode must be auto or manual")
    required = (["data_path", "label_path"] if d["mode"] == "auto" else
                [f"{s}_{k}_path" for s in ("train", "val", "test") for k in ("data", "label")])
    missing = [k for k in required if not d.get(k)]
    if missing:
        raise ValueError(f"missing required data paths: {missing}")
    for k in required:
        d[k] = str(Path(d[k]).expanduser().resolve())
    numeric_positive = {
        "model.initial_feature_maps": m["initial_feature_maps"],
        "training.learning_rate": t["learning_rate"], "training.batch_size": t["batch_size"],
        "training.max_epochs": t["max_epochs"], "training.early_stopping.patience": t["early_stopping"]["patience"],
        "training.scheduler.patience": t["scheduler"]["patience"],
        "training.checkpoint_frequency": t["checkpoint_frequency"],
    }
    if any(not isinstance(v, (int, float)) or v <= 0 for v in numeric_positive.values()):
        raise ValueError(f"these configuration values must be positive: {numeric_positive}")
    if not isinstance(t["seed"], int) or t["seed"] < 0 or not isinstance(t["num_workers"], int) or t["num_workers"] < 0:
        raise ValueError("seed and num_workers must be nonnegative integers")
    if t["optimizer"].lower() not in ("adam", "adamw"):
        raise ValueError("optimizer must be Adam or AdamW")
    if t["class_imbalance"] not in ("none", "weighted"):
        raise ValueError("class_imbalance must be none or weighted")
    if t["early_stopping"]["monitor"] != "validation_loss":
        raise ValueError("early_stopping.monitor must be validation_loss")
    if not 0 < float(t["scheduler"]["factor"]) < 1:
        raise ValueError("scheduler.factor must be between 0 and 1")
    if float(t["scheduler"]["min_lr"]) < 0 or float(t["early_stopping"]["min_improvement"]) < 0:
        raise ValueError("minimum learning rate and minimum improvement cannot be negative")
    expected_prep = PREPROCESSING
    if cfg["preprocessing"] != expected_prep:
        raise ValueError(f"preprocessing is fixed for compatibility and must equal {expected_prep}")
    cfg["run"]["output_root"] = str(Path(cfg["run"]["output_root"]).expanduser().resolve())
    return cfg


def dump_yaml(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state() -> Dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: Mapping[str, Any]) -> None:
    """Restore RNG snapshots after normalizing torch states to CPU byte tensors.

    Checkpoints are loaded onto the selected training device, but PyTorch's RNG
    restoration APIs require their serialized state tensors to remain on CPU.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").contiguous()
    torch.set_rng_state(torch_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [torch.as_tensor(item, dtype=torch.uint8, device="cpu").contiguous()
                       for item in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS requested but unavailable")
    return device


def worker_seed(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32); random.seed(torch.initial_seed() % 2**32)


def status_update(run: Path, state: str, cfg: Mapping[str, Any], **changes: Any) -> Dict[str, Any]:
    path = run / "status.json"
    old: Dict[str, Any] = {}
    if path.exists(): old = json.loads(path.read_text())
    old.update({"state": state, "updated_at": now(), "maximum_epochs": cfg["training"]["max_epochs"]})
    old.setdefault("created_at", now()); old.setdefault("completed_epochs", 0)
    old.setdefault("best_epoch", None); old.setdefault("best_validation_loss", None)
    old.setdefault("latest_validation_metrics", None); old.setdefault("checkpoint_paths", {})
    old.setdefault("stop_reason", None); old.update(changes)
    atomic_json(path, old); return old


def label_mapping(labels: Iterable[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(sorted(set(labels)))}


def pair_meta(pair: PairData) -> Dict[str, Any]:
    return {"data_path": str(pair.data_path), "label_path": str(pair.label_path),
            "data_checksum_sha256": pair.data_checksum, "label_checksum_sha256": pair.label_checksum,
            "row_count": len(pair.labels)}


def prepare_new_data(cfg: Mapping[str, Any], run: Path) -> Tuple[Dict[str, PairData], Dict[str, Any], Dict[str, int]]:
    d, seed = cfg["data"], cfg["training"]["seed"]
    if d["mode"] == "auto":
        source = load_pair(d["data_path"], d["label_path"])
        pieces, detail = automatic_split(source, seed=seed)
        manifest = {"mode": "auto", "seed": seed, "ratios": {"train": .70, "validation": .15, "test": .15},
                    "sources": {"source": pair_meta(source)}, "splits": detail}
    else:
        supplied = {name: load_pair(d[f"{name}_data_path"], d[f"{name}_label_path"])
                    for name in ("train", "val", "test")}
        pieces, detail = manual_split(supplied["train"], supplied["val"], supplied["test"])
        manifest = {"mode": "manual", "seed": seed, "sources": {k: pair_meta(v) for k, v in supplied.items()}, "splits": detail}
    mapping = label_mapping(pieces["train"].labels)
    for name, pair in pieces.items():
        manifest["splits"][name]["class_distribution"] = dict(sorted(Counter(pair.labels).items()))
    atomic_json(run / "split_manifest.json", manifest)
    atomic_json(run / "input_checksums.json", manifest["sources"])
    atomic_json(run / "label_mapping.json", mapping)
    return pieces, manifest, mapping


def verify_source(meta: Mapping[str, Any]) -> PairData:
    pair = load_pair(meta["data_path"], meta["label_path"])
    if pair.data_checksum != meta["data_checksum_sha256"] or pair.label_checksum != meta["label_checksum_sha256"]:
        raise ValueError(f"source files changed since training: {meta['data_path']}")
    return pair


def subset_pair(pair: PairData, indices: Sequence[int]) -> PairData:
    idx = np.asarray(indices, dtype=np.int64)
    return pair.subset(idx)


def load_saved_splits(run: Path) -> Tuple[Dict[str, PairData], Dict[str, Any], Dict[str, int]]:
    manifest = json.loads((run / "split_manifest.json").read_text())
    mapping = {str(k): int(v) for k, v in json.loads((run / "label_mapping.json").read_text()).items()}
    pieces: Dict[str, PairData] = {}
    if manifest["mode"] == "auto":
        source = verify_source(manifest["sources"]["source"])
        for name in ("train", "val", "test"):
            pieces[name] = subset_pair(source, manifest["splits"][name]["source_indices"])
    else:
        for name in ("train", "val", "test"):
            pieces[name] = verify_source(manifest["sources"][name])
    for name, pair in pieces.items():
        if list(pair.row_hashes) != manifest["splits"][name]["row_hashes"]:
            raise ValueError(f"saved {name} rows no longer match split manifest")
    return pieces, manifest, mapping


def arrays(pair: PairData, mapping: Mapping[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    unknown = sorted(set(pair.labels) - set(mapping))
    if unknown: raise ValueError(f"labels not present in checkpoint mapping: {unknown}")
    return normalize_rows(pair.values)[:, None, :], np.asarray([mapping[x] for x in pair.labels], dtype=np.int64)


def loader(pair: PairData, mapping: Mapping[str, int], batch: int, shuffle: bool,
           workers: int, seed: int) -> DataLoader:
    x, y = arrays(pair, mapping)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch,
                      shuffle=shuffle, num_workers=workers,
                      worker_init_fn=worker_seed if workers else None, persistent_workers=workers > 0)


def epoch_pass(model: nn.Module, dl: DataLoader, criterion: nn.Module, device: torch.device,
               optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
    model.train(optimizer is not None); total_loss = 0.0; truth = []; pred = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            if optimizer is not None: loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            truth.extend(y.detach().cpu().tolist()); pred.extend(logits.argmax(1).detach().cpu().tolist())
    p, r, f, _ = precision_recall_fscore_support(truth, pred, average="macro", zero_division=0)
    return {"loss": total_loss / len(dl.dataset), "accuracy": accuracy_score(truth, pred),
            "macro_precision": p, "macro_recall": r, "macro_f1": f}


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau,
                       completed: int, best_epoch: Optional[int], best: float, counter: int,
                       cfg: Mapping[str, Any], mapping: Mapping[str, int], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {"format_version": 1, "saved_at": now(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "completed_epoch": completed, "best_epoch": best_epoch, "best_metric": best,
            "early_stopping_counter": counter, "model_config": cfg["model"],
            "training_config": cfg["training"], "preprocessing_config": cfg["preprocessing"],
            "label_mapping": dict(mapping), "split_information": manifest, "rng_states": rng_state()}


def write_history(run: Path, history: Sequence[Mapping[str, Any]]) -> None:
    atomic_json(run / "metrics.json", list(history))
    if history:
        tmp = run / "metrics.csv.tmp"
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        os.replace(tmp, run / "metrics.csv")
        for stem, keys, ylabel in (("loss_curves", ("train_loss", "val_loss"), "loss"),
                                   ("accuracy_curves", ("train_accuracy", "val_accuracy"), "accuracy"),
                                   ("macro_f1_curves", ("train_macro_f1", "val_macro_f1"), "macro F1")):
            plt.figure()
            for key in keys: plt.plot([x[key] for x in history], label=key)
            plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
            plt.savefig(run / f"{stem}.png", dpi=140); plt.close()


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    cp = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state", "optimizer_state", "scheduler_state", "completed_epoch", "model_config",
                "preprocessing_config", "label_mapping", "split_information", "rng_states"}
    missing = required - set(cp)
    if missing: raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    if cp["preprocessing_config"] != PREPROCESSING: raise ValueError("checkpoint preprocessing is incompatible")
    return cp


def class_weights(labels: Sequence[str], mapping: Mapping[str, int], device: torch.device) -> torch.Tensor:
    counts = Counter(labels); n, k = len(labels), len(mapping)
    return torch.tensor([n / (k * counts[label]) for label, _ in sorted(mapping.items(), key=lambda x: x[1])],
                        dtype=torch.float32, device=device)


def train(run: Path, cfg: Dict[str, Any], resume_checkpoint: Optional[Path] = None) -> None:
    log = run / "logs" / "training.log"; log.parent.mkdir(exist_ok=True)
    def say(msg: str) -> None:
        print(msg, flush=True)
        with log.open("a") as f: f.write(f"{now()} {msg}\n")
    device = choose_device(cfg["training"]["device"]); seed_all(cfg["training"]["seed"])
    pieces, manifest, mapping = load_saved_splits(run)
    tr = cfg["training"]; model = ResNet((1, 3000), len(mapping), cfg["model"]["initial_feature_maps"]).to(device)
    count = sum(p.numel() for p in model.parameters()); (run / "parameter_count.txt").write_text(str(count) + "\n")
    opt_cls = Adam if tr["optimizer"].lower() == "adam" else AdamW
    optimizer = opt_cls(model.parameters(), lr=float(tr["learning_rate"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=float(tr["scheduler"]["factor"]),
                                  patience=int(tr["scheduler"]["patience"]), min_lr=float(tr["scheduler"]["min_lr"]))
    weights = class_weights(pieces["train"].labels, mapping, device) if tr["class_imbalance"] == "weighted" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_dl = loader(pieces["train"], mapping, tr["batch_size"], True, tr["num_workers"], tr["seed"])
    val_dl = loader(pieces["val"], mapping, tr["batch_size"], False, tr["num_workers"], tr["seed"])
    history = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else []
    completed, best_epoch, best, counter = 0, None, float("inf"), 0
    if resume_checkpoint:
        cp = load_checkpoint(resume_checkpoint, device)
        if (cp["model_config"] != cfg["model"] or cp["preprocessing_config"] != cfg["preprocessing"]
                or cp["label_mapping"] != mapping or cp["split_information"] != manifest):
            raise ValueError("resume rejected: model architecture, preprocessing, label mapping, or saved split changed")
        runtime_keys = {"max_epochs", "device", "num_workers"}
        incompatible_training = [key for key in cp["training_config"]
                                 if key not in runtime_keys and cp["training_config"][key] != tr[key]]
        if incompatible_training:
            raise ValueError(f"resume rejected: incompatible training settings changed: {incompatible_training}")
        model.load_state_dict(cp["model_state"]); optimizer.load_state_dict(cp["optimizer_state"]); scheduler.load_state_dict(cp["scheduler_state"])
        completed, best_epoch, best, counter = cp["completed_epoch"], cp.get("best_epoch"), cp.get("best_metric", float("inf")), cp.get("early_stopping_counter", 0)
        history = [row for row in history if row["epoch"] <= completed]
        write_history(run, history)
        restore_rng(cp["rng_states"])
    ckdir = run / "checkpoints"; ckdir.mkdir(exist_ok=True)
    status_update(run, "running", cfg, completed_epochs=completed, parameter_count=count, stop_reason=None)
    say(f"Device: {device}; parameters: {count:,}; resuming after epoch {completed}")
    initial_boundary = checkpoint_payload(model, optimizer, scheduler, completed, best_epoch, best, counter, cfg, mapping, manifest)
    initial_boundary_path = ckdir / ".initial_boundary.pt"
    atomic_torch_save(initial_boundary_path, initial_boundary)
    serialized_boundary_path = initial_boundary_path
    try:
        stop_state, stop_reason = "complete", "maximum_epochs_reached"
        for epoch in range(completed + 1, int(tr["max_epochs"]) + 1):
            started = time.perf_counter()
            tm = epoch_pass(model, train_dl, criterion, device, optimizer)
            vm = epoch_pass(model, val_dl, criterion, device)
            scheduler.step(vm["loss"])
            previous_best = best
            raw_better = vm["loss"] < previous_best
            significant = vm["loss"] < previous_best - float(tr["early_stopping"]["min_improvement"])
            if raw_better: best, best_epoch = vm["loss"], epoch
            counter = 0 if significant else counter + 1
            row = {"epoch": epoch, "train_loss": tm["loss"], "val_loss": vm["loss"],
                   "train_accuracy": tm["accuracy"], "val_accuracy": vm["accuracy"],
                   "train_macro_f1": tm["macro_f1"], "val_macro_f1": vm["macro_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"], "duration_seconds": time.perf_counter() - started}
            history.append(row); write_history(run, history)
            payload = checkpoint_payload(model, optimizer, scheduler, epoch, best_epoch, best, counter, cfg, mapping, manifest)
            if raw_better: atomic_torch_save(ckdir / "best.pt", payload)
            atomic_torch_save(ckdir / "latest.pt", payload)
            serialized_boundary_path = ckdir / "latest.pt"
            if epoch % int(tr["checkpoint_frequency"]) == 0: atomic_torch_save(ckdir / f"epoch_{epoch:04d}.pt", payload)
            status_update(run, "running", cfg, completed_epochs=epoch, best_epoch=best_epoch,
                          best_validation_loss=best, latest_validation_metrics=vm,
                          checkpoint_paths={"best": str(ckdir / "best.pt"), "latest": str(ckdir / "latest.pt")})
            say(f"Epoch {epoch}/{tr['max_epochs']} train_loss={tm['loss']:.6f} val_loss={vm['loss']:.6f} val_f1={vm['macro_f1']:.4f}")
            if counter >= int(tr["early_stopping"]["patience"]):
                stop_state, stop_reason = "early_stopped", "validation_loss_patience_exhausted"; break
        status_update(run, stop_state, cfg, completed_epochs=history[-1]["epoch"] if history else completed,
                      best_epoch=best_epoch, best_validation_loss=None if best == float("inf") else best, stop_reason=stop_reason)
    except KeyboardInterrupt:
        # Copy a serialized boundary: state_dict tensors held in memory share model storage.
        atomic_copy(serialized_boundary_path, ckdir / "interrupted.pt")
        boundary = load_checkpoint(ckdir / "interrupted.pt", torch.device("cpu"))
        history = [row for row in history if row["epoch"] <= boundary["completed_epoch"]]
        write_history(run, history)
        boundary_best = boundary["best_metric"]
        status_update(run, "paused", cfg, completed_epochs=boundary["completed_epoch"],
                      best_epoch=boundary["best_epoch"],
                      best_validation_loss=None if boundary_best == float("inf") else boundary_best,
                      checkpoint_paths={"best": str(ckdir / "best.pt"), "latest": str(ckdir / "latest.pt"),
                                        "interrupted": str(ckdir / "interrupted.pt")}, stop_reason="keyboard_interrupt")
        initial_boundary_path.unlink(missing_ok=True)
        say("Interrupted safely at the last completed epoch boundary."); return
    except Exception as exc:
        initial_boundary_path.unlink(missing_ok=True)
        status_update(run, "failed", cfg, stop_reason=f"training_failed: {exc}")
        raise
    initial_boundary_path.unlink(missing_ok=True)
    try:
        evaluate_checkpoint(ckdir / "best.pt", original=True)
    except Exception as exc:
        status_update(run, "failed", cfg, stop_reason=f"final_test_evaluation_failed: {exc}")
        raise


def evaluation_metrics(truth: Sequence[int], pred: Sequence[int], probabilities: np.ndarray,
                       labels: Sequence[str], loss: float, duration: float, parameter_count: int) -> Tuple[Dict[str, Any], list, np.ndarray, np.ndarray]:
    ids = list(range(len(labels))); raw = confusion_matrix(truth, pred, labels=ids)
    norm = np.divide(raw, raw.sum(axis=1, keepdims=True), out=np.zeros_like(raw, dtype=float), where=raw.sum(axis=1, keepdims=True) != 0)
    p, r, f, support = precision_recall_fscore_support(truth, pred, labels=ids, zero_division=0)
    no_predictions = [labels[i] for i in ids if raw[:, i].sum() == 0]
    metrics: Dict[str, Any] = {"test_loss": loss, "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "macro_precision": precision_score(truth, pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(truth, pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(truth, pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(truth, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
        "top_3_accuracy": (float(np.mean([target in np.argsort(row)[-3:]
                                           for target, row in zip(truth, probabilities)]))
                           if len(labels) >= 3 else None),
        "parameter_count": parameter_count, "evaluation_duration_seconds": duration,
        "samples_per_second": len(truth) / duration if duration else None,
        "zero_division_handling": {"value_used": 0, "classes_with_no_predictions": no_predictions,
                                    "occurred": bool(no_predictions)}, "sample_count": len(truth)}
    per_class = [{"class": labels[i], "precision": p[i], "recall": r[i], "f1": f[i], "sample_count": int(support[i])} for i in ids]
    return metrics, per_class, raw, norm


def save_matrix(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix): w.writerow([label, *row.tolist()])


def plot_matrix(path: Path, matrix: np.ndarray, labels: Sequence[str], title: str, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * .65), max(6, len(labels) * .6)))
    image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if len(labels) <= 20:
        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(len(labels)):
            for j in range(len(labels)): ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i,j] > threshold else "black", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_checkpoint(checkpoint: Path, original: bool, data_path: Optional[str] = None,
                        label_path: Optional[str] = None, device_spec: Optional[str] = None) -> Path:
    checkpoint = checkpoint.expanduser().resolve(); run = checkpoint.parent.parent
    cfg = validate_config(read_yaml(run / "config.yaml")); device = choose_device(device_spec or cfg["training"]["device"])
    cp = load_checkpoint(checkpoint, device); mapping = {str(k): int(v) for k, v in cp["label_mapping"].items()}
    if original:
        pieces, manifest, saved_mapping = load_saved_splits(run); pair = pieces["test"]
        if mapping != saved_mapping: raise ValueError("checkpoint label mapping differs from run label mapping")
        kind = "original_test"
    else:
        if not data_path or not label_path: raise ValueError("external evaluation requires data and label paths")
        pair = load_pair(data_path, label_path); kind = "external_test"
    x, y = arrays(pair, mapping); dl = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])
    model = ResNet((1, 3000), len(mapping), cp["model_config"]["initial_feature_maps"]).to(device); model.load_state_dict(cp["model_state"]); model.eval()
    criterion = nn.CrossEntropyLoss(); total = 0.; truth = []; probs = []; started = time.perf_counter()
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device); logits = model(xb)
            total += float(criterion(logits, yb)) * len(yb); truth.extend(yb.cpu().tolist()); probs.append(torch.softmax(logits, 1).cpu().numpy())
    duration = time.perf_counter() - started; probabilities = np.concatenate(probs); pred = probabilities.argmax(1)
    labels = [x for x, _ in sorted(mapping.items(), key=lambda z: z[1])]
    metrics, per_class, raw, norm = evaluation_metrics(truth, pred, probabilities, labels, total / len(y), duration, sum(p.numel() for p in model.parameters()))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); out = run / "evaluations" / f"{stamp}_{kind}"; out.mkdir(parents=True, exist_ok=False)
    metrics.update({"created_at": now(), "checkpoint": str(checkpoint), "evaluation_kind": kind,
                    "data_path": str(pair.data_path), "label_path": str(pair.label_path),
                    "data_checksum_sha256": pair.data_checksum, "label_checksum_sha256": pair.label_checksum})
    atomic_json(out / "metrics.json", metrics); atomic_json(out / "per_class_metrics.json", per_class)
    with (out / "per_class_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0])); w.writeheader(); w.writerows(per_class)
    save_matrix(out / "confusion_matrix_raw.csv", raw, labels); save_matrix(out / "confusion_matrix_normalized.csv", norm, labels)
    plot_matrix(out / "confusion_matrix_raw.png", raw, labels, "Confusion matrix", "d")
    plot_matrix(out / "confusion_matrix_normalized.png", norm, labels, "Row-normalized confusion matrix", ".2f")
    if original:
        # Convenience copies describe the final test while timestamped evaluations remain immutable.
        copies = [(out / "metrics.json", run / "test_metrics.json"),
                  (out / "per_class_metrics.csv", run / "test_per_class_metrics.csv")]
        copies.extend((out / n, run / f"test_{n}") for n in
                      ("confusion_matrix_raw.csv", "confusion_matrix_normalized.csv",
                       "confusion_matrix_raw.png", "confusion_matrix_normalized.png"))
        for source, destination in copies:
            if not destination.exists(): shutil.copy2(source, destination)
    print(f"Evaluation saved to {out}"); return out


def make_run(cfg: Dict[str, Any], run_name: Optional[str]) -> Path:
    name = run_name or cfg["run"].get("name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if Path(name).name != name or name in ("", ".", ".."): raise ValueError("run name must be a single safe path component")
    run = Path(cfg["run"]["output_root"]) / name; run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir(); (run / "checkpoints").mkdir()
    cfg["run"]["name"] = name; dump_yaml(run / "config.yaml", cfg)
    status_update(run, "initializing", cfg, stop_reason=None)
    try: prepare_new_data(cfg, run)
    except Exception as exc:
        status_update(run, "failed", cfg, stop_reason=f"initialization_failed: {exc}"); raise
    return run


def check_resume_config(saved: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {("training", "max_epochs"), ("training", "device"), ("training", "num_workers")}
    differences = []
    for section in saved:
        for key in saved[section]:
            if saved[section][key] != candidate[section][key] and (section, key) not in allowed:
                differences.append(f"{section}.{key}")
    if differences: raise ValueError(f"incompatible resume configuration changes: {differences}; only max_epochs, device, and num_workers may change")
    completed = json.loads((Path(saved["run"]["output_root"]) / saved["run"]["name"] / "status.json").read_text()).get("completed_epochs", 0)
    if candidate["training"]["max_epochs"] <= completed: raise ValueError(f"max_epochs must exceed completed epochs ({completed})")
    return candidate


def resume(run: Path, config_path: Optional[Path], checkpoint: Optional[Path]) -> None:
    run = run.expanduser().resolve(); saved = validate_config(read_yaml(run / "config.yaml"))
    cfg = check_resume_config(saved, validate_config(read_yaml(config_path))) if config_path else saved
    cp = checkpoint or ((run / "checkpoints" / "interrupted.pt") if (run / "checkpoints" / "interrupted.pt").exists() else (run / "checkpoints" / "latest.pt"))
    loaded = load_checkpoint(cp, torch.device("cpu"))
    if cfg["training"]["max_epochs"] < loaded["training_config"]["max_epochs"]:
        raise ValueError("maximum epochs may not be decreased on resume")
    dump_yaml(run / "config.yaml", cfg); train(run, cfg, cp)


def ask_path(prompt: str) -> str: return str(Path(input(prompt).strip()).expanduser().resolve())


def menu() -> None:
    while True:
        print("\n1. Train a new model\n2. Resume a paused or interrupted run\n3. Evaluate a checkpoint on its original saved test split\n4. Evaluate a checkpoint on another labeled test set\n5. Exit")
        choice = input("Select an action: ").strip()
        try:
            if choice == "1":
                config_path = Path(input(f"Configuration YAML [{HERE / 'configs/default.yaml'}]: ").strip() or HERE / "configs/default.yaml")
                raw = read_yaml(config_path); mode = input("Training data: 1) automatic split  2) separate train/validation/test pairs: ").strip()
                raw.setdefault("data", {})["mode"] = "auto" if mode == "1" else "manual"
                if mode == "1": raw["data"].update(data_path=ask_path("Data CSV: "), label_path=ask_path("Label CSV: "))
                else:
                    for split in ("train", "val", "test"):
                        raw["data"][f"{split}_data_path"] = ask_path(f"{split.title()} data CSV: ")
                        raw["data"][f"{split}_label_path"] = ask_path(f"{split.title()} label CSV: ")
                cfg = validate_config(raw); run = make_run(cfg, input("Run name (blank for timestamp): ").strip() or None); train(run, cfg)
            elif choice == "2":
                run = Path(ask_path("Run directory: ")); text = input("Optional resume YAML (blank uses saved config): ").strip()
                resume(run, Path(text).expanduser().resolve() if text else None, None)
            elif choice == "3": evaluate_checkpoint(Path(ask_path("Checkpoint path: ")), True)
            elif choice == "4": evaluate_checkpoint(Path(ask_path("Checkpoint path: ")), False, ask_path("Test data CSV: "), ask_path("Test label CSV: "))
            elif choice == "5": return
            else: print("Invalid selection.")
        except Exception as exc: print(f"ERROR: {exc}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    n = sub.add_parser("train"); n.add_argument("--config", required=True, type=Path); n.add_argument("--run-name")
    r = sub.add_parser("resume"); r.add_argument("--run", required=True, type=Path); r.add_argument("--config", type=Path); r.add_argument("--checkpoint", type=Path)
    e = sub.add_parser("evaluate-original"); e.add_argument("--checkpoint", required=True, type=Path); e.add_argument("--device")
    x = sub.add_parser("evaluate-other"); x.add_argument("--checkpoint", required=True, type=Path); x.add_argument("--data", required=True); x.add_argument("--labels", required=True); x.add_argument("--device")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None and len(sys.argv) == 1: menu(); return 0
    args = parser().parse_args(argv)
    try:
        if args.command == "train":
            cfg = validate_config(read_yaml(args.config)); run = make_run(cfg, args.run_name); train(run, cfg)
        elif args.command == "resume": resume(args.run, args.config, args.checkpoint)
        elif args.command == "evaluate-original": evaluate_checkpoint(args.checkpoint, True, device_spec=args.device)
        else: evaluate_checkpoint(args.checkpoint, False, args.data, args.labels, args.device)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("ENTROPY_DEBUG"): traceback.print_exc()
        return 1


if __name__ == "__main__": raise SystemExit(main())
