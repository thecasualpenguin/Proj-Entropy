"""Deterministic epoch-boundary training for the native flat WPP model."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .classification_metrics import classification_scores, per_class_scores
from .manifest_contract import sha256_file
from .model_factory import make_model, validate_model_config
from .wpp_reporting import REPORT_FILES, report_evaluation, write_evaluations
from .wpp_dataset import WPPDatasetBundle, WPPVideoDataset, collate_wpp_videos
from .wpp_normalization import (NormalizationError, WelfordPopulation, normalize_torch,
                                validate_statistics)

CHECKPOINT_CONTRACT = "native_wpp_checkpoint"
CHECKPOINT_VERSION = "1.0"
_REPORT_STATE = ".native-wpp-evaluations-state.json"
_REPORT_JOURNAL = ".native-wpp-evaluations-transaction.json"
_REPORT_ARCHIVES = ".native-wpp-evaluations-archives"
_REPORT_STATE_CONTRACT = "native_wpp_managed_evaluations"
_REPORT_STATE_VERSION = "1.0"


class TrainingError(ValueError):
    """A clear native-run configuration, provenance, or lifecycle failure."""


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingError(f"cannot read configuration: {path}") from exc
    if not isinstance(value, dict):
        raise TrainingError("configuration root must be a mapping")
    return value


def _merge_strict(default: Mapping[str, Any], supplied: Mapping[str, Any], where: str = "") -> dict[str, Any]:
    if not isinstance(supplied, Mapping):
        raise TrainingError(f"{where or 'configuration'} must be a mapping")
    extra = set(supplied) - set(default)
    if extra:
        raise TrainingError(f"unsupported configuration keys in {where or 'root'}: {sorted(extra)}")
    result: dict[str, Any] = {}
    for key, fallback in default.items():
        value = supplied.get(key, fallback)
        label = f"{where}.{key}" if where else key
        if isinstance(fallback, Mapping):
            result[key] = _merge_strict(fallback, value, label)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainingError(f"{name} must be finite")
    value = float(value)
    if value < 0 or (value == 0 and not allow_zero):
        raise TrainingError(f"{name} must be positive")
    return value


def validate_config(raw: Mapping[str, Any], default_path: str | Path) -> dict[str, Any]:
    config = _merge_strict(_read_yaml(Path(default_path)), raw)
    run, data, model, training, preprocessing = (config[name] for name in ("run", "data", "model", "training", "preprocessing"))
    if not isinstance(run["output_dir"], str) or not run["output_dir"].strip():
        raise TrainingError("run.output_dir must be a nonempty explicit path")
    for key in ("packed_bundle", "source_manifest", "index"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise TrainingError(f"data.{key} must be an explicit path")
    config["model"] = validate_model_config(model)
    _positive_int(training["seed"], "training.seed")
    training["learning_rate"] = _finite_positive(training["learning_rate"], "training.learning_rate")
    _positive_int(training["batch_size"], "training.batch_size")
    _positive_int(training["max_epochs"], "training.max_epochs")
    if training["optimizer"] not in ("Adam", "AdamW"):
        raise TrainingError("training.optimizer must be Adam or AdamW")
    _positive_int(training["checkpoint_frequency"], "training.checkpoint_frequency")
    if training["device"] not in ("auto", "cpu") and not (isinstance(training["device"], str) and training["device"].startswith("cuda")):
        raise TrainingError("training.device must be auto, cpu, or cuda[:N]")
    if training["num_workers"] != 0:
        raise TrainingError("native WPP runner owns dictionary batches; training.num_workers must be 0")
    early = training["early_stopping"]
    if set(early) != {"patience", "min_improvement"}:
        raise TrainingError("training.early_stopping keys are invalid")
    _positive_int(early["patience"], "training.early_stopping.patience")
    early["min_improvement"] = _finite_positive(early["min_improvement"], "training.early_stopping.min_improvement", allow_zero=True)
    scheduler = training["scheduler"]
    if set(scheduler) != {"factor", "patience", "min_lr"}:
        raise TrainingError("training.scheduler keys are invalid")
    scheduler["factor"] = _finite_positive(scheduler["factor"], "training.scheduler.factor")
    if scheduler["factor"] >= 1:
        raise TrainingError("training.scheduler.factor must be less than one")
    _positive_int(scheduler["patience"], "training.scheduler.patience")
    scheduler["min_lr"] = _finite_positive(scheduler["min_lr"], "training.scheduler.min_lr", allow_zero=True)
    expected_policy = {"method": "global_valid_train_zscore", "statistics_dtype": "float64",
                       "standard_deviation": "population", "ddof": 0,
                       "fit_split": "train", "valid_cells": "explicit_row_mask_and_time_mask",
                       "observed_zero": "included", "padding": "zero", "log_transform": None,
                       "clipping": None, "per_row_normalization": False}
    if preprocessing != expected_policy:
        raise TrainingError("preprocessing must be exactly the approved native WPP masked train-only policy")
    return config


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is unavailable")
    return device


def _json_primitive(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_primitive(item) for item in value]
    if isinstance(value, list):
        return [_json_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TrainingError(f"unsupported RNG state value: {type(value).__name__}")


def _as_tuple(value: Any) -> Any:
    return tuple(_as_tuple(item) for item in value) if isinstance(value, list) else value


def _rng_state(shuffle: torch.Generator) -> dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": _json_primitive(random.getstate()),
            "numpy": {"name": name, "keys": keys.tolist(), "position": int(position),
                      "has_gauss": int(has_gauss), "cached": float(cached)},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
            "shuffle_generator": shuffle.get_state()}


def _restore_rng(value: Mapping[str, Any], shuffle: torch.Generator) -> None:
    if not isinstance(value, Mapping) or set(value) != {"python", "numpy", "torch_cpu", "torch_cuda", "shuffle_generator"}:
        raise TrainingError("checkpoint RNG state is invalid")
    try:
        random.setstate(_as_tuple(value["python"]))
        np_state = value["numpy"]
        np.random.set_state((np_state["name"], np.asarray(np_state["keys"], dtype=np.uint32),
                             int(np_state["position"]), int(np_state["has_gauss"]), float(np_state["cached"])))
        torch.set_rng_state(value["torch_cpu"])
        if torch.cuda.is_available() and value["torch_cuda"]:
            torch.cuda.set_rng_state_all(list(value["torch_cuda"]))
        shuffle.set_state(value["shuffle_generator"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise TrainingError("checkpoint RNG state cannot be restored") from exc


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _bundle(config: Mapping[str, Any]) -> WPPDatasetBundle:
    data = config["data"]
    try:
        bundle = WPPDatasetBundle(data["packed_bundle"], source_manifest=data["source_manifest"], index=data["index"])
    except (OSError, ValueError) as exc:
        raise TrainingError(f"native WPP input verification failed: {exc}") from exc
    if not bundle.records_by_partition["train"] or not bundle.records_by_partition["validation"]:
        raise TrainingError("native training requires nonempty upstream train and validation partitions")
    return bundle


def _provenance(bundle: WPPDatasetBundle) -> dict[str, Any]:
    source_meta = bundle.source["metadata"]
    index_meta = bundle.extraction["metadata"]
    ledger = index_meta.get("native_workflow_ledger")
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("sha256"), str):
        raise TrainingError("trusted index has no native ledger binding")
    return {
        "bundle_path": str(bundle.directory),
        "source_manifest_path": str(bundle.source_manifest),
        "index_path": str(bundle.index),
        "bundle_csv_sha256": dict(bundle.metadata["output_checksums"]),
        "source_manifest_records_sha256": source_meta["records_sha256"],
        "source_manifest_metadata_sha256": sha256_file(bundle.source_manifest / "source_manifest.metadata.json"),
        "index_records_sha256": index_meta["records_sha256"],
        "index_metadata_sha256": sha256_file(bundle.index / "extraction_index.metadata.json"),
        "native_ledger_sha256": ledger["sha256"],
        "class_mapping": dict(bundle.metadata["class_mapping"]),
    }


def fit_train_statistics(dataset: WPPVideoDataset) -> dict[str, Any]:
    moments = WelfordPopulation(); expected = 0
    for position in range(len(dataset)):
        item = dataset[position]
        values = item["values"][None, :, :]
        rows = np.ones((1, item["row_count"]), dtype=bool)
        times = np.zeros((1, values.shape[2]), dtype=bool); times[:, :item["used_time_length"]] = True
        expected += int(item["row_count"]) * int(item["used_time_length"])
        moments.update(values, rows, times)
    return moments.freeze(expected_count=expected)


def _batches(dataset: WPPVideoDataset, batch_size: int, generator: torch.Generator, *, shuffle: bool) -> Iterable[dict[str, Any]]:
    order = torch.randperm(len(dataset), generator=generator).tolist() if shuffle else list(range(len(dataset)))
    for start in range(0, len(order), batch_size):
        yield collate_wpp_videos([dataset[index] for index in order[start:start + batch_size]])


def _tensor_batch(batch: Mapping[str, Any], device: torch.device, statistics: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(batch["values"]).to(device=device, dtype=torch.float32)
    rows = torch.from_numpy(batch["row_mask"]).to(device=device)
    times = torch.from_numpy(batch["time_mask"]).to(device=device)
    labels = torch.from_numpy(batch["labels"]).to(device=device, dtype=torch.long)
    return normalize_torch(values, rows, times, statistics), rows, times, labels


def evaluate_model(model: nn.Module, dataset: WPPVideoDataset, *, batch_size: int, device: torch.device,
                   statistics: Mapping[str, Any], class_names: list[str], retain_predictions: bool = False,
                   detailed: bool = True) -> dict[str, Any]:
    """Evaluate one partition, timing only its inference/loss loop.

    Aggregate scores are intentionally computed after the loop timer; callers
    timing an epoch still include those scores in their own epoch boundary.
    Per-class rows and the confusion matrix are optional because validation
    selection never needs them and reference epoch timing excludes them.
    """
    if not len(dataset):
        raise TrainingError(f"requested evaluation partition is empty: {dataset.partition}")
    model.eval(); truths: list[int] = []; predictions: list[int] = []; loss_sum = 0.0
    generator = torch.Generator(device="cpu"); generator.manual_seed(0)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in _batches(dataset, batch_size, generator, shuffle=False):
            values, rows, times, labels = _tensor_batch(batch, device, statistics)
            logits = model(values, rows, times, normalized=True)
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            truths.extend(labels.cpu().tolist()); predictions.extend(logits.argmax(dim=1).cpu().tolist())
    inference_duration = time.perf_counter() - started
    scores = classification_scores(truths, predictions)
    result: dict[str, Any] = {"partition": dataset.partition, "sample_count": len(truths), "cross_entropy": loss_sum / len(truths),
                              "metrics": scores, "class_names": class_names,
                              "_inference_duration_seconds": inference_duration}
    if detailed:
        matrix = [[sum(1 for truth, prediction in zip(truths, predictions) if truth == row and prediction == column)
                   for column in range(len(class_names))] for row in range(len(class_names))]
        result["per_class"] = per_class_scores(truths, predictions, class_names)
        result["confusion_matrix"] = matrix
    # Kept private to this process and consumed by the reporter; callers writing
    # native JSON must exclude these lists and timing implementation detail.
    if retain_predictions:
        result["_truths"] = truths
        result["_predictions"] = predictions
    return result


def _public_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Strip process-only prediction lists before persisting native JSON."""
    return {key: value for key, value in result.items() if not str(key).startswith("_")}


def _partition_identity(bundle: WPPDatasetBundle, partition: str) -> dict[str, str]:
    names = (f"{partition}-data.csv", f"{partition}-label-manifest.csv")
    checksums = bundle.metadata["output_checksums"]
    if any(name not in checksums for name in names):
        raise TrainingError("native bundle lacks partition checksum metadata")
    return {"data_path": str(bundle.directory / names[0]), "data_checksum": checksums[names[0]],
            "label_path": str(bundle.directory / names[1]), "label_checksum": checksums[names[1]]}


def _write_compatibility_report(report_root: Path, *, bundle: WPPDatasetBundle,
                                history: list[dict[str, Any]], result: Mapping[str, Any], checkpoint: Path,
                                parameter_count: int, evaluation_kind: str, duration_seconds: float) -> None:
    identity = _partition_identity(bundle, str(result["partition"]))
    summary, per_class, raw, normalized = report_evaluation(
        result, checkpoint=str(checkpoint.resolve()), parameter_count=parameter_count,
        evaluation_kind=evaluation_kind, duration_seconds=duration_seconds, **identity)
    write_evaluations(report_root / "evaluations", history=history, summary=summary, per_class=per_class,
                      raw=raw, normalized=normalized, class_names=result["class_names"])


def _report_path(run: Path, name: str) -> Path:
    if "/" in name or "\\" in name:
        raise TrainingError("invalid managed report path")
    path = run / name
    if path.is_symlink():
        raise TrainingError(f"managed report path is a symlink: {path}")
    return path


def _report_checksums(directory: Path) -> dict[str, str]:
    expected = set(REPORT_FILES)
    if directory.is_symlink() or not directory.is_dir():
        raise TrainingError("managed evaluations directory is not an ordinary directory")
    actual = {item.name for item in directory.iterdir()}
    if actual != expected or any((directory / name).is_symlink() or not (directory / name).is_file() for name in expected):
        raise TrainingError("managed evaluations directory is foreign or noncanonical")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sorted(expected)}


def _report_identity(run: Path) -> str:
    path = run / "provenance.json"
    if not path.is_file() or path.is_symlink():
        raise TrainingError("managed report requires a regular run provenance.json")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_report_state(run: Path) -> dict[str, Any] | None:
    path = _report_path(run, _REPORT_STATE)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("managed report state is malformed") from exc
    if not isinstance(value, dict) or value.get("contract") != _REPORT_STATE_CONTRACT or value.get("version") != _REPORT_STATE_VERSION:
        raise TrainingError("managed report state is unsupported")
    required = {"contract", "version", "generation", "provenance_sha256", "files"}
    if set(value) != required or not isinstance(value["generation"], int) or value["generation"] < 1:
        raise TrainingError("managed report state fields are invalid")
    if not isinstance(value["provenance_sha256"], str) or not isinstance(value["files"], dict):
        raise TrainingError("managed report state fields are invalid")
    return value


def _validate_legacy_test_record(run: Path, provenance: Mapping[str, Any]) -> bool:
    """Recognize only the previous native test.json layout, never arbitrary files."""
    legacy = run / "evaluations" / "test.json"
    try:
        value = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, Mapping) and value.get("provenance") == provenance and isinstance(value.get("checkpoint"), str)


def _recover_managed_report(run: Path) -> None:
    """Recover a journaled archive/publish interruption before training mutates state."""
    journal_path = _report_path(run, _REPORT_JOURNAL)
    if not journal_path.exists():
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        phase, archive_name, stage_name = journal["phase"], journal["archive"], journal["stage"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TrainingError("managed report transaction journal is malformed") from exc
    if (phase not in ("prepared", "archived", "published") or not isinstance(archive_name, str)
            or not isinstance(stage_name, str) or "/" in archive_name or "\\" in archive_name
            or not stage_name or "/" in stage_name or "\\" in stage_name):
        raise TrainingError("managed report transaction journal is invalid")
    evaluations, archives = _report_path(run, "evaluations"), _report_path(run, _REPORT_ARCHIVES)
    archive = (archives / archive_name) if archive_name else None
    stage = run / stage_name
    if ((archive is not None and archive.is_symlink()) or stage.is_symlink()
            or (archives.exists() and not archives.is_dir())):
        raise TrainingError("managed report transaction paths are unsafe")
    state_path = stage / "state.json"
    if phase in ("prepared", "archived"):
        # The initial-generation journal has no old archive. Its fully staged
        # public directory can safely be finalized with the staged state.
        if archive is None and evaluations.is_dir() and state_path.is_file():
            _atomic_json(_report_path(run, _REPORT_STATE), json.loads(state_path.read_text(encoding="utf-8")))
            shutil.rmtree(stage); journal_path.unlink(); return
        # For a refresh, preserve the prior known-good generation instead of
        # completing an interrupted replacement.
        if not evaluations.exists() and archive is not None and archive.is_dir():
            os.replace(archive, evaluations)
        if stage.exists():
            shutil.rmtree(stage)
        journal_path.unlink()
        return
    # Published data is authoritative only when the staged state can validate it.
    if not evaluations.is_dir() or not state_path.is_file():
        if not evaluations.exists() and archive is not None and archive.is_dir():
            os.replace(archive, evaluations)
        if stage.exists():
            shutil.rmtree(stage)
        journal_path.unlink()
        return
    _atomic_json(_report_path(run, _REPORT_STATE), json.loads(state_path.read_text(encoding="utf-8")))
    shutil.rmtree(stage); journal_path.unlink()


def _preflight_managed_report(run: Path, provenance: Mapping[str, Any]) -> None:
    _recover_managed_report(run)
    evaluations, state = _report_path(run, "evaluations"), _read_report_state(run)
    if not evaluations.exists():
        if state is not None:
            raise TrainingError("managed report state exists without evaluations directory")
        return
    if state is None:
        # The one-file legacy native result is an owned pre-report layout. It is
        # migratable on explicit resume, but arbitrary report contents are not.
        if {item.name for item in evaluations.iterdir()} == {"test.json"} and _validate_legacy_test_record(run, provenance):
            return
        raise TrainingError("existing evaluations directory is foreign or lacks managed ownership state")
    if state["provenance_sha256"] != _report_identity(run) or state["files"] != _report_checksums(evaluations):
        raise TrainingError("managed evaluations ownership or checksums do not match this run")


def _publish_managed_report(run: Path, *, bundle: WPPDatasetBundle, history: list[dict[str, Any]],
                            result: Mapping[str, Any], checkpoint: Path, parameter_count: int,
                            evaluation_kind: str, duration_seconds: float, provenance: Mapping[str, Any]) -> None:
    """Refresh an owned run report with archive+journal recovery, never blind overwrite."""
    _preflight_managed_report(run, provenance)
    evaluations, archives = _report_path(run, "evaluations"), _report_path(run, _REPORT_ARCHIVES)
    stage = Path(tempfile.mkdtemp(prefix=".native-wpp-evaluations-stage-", dir=run))
    try:
        _write_compatibility_report(stage, bundle=bundle, history=history, result=result, checkpoint=checkpoint,
                                    parameter_count=parameter_count, evaluation_kind=evaluation_kind,
                                    duration_seconds=duration_seconds)
        prior = _read_report_state(run)
        generation = 1 if prior is None else int(prior["generation"]) + 1
        state = {"contract": _REPORT_STATE_CONTRACT, "version": _REPORT_STATE_VERSION, "generation": generation,
                 "provenance_sha256": _report_identity(run), "files": _report_checksums(stage / "evaluations")}
        _atomic_json(stage / "state.json", state)
        journal_path = _report_path(run, _REPORT_JOURNAL)
        if evaluations.exists():
            archives.mkdir(exist_ok=True)
            if archives.is_symlink() or not archives.is_dir():
                raise TrainingError("managed report archive path is unsafe")
            archive_name = f"generation-{generation - 1:04d}"
            archive = archives / archive_name
            if archive.exists() or archive.is_symlink():
                raise TrainingError("managed report archive generation already exists")
            journal = {"phase": "prepared", "archive": archive_name, "stage": stage.name}
            _atomic_json(journal_path, journal)
            os.replace(evaluations, archive)
            journal["phase"] = "archived"; _atomic_json(_report_path(run, _REPORT_JOURNAL), journal)
            os.replace(stage / "evaluations", evaluations)
            journal["phase"] = "published"; _atomic_json(_report_path(run, _REPORT_JOURNAL), journal)
        else:
            # Journal initial publication too: a crash after publishing its
            # complete directory can still install the staged ownership state.
            journal = {"phase": "prepared", "archive": "", "stage": stage.name}
            _atomic_json(journal_path, journal)
            os.replace(stage / "evaluations", evaluations)
            journal["phase"] = "published"; _atomic_json(journal_path, journal)
        _atomic_json(_report_path(run, _REPORT_STATE), state)
        if journal_path.exists():
            journal_path.unlink()
        shutil.rmtree(stage)
    except Exception:
        # If a journal exists recovery restores the archived prior generation on
        # next resume. Leave it/stage intact rather than discarding recovery data.
        if not _report_path(run, _REPORT_JOURNAL).exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _checkpoint_payload(*, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int,
                        global_step: int, best_validation: float, stale_epochs: int, statistics: Mapping[str, Any],
                        config: Mapping[str, Any], provenance: Mapping[str, Any], shuffle: torch.Generator) -> dict[str, Any]:
    return {"contract": CHECKPOINT_CONTRACT, "version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "completed_epoch": int(epoch), "global_step": int(global_step),
            "best_validation_loss": float(best_validation), "stale_epochs": int(stale_epochs),
            "normalization": validate_statistics(statistics), "class_mapping": dict(provenance["class_mapping"]),
            "model_config": dict(config["model"]), "config": copy.deepcopy(dict(config)),
            "provenance": copy.deepcopy(dict(provenance)), "rng": _rng_state(shuffle)}


def _safe_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise TrainingError("this runner requires a PyTorch version supporting torch.load(weights_only=True)") from exc
    except Exception as exc:
        raise TrainingError("native checkpoint rejected by safe weights-only loader") from exc
    if not isinstance(payload, dict) or payload.get("contract") != CHECKPOINT_CONTRACT or payload.get("version") != CHECKPOINT_VERSION:
        raise TrainingError("unsupported or legacy checkpoint; native_wpp_checkpoint v1.0 is required")
    required = {"contract", "version", "model_state", "optimizer_state", "scheduler_state", "completed_epoch", "global_step",
                "best_validation_loss", "stale_epochs", "normalization", "class_mapping", "model_config", "config", "provenance", "rng"}
    if set(payload) != required:
        raise TrainingError("native checkpoint fields are invalid")
    validate_statistics(payload["normalization"])
    return payload


def _optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    cls = torch.optim.Adam if config["training"]["optimizer"] == "Adam" else torch.optim.AdamW
    return cls(model.parameters(), lr=config["training"]["learning_rate"])


def _scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> Any:
    spec = config["training"]["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=spec["factor"],
                                                        patience=spec["patience"], min_lr=spec["min_lr"])


def _class_names(mapping: Mapping[str, int]) -> list[str]:
    if not mapping or sorted(mapping.values()) != list(range(len(mapping))):
        raise TrainingError("native WPP class mapping must be contiguous")
    return [label for label, _ in sorted(mapping.items(), key=lambda pair: pair[1])]


def _compatible(saved: Mapping[str, Any], requested: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if saved["model_config"] != requested["model"] or saved["provenance"] != provenance:
        raise TrainingError("checkpoint model or trusted WPP input provenance is incompatible")
    if saved["class_mapping"] != provenance["class_mapping"]:
        raise TrainingError("checkpoint class mapping is incompatible")
    saved_config = saved["config"]
    if not isinstance(saved_config, Mapping):
        raise TrainingError("checkpoint configuration is invalid")
    left, right = copy.deepcopy(dict(saved_config)), copy.deepcopy(dict(requested))
    for config in (left, right):
        config["training"].pop("max_epochs", None)
        config["training"].pop("device", None)
        config["training"].pop("num_workers", None)
    if left != right:
        raise TrainingError("only training.max_epochs, device, and num_workers may differ on resume")
    if requested["training"]["max_epochs"] < saved["completed_epoch"]:
        raise TrainingError("training.max_epochs cannot be below completed_epoch")


def _lock(run: Path) -> Path:
    lock = run / ".native-wpp-run.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise TrainingError(f"run is already locked: {lock}") from exc
    return lock


def train(config: Mapping[str, Any], *, default_path: str | Path, resume_checkpoint: str | Path | None = None) -> Path:
    config = validate_config(config, default_path)
    run = Path(config["run"]["output_dir"]).expanduser().resolve()
    resume = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    if resume is None:
        if run.exists():
            raise TrainingError("run.output_dir must be a fresh path and is never overwritten")
        run.mkdir(parents=True)
    elif not run.is_dir() or not resume.is_file() or run not in resume.parents:
        raise TrainingError("resume requires an existing run output_dir and a checkpoint inside it")
    lock = _lock(run)
    run_mutation_started = False
    try:
        bundle = _bundle(config); provenance = _provenance(bundle); class_names = _class_names(provenance["class_mapping"])
        # Validate/recover a previously managed report before any resume can
        # update model, optimizer, history, or checkpoints.
        if resume is not None:
            _preflight_managed_report(run, provenance)
        train_set, validation_set = bundle.partition_view("train"), bundle.partition_view("validation")
        device = choose_device(config["training"]["device"])
        if resume is None:
            shuffle = _seed_everything(config["training"]["seed"])
            statistics = fit_train_statistics(train_set)
            model = make_model(config["model"], len(class_names)).to(device)
            # The scheduler is bound to this exact optimizer instance.
            optimizer = _optimizer(model, config); scheduler = _scheduler(optimizer, config)
            epoch = global_step = stale = 0; best = float("inf")
            _atomic_json(run / "config.json", config)
            _atomic_json(run / "provenance.json", provenance)
        else:
            payload = _safe_checkpoint(resume); _compatible(payload, config, provenance)
            statistics = validate_statistics(payload["normalization"])
            shuffle = _seed_everything(config["training"]["seed"])
            model = make_model(config["model"], len(class_names)).to(device)
            optimizer = _optimizer(model, config); scheduler = _scheduler(optimizer, config)
            model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"])
            _restore_rng(payload["rng"], shuffle)
            epoch, global_step = int(payload["completed_epoch"]), int(payload["global_step"])
            best, stale = float(payload["best_validation_loss"]), int(payload["stale_epochs"])
        run_mutation_started = True
        _atomic_json(run / "status.json", {"state": "running", "completed_epoch": epoch, "updated_at": _now()})
        history: list[dict[str, Any]] = []
        if resume is not None and (run / "history.json").is_file():
            try:
                prior_history = json.loads((run / "history.json").read_text(encoding="utf-8"))
                if not isinstance(prior_history, dict) or not isinstance(prior_history.get("epochs"), list):
                    raise ValueError("invalid history schema")
                history = list(prior_history["epochs"])
                if len(history) != epoch:
                    raise ValueError("history/checkpoint epoch mismatch")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise TrainingError("cannot resume with invalid native history") from exc
        while epoch < config["training"]["max_epochs"]:
            # These are intentionally the online, pre-update logits from the
            # ordinary training forwards.  Do not add a second train inference:
            # it would have different dropout/optimizer semantics.
            started = time.perf_counter()
            model.train(); train_loss = 0.0; samples = 0; train_truths: list[int] = []; train_predictions: list[int] = []
            for batch in _batches(train_set, config["training"]["batch_size"], shuffle, shuffle=True):
                values, rows, times, labels = _tensor_batch(batch, device, statistics)
                optimizer.zero_grad(set_to_none=True)
                logits = model(values, rows, times, normalized=True); loss = F.cross_entropy(logits, labels)
                train_truths.extend(labels.detach().cpu().tolist())
                train_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
                loss.backward(); optimizer.step()
                train_loss += float(loss.item()) * len(labels); samples += len(labels); global_step += 1
            # Reference epoch timing includes these seven online train scores,
            # but not per-class/confusion construction.
            train_metrics = classification_scores(train_truths, train_predictions)
            epoch += 1
            validation = evaluate_model(model, validation_set, batch_size=config["training"]["batch_size"], device=device,
                                        statistics=statistics, class_names=class_names, detailed=False)
            scheduler.step(validation["cross_entropy"])
            improved = validation["cross_entropy"] < best - config["training"]["early_stopping"]["min_improvement"]
            best, stale = (validation["cross_entropy"], 0) if improved else (best, stale + 1)
            # LR is post-scheduler. Build the row before stopping the timer;
            # history/checkpoint/report IO deliberately remains outside it.
            record = {"epoch": epoch, "train_cross_entropy": train_loss / samples,
                      "validation_cross_entropy": validation["cross_entropy"], "validation_metrics": validation["metrics"],
                      **{f"train_{metric}": train_metrics[metric] for metric in train_metrics},
                      **{f"validation_{metric}": validation["metrics"][metric] for metric in validation["metrics"]},
                      "learning_rate": float(optimizer.param_groups[0]["lr"]), "duration_seconds": None}
            record["duration_seconds"] = time.perf_counter() - started
            history.append(record); _atomic_json(run / "history.json", {"epochs": history})
            payload = _checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                                          global_step=global_step, best_validation=best, stale_epochs=stale,
                                          statistics=statistics, config=config, provenance=provenance, shuffle=shuffle)
            _atomic_torch(run / "checkpoints" / "last.pt", payload)
            if improved:
                _atomic_torch(run / "checkpoints" / "best.pt", payload)
            if epoch % config["training"]["checkpoint_frequency"] == 0:
                _atomic_torch(run / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
            _atomic_json(run / "status.json", {"state": "running", "completed_epoch": epoch, "updated_at": _now(),
                                                  "best_validation_loss": best})
            if stale >= config["training"]["early_stopping"]["patience"]:
                break
        best_path = run / "checkpoints" / "best.pt"
        if not best_path.is_file():
            raise TrainingError("no best native checkpoint was produced")
        best_payload = _safe_checkpoint(best_path)
        # Treat best.pt as an untrusted native checkpoint too: it may have been
        # replaced after validation selection but before post-selection testing.
        _compatible(best_payload, config, provenance)
        best_statistics = validate_statistics(best_payload["normalization"])
        if best_statistics != statistics:
            raise TrainingError("best checkpoint normalization statistics are incompatible")
        model.load_state_dict(best_payload["model_state"])
        test_set = bundle.partition_view("test")
        result = evaluate_model(model, test_set, batch_size=config["training"]["batch_size"], device=device,
                                statistics=statistics, class_names=class_names, retain_predictions=True) if len(test_set) else None
        test_duration = float(result["_inference_duration_seconds"]) if result is not None else 0.0
        if result is not None:
            _publish_managed_report(
                run, bundle=bundle, history=history, result=result, checkpoint=best_path,
                parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                evaluation_kind="original_test", duration_seconds=test_duration, provenance=provenance)
            # Kept outside evaluations/: that directory is exactly the fourteen
            # interoperable artifacts and must not carry native-only JSON.
            _atomic_json(run / "native_test_metrics.json", {"checkpoint": str(best_path), "provenance": provenance,
                                                               "normalization": statistics, **_public_evaluation(result)})
        _atomic_json(run / "status.json", {"state": "complete", "completed_epoch": epoch, "updated_at": _now(),
                                              "best_validation_loss": best, "test_evaluated": result is not None})
        return run
    except Exception:
        if run.exists() and run_mutation_started:
            _atomic_json(run / "status.json", {"state": "failed", "updated_at": _now()})
        raise
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def evaluate(checkpoint: str | Path, config: Mapping[str, Any], *, default_path: str | Path,
             partition: str, output_dir: str | Path) -> Path:
    if partition not in ("train", "validation", "test"):
        raise TrainingError("evaluation partition must be train, validation, or test")
    config = validate_config(config, default_path); output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise TrainingError("evaluation output_dir must be fresh and is never overwritten")
    payload = _safe_checkpoint(Path(checkpoint).expanduser().resolve())
    bundle = _bundle(config); provenance = _provenance(bundle); _compatible(payload, config, provenance)
    statistics = validate_statistics(payload["normalization"]); class_names = _class_names(provenance["class_mapping"])
    device = choose_device(config["training"]["device"])
    model = make_model(config["model"], len(class_names)).to(device); model.load_state_dict(payload["model_state"])
    result = evaluate_model(model, bundle.partition_view(partition), batch_size=config["training"]["batch_size"],
                            device=device, statistics=statistics, class_names=class_names, retain_predictions=True)
    duration = float(result["_inference_duration_seconds"])
    output.mkdir(parents=True)
    resolved_checkpoint = Path(checkpoint).resolve()
    _atomic_json(output / "metrics.json", {"checkpoint": str(resolved_checkpoint), "provenance": provenance,
                                             "normalization": statistics, **_public_evaluation(result)})
    _write_compatibility_report(
        output, bundle=bundle, history=[], result=result, checkpoint=resolved_checkpoint,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        evaluation_kind="original_test" if partition == "test" else f"native_{partition}_evaluation",
        duration_seconds=duration)
    return output


def _safe_fresh_report_output(path: str | Path, source_run: Path) -> Path:
    """Reject ambiguous/symlinked report destinations before creating anything."""
    output = Path(path).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise TrainingError("report output_dir must be a fresh path and is never overwritten")
    if output.name in ("", ".", ".."):
        raise TrainingError("report output_dir is unsafe")
    current = Path(output.anchor)
    for component in output.parts[1:]:
        current /= component
        if current.exists() and current.is_symlink():
            raise TrainingError(f"report output path contains a symlink: {current}")
    parent = output.parent
    if not parent.is_dir():
        raise TrainingError("report output parent must already exist")
    try:
        output.relative_to(source_run)
    except ValueError:
        return output
    raise TrainingError("report output_dir must not be inside the source run")


def _checkpoint_lr(payload: Mapping[str, Any]) -> float | None:
    """Read a post-scheduler checkpoint LR without constructing an optimizer."""
    state = payload.get("optimizer_state")
    if not isinstance(state, Mapping) or not isinstance(state.get("param_groups"), list) or not state["param_groups"]:
        return None
    group = state["param_groups"][0]
    value = group.get("lr") if isinstance(group, Mapping) else None
    return float(value) if _finite_number(value) and float(value) >= 0 else None


def _historical_history(run: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads((run / "history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot read native history for report export") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("epochs"), list):
        raise TrainingError("native history schema is invalid")
    return copy.deepcopy(value["epochs"])


def _assert_existing_test_matches(existing: Path, result: Mapping[str, Any]) -> None:
    """Guard export against silently reporting different frozen-test predictions."""
    if not existing.is_file():
        return
    try:
        saved = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("existing native test record is malformed") from exc
    if not isinstance(saved, Mapping):
        raise TrainingError("existing native test record is malformed")
    expected = {"cross_entropy": result["cross_entropy"], **result["metrics"]}
    for key, observed in expected.items():
        recorded = saved.get(key) if key == "cross_entropy" else saved.get("metrics", {}).get(key)
        if not _finite_number(recorded) or not _finite_number(observed):
            raise TrainingError(f"existing native test record lacks finite metric: {key}")
        # CPU/CUDA reductions can differ in their last bits; equality is not an
        # authenticity criterion, but a material mismatch must stop export.
        if not math.isclose(float(recorded), float(observed), rel_tol=1e-6, abs_tol=1e-8):
            raise TrainingError(f"frozen checkpoint test metric differs from existing record: {key}")


def export_report(run_dir: str | Path, output_dir: str | Path, *, default_path: str | Path,
                  reconstruct_train_metrics: bool, device_spec: str | None = None) -> Path:
    """Publish a fresh, immutable compatibility report for an existing native run.

    The source run is read only.  With reconstruction enabled, each epoch
    checkpoint is evaluated in ``eval``/``inference_mode`` on train data using
    its embedded, verified statistics; the results are explicitly documented as
    post-epoch checkpoint snapshots rather than online train measurements.
    """
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir() or run.is_symlink():
        raise TrainingError("report source run must be a non-symlink directory")
    output = _safe_fresh_report_output(output_dir, run)
    try:
        raw_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("report source run has no valid config.json") from exc
    config = validate_config(raw_config, default_path)
    if Path(config["run"]["output_dir"]).expanduser().resolve() != run:
        raise TrainingError("report source config does not bind this run directory")
    if device_spec is not None:
        config = copy.deepcopy(config); config["training"]["device"] = device_spec
    bundle = _bundle(config); provenance = _provenance(bundle); class_names = _class_names(provenance["class_mapping"])
    history = _historical_history(run)
    best_path = run / "checkpoints" / "best.pt"
    best_payload = _safe_checkpoint(best_path); _compatible(best_payload, config, provenance)
    statistics = validate_statistics(best_payload["normalization"])
    if len(history) != int((json.loads((run / "status.json").read_text(encoding="utf-8"))).get("completed_epoch", -1)):
        raise TrainingError("native history/status epoch count mismatch")
    device = choose_device(config["training"]["device"])
    model = make_model(config["model"], len(class_names)).to(device)
    reconstructed_epochs: list[int] = []
    if reconstruct_train_metrics:
        # Validate every checkpoint lineage before spending any of the requested
        # reconstruction inference budget.  Payloads remain CPU-only; neither
        # optimizer nor scheduler is instantiated for export.
        epoch_payloads: list[tuple[int, dict[str, Any]]] = []
        for expected_epoch, _record in enumerate(history, start=1):
            checkpoint = run / "checkpoints" / f"epoch_{expected_epoch:04d}.pt"
            payload = _safe_checkpoint(checkpoint); _compatible(payload, config, provenance)
            if (payload["model_config"] != best_payload["model_config"]
                    or payload["class_mapping"] != best_payload["class_mapping"]
                    or payload["provenance"] != best_payload["provenance"]):
                raise TrainingError("epoch checkpoint model, class mapping, or provenance differs from selected checkpoint")
            if validate_statistics(payload["normalization"]) != statistics:
                raise TrainingError("epoch checkpoint normalization differs from selected checkpoint")
            if int(payload["completed_epoch"]) != expected_epoch:
                raise TrainingError("epoch checkpoint does not match native history")
            epoch_payloads.append((expected_epoch, payload))
        train_set = bundle.partition_view("train")
        for expected_epoch, payload in epoch_payloads:
            record = history[expected_epoch - 1]
            model.load_state_dict(payload["model_state"])
            with torch.inference_mode():
                snapshot = evaluate_model(model, train_set, batch_size=config["training"]["batch_size"], device=device,
                                          statistics=statistics, class_names=class_names)
            for metric, value in snapshot["metrics"].items():
                if not _finite_number(record.get(f"train_{metric}")):
                    record[f"train_{metric}"] = value
            if not _finite_number(record.get("learning_rate")):
                record["learning_rate"] = _checkpoint_lr(payload)
            # No historical wall-clock interval was persisted. Leave it null.
            if "duration_seconds" not in record:
                record["duration_seconds"] = None
            reconstructed_epochs.append(expected_epoch)
    model.load_state_dict(best_payload["model_state"])
    with torch.inference_mode():
        result = evaluate_model(model, bundle.partition_view("test"), batch_size=config["training"]["batch_size"], device=device,
                                statistics=statistics, class_names=class_names, retain_predictions=True)
    # Match the reference summary boundary: inference/loss loop only, excluding
    # post-loop aggregate, per-class, and confusion-matrix construction.
    test_duration = float(result["_inference_duration_seconds"])
    # Older completed runs used evaluations/test.json; future runs keep this
    # native-only record at the run root so evaluations/ remains exactly 14 files.
    existing_native_test = run / "native_test_metrics.json"
    if not existing_native_test.is_file():
        existing_native_test = run / "evaluations" / "test.json"
    _assert_existing_test_matches(existing_native_test, result)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.report-staging-", dir=output.parent))
    try:
        _write_compatibility_report(stage, bundle=bundle, history=history, result=result, checkpoint=best_path,
                                    parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                                    evaluation_kind="original_test", duration_seconds=test_duration)
        _atomic_json(stage / "report_provenance.json", {
            "schema_version": "native_wpp_video_format_report/1.0",
            "source_run": str(run), "selected_checkpoint": str(best_path),
            "selected_checkpoint_epoch": int(best_payload["completed_epoch"]), "provenance": provenance,
            "normalization": statistics, "train_metric_source": (
                "frozen_post_epoch_checkpoint_snapshot_inference" if reconstruct_train_metrics else "unavailable_in_original_history"),
            "reconstructed_train_metric_epochs": reconstructed_epochs,
            "historical_train_cross_entropy": "retained unchanged from source history; online pre-update training loss",
            "historical_duration_seconds": "null when not recorded by source history",
            "learning_rate": "checkpoint optimizer post-scheduler value when absent from source history",
            "source_run_was_not_modified": True,
        })
        # Fresh destination plus same-parent rename prevents consumers seeing a
        # partial 14-file report tree.
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output
