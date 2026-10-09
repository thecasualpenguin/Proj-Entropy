#!/usr/bin/env python3
"""Standalone training/evaluation for native-row WPP real(0)/AIGC(1) models.

It intentionally neither imports nor modifies the legacy packet workflow.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "src"))
from wpp_data import (StreamingNormalizer, WPPVideo, assigned_prompt_split, load_manifest, make_windows,
                      normalize_video, prompt_group_split, sha256_file, windowing_record)
from wpp_deepaction_adapter import build_deepaction_index
from wpp_metrics import video_metrics
from wpp_models import make_wpp_model, video_logits

FORMAT_VERSION = 1
DEFAULT = HERE / "configs" / "wpp_default.yaml"


def now() -> str: return datetime.now(timezone.utc).isoformat()
def sha(obj: Any) -> str: return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)
def torch_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)
def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

def read_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    default = yaml.safe_load(DEFAULT.read_text())
    if not isinstance(raw, dict): raise ValueError("WPP config must be a mapping")
    unknown = set(raw) - set(default)
    if unknown: raise ValueError(f"unknown WPP config sections: {sorted(unknown)}")
    out = copy.deepcopy(default)
    for section, values in raw.items():
        if not isinstance(values, Mapping): raise ValueError(f"{section} must be a mapping")
        extra = set(values) - set(out[section])
        if extra: raise ValueError(f"unknown WPP config keys in {section}: {sorted(extra)}")
        out[section].update(values)
    if out["model"]["architecture"] not in ("resnet", "transformer"): raise ValueError("architecture must be resnet or transformer")
    if out["model"]["width"] != 64 or out["model"]["heads"] != 4 or out["model"]["dropout"] != .1: raise ValueError("approved WPP architecture is width=64, heads=4, dropout=.1")
    if out["data"]["window_seconds"] != 2: raise ValueError("native WPP windows are fixed at exactly 2 seconds")
    if out["training"]["seed"] != 42: raise ValueError("approved WPP split/training seed is fixed at 42")
    if (out["training"]["learning_rate"], out["training"]["early_stopping_patience"], out["training"]["batch_size"], out["training"]["accumulation_steps"]) != (.001, 15, 2, 8):
        raise ValueError("approved initial WPP settings are lr=.001, patience=15, batch=2, accumulation=8")
    if out["training"]["batch_size"] < 1 or out["training"]["accumulation_steps"] < 1: raise ValueError("batch/accumulation must be positive")
    if out["training"]["max_epochs"] > 100: raise ValueError("max_epochs may not exceed approved budget 100")
    if out["training"]["weight_decay"] < 0: raise ValueError("weight_decay must be nonnegative")
    if not out["data"]["manifest"]: raise ValueError("data.manifest is required")
    if out["data"]["input_format"] not in ("wpp_v1", "deepaction_manifest"): raise ValueError("unsupported WPP input_format")
    if out["data"]["input_format"] == "deepaction_manifest" and not out["data"]["dataset_root"]:
        raise ValueError("DeepAction adapter requires data.dataset_root")
    if out["data"]["strict_preflight"] and out["data"]["expected_video_count"] != 2605:
        raise ValueError("final strict WPP preflight requires expected_video_count: 2605")
    return out

def device_for(spec: str) -> torch.device:
    if spec == "auto": spec = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.device(spec)
    if d.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
    return d

def class_weights(train: list[WPPVideo], device: torch.device) -> torch.Tensor:
    counts = np.bincount([v.label for v in train], minlength=2)
    if np.any(counts == 0): raise ValueError("both real and AIGC require training videos")
    return torch.tensor([len(train) / (2 * int(n)) for n in counts], dtype=torch.float32, device=device)

def split_record(parts: Mapping[str, list[WPPVideo]]) -> dict:
    return {name: {"video_ids": [v.video_id for v in videos], "prompt_groups": sorted({v.prompt_group for v in videos}),
                   "class_counts": {"real": sum(v.label == 0 for v in videos), "aigc": sum(v.label == 1 for v in videos)}}
            for name, videos in parts.items()}

def evaluate(model, videos: list[WPPVideo], stats: Mapping[str, Any], weights: torch.Tensor, device: torch.device) -> dict:
    model.eval(); labels: list[int] = []; probs: list[list[float]] = []; per_video: list[dict[str, Any]] = []; weighted_sum = 0.; weight_total = 0.
    with torch.no_grad():
        for raw in videos:
            video = normalize_video(raw, stats); logits, _ = video_logits(model, video, make_windows(video), device)
            target = torch.tensor([video.label], device=device); per = F.cross_entropy(logits[None], target, reduction="none")
            w = float(weights[video.label]); weighted_sum += float(per) * w; weight_total += w
            probability = torch.softmax(logits, 0).cpu().tolist()
            labels.append(video.label); probs.append(probability)
            per_video.append({"video_id": video.video_id, "label": video.label, "probabilities": probability,
                              "window_count": len(make_windows(video)), "source": video.source,
                              "width": video.metadata.get("width"), "height": video.metadata.get("height"), "fps": video.metadata.get("fps")})
    source_meta = [{"video_id": v.video_id, "source": v.metadata.get("source_name", v.source),
                    "width": v.metadata.get("width"), "height": v.metadata.get("height"), "fps": v.metadata.get("fps")}
                   for v in videos]
    result = video_metrics(labels, probs, source_meta)
    result["weighted_ce"] = weighted_sum / weight_total; result["per_video"] = per_video
    return result

def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best, patience, cfg, stats, split, meta, weights):
    return {"format_version": FORMAT_VERSION, "created_at": now(), "epoch": epoch, "best": best, "patience": patience,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "amp_scaler_state": scaler.state_dict(), "model_config": cfg["model"], "training_config": cfg["training"],
            "normalizer": stats, "split": split, "dataset_meta": meta, "class_weights": weights.detach().cpu(),
            "config_sha256": sha(cfg), "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}

def _json_file(path: str | None, expected: type) -> Any:
    if not path: return expected()
    value = json.loads(Path(path).expanduser().read_text())
    if not isinstance(value, expected): raise ValueError(f"{path} has unexpected JSON type")
    return value

def train(cfg: dict, run: Path, resume: Path | None = None) -> None:
    device = device_for(cfg["training"]["device"]); seed_all(42)
    approved = _json_file(cfg["data"].get("approved_video_ids_path"), list)
    official = _json_file(cfg["data"].get("official_mapping_path"), dict)
    manifest_path = cfg["data"]["manifest"]; observed_failed: list[str] = []
    if cfg["data"]["input_format"] == "deepaction_manifest":
        built = build_deepaction_index(manifest_path, cfg["data"]["dataset_root"], run / "input_index.json", cfg["data"]["expected_video_count"], 42, cfg["data"].get("captions_csv"), cfg["data"]["strict_preflight"], cfg["data"].get("ffprobe_path", "/home/naoromi/miniforge3/envs/entropy/bin/ffprobe"))
        manifest_path = str(run / "input_index.json"); observed_failed = [entry["video_id"] for entry in built["provenance"]["failed"]]
        if observed_failed and (cfg["data"]["strict_preflight"] or not cfg["data"]["allow_incomplete_smoke"]):
            raise ValueError(f"DeepAction corpus is incomplete ({len(observed_failed)} failed); full training is blocked: {observed_failed}")
    videos, meta = load_manifest(manifest_path, expected_video_count=cfg["data"]["expected_video_count"],
                                 explicit_failed_video_ids=tuple(cfg["data"]["explicit_failed_video_ids"]) + tuple(observed_failed),
                                 approved_video_ids=approved, official_mapping=official or None)
    parts = assigned_prompt_split(videos, meta["split_assignments"]) if meta.get("split_assignments") else prompt_group_split(videos, 42)
    split = split_record(parts); dump(run / "split.json", split); dump(run / "dataset_meta.json", meta)
    normalizer = StreamingNormalizer(); stats = normalizer.fit_videos(parts["train"]); stats["manifest_sha256"] = meta["manifest_sha256"]
    dump(run / "normalizer.json", stats)
    # Persist exact nominal buckets and assigned native interval mass; no frame is duplicated.
    dump(run / "windowing.json", {split_name: [windowing_record(video) for video in split_videos]
                                   for split_name, split_videos in parts.items()})
    model = make_wpp_model(cfg["model"]).to(device); weights = class_weights(parts["train"], device)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg["training"]["scheduler_factor"], patience=cfg["training"]["scheduler_patience"], min_lr=cfg["training"]["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start = 1; best = {"macro_f1": -1., "weighted_ce": float("inf"), "epoch": None}; history = []; resumed_patience = 0
    if resume:
        cp = torch.load(resume, map_location=device, weights_only=False)
        if cp.get("format_version") != FORMAT_VERSION or cp.get("config_sha256") != sha(cfg) or cp.get("dataset_meta") != meta or cp.get("split") != split or cp.get("normalizer") != stats:
            raise ValueError("resume rejected: WPP config/dataset/split/normalizer changed")
        if not torch.equal(cp["class_weights"].to(device), weights): raise ValueError("resume rejected: video class weights changed")
        model.load_state_dict(cp["model_state"]); optimizer.load_state_dict(cp["optimizer_state"]); scheduler.load_state_dict(cp["scheduler_state"]); scaler.load_state_dict(cp["amp_scaler_state"])
        random.setstate(cp["rng"]["python"]); np.random.set_state(cp["rng"]["numpy"]); torch.set_rng_state(cp["rng"]["torch"].cpu())
        if cp["rng"].get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in cp["rng"]["cuda"]])
        start, best, resumed_patience = cp["epoch"] + 1, cp["best"], cp.get("patience", 0)
        history_path = run / "history.json"; history = json.loads(history_path.read_text()) if history_path.exists() else []
    run.mkdir(parents=True, exist_ok=True); (run / "checkpoints").mkdir(exist_ok=True)
    with (run / "config.yaml").open("w") as handle: yaml.safe_dump(cfg, handle, sort_keys=False)
    patience = resumed_patience; initial_accum = cfg["training"]["accumulation_steps"]; initial_batch = cfg["training"]["batch_size"]
    oom_events: list[dict[str, Any]] = []
    def rng_snapshot() -> dict[str, Any]:
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    def rng_restore(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
        if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    def update_group(group: list[WPPVideo], epoch: int, group_index: int) -> tuple[float, list[int], list[list[float]]]:
        # The denominator is actual video count, including a short final group.
        # Microbatch backwards therefore equal one full-group weighted-CE gradient.
        runtime_batch = initial_batch; retried = False
        while True:
            model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            optimizer_state, scaler_state, state_rng = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scaler.state_dict()), rng_snapshot()
            optimizer.zero_grad(set_to_none=True); total = 0.; truths: list[int] = []; probabilities: list[list[float]] = []
            try:
                for offset in range(0, len(group), runtime_batch):
                    numerator = torch.zeros((), device=device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        for raw in group[offset:offset + runtime_batch]:
                            video = normalize_video(raw, stats); logits, _ = video_logits(model, video, make_windows(video), device)
                            target = torch.tensor([video.label], device=device)
                            numerator = numerator + weights[video.label] * F.cross_entropy(logits[None], target)
                            truths.append(video.label); probabilities.append(torch.softmax(logits.detach(), 0).cpu().tolist())
                    total += float(numerator.detach())
                    scaler.scale(numerator / len(group)).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                return total / len(group), truths, probabilities
            except torch.OutOfMemoryError:
                # Retry the identical group exactly once from pre-update state; no video/frame is skipped.
                model.load_state_dict(model_state); optimizer.load_state_dict(optimizer_state); scaler.load_state_dict(scaler_state); rng_restore(state_rng); optimizer.zero_grad(set_to_none=True)
                if runtime_batch == 1: raise
                previous = runtime_batch; runtime_batch = 1; retried = True
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                oom_events.append({"epoch": epoch, "update_group": group_index, "previous_batch_size": previous,
                                   "runtime_batch_size": 1, "runtime_accumulation_steps": len(group),
                                   "effective_video_budget": initial_batch * initial_accum})
    for epoch in range(start, cfg["training"]["max_epochs"] + 1):
        epoch_started = time.perf_counter(); model.train(); shuffled = list(parts["train"]); random.Random(42 + epoch).shuffle(shuffled)
        running = 0.; seen = 0; online_truth: list[int] = []; online_probabilities: list[list[float]] = []; group_size = initial_batch * initial_accum
        # Every element remains a full video: no window sampling, resizing, or truncation.
        for group_index, offset in enumerate(range(0, len(shuffled), group_size)):
            group = shuffled[offset:offset + group_size]
            objective, truths, probabilities = update_group(group, epoch, group_index)
            running += objective * len(group); seen += len(group); online_truth.extend(truths); online_probabilities.extend(probabilities)
        if oom_events: dump(run / "oom_adjustments.json", oom_events)
        validation = evaluate(model, parts["validation"], stats, weights, device); scheduler.step(validation["weighted_ce"])
        row = {"epoch": epoch, "train_weighted_objective": running / seen, "train_online": video_metrics(online_truth, online_probabilities), "validation": validation,
               "learning_rate": optimizer.param_groups[0]["lr"], "duration_seconds": time.perf_counter() - epoch_started,
               "epoch_end_timestamp": now()}
        history.append(row); dump(run / "history.json", history)
        candidate = (validation["macro_f1"], -validation["weighted_ce"]); prior = (best["macro_f1"], -best["weighted_ce"])
        if candidate > prior:
            best = {"macro_f1": validation["macro_f1"], "weighted_ce": validation["weighted_ce"], "epoch": epoch}; patience = 0
            torch_dump(run / "checkpoints" / "best.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best, patience, cfg, stats, split, meta, weights))
        else: patience += 1
        torch_dump(run / "checkpoints" / "latest.pt", checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best, patience, cfg, stats, split, meta, weights))
        dump(run / "status.json", {"state": "running", "epoch": epoch, "best": best, "patience": patience, "device": str(device)})
        print(f"epoch={epoch} val_macro_f1={validation['macro_f1']:.4f} val_weighted_ce={validation['weighted_ce']:.6f}", flush=True)
        if patience >= cfg["training"]["early_stopping_patience"]: break
    cp = torch.load(run / "checkpoints" / "best.pt", map_location=device, weights_only=False); model.load_state_dict(cp["model_state"])
    # This is the sole canonical test call; training never reads test metrics.
    test = evaluate(model, parts["test"], stats, weights, device); dump(run / "test_once.json", test)
    dump(run / "status.json", {"state": "complete", "best": best, "test_evaluated_once": True, "device": str(device)})

def smoke(cfg: dict, architecture: str, video_count: int, epochs: int) -> None:
    """Real-train-only CUDA correctness smoke. It never evaluates validation/test or saves a final run."""
    if cfg["data"]["input_format"] != "deepaction_manifest" or not cfg["data"]["allow_incomplete_smoke"]:
        raise ValueError("smoke requires deepaction_manifest with allow_incomplete_smoke:true")
    root = Path(cfg["run"]["output_root"]).expanduser().resolve() / "_smoke_inputs"; root.mkdir(parents=True, exist_ok=True)
    built = build_deepaction_index(cfg["data"]["manifest"], cfg["data"]["dataset_root"], root / "input_index.json", cfg["data"]["expected_video_count"], 42, cfg["data"].get("captions_csv"), True, cfg["data"].get("ffprobe_path", "/home/naoromi/miniforge3/envs/entropy/bin/ffprobe"))
    failures = [entry["video_id"] for entry in built["provenance"]["failed"]]
    videos, meta = load_manifest(root / "input_index.json", expected_video_count=cfg["data"]["expected_video_count"], explicit_failed_video_ids=failures)
    parts = assigned_prompt_split(videos, meta["split_assignments"])
    # Select both labels from the already-assigned train prompts only; no corpus record is modified.
    selected: list[WPPVideo] = []
    for label in (0, 1): selected.extend([v for v in parts["train"] if v.label == label][:video_count])
    if len(selected) != 2 * video_count: raise ValueError("train split lacks both labels for requested smoke")
    stats = StreamingNormalizer().fit_videos(selected); device = device_for(cfg["training"]["device"])
    if device.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model_cfg = dict(cfg["model"], architecture=architecture); model = make_wpp_model(model_cfg).to(device).train()
    opt = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); weights = class_weights(selected, device)
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            objective = torch.zeros((), device=device)
            for raw in selected:
                video = normalize_video(raw, stats); logits, _ = video_logits(model, video, make_windows(video), device)
                objective += weights[video.label] * F.cross_entropy(logits[None], torch.tensor([video.label], device=device))
            objective = objective / len(selected)
        scaler.scale(objective).backward(); scaler.step(opt); scaler.update()
        suffix = f" peak_mib={torch.cuda.max_memory_allocated(device) / 2**20:.1f}" if device.type == "cuda" else ""
        print(f"real-train-only smoke architecture={architecture} epoch={epoch+1}/{epochs} objective={float(objective.detach()):.6f}{suffix}", flush=True)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train"); train_p.add_argument("--config", required=True, type=Path); train_p.add_argument("--run-name", required=True); train_p.add_argument("--resume", type=Path)
    smoke_p = sub.add_parser("smoke"); smoke_p.add_argument("--config", required=True, type=Path); smoke_p.add_argument("--architecture", choices=("resnet", "transformer"), required=True); smoke_p.add_argument("--videos-per-class", type=int, default=1); smoke_p.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv); cfg = read_config(args.config)
    if args.command == "smoke":
        if args.videos_per_class < 1 or args.epochs < 1: raise ValueError("smoke counts must be positive")
        smoke(cfg, args.architecture, args.videos_per_class, args.epochs); return 0
    root = Path(cfg["run"]["output_root"]).expanduser().resolve(); run = root / args.run_name
    if args.resume is None and run.exists(): raise ValueError("run directory exists")
    if args.resume is not None and (run / "status.json").is_file() and json.loads((run / "status.json").read_text()).get("state") == "complete":
        raise ValueError("completed WPP runs cannot be resumed: canonical test evaluation is exactly once")
    train(cfg, run, args.resume); return 0
if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc: print(f"ERROR: {exc}", file=sys.stderr); raise
