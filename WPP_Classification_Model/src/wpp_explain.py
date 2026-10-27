"""Read-only integrated-gradients and bounded attention diagnostics for native WPP."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .model_factory import make_model
from .wpp_dataset import PARTITIONS, WPPDatasetBundle, collate_wpp_videos
from .wpp_explain_types import (AttentionSummary, DataRoots, ExplainerSession, Prediction,
                                ProgressCallback, ProgressEvent, VideoExplanation, VideoSummary)
from .wpp_media import MediaError, resolve_source_video, source_media_timing, wpp_row_boxes
from .wpp_normalization import normalize_torch, validate_statistics
from .wpp_training import TrainingError, _provenance, _safe_checkpoint


class ExplanationError(ValueError):
    """The selected checkpoint, device, or explanation request is invalid."""


def _emit(callback: ProgressCallback | None, stage: str, message: str,
          completed: int | None = None, total: int | None = None) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, message, completed, total))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(checkpoint: str | Path, run_dir: str | Path | None) -> Path:
    if str(checkpoint) == "best":
        if run_dir is None:
            raise ExplanationError("checkpoint='best' requires an explicit run_dir")
        result = Path(run_dir).expanduser().resolve() / "checkpoints" / "best.pt"
    else:
        result = Path(checkpoint).expanduser().resolve()
    if not result.is_file():
        raise ExplanationError(f"checkpoint is missing: {result}")
    return result


def _requested_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(spec)
    except (TypeError, RuntimeError) as exc:
        raise ExplanationError("device must be auto, cpu, cuda[:N], or mps") from exc
    if device.type not in ("cpu", "cuda", "mps"):
        raise ExplanationError("device must be auto, cpu, cuda[:N], or mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExplanationError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ExplanationError("MPS was requested but is unavailable")
    return device


def _accelerator_runtime_error(exc: RuntimeError, device: torch.device) -> bool:
    """Only recover known accelerator/runtime failures, never validation bugs."""
    if device.type == "cpu":
        return False
    text = str(exc).lower()
    markers = ("out of memory", "cuda", "cudnn", "cublas", "hip", "mps", "not implemented",
               "not supported", "device-side", "driver", "invalid device")
    return any(marker in text for marker in markers)


def _make_model(payload: Mapping[str, Any], class_count: int, device: torch.device) -> Any:
    model = make_model(payload["model_config"], class_count)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model.to(device)


def open_explainer(checkpoint: str | Path = "best", *, run_dir: str | Path | None = None,
                   roots: DataRoots, device: str = "auto",
                   progress: ProgressCallback | None = None) -> ExplainerSession:
    """Open a strictly verified, read-only explanation session.

    ``roots`` is intentionally explicit so a verified dataset can be relocated
    without trusting stale absolute paths embedded in a training checkpoint.
    """
    path = _checkpoint_path(checkpoint, run_dir)
    _emit(progress, "open", "validating checkpoint and immutable data contracts")
    try:
        payload = _safe_checkpoint(path)
        bundle = WPPDatasetBundle(roots.packed_bundle, source_manifest=roots.source_manifest, index=roots.index)
        provenance = _provenance(bundle)
        statistics = validate_statistics(payload["normalization"])
    except (OSError, ValueError, TrainingError) as exc:
        raise ExplanationError(f"cannot open explanation inputs: {exc}") from exc
    saved_provenance = payload["provenance"]
    relocated_paths = {"bundle_path", "source_manifest_path", "index_path"}
    if not isinstance(saved_provenance, Mapping) or set(saved_provenance) != set(provenance):
        raise ExplanationError("checkpoint provenance schema does not match the supplied immutable roots")
    saved_immutable = {key: value for key, value in saved_provenance.items() if key not in relocated_paths}
    supplied_immutable = {key: value for key, value in provenance.items() if key not in relocated_paths}
    # Native training records absolute locations as audit metadata.  The bytes
    # bound by their hashes/checksums and the class mapping—not those locations—
    # establish identity when a read-only bundle is relocated.
    if saved_immutable != supplied_immutable or payload["class_mapping"] != provenance["class_mapping"]:
        raise ExplanationError("checkpoint immutable provenance/class mapping does not match the supplied roots")
    model_config = payload["model_config"]
    if payload["config"].get("model") != model_config:
        raise ExplanationError("checkpoint model configuration is internally inconsistent")
    class_names = tuple(label for label, _ in sorted(provenance["class_mapping"].items(), key=lambda item: item[1]))
    chosen = _requested_device(device)
    fallback: str | None = None
    try:
        model = _make_model(payload, len(class_names), chosen)
    except RuntimeError as exc:
        if not _accelerator_runtime_error(exc, chosen):
            raise
        fallback = f"{chosen}: {exc}"
        chosen = torch.device("cpu")
        model = _make_model(payload, len(class_names), chosen)
    session = ExplainerSession(model=model, bundle=bundle, checkpoint_path=path, checkpoint_sha256=_sha256(path),
                               normalization=statistics, class_names=class_names, roots=roots,
                               requested_device=device, device=chosen, fallback_reason=fallback,
                               metadata={"checkpoint_contract": payload["contract"],
                                         "checkpoint_version": payload["version"],
                                         "provenance": provenance, "model_config": dict(model_config)})
    _emit(progress, "open", f"ready on {chosen}" + (" (CPU fallback)" if fallback else ""))
    return session


def list_videos(session: ExplainerSession, partition: str) -> list[VideoSummary]:
    if partition not in PARTITIONS:
        raise ExplanationError(f"partition must be one of {PARTITIONS}")
    dataset = session.bundle.partition_view(partition)
    return [VideoSummary(video_id=record.video_id, partition=record.partition, label=record.label,
                         class_id=dataset.class_mapping[record.label], source_relative_path=record.source_relative_path,
                         original_frame_count=record.original_frames, used_frame_count=record.used_frames,
                         row_count=record.rows, time_truncated=record.truncated)
            for record in dataset.records]


def _select_item(session: ExplainerSession, partition: str, video_id: str) -> tuple[VideoSummary, dict[str, Any]]:
    if partition not in PARTITIONS:
        raise ExplanationError(f"partition must be one of {PARTITIONS}")
    dataset = session.bundle.partition_view(partition)
    for index, record in enumerate(dataset.records):
        if record.video_id == video_id:
            item = dataset[index]
            return (VideoSummary(video_id=record.video_id, partition=record.partition, label=record.label,
                                 class_id=dataset.class_mapping[record.label], source_relative_path=record.source_relative_path,
                                 original_frame_count=record.original_frames, used_frame_count=record.used_frames,
                                 row_count=record.rows, time_truncated=record.truncated), item)
    raise ExplanationError(f"video_id is not in {partition}: {video_id}")


def _batch_tensors(item: Mapping[str, Any], session: ExplainerSession) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = collate_wpp_videos([item])
    raw = torch.from_numpy(batch["values"]).to(device=session.device, dtype=torch.float32)
    rows = torch.from_numpy(batch["row_mask"]).to(device=session.device)
    times = torch.from_numpy(batch["time_mask"]).to(device=session.device)
    normalized = normalize_torch(raw, rows, times, session.normalization)
    return raw, normalized, rows, times


def _integrated_gradients(model: Any, normalized: torch.Tensor, rows: torch.Tensor, times: torch.Tensor,
                          target: int, steps: int, callback: ProgressCallback | None) -> tuple[torch.Tensor, dict[str, float]]:
    if steps < 2:
        raise ExplanationError("ig_steps must be at least 2 for trapezoid integration")
    with torch.no_grad():
        baseline_logit = float(model(torch.zeros_like(normalized), rows, times, normalized=True)[0, target].item())
        input_logit = float(model(normalized, rows, times, normalized=True)[0, target].item())
    total = torch.zeros_like(normalized)
    previous: torch.Tensor | None = None
    for position, alpha in enumerate(torch.linspace(0.0, 1.0, steps, device=normalized.device, dtype=normalized.dtype)):
        interpolation = (normalized * alpha).detach().requires_grad_(True)
        logit = model(interpolation, rows, times, normalized=True)[0, target]
        gradient, = torch.autograd.grad(logit, interpolation, retain_graph=False, create_graph=False)
        if previous is not None:
            total.add_(previous.add(gradient).mul_(0.5))
        previous = gradient.detach()
        _emit(callback, "integrated_gradients", "integrating selected logit", position + 1, steps)
    attribution = normalized * (total / float(steps - 1))
    valid = rows[:, :, None] & times[:, None, :]
    attribution = torch.where(valid, attribution, torch.zeros((), device=attribution.device, dtype=attribution.dtype))
    attribution_sum = float(attribution.sum().item())
    delta = input_logit - baseline_logit
    return attribution, {"input_logit": input_logit, "baseline_logit": baseline_logit,
                         "logit_delta": delta, "attribution_sum": attribution_sum,
                         "completeness_residual": delta - attribution_sum, "steps": float(steps)}


def _qk_from_attention_input(layer: Any, tokens: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attention = layer.self_attn
    if attention.in_proj_weight is None:
        raise ExplanationError("separate Q/K projection weights are unsupported by this native checkpoint")
    projected = F.linear(tokens[0, valid], attention.in_proj_weight, attention.in_proj_bias)
    dimension = attention.embed_dim
    queries, keys, _ = projected.split(dimension, dim=-1)
    heads, head_dimension = attention.num_heads, dimension // attention.num_heads
    queries = queries.reshape(-1, heads, head_dimension).permute(1, 0, 2).contiguous()
    keys = keys.reshape(-1, heads, head_dimension).permute(1, 0, 2).contiguous()
    return queries, keys


def _attention_rollout(qks: list[tuple[torch.Tensor, torch.Tensor]], *, chunk_queries: int,
                       callback: ProgressCallback | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if chunk_queries <= 0:
        raise ExplanationError("attention_chunk_queries must be positive")
    token_count = qks[0][0].shape[1]
    if not token_count or any(q.shape[1] != token_count or k.shape[1] != token_count for q, k in qks):
        raise ExplanationError("captured attention token counts are inconsistent")
    relevance = qks[0][0].new_full((token_count,), 1.0 / token_count)
    incoming: list[torch.Tensor] = [q.new_zeros(q.shape[:2]) for q, _ in qks]
    for reverse_index, (queries, keys) in enumerate(reversed(qks)):
        layer_index = len(qks) - 1 - reverse_index
        heads, _, head_dimension = queries.shape
        next_relevance = relevance * 0.5  # residual identity contribution, no NxN identity allocation
        incoming_layer = incoming[layer_index]
        for start in range(0, token_count, chunk_queries):
            stop = min(token_count, start + chunk_queries)
            scores = torch.matmul(queries[:, start:stop], keys.transpose(-2, -1)) / math.sqrt(head_dimension)
            weights = torch.softmax(scores, dim=-1)
            # Mean-head attention contribution to r @ (I + mean_heads(A)) / 2.
            next_relevance.add_((weights * relevance[start:stop][None, :, None]).sum(dim=(0, 1)) / (2.0 * heads))
            incoming_layer.add_(weights.sum(dim=1) / token_count)
        relevance = next_relevance
        _emit(callback, "attention_rollout", "streaming masked attention rollout", reverse_index + 1, len(qks))
    return relevance, incoming


def _trace_attention(model: Any, normalized: torch.Tensor, rows: torch.Tensor, times: torch.Tensor,
                     *, chunk_queries: int, callback: ProgressCallback | None) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    with torch.no_grad():
        ordinary = model(normalized, rows, times, normalized=True)
        traced, inputs = model.forward_with_attention_inputs(normalized, rows, times, normalized=True)
    if not torch.allclose(ordinary, traced, rtol=1e-5, atol=1e-6):
        difference = float((ordinary - traced).abs().max().item())
        raise ExplanationError(f"attention trace changed ordinary inference logits (max difference {difference})")
    token_mask = model._token_mask(rows, times)[0]
    with torch.no_grad():
        qks = [_qk_from_attention_input(layer, value, token_mask) for layer, value in zip(model.layers, inputs)]
        if len(qks) != len(model.layers):
            raise ExplanationError("attention trace did not capture every encoder layer")
        relevance, incoming = _attention_rollout(qks, chunk_queries=chunk_queries, callback=callback)
    return ordinary, incoming, relevance


def _scatter_token_values(values: torch.Tensor, token_mask: torch.Tensor, rows: int, patches: int) -> np.ndarray:
    result = torch.zeros(rows * patches, device=values.device, dtype=values.dtype)
    result[token_mask] = values
    return result.reshape(rows, patches).detach().cpu().numpy().astype(np.float32, copy=False)


def _attention_summary(model: Any, normalized: torch.Tensor, rows: torch.Tensor, times: torch.Tensor,
                       *, row_count: int, used_frames: int, chunk_queries: int,
                       callback: ProgressCallback | None) -> tuple[torch.Tensor, AttentionSummary]:
    logits, incoming, relevance = _trace_attention(model, normalized, rows, times,
                                                    chunk_queries=chunk_queries, callback=callback)
    token_mask = model._token_mask(rows, times)[0]
    patches = model.patches_per_row
    rollout_tokens = _scatter_token_values(relevance, token_mask, row_count, patches)
    incoming_tokens = np.stack([
        np.stack([_scatter_token_values(head, token_mask, row_count, patches) for head in layer], axis=0)
        for layer in incoming
    ], axis=0)
    rollout_frames = np.repeat(rollout_tokens, model.patch_size, axis=1)[:, :used_frames]
    return logits, AttentionSummary(rollout_tokens=rollout_tokens, rollout_frames=rollout_frames,
                                    frame_strip=rollout_frames.sum(axis=0), incoming_tokens=incoming_tokens,
                                    patch_size=model.patch_size, layers=len(incoming),
                                    heads=int(incoming_tokens.shape[1]))


def _artifact_directory(session: ExplainerSession, item: Mapping[str, Any]) -> Path:
    root_ref = session.bundle.extraction["metadata"].get("artifact_root_ref")
    if not isinstance(root_ref, str):
        raise ExplanationError("extraction index lacks artifact_root_ref")
    directory = (session.bundle.index.parent / root_ref / item["artifact_relpath"]).resolve()
    if not directory.is_dir():
        raise ExplanationError(f"native artifact is missing: {directory}")
    return directory


def _compute(session: ExplainerSession, item: Mapping[str, Any], *, ig_steps: int,
             chunk_queries: int, callback: ProgressCallback | None,
             target_class_id: int | None = None) -> tuple[Prediction, np.ndarray, dict[str, float], AttentionSummary]:
    _, normalized, rows, times = _batch_tensors(item, session)
    with torch.no_grad():
        logits = session.model(normalized, rows, times, normalized=True)
        class_id = int(logits.argmax(dim=1).item())
        probabilities = torch.softmax(logits, dim=1)
    target = class_id if target_class_id is None else target_class_id
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < logits.shape[1]:
        raise ExplanationError(f"target_class_id must be an integer in [0, {logits.shape[1]})")
    ig, diagnostics = _integrated_gradients(session.model, normalized, rows, times, target, ig_steps, callback)
    diagnostics["target_class_id"] = float(target)
    diagnostics["target_logit"] = float(logits[0, target].item())
    traced_logits, attention = _attention_summary(session.model, normalized, rows, times, row_count=int(item["row_count"]),
                                                  used_frames=int(item["used_time_length"]), chunk_queries=chunk_queries,
                                                  callback=callback)
    if not torch.allclose(logits, traced_logits, rtol=1e-5, atol=1e-6):
        raise ExplanationError("attention trace logits differ from the original prediction")
    truth = int(item["class_id"])
    prediction = Prediction(predicted_class_id=class_id, predicted_label=session.class_names[class_id],
                            predicted_logit=float(logits[0, class_id].item()),
                            predicted_probability=float(probabilities[0, class_id].item()),
                            ground_truth_class_id=truth, ground_truth_label=session.class_names[truth],
                            correct=class_id == truth)
    return prediction, ig[0, :int(item["row_count"]), :int(item["used_time_length"])].detach().cpu().numpy().astype(np.float32), diagnostics, attention


def _move_to_cpu(session: ExplainerSession) -> None:
    payload = _safe_checkpoint(session.checkpoint_path)
    session.model = _make_model(payload, len(session.class_names), torch.device("cpu"))
    session.device = torch.device("cpu")


def explain_video(session: ExplainerSession, partition: str, video_id: str, *, ig_steps: int = 32,
                  display_mode: str = "positive", attention_chunk_queries: int | None = None,
                  target_class_id: int | None = None,
                  progress: ProgressCallback | None = None) -> VideoExplanation:
    """Explain one video without changing any input artifact or checkpoint.

    ``target_class_id`` selects the IG logit; omitted uses the prediction.
    Prediction and attention diagnostics always remain prediction/nonclass-specific.
    """
    if display_mode not in ("positive", "signed"):
        raise ExplanationError("display_mode must be 'positive' or 'signed'")
    summary, item = _select_item(session, partition, video_id)
    chunk = session.default_attention_chunk_queries if attention_chunk_queries is None else attention_chunk_queries
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ExplanationError("attention_chunk_queries must be a positive integer")
    _emit(progress, "media", "hashing source video and validating presentation timestamps")
    source = resolve_source_video(session.roots.source_video_root, summary.source_relative_path)
    media_url = session.roots.media_url_resolver(source) if session.roots.media_url_resolver else None
    try:
        media = source_media_timing(source_path=source, expected_sha256=item["source_sha256"],
                                    artifact_dir=_artifact_directory(session, item),
                                    expected_frames=summary.original_frame_count,
                                    retained_frames=summary.used_frame_count, media_url=media_url)
        boxes = wpp_row_boxes(rows=summary.row_count, frame_height=media.height)
    except MediaError as exc:
        raise ExplanationError(str(exc)) from exc
    _emit(progress, "predict", f"predicting and explaining {video_id}")
    try:
        prediction, ig, diagnostics, attention = _compute(session, item, ig_steps=ig_steps,
                                                           chunk_queries=chunk, callback=progress,
                                                           target_class_id=target_class_id)
    except RuntimeError as exc:
        if not _accelerator_runtime_error(exc, session.device):
            raise
        previous = session.device
        reason = f"{previous}: {exc}"
        _move_to_cpu(session)
        session.fallback_reason = reason
        _emit(progress, "device_fallback", f"{reason}; rerunning this video on CPU")
        prediction, ig, diagnostics, attention = _compute(session, item, ig_steps=ig_steps,
                                                           chunk_queries=chunk, callback=progress,
                                                           target_class_id=target_class_id)
    metadata: dict[str, Any] = {
        "method": {
            "integrated_gradients": "trapezoid, selected whole-video logit (predicted by default), train-mean raw baseline / normalized zero, fixed masks",
            "ig_steps": ig_steps, "attention": "masked mean-head residual-normalized rollout from masked-mean pooling",
            "attention_is_class_specific": False, "attention_chunk_queries": chunk,
            "attention_incoming": "mean over valid query tokens, per layer/head/key token",
        },
        "ig_diagnostics": diagnostics,
        "normalization": dict(session.normalization),
        "display": {"mode": display_mode, "ig_visualization": "positive clamp by default; signed numeric array retained",
                    "attention_visualization": "separate attention unit/scale; do not share an IG scale"},
        "device": {"requested": session.requested_device, "used": str(session.device), "fallback_reason": session.fallback_reason},
        "row_boxes_source_pixels": boxes.tolist(),
        "limitations": ["Integrated gradients measures this input-baseline path; it is not proof of causal fidelity.",
                        "Attention rollout and incoming-attention summaries are attention diagnostics, not class-specific explanations.",
                        "No source media URL is assumed playable unless an explicit media_url_resolver is supplied."],
        "checkpoint_sha256": session.checkpoint_sha256,
    }
    return VideoExplanation(video=summary, prediction=prediction, media=media, ig_signed=ig,
                            ig_frame_strip_signed=ig.sum(axis=0), attention=attention,
                            metadata=metadata, display_mode=display_mode)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def export_explanation(explanation: VideoExplanation, output_dir: str | Path, *, write_png: bool = True,
                       write_svg: bool = True) -> Path:
    """Write compact numeric arrays and honest metadata to a fresh directory."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ExplanationError("export output_dir must be fresh and is never overwritten")
    destination.mkdir(parents=True)
    data_dir, figure_dir = destination / "data", destination / "figures"
    data_dir.mkdir(); figure_dir.mkdir()
    np.savez_compressed(data_dir / "arrays.npz", ig_signed=explanation.ig_signed,
                        ig_frame_strip_signed=explanation.ig_frame_strip_signed,
                        attention_rollout_tokens=explanation.attention.rollout_tokens,
                        attention_rollout_frames=explanation.attention.rollout_frames,
                        attention_frame_strip=explanation.attention.frame_strip,
                        attention_incoming_tokens=explanation.attention.incoming_tokens,
                        source_pts_seconds=explanation.media.pts_seconds,
                        media_seconds=explanation.media.media_seconds)
    payload = {"video": asdict(explanation.video), "prediction": asdict(explanation.prediction),
               "media": asdict(explanation.media), "metadata": explanation.metadata,
               "arrays": "data/arrays.npz", "display_mode": explanation.display_mode}
    (destination / "metadata.json").write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_png or write_svg:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), constrained_layout=True)
        axes[0].plot(explanation.ig_frame_strip_signed, color="tab:red")
        axes[0].set(title="Signed integrated-gradients frame strip", xlabel="presentation frame", ylabel="logit attribution")
        axes[1].plot(explanation.attention.frame_strip, color="tab:blue")
        axes[1].set(title="Attention-rollout frame strip (separate unit)", xlabel="presentation frame", ylabel="rollout mass repeated per patch")
        if write_png:
            fig.savefig(figure_dir / "overview.png", dpi=160)
        if write_svg:
            fig.savefig(figure_dir / "overview.svg")
        plt.close(fig)
    return destination
