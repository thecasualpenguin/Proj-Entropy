"""Lazy notebook rendering/controls for backend class-average WPP attribution.

All attribution, cache validation, native mass rebinning, and numeric CSV export
are deliberately owned by :mod:`src.wpp_class_attribution`.  This module is a
thin, import-safe plotting/widget layer over its ``AttributionAggregate``.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .wpp_local_paths import local_config_path, normalize_local_path_text

DEEPACTION_CLASSES = (
    "BDAnimateDiffLightning", "CogVideoX5B", "Pexels", "RunwayML",
    "StableDiffusion", "Veo", "VideoPoet",
)
EXPECTED_VIDEO_COUNTS = {"train": 1810, "test": 385}
DEFAULT_ROOTS = {
    "packed_bundle": "/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-packed",
    "source_manifest": "/mnt/d/projects/proj-entropy/datasets/deepaction_v1_manifest_seed42_quarantine_v1",
    "index": "/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1/extraction-index",
    "source_video_root": "/mnt/d/projects/proj-entropy/datasets/deepaction_v1",
    "run_dir": "/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-training",
    "checkpoint": "/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-training/checkpoints/best.pt",
    "cache_dir": "/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-training/class_attributions",
}


def _backend():
    # No torch/backend import merely from importing this UI module.
    from . import wpp_class_attribution
    return wpp_class_attribution


def _single_backend():
    # Single-video reads are strictly cache-only and intentionally have their own
    # lazy import; opening class-average controls does not require this module.
    from . import wpp_single_attribution
    return wpp_single_attribution


@dataclass(frozen=True)
class UIOptions:
    classes: tuple[str, ...] = DEEPACTION_CLASSES
    grouping: str = "gt"                    # gt | pred
    split_layout: str = "separate"          # separate | combined
    splits: tuple[str, ...] = ("train", "test")
    correctness: str = "all"
    pooled_misclassified: bool = False
    ig_target: str = "predicted"
    sign: str = "abs"
    magnitude: str = "per_video_normalized"
    # UI/backend canonical order is vertical, time: 32 physical bands × 192 frame-progress bins.
    time_bins: int = 192
    band_bins: int = 32
    shared_scale_within_method: bool = True
    preview: bool = False

    def validate(self) -> None:
        if not self.classes:
            raise ValueError("choose at least one class")
        if self.grouping not in ("gt", "pred") or self.split_layout not in ("separate", "combined"):
            raise ValueError("invalid grouping or split layout")
        if not self.splits or not set(self.splits) <= {"train", "test"}:
            raise ValueError("splits may contain train and/or test only; validation is excluded")
        if self.correctness not in ("all", "correct", "incorrect"):
            raise ValueError("invalid correctness filter")
        if self.ig_target not in ("predicted", "truth") or self.sign not in ("abs", "signed"):
            raise ValueError("invalid IG target or sign")
        if self.magnitude not in ("raw", "per_video_normalized"):
            raise ValueError("magnitude must be raw or per_video_normalized")
        if self.time_bins < 1 or self.band_bins < 1:
            raise ValueError("grid dimensions must be positive")


@dataclass(frozen=True)
class SingleViewOptions:
    ig_target: str = "predicted"
    sign: str = "abs"
    value_mode: str = "display_normalized"

    def validate(self) -> None:
        if self.ig_target not in ("predicted", "truth") or self.sign not in ("abs", "signed"):
            raise ValueError("single-video IG target/sign is invalid")
        if self.value_mode not in ("display_normalized", "raw"):
            raise ValueError("single-video value mode must be display_normalized or raw")


def aggregate_cached(cache_dir: str | Path, options: UIOptions):
    """Ask backend to aggregate an already-built cache; never runs a model."""
    options.validate()
    backend = _backend()
    # Combined is meaningful only for both selected splits. A one-split request
    # remains separate then is filtered at rendering, avoiding a hidden other split.
    split = "combined" if options.split_layout == "combined" and set(options.splits) == {"train", "test"} else "separate"
    return backend.aggregate_class_attributions(
        cache_dir, split=split, group_by=options.grouping, correctness=options.correctness,
        classes=options.classes, ig_target=options.ig_target, sign=options.sign,
        # Backend canonical order is (physical vertical bands, frame-progress bins).
        magnitude=options.magnitude, grid=(options.band_bins, options.time_bins),
        include_pooled_misclassified=options.pooled_misclassified, preview=options.preview,
    )


def _groups_for(result: Any, options: UIOptions) -> tuple[list[str], list[str]]:
    sections = ["combined"] if result.metadata["settings"]["split"] == "combined" else list(options.splits)
    labels = list(options.classes) + (["pooled_misclassified"] if options.pooled_misclassified else [])
    return sections, labels


def _find_group(result: Any, section: str, label: str, grouping: str):
    key = f"{section}|pooled_misclassified" if label == "pooled_misclassified" else f"{section}|{grouping}:{label}"
    return result.groups.get(key)


def interpretation_label(options: UIOptions, *, pooled: bool = False) -> str:
    """Make target/cohort explicit, including the special pooled-error row."""
    target = "Predicted-logit IG" if options.ig_target == "predicted" else "Ground-truth-logit IG"
    if pooled:
        return f"{target} among pooled misclassified videos"
    cohort = "ground-truth class cohort" if options.grouping == "gt" else "predicted-class cohort"
    return f"{target} within {cohort}"


def validate_default_session_cache(cache_dir: str | Path, training_root: str | Path) -> str | None:
    """Reject a non-approved manifest only at the approved default location.

    Custom cache locations retain the backend's generic contract.  This UI guard
    never substitutes its own expected records for the backend manifest.
    """
    cache = local_config_path(cache_dir).resolve()
    default_cache = local_config_path(DEFAULT_ROOTS["cache_dir"]).resolve()
    root = local_config_path(training_root).resolve()
    default_root = local_config_path(DEFAULT_ROOTS["run_dir"]).resolve()
    if cache != default_cache or root != default_root:
        return None
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        return None  # backend will give the actionable missing/incomplete-cache error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["expected_videos"]
        fingerprint = manifest["fingerprint"]
        counts = {split: sum(1 for item in expected if item.get("partition") == split) for split in EXPECTED_VIDEO_COUNTS}
        class_names = tuple(fingerprint["class_names"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("default class-attribution cache has an unreadable/non-default manifest") from exc
    if counts != EXPECTED_VIDEO_COUNTS or len(expected) != sum(EXPECTED_VIDEO_COUNTS.values()):
        raise ValueError(f"default class-attribution cache is non-default: expected {EXPECTED_VIDEO_COUNTS}, found {counts}")
    if class_names != DEEPACTION_CLASSES:
        raise ValueError("default class-attribution cache fingerprint has a non-DeepAction class order")
    return "Approved default session: train 1810 + test 385 expected."


def render_aggregate(result: Any, options: UIOptions):
    """Render backend aggregate maps, keeping empty backend zero-grids blank."""
    import matplotlib.pyplot as plt

    sections, labels = _groups_for(result, options)
    methods = ("ig", "attention")
    figure, axes = plt.subplots(len(labels), len(sections) * 2, figsize=(4.3 * len(sections) * 2, 2.7 * len(labels)),
                                squeeze=False, constrained_layout=True)
    signed = options.sign == "signed"
    scales: dict[str, float] = {"ig": 0.0, "attention": 0.0}
    for section in sections:
        for label in labels:
            group = _find_group(result, section, label, options.grouping)
            if group is None or group.count == 0:
                continue
            scales["ig"] = max(scales["ig"], float(abs(group.ig_grid).max()) if signed else float(group.ig_grid.max()))
            scales["attention"] = max(scales["attention"], float(group.attention_grid.max()))
    for row, label in enumerate(labels):
        for section_index, section in enumerate(sections):
            group = _find_group(result, section, label, options.grouping)
            for method_index, method in enumerate(methods):
                axis = axes[row][section_index * 2 + method_index]
                count = 0 if group is None else group.count
                if count == 0:
                    axis.set_facecolor("#f5f5f5")
                    axis.text(.5, .5, "No matching videos", ha="center", va="center", transform=axis.transAxes)
                else:
                    values = group.ig_grid if method == "ig" else group.attention_grid
                    scale = scales[method] if options.shared_scale_within_method else float(abs(values).max() if method == "ig" and signed else values.max())
                    scale = scale or 1.0
                    axis.imshow(values, aspect="auto", origin="upper", interpolation="nearest",
                                cmap="RdBu_r" if method == "ig" and signed else "magma",
                                vmin=-scale if method == "ig" and signed else 0, vmax=scale)
                title = "IG" if method == "ig" else "rollout (not class-specific)"
                active_scale = scales[method] if options.shared_scale_within_method else (scale if count else 0.0)
                scale_label = f"scale ±{active_scale:.3g}" if method == "ig" and signed else f"scale 0–{active_scale:.3g}"
                if label == "pooled_misclassified":
                    pretty = interpretation_label(options, pooled=True)
                else:
                    pretty = label
                # Numeric scale labels make exported PNGs interpretable even
                # outside the live notebook (without adding 28 colorbar axes).
                axis.set_title(f"{section} · {title} ({scale_label})\n{pretty} (n={count})", fontsize=9)
                axis.set_xticks([0, max(0, options.time_bins - 1)], ["0%", "100%"])
                axis.set_yticks([0, max(0, options.band_bins - 1)], ["0%", "100%"])
                axis.set_xlabel("relative retained presentation-frame progress")
                axis.set_ylabel("normalized physical WPP vertical position")
    figure.suptitle(f"{interpretation_label(options)}. Coordinates align input geometry/progress, not semantic scenes or actions.", fontsize=10)
    return figure


def _download_link(name: str, payload: bytes, mimetype: str) -> str:
    return f'<a download="{name}" href="data:{mimetype};base64,{base64.b64encode(payload).decode("ascii")}">Download {name}</a>'


def export_rendered_aggregate(result: Any, options: UIOptions, output_dir: str | Path) -> Path:
    """Stage backend CSV + UI PNG/metadata, then atomically publish directory/ZIP.

    Both final paths are preflighted before the backend is allowed to write, so a
    ZIP collision cannot leave a partial export directory behind.
    """
    backend = _backend()
    final = Path(output_dir).expanduser().resolve()
    archive = final.with_suffix(".zip")
    if final.exists():
        raise FileExistsError(f"export output already exists: {final}")
    if archive.exists():
        raise FileExistsError(f"export ZIP already exists: {archive}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    stage.rmdir()  # backend requires a fresh, absent output directory
    descriptor, temporary_archive_name = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".tmp", dir=final.parent)
    os.close(descriptor)
    temporary_archive = Path(temporary_archive_name)
    published = False
    try:
        destination = backend.export_class_attributions(result, stage, zip_output=False)
        figure = render_aggregate(result, options)
        (destination / "ui_rendering_metadata.json").write_text(json.dumps({
            "ig_target_and_cohort": interpretation_label(options),
            "pooled_misclassified_ig_target_and_cohort": interpretation_label(options, pooled=True),
            "ui_options": options.__dict__, "grid_shape_vertical_time": [options.band_bins, options.time_bins],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            figure.savefig(destination / "class_attribution_maps.png", dpi=160)
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(destination))
        os.replace(destination, final)
        published = True
        os.replace(temporary_archive, archive)
        return final
    except BaseException:
        if published and final.exists():
            shutil.rmtree(final)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if temporary_archive.exists():
            temporary_archive.unlink()


def list_single_cached_records(cache_dir: str | Path):
    """Return backend-validated cached records only; never starts inference."""
    return _single_backend().list_single_video_records(cache_dir)


def load_single_cached_attribution(cache_dir: str | Path, video_id: str, options: SingleViewOptions):
    """Load one validated native record from cache without touching a model/GPU."""
    options.validate()
    return _single_backend().load_single_video_attribution(cache_dir, video_id, ig_target=options.ig_target,
        sign=options.sign, value_mode=options.value_mode)


def _single_record_payload(attribution: Any) -> dict[str, Any]:
    record = attribution.record
    return {
        "video_id": record.video_id, "split": record.split, "truth_class_id": record.truth_class_id,
        "truth_label": record.truth_label, "predicted_class_id": record.predicted_class_id,
        "predicted_label": record.predicted_label, "correct": record.correct,
        "native_wpp_rows": record.native_wpp_rows, "retained_frame_count": record.retained_frame_count,
        "ig_target": attribution.ig_target, "truth_target_source": attribution.truth_target_source,
        "sign": attribution.sign, "value_mode": attribution.value_mode,
        "ig_display_scale": float(attribution.ig_display_scale),
        "attention_display_scale": float(attribution.attention_display_scale),
        "metadata": attribution.metadata,
        "axis_semantics": {"x": "native retained presentation-frame index (0-based)",
                            "y": "native WPP row index (0-based, top row is 0)"},
        "units": {"ig": "integrated-gradient attribution", "attention_frame": "display-expanded native token mass",
                  "attention_token": "native attention rollout token mass"},
    }


def render_single_video_attribution(attribution: Any):
    """Render native [WPP row, frame] maps with no alignment, resize, or averaging."""
    import matplotlib.pyplot as plt
    import numpy as np

    record = attribution.record
    ig = np.asarray(attribution.ig_values)
    attention = np.asarray(attribution.attention_frame_values)
    if ig.ndim != 2 or attention.ndim != 2 or ig.shape != attention.shape:
        raise ValueError("single-video cached IG and expanded rollout must be matching native 2-D maps")
    signed = attribution.sign == "signed"
    # Values are already backend-selected raw or existing-explorer max-normalized;
    # scales remain separate because IG and rollout have different units.
    ig_scale = max(float(np.abs(ig).max()) if signed else float(ig.max()), 1.0) if attribution.value_mode == "display_normalized" else max(float(attribution.ig_display_scale), 1.0)
    attention_scale = max(float(attention.max()), 1.0) if attribution.value_mode == "display_normalized" else max(float(attribution.attention_display_scale), 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    panels = ((axes[0], ig, "IG", "RdBu_r" if signed else "magma", -ig_scale if signed else 0.0, ig_scale),
              (axes[1], attention, "attention rollout (not class-specific)", "magma", 0.0, attention_scale))
    for axis, values, name, cmap, minimum, maximum in panels:
        axis.imshow(values, origin="upper", aspect="auto", interpolation="nearest", cmap=cmap, vmin=minimum, vmax=maximum)
        mode = "existing-explorer max-normalized" if attribution.value_mode == "display_normalized" else "raw"
        axis.set_title(f"{name} · {mode} · display scale {'±' if minimum < 0 else ''}{maximum:.4g}")
        axis.set_xlabel("native retained presentation-frame index")
        axis.set_ylabel("native WPP row index (top row = 0)")
    correctness = "correct" if record.correct else "incorrect"
    figure.suptitle(f"{record.video_id} · {record.split} · truth={record.truth_label} · predicted={record.predicted_label} · {correctness}\n"
                   f"{('Predicted' if attribution.ig_target == 'predicted' else 'Ground-truth')}-logit IG; rows={ig.shape[0]}, frames={ig.shape[1]} (n=1)")
    return figure


def export_single_video_attribution(attribution: Any, output_dir: str | Path) -> Path:
    """Atomically publish native numeric CSV/JSON, PNG, and ZIP for one cached record."""
    import numpy as np
    final = Path(output_dir).expanduser().resolve(); archive = final.with_suffix(".zip")
    if final.exists(): raise FileExistsError(f"export output already exists: {final}")
    if archive.exists(): raise FileExistsError(f"export ZIP already exists: {archive}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent)); stage.rmdir()
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".tmp", dir=final.parent); os.close(descriptor)
    temporary_archive = Path(temp_name); published = False
    try:
        stage.mkdir()
        np.savetxt(stage / "ig_values.csv", attribution.ig_values, delimiter=",")
        np.savetxt(stage / "attention_frame_values.csv", attribution.attention_frame_values, delimiter=",")
        np.savetxt(stage / "attention_token_mass.csv", attribution.attention_token_mass, delimiter=",")
        (stage / "metadata.json").write_text(json.dumps(_single_record_payload(attribution), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        figure = render_single_video_attribution(attribution)
        try: figure.savefig(stage / "single_video_maps.png", dpi=160)
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(stage.iterdir()):
                if path.is_file(): handle.write(path, path.name)
        os.replace(stage, final); published = True; os.replace(temporary_archive, archive)
        return final
    except BaseException:
        if published and final.exists(): shutil.rmtree(final)
        raise
    finally:
        if stage.exists(): shutil.rmtree(stage)
        if temporary_archive.exists(): temporary_archive.unlink()


class _SplitSelection:
    def __init__(self, widgets: Any):
        self.train = widgets.Checkbox(value=True, description="train")
        self.test = widgets.Checkbox(value=True, description="test")
        self.widget = widgets.HBox([widgets.Label("Splits:"), self.train, self.test])
    @property
    def value(self) -> tuple[str, ...]:
        return tuple(name for name, widget in (("train", self.train), ("test", self.test)) if widget.value)


class ClassAttributionUI:
    """Explicit-build, cached-redraw notebook facade over backend dataclasses."""
    def __init__(self, training_root: str | Path = DEFAULT_ROOTS["run_dir"], *, roots: Mapping[str, str | Path] | None = None,
                 cache_dir: str | Path | None = None, device: str = "auto", ig_steps: int = 32,
                 attention_chunk_queries: int = 128):
        # Adapt WSL-style approved defaults before any downstream .resolve().
        # This is configuration-only: cache manifest/provenance strings are never
        # read, normalized, or rewritten here.
        self.training_root = local_config_path(training_root)
        configured_roots = DEFAULT_ROOTS if roots is None else roots
        self.roots = {name: local_config_path(value) for name, value in configured_roots.items()}
        self.cache_dir = (local_config_path(cache_dir) if cache_dir is not None
                          else self.training_root / "class_attributions")
        self.device, self.ig_steps, self.attention_chunk_queries = device, ig_steps, attention_chunk_queries
        self._widgets: dict[str, Any] = {}
        self._single_listing: Any = None

    def options(self) -> UIOptions:
        w = self._widgets
        return UIOptions(classes=tuple(w["classes"].value), grouping=w["grouping"].value,
                         split_layout=w["layout"].value, splits=w["splits"].value,
                         correctness=w["correctness"].value, pooled_misclassified=w["pooled"].value,
                         ig_target=w["target"].value, sign=w["sign"].value, magnitude=w["magnitude"].value,
                         time_bins=w["time_bins"].value, band_bins=w["band_bins"].value,
                         shared_scale_within_method=w["scale"].value == "shared", preview=w["preview"].value)

    def single_options(self) -> SingleViewOptions:
        w = self._widgets
        return SingleViewOptions(ig_target=w["single_target"].value, sign=w["single_sign"].value,
                                 value_mode=w["single_values"].value)

    def _refresh_single_records(self, *_: Any) -> None:
        """Explicitly refresh validated cache listings; never build/reopen a model."""
        w = self._widgets
        try:
            listing = list_single_cached_records(self.cache_dir)
            self._single_listing = listing
            split, label, correctness = w["single_split"].value, w["single_class"].value, w["single_correctness"].value
            records = [record for record in listing.records
                       if (split == "all" or record.split == split)
                       and (label == "all" or record.truth_label == label or record.predicted_label == label)
                       and (correctness == "all" or (correctness == "correct") == bool(record.correct))]
            options = [(f"{record.video_id} · {record.split} · truth={record.truth_label} · pred={record.predicted_label} · "
                        f"{'correct' if record.correct else 'incorrect'}", record.video_id) for record in records]
            # A refresh/filter must not silently change a still-valid selected
            # record (especially while a live cache publishes other videos).
            selected = w["single_video"].value
            available_ids = {video_id for _label, video_id in options}
            w["single_video"].options = options
            if selected in available_ids:
                w["single_video"].value = selected
            elif options:
                w["single_video"].value = options[0][1]
            else:
                w["single_video"].value = None
            state = "publication in progress or cache missing" if listing.publication_in_progress_or_missing else listing.cache_status
            w["single_status"].value = (f"Cached valid records: {len(records)} filtered / {listing.available_videos} available / "
                                        f"{listing.expected_videos} expected; {listing.unavailable_videos} unavailable; "
                                        f"{len(listing.invalid_records)} invalid; {state}. Refresh is cache-only and does not run inference.")
            if not options:
                w["single_status"].value += " No selectable cached record matches; refresh after a record completes or change filters."
        except Exception as exc:
            w["single_status"].value = f"Cannot list cached single-video records: {exc}. Generate/resume cache elsewhere, then refresh; no model was started."

    def _draw_single(self, *_: Any) -> None:
        from IPython.display import display
        w = self._widgets; output = w["single_output"]
        with output:
            output.clear_output(wait=True)
            try:
                video_id = w["single_video"].value
                if not video_id:
                    raise ValueError("select a validated cached video and press Refresh cached records first")
                attribution = load_single_cached_attribution(self.cache_dir, video_id, self.single_options())
                figure = render_single_video_attribution(attribution); display(figure)
                import matplotlib.pyplot as plt
                plt.close(figure)
                with tempfile.TemporaryDirectory() as temporary:
                    destination = export_single_video_attribution(attribution, Path(temporary) / "single-video")
                    archive = destination.with_suffix(".zip").read_bytes()
                    files = [(path.name, path.read_bytes()) for path in sorted(destination.iterdir()) if path.is_file() and path.suffix != ".png"]
                    png = (destination / "single_video_maps.png").read_bytes()
                links = " · ".join(_download_link(name, data, "application/json" if name.endswith(".json") else "text/csv") for name, data in files)
                display(w["html"](_download_link("single_video_maps.png", png, "image/png") + " · " + links + " · " +
                                      _download_link("single_video_export.zip", archive, "application/zip")))
            except Exception as exc:
                w["single_status"].value = f"Cannot render selected cached video: {exc}"

    def _switch_view(self, *_: Any) -> None:
        single = self._widgets["view"].value == "single"
        self._widgets["class_panel"].layout.display = "none" if single else ""
        self._widgets["single_panel"].layout.display = "" if single else "none"

    def _status_from_result(self, result: Any, approved_note: str | None = None) -> str:
        data = result.metadata
        prefix = (approved_note + " ") if approved_note else ""
        return (f"Cache path: {self.cache_dir.resolve()}. " + prefix +
                f"Cache {'preview' if data['preview'] else 'complete'}: "
                f"{data['available_videos']}/{data['expected_videos']} cached, "
                f"{data['missing_videos']} missing. Cached aggregation only; no model rerun.")

    def _redraw(self, *_: Any) -> None:
        from IPython.display import display
        output = self._widgets["output"]
        with output:
            output.clear_output(wait=True)
            try:
                options = self.options()
                # Always ask the backend again: a concurrent explicit resume can
                # change manifest/video content without changing UI controls.
                approved_note = validate_default_session_cache(self.cache_dir, self.training_root)
                result = aggregate_cached(self.cache_dir, options)
                self._widgets["status"].value = self._status_from_result(result, approved_note)
                figure = render_aggregate(result, options); display(figure)
                png = io.BytesIO(); figure.savefig(png, format="png", dpi=160)
                import matplotlib.pyplot as plt
                plt.close(figure)
                # The backend is authoritative for CSV/settings metadata. For a
                # notebook-local download, use its exporter in a temporary dir.
                with tempfile.TemporaryDirectory() as temporary:
                    destination = export_rendered_aggregate(result, options, Path(temporary) / "export")
                    archive = destination.with_suffix(".zip").read_bytes()
                    metadata = (destination / "metadata.json").read_bytes()
                    ui_metadata = (destination / "ui_rendering_metadata.json").read_bytes()
                    csv_parts = [(path.name, path.read_bytes()) for path in sorted((destination / "data").glob("*.csv"))]
                csv_note = " · ".join(_download_link(name, data, "text/csv") for name, data in csv_parts)
                display(self._widgets["html"](_download_link("class_attribution_maps.png", png.getvalue(), "image/png") + " · " +
                    csv_note + " · " + _download_link("class_attribution_metadata.json", metadata, "application/json") + " · " +
                    _download_link("ui_rendering_metadata.json", ui_metadata, "application/json") + " · " +
                    _download_link("class_attribution_export.zip", archive, "application/zip")))
            except Exception as exc:
                self._widgets["status"].value = (
                    f"Unable to aggregate cached results at {self.cache_dir.resolve()}: {exc}"
                )

    def _build(self, *_: Any) -> None:
        """Only this explicit action imports backend/Torch and can launch work."""
        w = self._widgets; w["status"].value = "Generating/resuming full train + test cache…"
        try:
            from .wpp_explain_types import DataRoots
            backend = _backend()
            roots = DataRoots(packed_bundle=self.roots["packed_bundle"], source_manifest=self.roots["source_manifest"],
                              index=self.roots["index"], source_video_root=self.roots["source_video_root"])
            def progress(event: Any) -> None:
                completed = "" if event.completed is None else f" {event.completed}/{event.total}"
                w["status"].value = f"{event.stage}:{completed} {event.message}"
            built = backend.build_class_attribution_cache(self.training_root, roots=roots, cache_dir=self.cache_dir,
                splits=("train", "test"), ig_steps=w["ig_steps"].value,
                attention_chunk_queries=w["chunk_queries"].value, device=w["device"].value, progress=progress)
            w["status"].value = f"Cache {'complete' if built.complete else 'incomplete'}: {built.completed}/{built.total}; reused {built.reused}, computed {built.computed}."
            if built.complete: self._redraw()
        except Exception as exc:
            w["status"].value = f"Cache build stopped (safe to resume): {exc}"

    def widget(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display as _display  # validate the output transport at construction time
        except ImportError as exc:
            raise RuntimeError(
                "Interactive class attribution requires optional notebook dependencies; "
                "install them with `conda run -n entropy python -m pip install -r requirements-notebook.txt`."
            ) from exc
        w = self._widgets; w["html"] = widgets.HTML
        w["view"] = widgets.ToggleButtons(options=[("Class averages", "class"), ("Single cached video", "single")], value="class", description="View")
        # Class-average-only controls (the default preserved 28-map view).
        w["classes"] = widgets.SelectMultiple(options=DEEPACTION_CLASSES, value=DEEPACTION_CLASSES, description="Classes")
        w["grouping"] = widgets.Dropdown(options=[("ground truth", "gt"), ("predicted", "pred")], value="gt", description="Group by")
        w["layout"] = widgets.Dropdown(options=[("train/test separate", "separate"), ("combined", "combined")], value="separate", description="Layout")
        w["splits"] = _SplitSelection(widgets)
        w["correctness"] = widgets.Dropdown(options=["all", "correct", "incorrect"], description="Correctness")
        w["pooled"] = widgets.Checkbox(value=False, description="Add pooled misclassified maps")
        w["target"] = widgets.Dropdown(options=[("predicted logit", "predicted"), ("ground-truth logit", "truth")], value="predicted", description="IG target")
        w["sign"] = widgets.Dropdown(options=[("magnitude (abs before rebin)", "abs"), ("signed", "signed")], value="abs", description="IG display")
        w["magnitude"] = widgets.Dropdown(options=[("per-video unit total", "per_video_normalized"), ("raw attribution", "raw")], value="per_video_normalized", description="Aggregate")
        w["scale"] = widgets.Dropdown(options=[("shared within method", "shared"), ("independent maps", "independent")], value="shared", description="Color scale")
        w["time_bins"] = widgets.BoundedIntText(value=192, min=1, max=1024, description="Grid frame bins")
        w["band_bins"] = widgets.BoundedIntText(value=32, min=1, max=512, description="Grid WPP bins")
        w["ig_steps"] = widgets.BoundedIntText(value=self.ig_steps, min=2, max=4096, description="IG steps")
        w["chunk_queries"] = widgets.BoundedIntText(value=self.attention_chunk_queries, min=1, max=4096, description="Attention chunk")
        w["device"] = widgets.Dropdown(options=["auto", "cuda", "cpu", "mps"], value=self.device, description="Device")
        w["preview"] = widgets.Checkbox(value=False, description="Explicit partial-cache preview")
        w["status"] = widgets.HTML(
            f"Cache path: {self.cache_dir.resolve()}. No attribution runs on open. "
            "Full train/test cache is required unless preview is explicitly checked."
        )
        w["output"] = widgets.Output(); build = widgets.Button(description="Generate/resume full train + test cache", button_style="warning")
        redraw = widgets.Button(description="Generate cached aggregate")
        build.on_click(self._build); redraw.on_click(self._redraw)
        for control in (w["classes"], w["grouping"], w["layout"], w["correctness"], w["pooled"], w["target"], w["sign"], w["magnitude"], w["scale"], w["time_bins"], w["band_bins"], w["preview"]):
            control.observe(self._redraw, names="value")
        for control in (w["splits"].train, w["splits"].test): control.observe(self._redraw, names="value")
        w["class_panel"] = widgets.VBox([widgets.HTML("<b>Class-average WPP attribution — train/test only (validation excluded)</b>"),
            widgets.HBox([w["classes"], widgets.VBox([w["grouping"], w["layout"], w["splits"].widget, w["correctness"], w["pooled"]])]),
            widgets.HBox([w["target"], w["sign"], w["magnitude"], w["scale"]]),
            widgets.HBox([w["ig_steps"], w["chunk_queries"], w["device"], w["time_bins"], w["band_bins"]]),
            w["preview"], widgets.HBox([build, redraw]), w["status"], w["output"]])
        # Native single-video controls: no rebinning/averaging and no build button.
        w["single_split"] = widgets.Dropdown(options=[("all", "all"), ("train", "train"), ("test", "test")], value="all", description="Split")
        w["single_class"] = widgets.Dropdown(options=[("all classes", "all")] + [(name, name) for name in DEEPACTION_CLASSES], value="all", description="Class")
        w["single_correctness"] = widgets.Dropdown(options=["all", "correct", "incorrect"], value="all", description="Correctness")
        w["single_video"] = widgets.Dropdown(options=[], description="Cached video")
        w["single_target"] = widgets.Dropdown(options=[("predicted logit", "predicted"), ("ground-truth logit", "truth")], value="predicted", description="IG target")
        w["single_sign"] = widgets.Dropdown(options=[("magnitude (absolute IG)", "abs"), ("signed", "signed")], value="abs", description="IG display")
        w["single_values"] = widgets.Dropdown(options=[("existing-explorer normalized display", "display_normalized"), ("raw attribution values", "raw")], value="display_normalized", description="Values")
        w["single_status"] = widgets.HTML("Single view is cache-only. Press Refresh cached records; it will not start a model, GPU, or cache build.")
        w["single_output"] = widgets.Output(); refresh = widgets.Button(description="Refresh cached records")
        draw_single = widgets.Button(description="Show selected native video")
        refresh.on_click(self._refresh_single_records); draw_single.on_click(self._draw_single)
        w["single_panel"] = widgets.VBox([widgets.HTML("<b>Single cached native video — no coordinate alignment, resizing, or averaging</b>"),
            widgets.HBox([w["single_split"], w["single_class"], w["single_correctness"], refresh]), w["single_video"],
            widgets.HBox([w["single_target"], w["single_sign"], w["single_values"], draw_single]), w["single_status"], w["single_output"]])
        w["single_panel"].layout.display = "none"
        w["view"].observe(self._switch_view, names="value")
        return widgets.VBox([w["view"], w["class_panel"], w["single_panel"]])


def open_class_attribution_ui(**kwargs: Any):
    """Create inert controls; no model/cache build occurs until the build button."""
    return ClassAttributionUI(**kwargs).widget()


# Compatibility name for early UI consumers; it is the renderer option type,
# not a second aggregation implementation.
AggregateOptions = UIOptions

__all__ = ["AggregateOptions", "ClassAttributionUI", "DEFAULT_ROOTS", "DEEPACTION_CLASSES", "EXPECTED_VIDEO_COUNTS", "SingleViewOptions", "UIOptions", "aggregate_cached", "export_rendered_aggregate", "export_single_video_attribution", "interpretation_label", "list_single_cached_records", "load_single_cached_attribution", "local_config_path", "normalize_local_path_text", "open_class_attribution_ui", "render_aggregate", "render_single_video_attribution", "validate_default_session_cache"]
