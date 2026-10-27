import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.wpp_class_attribution_ui import (
    ClassAttributionUI, DEEPACTION_CLASSES, DEFAULT_ROOTS, SingleViewOptions, UIOptions, aggregate_cached,
    export_rendered_aggregate, export_single_video_attribution, interpretation_label,
    load_single_cached_attribution, render_aggregate, render_single_video_attribution,
    validate_default_session_cache,
)


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.exports = 0

    def aggregate_class_attributions(self, cache_dir, **kwargs):
        self.calls.append((cache_dir, kwargs))
        # Backend grids are [physical vertical bands, frame progress].
        group = SimpleNamespace(count=1, ig_grid=np.ones(kwargs["grid"]),
                                attention_grid=np.ones(kwargs["grid"]) * .5)
        groups = {}
        for label in kwargs["classes"]:
            groups[f"train|gt:{label}"] = group
            groups[f"test|gt:{label}"] = group
        return SimpleNamespace(groups=groups, metadata={"settings": {"split": kwargs["split"]}, "preview": False,
                                                        "available_videos": 2195, "expected_videos": 2195, "missing_videos": 0})

    def export_class_attributions(self, result, output_dir, *, zip_output=False):
        self.exports += 1
        destination = Path(output_dir); destination.mkdir(parents=True)
        (destination / "data").mkdir()
        (destination / "data" / "group-0000-ig.csv").write_text("1,2\n", encoding="utf-8")
        (destination / "metadata.json").write_text(json.dumps(result.metadata), encoding="utf-8")
        return destination


class ClassAttributionUITests(unittest.TestCase):
    def test_defaults_match_approved_session_and_grid(self):
        options = UIOptions()
        self.assertEqual(DEEPACTION_CLASSES, options.classes)
        self.assertEqual("gt", options.grouping)
        self.assertEqual(("train", "test"), options.splits)
        self.assertEqual("predicted", options.ig_target)
        self.assertEqual("abs", options.sign)
        self.assertEqual("per_video_normalized", options.magnitude)
        self.assertEqual((192, 32), (options.time_bins, options.band_bins))
        self.assertFalse(options.pooled_misclassified)
        self.assertTrue(DEFAULT_ROOTS["cache_dir"].endswith("deepaction-native-v1-training/class_attributions"))
        with self.assertRaisesRegex(ValueError, "validation"):
            UIOptions(splits=("validation",)).validate()

    def test_adapter_uses_backend_aggregate_and_backend_grid_order(self):
        backend = FakeBackend()
        options = UIOptions(classes=("Pexels",), time_bins=192, band_bins=32)
        with patch("src.wpp_class_attribution_ui._backend", return_value=backend):
            result = aggregate_cached("/cache", options)
        path, call = backend.calls[0]
        self.assertEqual("/cache", path)
        self.assertEqual((32, 192), call["grid"])
        # Asymmetric default guards accidental UI/backend transpose: rows are
        # normalized physical WPP bands, columns are frame-progress bins.
        self.assertEqual((32, 192), result.groups["train|gt:Pexels"].ig_grid.shape)
        self.assertEqual((32, 192), result.groups["train|gt:Pexels"].attention_grid.shape)
        self.assertEqual("per_video_normalized", call["magnitude"])
        self.assertFalse(call["include_pooled_misclassified"])
        self.assertEqual("separate", call["split"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib is optional")
    def test_headless_renderer_has_default_28_maps_and_blanks_empty_groups(self):
        backend = FakeBackend()
        options = UIOptions()
        result = backend.aggregate_class_attributions("/cache", classes=options.classes, grid=(32, 192), split="separate")
        del result.groups["test|gt:VideoPoet"]
        figure = render_aggregate(result, options)
        try:
            self.assertEqual(28, len(figure.axes))
            self.assertIn("relative retained presentation-frame progress", figure.axes[0].get_xlabel())
            self.assertIn("No matching videos", [text.get_text() for text in figure.axes[-1].texts])
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib is optional")
    def test_rendered_export_delegates_numeric_files_and_adds_png_zip(self):
        backend = FakeBackend(); options = UIOptions(classes=("Pexels",), splits=("train",))
        result = backend.aggregate_class_attributions("/cache", classes=options.classes, grid=(32, 192), split="separate")
        with tempfile.TemporaryDirectory() as directory, patch("src.wpp_class_attribution_ui._backend", return_value=backend):
            output = export_rendered_aggregate(result, options, Path(directory) / "export")
            self.assertTrue((output / "class_attribution_maps.png").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertTrue((output / "ui_rendering_metadata.json").is_file())
            self.assertTrue(output.with_suffix(".zip").is_file())

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib is optional")
    def test_pooled_interpretation_is_in_figure_and_exportable_scale_is_numeric(self):
        backend = FakeBackend(); options = UIOptions(classes=("Pexels",), splits=("train",), ig_target="truth",
                                                     grouping="pred", pooled_misclassified=True)
        result = backend.aggregate_class_attributions("/cache", classes=options.classes, grid=(32, 192), split="separate")
        group = result.groups["train|gt:Pexels"]
        result.groups["train|pooled_misclassified"] = group
        figure = render_aggregate(result, options)
        try:
            titles = [axis.get_title() for axis in figure.axes]
            self.assertTrue(any("Ground-truth-logit IG among pooled misclassified videos" in title for title in titles))
            self.assertTrue(all("scale " in title for title in titles))
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    def test_export_preflights_zip_before_backend_writes(self):
        backend = FakeBackend(); options = UIOptions(classes=("Pexels",), splits=("train",))
        result = backend.aggregate_class_attributions("/cache", classes=options.classes, grid=(32, 192), split="separate")
        with tempfile.TemporaryDirectory() as directory, patch("src.wpp_class_attribution_ui._backend", return_value=backend):
            output = Path(directory) / "export"
            output.with_suffix(".zip").write_bytes(b"foreign")
            with self.assertRaisesRegex(FileExistsError, "ZIP"):
                export_rendered_aggregate(result, options, output)
            self.assertFalse(output.exists())
            self.assertEqual(0, backend.exports)

    def test_default_cache_count_and_fingerprint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"; cache = root / "class_attributions"; cache.mkdir(parents=True)
            expected = ([{"partition": "train"}] * 1810) + ([{"partition": "test"}] * 385)
            (cache / "manifest.json").write_text(json.dumps({"expected_videos": expected,
                "fingerprint": {"class_names": list(DEEPACTION_CLASSES)}}), encoding="utf-8")
            with patch.dict("src.wpp_class_attribution_ui.DEFAULT_ROOTS", {"run_dir": str(root), "cache_dir": str(cache)}):
                self.assertIn("train 1810", validate_default_session_cache(cache, root))
                (cache / "manifest.json").write_text(json.dumps({"expected_videos": expected[:-1],
                    "fingerprint": {"class_names": list(DEEPACTION_CLASSES)}}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-default"):
                    validate_default_session_cache(cache, root)
            self.assertFalse(hasattr(ClassAttributionUI(), "results"))

    def _single_attribution(self, *, value_mode="display_normalized", sign="abs"):
        record = SimpleNamespace(video_id="cached-video", split="train", truth_class_id=0, truth_label="Pexels",
                                 predicted_class_id=1, predicted_label="Veo", correct=False,
                                 native_wpp_rows=2, retained_frame_count=3)
        return SimpleNamespace(record=record, ig_target="predicted", truth_target_source="predicted",
            sign=sign, value_mode=value_mode, ig_values=np.array([[.25, .5, 1.], [.1, .2, .3]]),
            attention_frame_values=np.array([[.5, 1., 1.], [.2, .4, .4]]),
            attention_token_mass=np.array([[.5, 1.], [.2, .4]]), ig_display_scale=8., attention_display_scale=.7,
            metadata={"cache": "fixture"})

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib is optional")
    def test_single_view_preserves_native_shape_axes_and_exports_modes_scales(self):
        attribution = self._single_attribution()
        figure = render_single_video_attribution(attribution)
        try:
            self.assertEqual(2, len(figure.axes))
            self.assertIn("native retained presentation-frame index", figure.axes[0].get_xlabel())
            self.assertIn("native WPP row index", figure.axes[0].get_ylabel())
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)
        with tempfile.TemporaryDirectory() as directory:
            output = export_single_video_attribution(attribution, Path(directory) / "single")
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("display_normalized", metadata["value_mode"])
            self.assertEqual(8., metadata["ig_display_scale"])
            self.assertEqual((2, 3), np.loadtxt(output / "ig_values.csv", delimiter=",").shape)
            self.assertTrue((output / "attention_token_mass.csv").is_file())
            self.assertTrue(output.with_suffix(".zip").is_file())

    def test_single_loader_delegates_only_cache_read_options(self):
        observed = {}
        fake = SimpleNamespace(load_single_video_attribution=lambda cache, video, **kwargs: observed.update(cache=cache, video=video, **kwargs) or "loaded")
        with patch("src.wpp_class_attribution_ui._single_backend", return_value=fake):
            self.assertEqual("loaded", load_single_cached_attribution("/cache", "video", SingleViewOptions()))
        self.assertEqual({"cache": "/cache", "video": "video", "ig_target": "predicted", "sign": "abs", "value_mode": "display_normalized"}, observed)

    def test_notebook_is_valid_and_code_cells_parse(self):
        notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "wpp_class_attribution.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(4, notebook["nbformat"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "wpp_class_attribution.ipynb", "exec")


if __name__ == "__main__":
    unittest.main()
