from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import src.wpp_class_attribution as attribution
from src.wpp_class_attribution import (
    CACHE_SCHEMA_VERSION,
    ClassAttributionError,
    aggregate_class_attributions,
    build_class_attribution_cache,
)
from src.wpp_class_attribution_ui import (
    ClassAttributionUI,
    UIOptions,
    aggregate_cached,
    export_rendered_aggregate,
)


class _Partition:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class ClassAttributionIntegrationTests(unittest.TestCase):
    def _cache(self, directory: Path, records, *, class_names=("a", "b"), splits=("train", "test")) -> Path:
        root = directory / "cache"
        videos = root / "videos"
        videos.mkdir(parents=True)
        fingerprint = {"class_names": list(class_names), "splits": list(splits), "fixture": "integration",
                       "model_config": {"patch_size": 16}}
        expected = []
        for item in records:
            native = np.asarray(item["signed"], dtype=np.float32)
            if native.ndim == 1:
                native = native[None, :]
            rows, frames = native.shape
            expected.append({"video_id": item["video_id"], "partition": item["partition"],
                             "class_id": item["gt"], "source_sha256": item.get("source_sha256", "a" * 64),
                             "label": class_names[item["gt"]], "row_count": rows, "used_frames": frames})
        manifest = {"schema": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint, "expected_videos": expected,
                    "status": "complete", "completed": len(records), "total": len(records)}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for item in records:
            signed = np.asarray(item["signed"], dtype=np.float32)
            if signed.ndim == 1:
                signed = signed[None, :]
            rows, frames = signed.shape
            low = np.asarray(item.get("vertical_low", np.arange(rows) / rows), dtype=np.float64)
            high = np.asarray(item.get("vertical_high", (np.arange(rows) + 1) / rows), dtype=np.float64)
            predicted = int(item["pred"])
            truth = int(item["gt"])
            metadata = {
                "cache_fingerprint": fingerprint, "video_id": item["video_id"], "partition": item["partition"],
                "ground_truth_class_id": truth, "ground_truth_label": class_names[truth],
                "predicted_class_id": predicted, "predicted_label": class_names[predicted],
                "correct": predicted == truth, "used_frames": frames, "row_count": rows, "patch_size": 16,
                "source_sha256": item.get("source_sha256", "a" * 64),
                "ig_truth_source": "predicted_reused" if predicted == truth else "separate_truth_logit",
            }
            arrays = {
                "ig_pred_signed": signed,
                "valid_mask": np.ones(signed.shape, dtype=np.bool_),
                # Rollout is a probability mass over the whole native video, not one unit per row.
                "attention_token_mass": np.asarray(item.get("attention", np.full((rows, 1), 1.0 / rows)), dtype=np.float32),
                "attention_token_mask": np.ones((rows, 1), dtype=np.bool_),
                "vertical_low": low,
                "vertical_high": high,
            }
            if predicted != truth:
                truth_signed = np.asarray(item.get("truth_signed", signed), dtype=np.float32)
                if truth_signed.ndim == 1:
                    truth_signed = truth_signed[None, :]
                arrays["ig_truth_signed"] = truth_signed
            stem = hashlib.sha256(item["video_id"].encode("utf-8")).hexdigest()
            npz_path = videos / f"{stem}.npz"
            np.savez_compressed(npz_path, **arrays)
            metadata["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            (videos / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
        return root

    def test_combined_split_and_pooled_mistakes_are_equal_video_means(self):
        records = [
            {"video_id": "train-one", "partition": "train", "gt": 0, "pred": 0, "signed": [1.0]},
            {"video_id": "train-two", "partition": "train", "gt": 0, "pred": 0, "signed": [3.0]},
            {"video_id": "test-one", "partition": "test", "gt": 0, "pred": 0, "signed": [9.0]},
            {"video_id": "mistake-a-one", "partition": "train", "gt": 0, "pred": 1, "signed": [1.0]},
            {"video_id": "mistake-a-two", "partition": "train", "gt": 0, "pred": 1, "signed": [3.0]},
            {"video_id": "mistake-b", "partition": "test", "gt": 1, "pred": 0, "signed": [10.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), records)
            combined = aggregate_class_attributions(root, split="combined", group_by="gt", correctness="correct", grid=(1, 1))
            # Three individual a videos, not an equal weighting of the train and test split means.
            self.assertEqual(3, combined.groups["combined|gt:a"].count)
            self.assertAlmostEqual(13.0 / 3.0, float(combined.groups["combined|gt:a"].ig_grid[0, 0]))
            mistakes = aggregate_class_attributions(root, split="combined", group_by="gt", correctness="incorrect", grid=(1, 1))
            # (1 + 3 + 10) / 3, rather than the unweighted mean of class-a's mean (2) and class-b's mean (10).
            pooled = mistakes.groups["combined|pooled_misclassified"]
            self.assertEqual(3, pooled.count)
            self.assertAlmostEqual(14.0 / 3.0, float(pooled.ig_grid[0, 0]))

    def test_gt_and_predicted_cohorts_select_different_videos(self):
        records = [
            {"video_id": "a-as-b", "partition": "train", "gt": 0, "pred": 1, "signed": [2.0]},
            {"video_id": "b-as-a", "partition": "train", "gt": 1, "pred": 0, "signed": [5.0]},
            {"video_id": "a-as-a", "partition": "test", "gt": 0, "pred": 0, "signed": [7.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), records)
            truth = aggregate_class_attributions(root, split="combined", group_by="gt", classes=["a"], grid=(1, 1))
            predicted = aggregate_class_attributions(root, split="combined", group_by="pred", classes=["a"], grid=(1, 1))
            self.assertEqual(2, truth.groups["combined|gt:a"].count)
            self.assertAlmostEqual(4.5, float(truth.groups["combined|gt:a"].ig_grid[0, 0]))
            self.assertEqual(2, predicted.groups["combined|pred:a"].count)
            self.assertAlmostEqual(6.0, float(predicted.groups["combined|pred:a"].ig_grid[0, 0]))

    def test_native_l1_normalization_precedes_signed_coarse_bin_cancellation(self):
        records = [
            {"video_id": "small", "partition": "train", "gt": 0, "pred": 0, "signed": [1.0, -1.0]},
            {"video_id": "large", "partition": "test", "gt": 0, "pred": 0, "signed": [10.0, -10.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), records)
            raw = aggregate_class_attributions(root, split="combined", sign="signed", magnitude="raw", grid=(1, 2))
            unit = aggregate_class_attributions(root, split="combined", sign="signed", magnitude="per_video_normalized", grid=(1, 2))
            np.testing.assert_allclose(raw.groups["combined|gt:a"].ig_grid[0], [5.5, -5.5])
            np.testing.assert_allclose(unit.groups["combined|gt:a"].ig_grid[0], [0.5, -0.5])
            # A one-bin signed map cancels, but it must still use the nonzero native L1 denominator.
            cancelled = aggregate_class_attributions(root, split="combined", sign="signed", magnitude="per_video_normalized", grid=(1, 1))
            absolute = aggregate_class_attributions(root, split="combined", sign="abs", magnitude="per_video_normalized", grid=(1, 1))
            self.assertTrue(np.isfinite(cancelled.groups["combined|gt:a"].ig_grid).all())
            self.assertAlmostEqual(0.0, float(cancelled.groups["combined|gt:a"].ig_grid[0, 0]))
            self.assertAlmostEqual(1.0, float(absolute.groups["combined|gt:a"].ig_grid[0, 0]))

    def test_partial_last_ctu_band_and_zero_mass_empty_groups(self):
        partial = {"video_id": "partial", "partition": "train", "gt": 0, "pred": 0,
                   "signed": [[0.0], [10.0]], "vertical_low": [0.0, 0.64], "vertical_high": [0.64, 1.0]}
        zero = {"video_id": "zero", "partition": "test", "gt": 0, "pred": 0, "signed": [0.0]}
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), [partial, zero])
            geometry = aggregate_class_attributions(root, split="separate", classes=["a"], grid=(3, 1))
            # The last physical band is 36% of the coded height, not an equal-height second row.
            np.testing.assert_allclose(geometry.groups["train|gt:a"].ig_grid[:, 0], [0.0, 20.0 / 27.0, 250.0 / 27.0])
            zero_result = aggregate_class_attributions(root, split="separate", classes=["b"], magnitude="per_video_normalized", grid=(1, 1))
            empty = zero_result.groups["test|gt:b"]
            self.assertEqual(0, empty.count)
            self.assertTrue(np.isfinite(empty.ig_grid).all())
            self.assertEqual(0.0, float(empty.ig_grid.sum()))
            zero_a = aggregate_class_attributions(root, split="separate", classes=["a"], magnitude="per_video_normalized", grid=(1, 1))
            self.assertTrue(np.isfinite(zero_a.groups["test|gt:a"].ig_grid).all())
            self.assertEqual(0.0, float(zero_a.groups["test|gt:a"].ig_grid.sum()))

    def test_cancelled_cache_resumes_after_corruption_and_rejects_incompatible_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            artifact_root = base / "artifacts"
            records = []
            for name in ("one", "two"):
                (artifact_root / name).mkdir(parents=True)
                (artifact_root / name / "metadata.json").write_text(json.dumps({"coded_height": 64, "ctu_size": 64}), encoding="utf-8")
                records.append({"video_id": name, "partition": "train", "artifact_relpath": name,
                                "row_count": 1, "used_time_length": 2, "class_id": 0})
            bundle = SimpleNamespace(
                partition_view=lambda split: _Partition(records if split == "train" else []),
                extraction={"metadata": {"artifact_root_ref": "artifacts"}}, index=base / "index.json",
            )
            session = SimpleNamespace(checkpoint_sha256="checkpoint", metadata={"checkpoint_contract": "native", "checkpoint_version": "1", "provenance": {}, "model_config": {}},
                                      class_names=("a", "b"), normalization={"mean": 0.0, "std": 1.0}, bundle=bundle)

            def compute(fake_session, item, **_kwargs):
                low, high, _ = attribution._artifact_geometry(fake_session, item)
                metadata = {"video_id": item["video_id"], "partition": item["partition"], "geometry_fingerprint": attribution._geometry_fingerprint(fake_session, item)}
                arrays = {"ig_pred_signed": np.ones((1, 2), dtype=np.float32), "valid_mask": np.ones((1, 2), dtype=np.bool_),
                          "attention_token_mass": np.ones((1, 1), dtype=np.float32), "attention_token_mask": np.ones((1, 1), dtype=np.bool_),
                          "vertical_low": low, "vertical_high": high}
                return metadata, arrays

            destination = base / "resumable-cache"
            cancelled = iter((False, True))
            with mock.patch.object(attribution, "open_explainer", return_value=session), mock.patch.object(attribution, "_compute_one", side_effect=compute):
                first = build_class_attribution_cache(base / "run", roots=object(), cache_dir=destination, splits=("train",), ig_steps=2,
                                                      should_cancel=lambda: next(cancelled))
                self.assertFalse(first.complete)
                self.assertEqual((1, 1), (first.completed, first.computed))
                first_npz = destination / "videos" / f"{hashlib.sha256(b'one').hexdigest()}.npz"
                first_npz.write_bytes(b"corrupt")
                resumed = build_class_attribution_cache(base / "run", roots=object(), cache_dir=destination, splits=("train",), ig_steps=2)
                self.assertTrue(resumed.complete)
                self.assertEqual((2, 0, 2), (resumed.completed, resumed.reused, resumed.computed))
                with self.assertRaisesRegex(ClassAttributionError, "different checkpoint, data, splits, or settings"):
                    build_class_attribution_cache(base / "run", roots=object(), cache_dir=destination, splits=("train",), ig_steps=3)

    def test_geometry_clips_the_final_ctu_to_decoded_display_height(self):
        """Padded coded 256px input has only 48 display pixels in its last CTU."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts" / "padded"; artifact.mkdir(parents=True)
            (artifact / "metadata.json").write_text(json.dumps({
                "coded_height": 256, "decoded_display_height": 240, "ctu_size": 64,
            }), encoding="utf-8")
            session = SimpleNamespace(bundle=SimpleNamespace(
                extraction={"metadata": {"artifact_root_ref": "artifacts"}}, index=root / "index.json"))
            low, high, ctu = attribution._artifact_geometry(session, {"artifact_relpath": "padded", "row_count": 4})
            self.assertEqual(64, ctu)
            np.testing.assert_allclose(low, [0.0, 64.0 / 240.0, 128.0 / 240.0, 192.0 / 240.0])
            np.testing.assert_allclose(high, [64.0 / 240.0, 128.0 / 240.0, 192.0 / 240.0, 1.0])

    def test_aggregate_rejects_semantically_corrupt_archives_and_manifests(self):
        records = [{"video_id": "one", "partition": "train", "gt": 0, "pred": 0, "signed": [1.0, 2.0]}]
        corruptions = {
            "nan IG": lambda arrays, meta: arrays.__setitem__("ig_pred_signed", np.array([[np.nan, 2.0]], dtype=np.float32)),
            "negative rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.array([[-1.0]], dtype=np.float32)),
            "nonunit rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.array([[2.0]], dtype=np.float32)),
            "wrong masks": lambda arrays, meta: arrays.__setitem__("valid_mask", np.array([[True, False]], dtype=np.bool_)),
            "wrong partition": lambda arrays, meta: meta.__setitem__("partition", "test"),
            "wrong class metadata": lambda arrays, meta: meta.update(ground_truth_class_id=1, ground_truth_label="b"),
            "wrong source hash": lambda arrays, meta: meta.__setitem__("source_sha256", "b" * 64),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self._cache(Path(directory), records)
                stem = hashlib.sha256(b"one").hexdigest(); npz_path = root / "videos" / f"{stem}.npz"; meta_path = root / "videos" / f"{stem}.json"
                with np.load(npz_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                metadata = json.loads(meta_path.read_text(encoding="utf-8")); corrupt(arrays, metadata)
                np.savez_compressed(npz_path, **arrays); meta_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ClassAttributionError, msg=name):
                    aggregate_class_attributions(root, split="separate", grid=(1, 2))
        for name, mutate in {
            "total differs from expected": lambda manifest: manifest.update(total=2, completed=2),
            "duplicate expected video": lambda manifest: manifest.update(expected_videos=manifest["expected_videos"] * 2, total=2, completed=2),
        }.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self._cache(Path(directory), records); manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")); mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ClassAttributionError, msg=name):
                    aggregate_class_attributions(root, grid=(1, 2))

    def test_schema_v2_requires_unit_rollout_mass_and_a_valid_npz_checksum(self):
        """Zero/row-repeated rollout mass and optional checksums must never become fixture compatibility paths."""
        records = [{"video_id": "two-rows", "partition": "train", "gt": 0, "pred": 0,
                    "signed": [[1.0], [2.0]]}]
        cases = {
            "zero native rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.zeros((2, 1), dtype=np.float32)),
            "one unit per row rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.ones((2, 1), dtype=np.float32)),
            "missing checksum": lambda arrays, meta: meta.pop("npz_sha256"),
            "malformed checksum": lambda arrays, meta: meta.__setitem__("npz_sha256", "not-a-64-character-hex-digest"),
            # The altered cells remain finite/shape-valid, so only the immutable NPZ digest detects this tamper.
            "finite numeric bytes changed": lambda arrays, meta: arrays.__setitem__("ig_pred_signed", np.array([[9.0], [2.0]], dtype=np.float32)),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self._cache(Path(directory), records)
                stem = hashlib.sha256(b"two-rows").hexdigest(); npz_path = root / "videos" / f"{stem}.npz"; meta_path = root / "videos" / f"{stem}.json"
                with np.load(npz_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                metadata = json.loads(meta_path.read_text(encoding="utf-8")); corrupt(arrays, metadata)
                if name not in {"missing checksum", "malformed checksum"}:
                    np.savez_compressed(npz_path, **arrays)
                    # Preserve checksum validity for the rollout semantics cases; retain the old digest for numeric tamper.
                    if name != "finite numeric bytes changed":
                        metadata["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
                meta_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ClassAttributionError, msg=name):
                    aggregate_class_attributions(root, grid=(2, 1))

    def test_build_recomputes_readable_semantic_cache_corruption(self):
        """Resume must not reuse caches that load successfully but violate the record contract."""
        variants = {
            "nan IG": lambda arrays, meta: arrays.__setitem__("ig_pred_signed", np.array([[np.nan, 1.0]], dtype=np.float32)),
            "negative rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.array([[-1.0]], dtype=np.float32)),
            "nonunit rollout": lambda arrays, meta: arrays.__setitem__("attention_token_mass", np.array([[2.0]], dtype=np.float32)),
            "wrong masks": lambda arrays, meta: arrays.__setitem__("attention_token_mask", np.array([[False]], dtype=np.bool_)),
            "wrong partition": lambda arrays, meta: meta.__setitem__("partition", "test"),
            "wrong class metadata": lambda arrays, meta: meta.update(ground_truth_class_id=1, ground_truth_label="b"),
            "wrong source hash": lambda arrays, meta: meta.__setitem__("source_sha256", "b" * 64),
        }
        for name, corrupt in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                base = Path(directory); artifact = base / "artifacts" / "one"; artifact.mkdir(parents=True)
                (artifact / "metadata.json").write_text(json.dumps({"coded_height": 64, "decoded_display_height": 64, "ctu_size": 64}), encoding="utf-8")
                item = {"video_id": "one", "partition": "train", "artifact_relpath": "one", "row_count": 1,
                        "used_time_length": 2, "class_id": 0, "source_sha256": "a" * 64}
                bundle = SimpleNamespace(partition_view=lambda split: _Partition([item] if split == "train" else []),
                    extraction={"metadata": {"artifact_root_ref": "artifacts"}}, index=base / "index.json")
                session = SimpleNamespace(checkpoint_sha256="checkpoint", class_names=("a", "b"), normalization={"mean": 0.0, "std": 1.0},
                    metadata={"checkpoint_contract": "native", "checkpoint_version": "1", "provenance": {}, "model_config": {}}, bundle=bundle)

                def compute(fake_session, current, **_kwargs):
                    low, high, _ = attribution._artifact_geometry(fake_session, current)
                    return ({"video_id": "one", "partition": "train", "ground_truth_class_id": 0, "ground_truth_label": "a",
                             "predicted_class_id": 0, "predicted_label": "a", "correct": True, "row_count": 1,
                             "used_frames": 2, "patch_size": 16, "source_sha256": "a" * 64},
                            {"ig_pred_signed": np.ones((1, 2), dtype=np.float32), "valid_mask": np.ones((1, 2), dtype=np.bool_),
                             "attention_token_mass": np.ones((1, 1), dtype=np.float32), "attention_token_mask": np.ones((1, 1), dtype=np.bool_),
                             "vertical_low": low, "vertical_high": high})

                cache = base / "cache"
                with mock.patch.object(attribution, "open_explainer", return_value=session), mock.patch.object(attribution, "_compute_one", side_effect=compute) as computed:
                    build_class_attribution_cache(base / "run", roots=object(), cache_dir=cache, splits=("train",), ig_steps=2)
                    stem = hashlib.sha256(b"one").hexdigest(); npz_path = cache / "videos" / f"{stem}.npz"; meta_path = cache / "videos" / f"{stem}.json"
                    with np.load(npz_path, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    metadata = json.loads(meta_path.read_text(encoding="utf-8")); corrupt(arrays, metadata)
                    np.savez_compressed(npz_path, **arrays); meta_path.write_text(json.dumps(metadata), encoding="utf-8")
                    resumed = build_class_attribution_cache(base / "run", roots=object(), cache_dir=cache, splits=("train",), ig_steps=2)
                self.assertEqual((0, 1), (resumed.reused, resumed.computed))
                self.assertEqual(2, computed.call_count)

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib is optional")
    def test_ui_cached_redraw_options_and_export_match_backend_arrays(self):
        """The interactive redraw is cache-only and its downloads retain backend grids/counts."""
        records = [
            {"video_id": "train-error", "partition": "train", "gt": 0, "pred": 1, "signed": [1.0, -2.0], "truth_signed": [4.0, 5.0]},
            {"video_id": "test-correct", "partition": "test", "gt": 1, "pred": 1, "signed": [3.0, -1.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = self._cache(Path(directory), records)
            selections = [
                UIOptions(classes=("a", "b"), grouping="gt", split_layout="separate", splits=("train", "test"),
                          sign="abs", ig_target="predicted", magnitude="per_video_normalized", band_bins=2, time_bins=3),
                UIOptions(classes=("a", "b"), grouping="pred", split_layout="combined", splits=("train", "test"),
                          correctness="incorrect", pooled_misclassified=True, sign="signed", ig_target="truth", magnitude="raw", band_bins=2, time_bins=3),
            ]
            for options in selections:
                result = aggregate_cached(root, options)
                self.assertEqual((options.band_bins, options.time_bins), next(iter(result.groups.values())).ig_grid.shape)
                self.assertEqual(options.grouping, result.metadata["settings"]["group_by"])
                self.assertEqual(options.sign, result.metadata["settings"]["sign"])
                self.assertEqual(options.ig_target, result.metadata["settings"]["ig_target"])
                self.assertEqual(options.magnitude, result.metadata["settings"]["magnitude"])
            options = selections[0]
            result = aggregate_cached(root, options)

            class Value:
                def __init__(self, value): self.value = value
            class Output:
                def __enter__(self): return self
                def __exit__(self, *_args): return False
                def clear_output(self, **_kwargs): pass
            ui = ClassAttributionUI(training_root=Path(directory) / "no-run", roots={}, cache_dir=root)
            # Construction stays inert; the UI deliberately has no stale-result memo.
            self.assertFalse(hasattr(ui, "results"))
            if __import__("importlib").util.find_spec("IPython"):
                ui._widgets = {
                    "classes": Value(options.classes), "grouping": Value(options.grouping), "layout": Value(options.split_layout),
                    "splits": Value(options.splits), "correctness": Value(options.correctness), "pooled": Value(options.pooled_misclassified),
                    "target": Value(options.ig_target), "sign": Value(options.sign), "magnitude": Value(options.magnitude),
                    "time_bins": Value(options.time_bins), "band_bins": Value(options.band_bins), "scale": Value("shared"),
                    "preview": Value(False), "status": Value(""), "output": Output(), "html": lambda text: text,
                }
                # _redraw only aggregates/exports the supplied cache; it never calls the build action.
                with mock.patch("IPython.display.display"):
                    ui._redraw()
                self.assertIn("Cached aggregation only; no model rerun", ui._widgets["status"].value)

            exported = export_rendered_aggregate(result, options, Path(directory) / "export")
            metadata = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
            for group in metadata["groups"]:
                expected = result.groups[group["key"]]
                self.assertEqual(expected.count, group["count"])
                actual = np.loadtxt(exported / group["ig_csv"], delimiter=",")
                np.testing.assert_allclose(np.atleast_2d(actual), expected.ig_grid)
            self.assertTrue((exported / "class_attribution_maps.png").is_file())
            self.assertTrue((exported / "ui_rendering_metadata.json").is_file())
            with __import__("zipfile").ZipFile(exported.with_suffix(".zip")) as archive:
                self.assertIn("metadata.json", archive.namelist())
                self.assertIn("class_attribution_maps.png", archive.namelist())
                self.assertIn("ui_rendering_metadata.json", archive.namelist())

    def test_single_refresh_preserves_selected_available_video(self):
        """A cache-only refresh must not discard a still-filtered selected record."""
        records = [
            SimpleNamespace(video_id="first", split="train", truth_label="a", predicted_label="b", correct=False),
            SimpleNamespace(video_id="second", split="test", truth_label="b", predicted_label="b", correct=True),
        ]
        listing = SimpleNamespace(records=tuple(records), cache_status="building", available_videos=2,
                                  expected_videos=3, unavailable_videos=1, publication_in_progress_or_missing=1,
                                  invalid_records={})
        class Value:
            def __init__(self, value=None): self.value = value
        class Choice(Value):
            def __init__(self): super().__init__(None); self.options = []
        ui = ClassAttributionUI(training_root="/no-run", roots={}, cache_dir="/synthetic-cache")
        ui._widgets = {"single_split": Value("all"), "single_class": Value("all"), "single_correctness": Value("all"),
                       "single_video": Choice(), "single_status": Value("")}
        with mock.patch("src.wpp_class_attribution_ui.list_single_cached_records", return_value=listing):
            ui._refresh_single_records()
            ui._widgets["single_video"].value = "second"
            ui._refresh_single_records()
        self.assertEqual("second", ui._widgets["single_video"].value)


if __name__ == "__main__":
    unittest.main()
