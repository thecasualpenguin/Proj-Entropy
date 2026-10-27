from __future__ import annotations

import json
import sys
from fractions import Fraction
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from src.wpp_class_attribution import (CACHE_SCHEMA_VERSION, ClassAttributionError, _CacheWriterLock,
                                       _artifact_geometry, _geometry_fingerprint_from,
                                       aggregate_class_attributions, build_class_attribution_cache,
                                       export_class_attributions)
from src.wpp_explain_types import DataRoots
from src.wpp_media import ValidatedSourceMediaFrames
from src.native_wpp_workflow import PresentationProbe


class ClassAttributionAggregationTests(unittest.TestCase):
    def _cache(self, directory: Path, *, correct: bool = False) -> Path:
        root = directory / "cache"; (root / "videos").mkdir(parents=True)
        fingerprint = {"class_names": ["a", "b"], "splits": ["train"], "marker": "test"}
        video_id = "video-1"
        # The known safe cache filename is simply SHA256(video_id); avoid depending on a private helper in the test.
        import hashlib
        stem = hashlib.sha256(video_id.encode()).hexdigest()
        manifest = {"schema": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint,
                    "expected_videos": [{"video_id": video_id, "partition": "train", "source_sha256": "a" * 64,
                                         "class_id": 0, "label": "a", "row_count": 1, "used_frames": 2}],
                    "status": "complete", "completed": 1, "total": 1}
        (root / "manifest.json").write_text(json.dumps(manifest))
        metadata = {"cache_fingerprint": fingerprint, "video_id": video_id, "partition": "train", "source_sha256": "a" * 64,
                    "ground_truth_class_id": 0, "ground_truth_label": "a",
                    "predicted_class_id": 0 if correct else 1, "predicted_label": "a" if correct else "b",
                    "correct": correct, "row_count": 1, "used_frames": 2, "patch_size": 16,
                    "ig_truth_source": "predicted_reused" if correct else "separate_truth_logit"}
        # Both signed cells share a single coarse cell.  Correct abs behavior is 2, not abs(1 + -1) == 0.
        arrays = {"ig_pred_signed": np.array([[1.0, -1.0]], dtype=np.float32),
                  "valid_mask": np.ones((1, 2), dtype=np.bool_),
                  "attention_token_mass": np.array([[1.0]], dtype=np.float32),
                  "attention_token_mask": np.array([[True]]),
                  "vertical_low": np.array([0.0]), "vertical_high": np.array([1.0]),
                  "ig_truth_signed": np.array([[3.0, 4.0]], dtype=np.float32)}
        npz_path = root / "videos" / f"{stem}.npz"; np.savez_compressed(npz_path, **arrays)
        metadata["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        (root / "videos" / f"{stem}.json").write_text(json.dumps(metadata))
        return root

    def test_absolute_is_native_before_coarse_resampling_and_attention_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = aggregate_class_attributions(self._cache(Path(temporary)), split="combined", group_by="pred",
                                                   ig_target="pred", sign="abs", grid=(1, 1))
            group = result.groups["combined|pred:b"]
            self.assertEqual(group.count, 1)
            self.assertAlmostEqual(float(group.ig_grid[0, 0]), 2.0)
            # Native rollout token has unit mass although its patch covers two presentation frames.
            self.assertAlmostEqual(float(group.attention_grid.sum()), 1.0)
            self.assertIn("native signed cells before overlap resampling", result.metadata["semantics"]["ig"])

    def test_truth_target_uses_separate_misclassified_cache_and_empty_groups_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = aggregate_class_attributions(self._cache(Path(temporary)), split="separate", group_by="gt",
                                                   ig_target="truth", sign="signed", grid=(2, 2))
            self.assertAlmostEqual(float(result.groups["train|gt:a"].ig_grid.sum()), 7.0)
            self.assertEqual(result.groups["train|gt:b"].count, 0)
            self.assertEqual(result.groups["train|pooled_misclassified"].count, 1)

    def test_incomplete_cache_needs_explicit_preview_and_export_is_numeric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._cache(Path(temporary))
            manifest_path = root / "manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["status"] = "cancelled"; manifest["completed"] = 0; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ClassAttributionError, "incomplete"):
                aggregate_class_attributions(root)
            result = aggregate_class_attributions(root, preview=True, split="combined", grid=(2, 2))
            export = Path(temporary) / "export"; export_class_attributions(result, export, zip_output=True)
            self.assertTrue((export / "metadata.json").is_file())
            self.assertTrue(list((export / "data").glob("*-ig.csv")))
            self.assertTrue(export.with_suffix(".zip").is_file())


class ClassAttributionCacheBuildTests(unittest.TestCase):
    def _session(self):
        return SimpleNamespace(
            checkpoint_sha256="checkpoint", class_names=("a", "b"), normalization={"mean": 0.0, "std": 1.0},
            metadata={"checkpoint_contract": "native_wpp_checkpoint", "checkpoint_version": "1.0",
                      "provenance": {"bundle": "bound"}, "model_config": {"patch_size": 16}})

    @staticmethod
    def _items():
        return [("train", {"video_id": "one", "partition": "train", "class_id": 0, "label": "a", "source_sha256": "a" * 64,
                           "row_count": 1, "used_time_length": 1}),
                ("test", {"video_id": "two", "partition": "test", "class_id": 0, "label": "a", "source_sha256": "b" * 64,
                          "row_count": 1, "used_time_length": 1})]

    @staticmethod
    def _compute(_session, item, **_kwargs):
        metadata = {"video_id": item["video_id"], "partition": item["partition"],
                    "predicted_class_id": 0, "predicted_label": "a", "correct": True,
                    "patch_size": 16, "ig_truth_source": "predicted_reused"}
        arrays = {"ig_pred_signed": np.ones((1, 1), dtype=np.float32), "valid_mask": np.ones((1, 1), dtype=np.bool_),
                  "attention_token_mass": np.ones((1, 1), dtype=np.float32), "attention_token_mask": np.ones((1, 1), dtype=np.bool_),
                  "vertical_low": np.array([0.0]), "vertical_high": np.array([1.0])}
        return metadata, arrays

    def test_one_video_pilot_pauses_then_resumes_without_changing_full_expected_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"; cache = root / "class_attributions"
            roots = DataRoots("bundle", "manifest", "index", "source")
            with patch("src.wpp_class_attribution.open_explainer", return_value=self._session()), \
                 patch("src.wpp_class_attribution._record_items", return_value=self._items()), \
                 patch("src.wpp_class_attribution._artifact_geometry", return_value=(np.array([0.0]), np.array([1.0]), 64)), \
                 patch("src.wpp_class_attribution._compute_one", side_effect=self._compute) as compute:
                first = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu",
                                                      max_new_videos=1, prioritize_video_ids=["two"])
                self.assertFalse(first.complete); self.assertEqual((first.computed, first.total), (1, 2))
                manifest = json.loads((cache / "manifest.json").read_text())
                self.assertEqual((manifest["status"], len(manifest["expected_videos"])), ("paused", 2))
                second = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu")
                self.assertTrue(second.complete); self.assertEqual((second.reused, second.computed), (1, 1))
                self.assertEqual(compute.call_count, 2)
                # A fully valid cache stays byte-compatible and is reused without recomputation.
                third = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu")
                self.assertTrue(third.complete); self.assertEqual((third.reused, third.computed), (2, 0))
                self.assertEqual(compute.call_count, 2)

    def test_paused_manifest_counts_a_valid_later_cached_record_before_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"; cache = root / "class_attributions"; roots = DataRoots("bundle", "manifest", "index", "source")
            items = self._items() + [("test", {"video_id": "three", "partition": "test", "class_id": 0, "label": "a", "source_sha256": "c" * 64,
                                                  "row_count": 1, "used_time_length": 1})]
            with patch("src.wpp_class_attribution.open_explainer", return_value=self._session()), \
                 patch("src.wpp_class_attribution._record_items", return_value=items), \
                 patch("src.wpp_class_attribution._artifact_geometry", return_value=(np.array([0.0]), np.array([1.0]), 64)), \
                 patch("src.wpp_class_attribution._compute_one", side_effect=self._compute):
                # Seed only canonical-later ``three``.
                first = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu",
                                                      max_new_videos=1, prioritize_video_ids=["three"])
                self.assertEqual((1, 1), (first.completed, first.computed))
                # Cache ``one`` then stop before uncached ``two``; ``three`` must remain counted.
                paused = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu", max_new_videos=1)
                self.assertFalse(paused.complete)
                self.assertEqual((2, 3, 1), (paused.completed, paused.total, paused.computed))
                manifest = json.loads((cache / "manifest.json").read_text())
                self.assertEqual(("paused", 2), (manifest["status"], manifest["completed"]))

    def test_geometry_and_fingerprint_do_not_depend_on_final_frame_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); artifact = root / "artifacts" / "one"; artifact.mkdir(parents=True)
            (artifact / "metadata.json").write_text(json.dumps({"ctu_size": 64}), encoding="utf-8")
            source = root / "sources" / "class" / "a.mp4"; source.parent.mkdir(parents=True); source.write_bytes(b"source")
            times = tuple(Fraction(index, 30) for index in range(184))
            probe = PresentationProbe(3840, 2160, 3840, 2160, "1:1", Fraction(1, 30), times)
            common = dict(source_path=source.resolve(), probe=probe, artifact_presentation_times=times,
                          rotation_degrees=0, sample_aspect_ratio="1:1")
            with_duration = ValidatedSourceMediaFrames(**common, frame_durations={times[-1]: Fraction(1, 30)})
            without_duration = ValidatedSourceMediaFrames(**common, frame_durations={})
            session = SimpleNamespace(roots=DataRoots("bundle", "manifest", "index", root / "sources"), bundle=SimpleNamespace(
                extraction={"metadata": {"artifact_root_ref": "artifacts"}}, index=root / "index.json"))
            item = {"video_id": "one", "artifact_relpath": "one", "source_relative_path": "class/a.mp4",
                    "source_sha256": "a" * 64, "original_frame_count": 184, "used_time_length": 184, "row_count": 34}
            with patch("src.wpp_class_attribution.validate_source_media_frames", side_effect=(with_duration, without_duration)) as validate:
                duration_geometry = _artifact_geometry(session, item)
                missing_duration_geometry = _artifact_geometry(session, item)
            self.assertEqual(2, validate.call_count)
            np.testing.assert_array_equal(duration_geometry[0], missing_duration_geometry[0])
            np.testing.assert_array_equal(duration_geometry[1], missing_duration_geometry[1])
            self.assertEqual(_geometry_fingerprint_from(duration_geometry), _geometry_fingerprint_from(missing_duration_geometry))

    def test_exclusive_cache_writer_lock_rejects_another_live_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = _CacheWriterLock(Path(temporary) / "cache"); second = _CacheWriterLock(Path(temporary) / "cache")
            first.acquire()
            try:
                with self.assertRaisesRegex(ClassAttributionError, "another class-attribution cache builder"):
                    second.acquire()
            finally:
                first.release()
            second.acquire(); second.release()

    def test_corrupt_npz_is_recomputed_and_changed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"; cache = root / "class_attributions"; roots = DataRoots("bundle", "manifest", "index", "source")
            with patch("src.wpp_class_attribution.open_explainer", return_value=self._session()), \
                 patch("src.wpp_class_attribution._record_items", return_value=self._items()), \
                 patch("src.wpp_class_attribution._artifact_geometry", return_value=(np.array([0.0]), np.array([1.0]), 64)), \
                 patch("src.wpp_class_attribution._compute_one", side_effect=self._compute) as compute:
                build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu", max_new_videos=1)
                npz = next((cache / "videos").glob("*.npz")); npz.write_bytes(b"not an npz")
                resumed = build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu", max_new_videos=1)
                self.assertEqual((resumed.reused, resumed.computed), (0, 1))
                with self.assertRaisesRegex(ClassAttributionError, "different checkpoint"):
                    build_class_attribution_cache(root, roots=roots, cache_dir=cache, device="cpu", ig_steps=8)
                self.assertEqual(compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
