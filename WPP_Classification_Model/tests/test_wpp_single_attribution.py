from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from src.wpp_class_attribution import CACHE_SCHEMA_VERSION, ClassAttributionError
from src.wpp_single_attribution import list_single_video_records, load_single_video_attribution


class SingleVideoAttributionTests(unittest.TestCase):
    def _cache(self, directory: Path, *, correct: bool = False, include_record: bool = True,
               status: str = "building", zero_ig: bool = False) -> tuple[Path, Path, Path]:
        root = directory / "cache"
        (root / "videos").mkdir(parents=True)
        fingerprint = {"class_names": ["truth", "prediction"], "splits": ["train"],
                       "model_config": {"patch_size": 2}, "marker": "single-test"}
        expected = [
            {"video_id": "ready", "partition": "train", "source_sha256": "a" * 64,
             "class_id": 0, "label": "truth", "row_count": 2, "used_frames": 3},
            {"video_id": "publishing", "partition": "train", "source_sha256": "b" * 64,
             "class_id": 1, "label": "prediction", "row_count": 1, "used_frames": 1},
        ]
        (root / "manifest.json").write_text(json.dumps({
            "schema": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint, "expected_videos": expected,
            "status": status, "completed": 1 if include_record else 0, "total": 2,
        }), encoding="utf-8")
        stem = hashlib.sha256(b"ready").hexdigest()
        meta_path, npz_path = root / "videos" / f"{stem}.json", root / "videos" / f"{stem}.npz"
        if include_record:
            signed = np.zeros((2, 3), dtype=np.float32) if zero_ig else np.array([[-2.0, 1.0, 0.0], [3.0, -1.0, .5]], dtype=np.float32)
            arrays = {
                "ig_pred_signed": signed,
                "valid_mask": np.ones((2, 3), dtype=np.bool_),
                # Valid token masses sum to one. The second token covers only frame 2.
                "attention_token_mass": np.array([[.1, .2], [.3, .4]], dtype=np.float32),
                "attention_token_mask": np.ones((2, 2), dtype=np.bool_),
                "vertical_low": np.array([0.0, .5], dtype=np.float64),
                "vertical_high": np.array([.5, 1.0], dtype=np.float64),
            }
            if not correct:
                arrays["ig_truth_signed"] = np.array([[4.0, -5.0, 1.0], [2.0, 0.0, -3.0]], dtype=np.float32)
            np.savez_compressed(npz_path, **arrays)
            meta = {
                "cache_fingerprint": fingerprint, "video_id": "ready", "partition": "train",
                "source_sha256": "a" * 64, "ground_truth_class_id": 0, "ground_truth_label": "truth",
                "predicted_class_id": 0 if correct else 1,
                "predicted_label": "truth" if correct else "prediction", "correct": correct,
                "row_count": 2, "used_frames": 3, "patch_size": 2,
                "ig_truth_source": "predicted_reused" if correct else "separate_truth_logit",
                "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return root, meta_path, npz_path

    def test_native_axes_abs_display_scale_and_piecewise_attention_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, _npz = self._cache(Path(temporary))
            result = load_single_video_attribution(root, "ready")
            np.testing.assert_array_equal(result.row_indices, [0, 1])
            np.testing.assert_array_equal(result.frame_indices, [0, 1, 2])
            # Contract exposes native shape directly; no 32x192/rebinning is introduced.
            self.assertEqual(result.ig_values.shape, (2, 3))
            np.testing.assert_allclose(result.ig_values, [[2 / 3, 1 / 3, 0], [1, 1 / 3, 1 / 6]])
            self.assertEqual(result.ig_display_scale, 3.0)
            # Token values [0.1, 0.2] and [0.3, 0.4] repeat through frame patches [0,1] and [2].
            np.testing.assert_allclose(result.attention_frame_raw, [[.1, .1, .2], [.3, .3, .4]])
            np.testing.assert_allclose(result.attention_frame_values, [[.25, .25, .5], [.75, .75, 1]])
            np.testing.assert_allclose(result.attention_mass_per_frame, [[.05, .05, .2], [.15, .15, .4]])
            self.assertAlmostEqual(float(result.attention_mass_per_frame.sum()), 1.0)
            self.assertIn("not per-frame conserved mass", result.metadata["attention_units"])

    def test_truth_target_reuses_predicted_for_correct_and_raw_preserves_native_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, _npz = self._cache(Path(temporary), correct=True)
            result = load_single_video_attribution(root, "ready", ig_target="truth", sign="signed", value_mode="raw")
            self.assertEqual(result.truth_target_source, "predicted_reused")
            np.testing.assert_allclose(result.ig_values, [[-2, 1, 0], [3, -1, .5]])
            np.testing.assert_allclose(result.ig_native_signed, result.ig_values)
            self.assertEqual(result.metadata["value_mode"], "raw")

    def test_truth_target_uses_misclassified_truth_array_and_scale_is_per_video_max_abs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, _npz = self._cache(Path(temporary), correct=False)
            result = load_single_video_attribution(root, "ready", ig_target="truth", sign="signed")
            self.assertEqual(result.truth_target_source, "separate_truth_logit")
            self.assertEqual(result.ig_display_scale, 5.0)
            np.testing.assert_allclose(result.ig_values, [[.8, -1, .2], [.4, 0, -.6]])

    def test_partial_cache_lists_valid_record_and_marks_unpublished_selection_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, _npz = self._cache(Path(temporary), status="building")
            listing = list_single_video_records(root)
            self.assertEqual(listing.cache_status, "building")
            self.assertEqual((listing.expected_videos, listing.available_videos), (2, 1))
            self.assertEqual(listing.publication_in_progress_or_missing, 1)
            self.assertEqual(listing.records[0].video_id, "ready")
            with self.assertRaisesRegex(ClassAttributionError, "not available"):
                load_single_video_attribution(root, "publishing")

    def test_patch_size_metadata_tampering_is_invalid_and_cannot_change_frame_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, meta_path, _npz = self._cache(Path(temporary))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["patch_size"] = 1  # NPZ checksum intentionally remains valid.
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            listing = list_single_video_records(root)
            self.assertIn("ready", listing.invalid_records)
            with self.assertRaisesRegex(ClassAttributionError, "patch size"):
                load_single_video_attribution(root, "ready")

    def test_present_corruption_is_visible_in_listing_and_selected_record_rejects_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, npz = self._cache(Path(temporary))
            npz.write_bytes(b"tampered")
            listing = list_single_video_records(root)
            self.assertEqual(listing.available_videos, 0)
            self.assertIn("ready", listing.invalid_records)
            with self.assertRaises(ClassAttributionError):
                load_single_video_attribution(root, "ready")

    def test_zero_scale_is_defined_as_zero_not_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _meta, _npz = self._cache(Path(temporary), zero_ig=True)
            result = load_single_video_attribution(root, "ready")
            self.assertEqual(result.ig_display_scale, 0.0)
            self.assertTrue(np.array_equal(result.ig_values, np.zeros((2, 3), dtype=np.float32)))
            self.assertTrue(np.isfinite(result.ig_values).all())


if __name__ == "__main__":
    unittest.main()
