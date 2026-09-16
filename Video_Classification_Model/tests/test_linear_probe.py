from __future__ import annotations

import copy
import io
import json
import sys
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import linear_probe as probe
from model_resnet_tsc import ResNet
from supervised_data import FEATURE_COLUMNS, automatic_split, load_pair


def write_pair(folder: Path, rows: np.ndarray, labels: list[str]) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    data, label = folder / "data.csv", folder / "labels.csv"
    pd.DataFrame(rows, columns=FEATURE_COLUMNS).to_csv(data, index=False)
    pd.DataFrame({"label": labels}).to_csv(label, index=False)
    return data, label


def sample_rows(n: int) -> np.ndarray:
    base = np.arange(3000, dtype=np.float64)
    return np.stack([base + index * .01 for index in range(n)])


class LinearProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
        self.data, self.labels = write_pair(self.root, sample_rows(30), labels)
        source = load_pair(self.data, self.labels)
        pieces, details = automatic_split(source, seed=4)
        self.manifest = {"mode": "auto", "sources": {"source": {
            "data_path": str(source.data_path), "label_path": str(source.label_path),
            "data_checksum_sha256": source.data_checksum, "label_checksum_sha256": source.label_checksum,
            "row_count": len(source.labels)}}, "splits": details}
        model = ResNet((1, 3000), 3, 1)
        self.checkpoint = self.root / "run" / "checkpoints" / "best.pt"
        self.checkpoint.parent.mkdir(parents=True)
        torch.save({"format_version": 1, "completed_epoch": 7, "model_state": model.state_dict(), "model_config": {"initial_feature_maps": 1},
                    "preprocessing_config": probe.PREPROCESSING, "label_mapping": {"a": 0, "b": 1, "c": 2},
                    "split_information": self.manifest}, self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def test_device_defaults_to_auto_and_reports_resolved_device(self):
        self.assertEqual(probe.parser().parse_args(["--checkpoint", "checkpoint.pt"]).device, "auto")
        output = io.StringIO()
        with mock.patch.object(probe, "choose_device", return_value=torch.device("mps")) as choose, \
             mock.patch.object(probe, "_load_checkpoint", side_effect=RuntimeError("stop")), \
             redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                probe.run_linear_probe(self.checkpoint)
        choose.assert_called_once_with("auto")
        self.assertEqual(output.getvalue(), "Resolved device: mps\n")

    def test_relocated_source_requires_checksum_and_split_integrity(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        pieces, mapping = probe.load_recorded_splits(cp, str(self.data), str(self.labels))
        self.assertEqual(mapping["a"], 0)
        self.assertEqual(sum(len(pair.labels) for pair in pieces.values()), 30)
        changed = sample_rows(30); changed[0, 0] += 1
        bad_data, _ = write_pair(self.root / "changed", changed, ["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            probe.load_recorded_splits(cp, str(bad_data), str(self.labels))

    def test_rejects_overlapping_or_tampered_saved_splits(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        overlap = copy.deepcopy(cp)
        overlap["split_information"]["splits"]["val"]["source_indices"][0] = overlap["split_information"]["splits"]["train"]["source_indices"][0]
        overlap["split_information"]["splits"]["val"]["row_hashes"][0] = overlap["split_information"]["splits"]["train"]["row_hashes"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            probe.load_recorded_splits(overlap)
        tampered = copy.deepcopy(cp)
        tampered["split_information"]["splits"]["test"]["row_hashes"].pop()
        with self.assertRaisesRegex(ValueError, "row hashes"):
            probe.load_recorded_splits(tampered)

    def test_rejects_omitted_automatic_unique_row(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        omitted = copy.deepcopy(cp)
        omitted["split_information"]["splits"]["test"]["source_indices"].pop()
        omitted["split_information"]["splits"]["test"]["row_hashes"].pop()
        omitted["split_information"]["splits"]["test"]["row_count"] -= 1
        with self.assertRaisesRegex(ValueError, "incomplete"):
            probe.load_recorded_splits(omitted)

    def test_manual_split_allows_duplicates_within_one_source(self):
        train_rows = sample_rows(3); train_rows[1] = train_rows[0]
        train = load_pair(*write_pair(self.root / "manual-train", train_rows, ["a", "a", "b"]))
        val = load_pair(*write_pair(self.root / "manual-val", sample_rows(2) + 100, ["a", "b"]))
        test = load_pair(*write_pair(self.root / "manual-test", sample_rows(2) + 200, ["a", "b"]))
        def meta(pair):
            return {"data_path": str(pair.data_path), "label_path": str(pair.label_path),
                    "data_checksum_sha256": pair.data_checksum, "label_checksum_sha256": pair.label_checksum,
                    "row_count": len(pair.labels)}
        def detail(pair):
            return {"source_indices": pair.source_indices.tolist(), "row_hashes": list(pair.row_hashes),
                    "row_count": len(pair.labels)}
        manual = {"label_mapping": {"a": 0, "b": 1}, "split_information": {
            "mode": "manual", "sources": {"train": meta(train), "val": meta(val), "test": meta(test)},
            "splits": {"train": detail(train), "val": detail(val), "test": detail(test)}}}
        pieces, _ = probe.load_recorded_splits(manual)
        self.assertEqual(len(pieces["train"].row_hashes), 3)

    def test_malformed_checkpoint_version_labels_and_numeric_inputs(self):
        bad_version = self.root / "bad-version.pt"
        raw = torch.load(self.checkpoint, weights_only=False); raw["format_version"] = 2; torch.save(raw, bad_version)
        with self.assertRaisesRegex(ValueError, "format_version"):
            probe._load_checkpoint(bad_version, torch.device("cpu"))
        bad_labels = self.root / "bad-labels.pt"
        raw["format_version"] = 1; raw["label_mapping"] = {"a": 0, "b": 2}; torch.save(raw, bad_labels)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            probe._load_checkpoint(bad_labels, torch.device("cpu"))
        for c_values, max_iter in (((float("nan"),), 1), ((0,), 1), ((1,), 0)):
            with self.assertRaises(ValueError):
                probe.run_linear_probe(self.checkpoint, c_values=c_values, max_iter=max_iter)

    def test_probe_selects_without_test_labels_and_writes_artifacts(self):
        # This is a parent-run final-test artifact: it is a display-only reference,
        # not an input to the selection fits below.
        reference = {"evaluation_kind": "original_test", "checkpoint": str(self.checkpoint.resolve()),
                     "accuracy": .8, "balanced_accuracy": .75, "macro_f1": .7, "test_loss": 1.2}
        (self.checkpoint.parent.parent / "test_metrics.json").write_text(json.dumps(reference))
        def fake_features(_model, pair, _device, _batch):
            # All blocks have separable train/validation features.  The test
            # labels are intentionally irrelevant to C selection.
            y = np.asarray([{ "a": 0, "b": 1, "c": 2}[label] for label in pair.labels])
            feature = np.eye(3, dtype=float)[y]
            return {1: feature, 2: feature * 2, 3: feature * 3}

        fit_sizes = []
        original_fit = probe._fit_probe
        def observing_fit(x, y, c, max_iter):
            fit_sizes.append(len(x))
            return original_fit(x, y, c, max_iter)
        with mock.patch.object(probe, "reconstruct_encoder", return_value=object()), \
             mock.patch.object(probe, "extract_block_features", side_effect=fake_features), \
             mock.patch.object(probe, "_fit_probe", side_effect=observing_fit):
            out = probe.run_linear_probe(self.checkpoint, c_values=(1.0, 0.1), output_root=self.root / "artifacts")
        # The six selection fits use training features only; the three later
        # fits are the explicitly permitted train+validation refits.
        train_count = len(automatic_split(load_pair(self.data, self.labels), seed=4)[0]["train"].labels)
        self.assertEqual(fit_sizes[:6], [train_count] * 6)
        rows = json.loads((out / "selection_metrics.json").read_text())
        self.assertEqual(len(rows), 6)
        for row in rows:
            for split in ("train", "validation"):
                for metric in ("accuracy", "macro_f1", "multinomial_log_loss"):
                    self.assertIn(f"{split}_{metric}", row)
                    self.assertTrue(np.isfinite(row[f"{split}_{metric}"]))
        for block in (1, 2, 3):
            metrics = json.loads((out / f"block{block}" / "final_metrics.json").read_text())
            self.assertEqual(metrics["selected_C"], .1)  # smaller C breaks tied validation F1
            self.assertIn("multinomial_log_loss", metrics)
            self.assertTrue(np.isfinite(metrics["multinomial_log_loss"]))
            self.assertTrue((out / f"block{block}" / "scaler.joblib").exists())
            self.assertTrue((out / f"block{block}" / "confusion_matrix_raw.csv").exists())
        for name in ("selection_accuracy_vs_C.png", "selection_macro_f1_vs_C.png",
                     "selection_log_loss_vs_C.png", "final_test_metrics_by_block.png"):
            self.assertTrue((out / name).is_file(), name)
        metadata = json.loads((out / "metadata.json").read_text())
        saved_reference = metadata["original_resnet_test_reference"]
        self.assertEqual(saved_reference["status"], "available")
        self.assertEqual(saved_reference["values"]["multinomial_log_loss"], 1.2)
        self.assertEqual(saved_reference["source_keys"]["multinomial_log_loss"], "test_loss")

    def test_log_loss_uses_full_label_order_and_reference_failures_are_safe(self):
        class FakeProbe:
            classes_ = np.asarray([0, 2])
        probabilities = probe._full_probabilities(FakeProbe(), np.asarray([[.9, .1], [.2, .8]]), 3)
        values = probe._classification_values(np.asarray([0, 2]), np.asarray([0, 2]), probabilities, 3)
        self.assertTrue(np.isfinite(values["multinomial_log_loss"]))
        self.assertAlmostEqual(values["multinomial_log_loss"], (-np.log(.9) - np.log(.8)) / 2)
        # Missing, malformed, and mismatched-source artifacts are deliberately
        # non-fatal and are recorded as unavailable reference states.
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve())["status"], "unavailable")
        source = self.checkpoint.parent.parent / "test_metrics.json"
        source.write_text("not json")
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve())["status"], "malformed")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": "/wrong.pt", "accuracy": .5}))
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve())["status"], "invalid_provenance")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(self.checkpoint.resolve()),
                                      "accuracy": .5, "macro_f1": "not-a-number"}))
        partial = probe._load_original_test_reference(self.checkpoint.resolve())
        self.assertEqual(partial["status"], "available")
        self.assertEqual(partial["values"], {"accuracy": .5})
        self.assertEqual(set(partial["omitted_metrics"]), {"balanced_accuracy", "macro_f1", "multinomial_log_loss"})

    def test_reference_accepts_strict_run_relative_relocation_and_rejects_mismatches(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        source = self.checkpoint.parent.parent / "test_metrics.json"
        # The old project-root prefix is stale, but the run name and checkpoint
        # suffix identify this exact run/checkpoint and epoch structurally.
        stale = Path("/old/Proj-Entropy") / self.checkpoint.parent.parent.name / "checkpoints" / self.checkpoint.name
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(stale),
                                      "checkpoint_epoch": 7, "sample_count": cp["split_information"]["splits"]["test"]["row_count"],
                                      "accuracy": .5}))
        accepted = probe._load_original_test_reference(self.checkpoint.resolve(), cp)
        self.assertEqual(accepted["status"], "available")
        self.assertEqual(accepted["validation"]["method"], "relocated_run_relative")
        self.assertEqual(accepted["validation"]["details"]["current_checkpoint_epoch"], 7)
        wrong_run = Path("/old/Proj-Entropy/another-run/checkpoints/best.pt")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(wrong_run),
                                      "checkpoint_epoch": 7, "accuracy": .5}))
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve(), cp)["status"], "invalid_provenance")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(stale.with_name("latest.pt")),
                                      "checkpoint_epoch": 7, "accuracy": .5}))
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve(), cp)["status"], "invalid_provenance")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(stale),
                                      "checkpoint_epoch": 6, "accuracy": .5}))
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve(), cp)["status"], "invalid_provenance")
        source.write_text(json.dumps({"evaluation_kind": "original_test", "checkpoint": str(stale),
                                      "checkpoint_epoch": 7, "sample_count": 999, "accuracy": .5}))
        self.assertEqual(probe._load_original_test_reference(self.checkpoint.resolve(), cp)["status"], "invalid_provenance")

    def test_extract_features_preserves_order_dimensions_and_gap(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        model = probe.reconstruct_encoder(cp, torch.device("cpu"))
        pair = load_pair(self.data, self.labels).subset([2, 0, 1])
        features = probe.extract_block_features(model, pair, torch.device("cpu"), batch_size=2)
        self.assertEqual(set(features), {1, 2, 3})
        self.assertEqual(features[1].shape, (3, 1))
        self.assertEqual(features[2].shape, (3, 2))
        self.assertEqual(features[3].shape, (3, 2))
        values = torch.from_numpy(probe.normalize_rows(pair.values)[:, None, :])
        with torch.inference_mode():
            one = model.layer1(values); two = model.layer2(one); three = model.layer3(two)
        np.testing.assert_allclose(features[1], one.mean(2).numpy())
        np.testing.assert_allclose(features[2], two.mean(2).numpy())
        np.testing.assert_allclose(features[3], model.gap(three).flatten(1).numpy())

    def test_encoder_is_frozen_and_block_three_matches_gap(self):
        cp = probe._load_checkpoint(self.checkpoint, torch.device("cpu"))
        model = probe.reconstruct_encoder(cp, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertTrue(all(not item.requires_grad for item in model.parameters()))
        x = torch.randn(2, 1, 3000)
        with torch.inference_mode():
            block3 = model.layer3(model.layer2(model.layer1(x)))
            np.testing.assert_allclose(block3.mean(2).numpy(), model.gap(block3).flatten(1).numpy())


if __name__ == "__main__":
    unittest.main()
