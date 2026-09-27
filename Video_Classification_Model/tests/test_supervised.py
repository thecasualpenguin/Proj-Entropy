from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import run_pipeline as app
from supervised_data import (FEATURE_COLUMNS, automatic_split, load_pair, manual_split,
                             normalize_rows)


def write_pair(folder: Path, stem: str, rows: np.ndarray, labels, columns=FEATURE_COLUMNS):
    data, label = folder / f"{stem}-data.csv", folder / f"{stem}-labels.csv"
    pd.DataFrame(rows, columns=columns).to_csv(data, index=False)
    pd.DataFrame({"class_label": labels}).to_csv(label, index=False)
    return data, label


def rows(n: int, offset: float = 0) -> np.ndarray:
    base = np.arange(3000, dtype=np.float64)
    return np.stack([base + offset + i * 0.01 for i in range(n)])


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory(); self.tmp = Path(self.tmp_obj.name)

    def tearDown(self): self.tmp_obj.cleanup()

    def test_schema_label_normalization_and_preprocessing(self):
        data, labels = write_pair(self.tmp, "ok", rows(2), [" Sports ", "sPoRtS"])
        pair = load_pair(data, labels)
        self.assertEqual(pair.labels.tolist(), ["sports", "sports"])
        z = normalize_rows(pair.values)
        np.testing.assert_allclose(z.mean(1), 0, atol=2e-6)
        np.testing.assert_allclose(z.std(1), 1, atol=2e-6)
        self.assertEqual(z.dtype, np.float32)

    def test_rejects_wrong_order_missing_nonfinite_negative_and_constant(self):
        cases = []
        wrong = list(FEATURE_COLUMNS); wrong[0], wrong[1] = wrong[1], wrong[0]
        cases.append((rows(1), wrong))
        for i, matrix in enumerate((np.full((1, 3000), np.nan), np.full((1, 3000), -1.), np.ones((1, 3000)))):
            d, l = write_pair(self.tmp, f"bad{i}", matrix, ["x"])
            with self.assertRaises(ValueError): load_pair(d, l)
        d, l = write_pair(self.tmp, "order", rows(1), ["x"], wrong)
        with self.assertRaises(ValueError): load_pair(d, l)

    def test_duplicate_policy_and_deterministic_stratification(self):
        x = rows(60); labels = ["A"] * 20 + [" B "] * 20 + ["c"] * 20
        # Same-label duplicate: retain first.
        x[19] = x[0]
        d, l = write_pair(self.tmp, "dup", x, labels)
        pair = load_pair(d, l)
        a, detail_a = automatic_split(pair, 42); b, detail_b = automatic_split(pair, 42)
        self.assertEqual(detail_a, detail_b)
        self.assertEqual(sum(len(v.labels) for v in a.values()), 59)
        self.assertFalse(set(a["train"].row_hashes) & set(a["test"].row_hashes))
        # Conflicting duplicate is fatal.
        labels[19] = "c"
        d2, l2 = write_pair(self.tmp, "conflict", x, labels)
        with self.assertRaisesRegex(ValueError, "conflicting labels"): automatic_split(load_pair(d2, l2), 42)

    def test_manual_overlap_and_unseen_labels(self):
        train = load_pair(*write_pair(self.tmp, "train", rows(3), ["a", "b", "a"]))
        val_overlap = load_pair(*write_pair(self.tmp, "vo", rows(2), ["a", "b"]))
        test = load_pair(*write_pair(self.tmp, "test", rows(2, 50), ["a", "b"]))
        with self.assertRaisesRegex(ValueError, "overlap"): manual_split(train, val_overlap, test)
        val = load_pair(*write_pair(self.tmp, "val", rows(2, 20), ["a", "new"]))
        with self.assertRaisesRegex(ValueError, "absent"): manual_split(train, val, test)


class MetricsAndLifecycleTests(unittest.TestCase):
    def test_metrics_class_coverage_and_zero_division(self):
        # Predictions into absent configured classes remain errors, while macro
        # and weighted aggregates average only the ground-truth-present class.
        truth = [0, 0]; pred = [2, 2]
        probs = np.asarray([[.1, .1, .8], [.1, .1, .8]])
        metrics, per_class, raw, norm = app.evaluation_metrics(truth, pred, probs, ["a", "b", "c"], .5, 2., 99)
        self.assertEqual(metrics["parameter_count"], 99)
        self.assertTrue(metrics["zero_division_handling"]["occurred"])
        self.assertEqual(raw.shape, (3, 3)); self.assertEqual(norm.shape, (3, 3))
        self.assertEqual(len(per_class), 3)
        self.assertEqual(metrics["class_coverage"]["absent_classes"], ["b", "c"])
        self.assertEqual(metrics["accuracy"], 0.0)
        for key in ("macro_precision", "weighted_precision", "macro_recall", "weighted_recall", "macro_f1", "weighted_f1"):
            self.assertEqual(metrics[key], 0.0)
        self.assertNotIn("top_3_accuracy", metrics)
        self.assertNotIn("balanced_accuracy", metrics)

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"; torch.save({"model_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing fields"): app.load_checkpoint(path, torch.device("cpu"))

    def test_partial_legacy_history_preserves_known_best_metrics_without_inventing_others(self):
        existing = {"train": {"macro_f1": {"value": .9, "epoch": 3}},
                    "validation": {"weighted_f1": {"value": .8, "epoch": 2}}}
        partial = [{"epoch": 4, "train_accuracy": .6, "validation_accuracy": .7},
                   {"epoch": 1, "train_accuracy": .6, "validation_accuracy": .5}]
        merged = app.preserve_best_observed(existing, app.best_observed_history(partial))
        self.assertEqual(merged["train"]["macro_f1"], {"value": .9, "epoch": 3})
        self.assertEqual(merged["validation"]["weighted_f1"], {"value": .8, "epoch": 2})
        self.assertEqual(merged["train"]["accuracy"], {"value": .6, "epoch": 1})
        self.assertEqual(merged["validation"]["accuracy"], {"value": .7, "epoch": 4})
        self.assertNotIn("weighted_precision", merged["train"])
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "metrics.json").write_text(json.dumps(partial))
            self.assertEqual(app.saved_history_for_summary(run), partial)

    def test_interruption_resume_increase_and_repeat_original_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data, labels = write_pair(root, "all", rows(60), ["a"]*20+["b"]*20+["c"]*20)
            raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
            raw["run"].update(output_root=str(root / "runs"), name="lifecycle")
            raw["data"].update(mode="auto", data_path=str(data), label_path=str(labels))
            raw["model"]["initial_feature_maps"] = 1
            raw["training"].update(max_epochs=1, batch_size=16, device="cpu")
            cfg = app.validate_config(raw); run = app.make_run(cfg, None)
            with mock.patch.object(app, "epoch_pass", side_effect=KeyboardInterrupt): app.train(run, cfg)
            status = json.loads((run / "status.json").read_text())
            self.assertEqual(status["state"], "paused")
            cp = app.load_checkpoint(run / "checkpoints" / "interrupted.pt", torch.device("cpu"))
            self.assertEqual(cp["completed_epoch"], 0)

            candidate = copy.deepcopy(cfg); candidate["training"]["max_epochs"] = 2
            resume_yaml = root / "resume.yaml"; resume_yaml.write_text(yaml.safe_dump(candidate, sort_keys=False))
            fake = {"loss": .5, **{metric: .5 for metric in app.CLASSIFICATION_METRICS}}
            with mock.patch.object(app, "epoch_pass", return_value=fake), mock.patch.object(app, "evaluate_checkpoint"):
                app.resume(run, resume_yaml, None)
            self.assertEqual(json.loads((run / "status.json").read_text())["completed_epochs"], 2)
            self.assertTrue((run / "checkpoints" / "best.pt").exists())
            first = app.evaluate_checkpoint(run / "checkpoints" / "best.pt", True, device_spec="cpu")
            second = app.evaluate_checkpoint(run / "checkpoints" / "best.pt", True, device_spec="cpu")
            expected = {"per_epoch_metrics.csv", "per_epoch_metrics.json", "summarized_metrics.json",
                        "accuracy_curves.png", "loss_curves.png", "precision_curves.png", "recall_curves.png", "f1_curves.png",
                        "test_confusion_matrix_normalized.csv", "test_confusion_matrix_normalized.png",
                        "test_confusion_matrix_raw.csv", "test_confusion_matrix_raw.png",
                        "test_per_class_metrics.csv", "test_per_class_metrics.json"}
            self.assertEqual(first, run / "evaluations")
            self.assertEqual(second, first)
            self.assertTrue(expected <= {path.name for path in first.iterdir()})
            self.assertFalse((run / "metrics.json").exists())
            self.assertFalse((run / "test_metrics.json").exists())
            history = json.loads((first / "per_epoch_metrics.json").read_text())
            self.assertTrue(all(f"{split}_{metric}" in history[0]
                                for split in ("train", "validation")
                                for metric in app.CLASSIFICATION_METRICS))
            summary = json.loads((first / "summarized_metrics.json").read_text())
            maxima = summary["best_observed_per_epoch_metrics"]
            self.assertEqual(maxima["train"]["accuracy"], {"value": .5, "epoch": 1})
            self.assertEqual(maxima["validation"]["weighted_f1"], {"value": .5, "epoch": 1})

            # A standalone original-test evaluation with unavailable history
            # must retain the already summarized best-observed section.
            before_standalone = json.loads((first / "summarized_metrics.json").read_text())["best_observed_per_epoch_metrics"]
            (first / "per_epoch_metrics.json").unlink()
            app.evaluate_checkpoint(run / "checkpoints" / "best.pt", True, device_spec="cpu")
            after_standalone = json.loads((first / "summarized_metrics.json").read_text())["best_observed_per_epoch_metrics"]
            self.assertEqual(after_standalone, before_standalone)

            original_summary = (first / "summarized_metrics.json").read_text()
            other = app.evaluate_checkpoint(run / "checkpoints" / "best.pt", False, str(data), str(labels), "cpu")
            self.assertEqual(other, first / "external_test")
            self.assertTrue((other / "summarized_metrics.json").exists())
            self.assertEqual((first / "summarized_metrics.json").read_text(), original_summary)


if __name__ == "__main__": unittest.main()
