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
    def test_metrics_top3_confusions_and_zero_division(self):
        truth = [0, 1, 2, 2]; pred = [0, 0, 0, 0]
        probs = np.asarray([[.8,.1,.1], [.7,.2,.1], [.6,.2,.2], [.6,.1,.3]])
        metrics, per_class, raw, norm = app.evaluation_metrics(truth, pred, probs, ["a","b","c"], .5, 2., 99)
        self.assertEqual(metrics["parameter_count"], 99)
        self.assertTrue(metrics["zero_division_handling"]["occurred"])
        self.assertEqual(raw.shape, (3,3)); self.assertEqual(norm.shape, (3,3))
        self.assertEqual(len(per_class), 3); self.assertEqual(metrics["top_3_accuracy"], 1.0)

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.pt"; torch.save({"model_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "missing fields"): app.load_checkpoint(path, torch.device("cpu"))

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
            fake = {"loss": .5, "accuracy": .5, "macro_precision": .5, "macro_recall": .5, "macro_f1": .5}
            with mock.patch.object(app, "epoch_pass", return_value=fake), mock.patch.object(app, "evaluate_checkpoint"):
                app.resume(run, resume_yaml, None)
            self.assertEqual(json.loads((run / "status.json").read_text())["completed_epochs"], 2)
            self.assertTrue((run / "checkpoints" / "best.pt").exists())
            first = app.evaluate_checkpoint(run / "checkpoints" / "best.pt", True, device_spec="cpu")
            second = app.evaluate_checkpoint(run / "checkpoints" / "best.pt", True, device_spec="cpu")
            self.assertNotEqual(first, second); self.assertTrue((first / "metrics.json").exists()); self.assertTrue((second / "metrics.json").exists())


if __name__ == "__main__": unittest.main()
