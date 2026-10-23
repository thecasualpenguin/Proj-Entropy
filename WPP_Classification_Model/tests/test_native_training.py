from __future__ import annotations

import copy
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from src.model_factory import make_model
from src import wpp_training
from src.wpp_training import TrainingError, evaluate, train, validate_config
import test_wpp_dataset as _dataset_tests


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing complete, ledger-verified synthetic native fixture.
        self.fixture = _dataset_tests.WPPDatasetTests(); self.fixture.setUp(); self.root = self.fixture.root
        specs = [
            ("a.mp4", "train", "alpha", "g-a", np.array([[0, 2, 3]], dtype=np.int64)),
            ("b.mp4", "train", "beta", "g-b", np.array([[5, 6]], dtype=np.int64)),
            ("c.mp4", "validation", "alpha", "g-c", np.array([[7, 8]], dtype=np.int64)),
            ("d.mp4", "test", "beta", "g-d", np.array([[9, 10]], dtype=np.int64)),
        ]
        self.manifest, self.index = self.fixture.build_complete_native(specs)
        from src.wpp_dataset import pack_wpp_dataset
        self.bundle = self.root / "bundle"; pack_wpp_dataset(self.manifest, self.index, self.bundle)

    def tearDown(self):
        self.fixture.tearDown()

    def config(self, epochs=1):
        raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
        raw["run"]["output_dir"] = str(self.root / "run")
        raw["data"].update({"packed_bundle": str(self.bundle), "source_manifest": str(self.manifest), "index": str(self.index)})
        raw["model"].update({"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.1})
        raw["training"].update({"max_epochs": epochs, "batch_size": 1, "device": "cpu"})
        return raw

    def test_factory_rejects_legacy_kind(self):
        config = self.config()["model"]
        self.assertEqual(make_model(config, 2).num_classes, 2)
        config["kind"] = "resnet"
        with self.assertRaisesRegex(ValueError, "legacy"):
            make_model(config, 2)

    def test_tiny_train_resume_and_evaluate_never_refits_statistics(self):
        raw = self.config(epochs=1)
        run = train(raw, default_path=ROOT / "configs" / "default.yaml")
        self.assertTrue((run / "checkpoints" / "best.pt").is_file())
        saved = validate_config(raw, ROOT / "configs" / "default.yaml")
        resumed = copy.deepcopy(saved); resumed["training"]["max_epochs"] = 2
        with mock.patch("src.wpp_training.fit_train_statistics", side_effect=AssertionError("must not refit")):
            train(resumed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")
        status = yaml.safe_load((run / "status.json").read_text())
        self.assertEqual(status["state"], "complete"); self.assertEqual(status["completed_epoch"], 2)
        # With dropout and shuffled one-video batches, epoch-boundary resume must
        # restore both global and shuffle RNG state exactly.
        uninterrupted = self.config(epochs=2); uninterrupted["run"]["output_dir"] = str(self.root / "uninterrupted")
        full_run = train(uninterrupted, default_path=ROOT / "configs" / "default.yaml")
        resumed_state = torch.load(run / "checkpoints" / "last.pt", weights_only=True)["model_state"]
        full_state = torch.load(full_run / "checkpoints" / "last.pt", weights_only=True)["model_state"]
        self.assertEqual(set(resumed_state), set(full_state))
        self.assertTrue(all(torch.equal(resumed_state[key], full_state[key]) for key in resumed_state))
        output = self.root / "evaluation"
        evaluate(run / "checkpoints" / "best.pt", resumed, default_path=ROOT / "configs" / "default.yaml",
                 partition="test", output_dir=output)
        report = yaml.safe_load((output / "metrics.json").read_text())
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["normalization"]["fit_split"], "train")
        self.assertEqual(report["normalization"]["count"], 5)  # 3 + 2 valid train cells, including zero

    def test_replaced_best_checkpoint_is_rejected_before_test_evaluation(self):
        donor = self.config(epochs=1); donor["run"]["output_dir"] = str(self.root / "donor")
        donor["model"]["dropout"] = 0.0
        donor_run = train(donor, default_path=ROOT / "configs" / "default.yaml")
        target = self.config(epochs=1); target_run = Path(target["run"]["output_dir"])
        original_publish = wpp_training._atomic_torch
        original_evaluate = wpp_training.evaluate_model
        partitions = []

        def replace_best(path, payload):
            original_publish(path, payload)
            if path == target_run / "checkpoints" / "best.pt":
                shutil.copyfile(donor_run / "checkpoints" / "best.pt", path)

        def record_evaluation(model, dataset, **kwargs):
            partitions.append(dataset.partition)
            return original_evaluate(model, dataset, **kwargs)

        with mock.patch("src.wpp_training._atomic_torch", side_effect=replace_best), \
             mock.patch("src.wpp_training.evaluate_model", side_effect=record_evaluation):
            with self.assertRaisesRegex(TrainingError, "incompatible"):
                train(target, default_path=ROOT / "configs" / "default.yaml")
        self.assertEqual(partitions, ["validation"])

    def test_legacy_checkpoint_and_provenance_mismatch_are_rejected(self):
        raw = self.config(epochs=1); run = train(raw, default_path=ROOT / "configs" / "default.yaml")
        bad = self.root / "legacy.pt"; bad.write_bytes(b"not a native checkpoint")
        with self.assertRaisesRegex(TrainingError, "safe weights-only"):
            evaluate(bad, raw, default_path=ROOT / "configs" / "default.yaml", partition="test",
                     output_dir=self.root / "bad-evaluation")
        changed = copy.deepcopy(raw); changed["model"]["dropout"] = 0.0
        with self.assertRaisesRegex(TrainingError, "incompatible"):
            train(changed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")


if __name__ == "__main__":
    unittest.main()
