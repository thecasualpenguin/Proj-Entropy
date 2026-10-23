from __future__ import annotations

import copy
import csv
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from src import wpp_training
from src.wpp_reporting import EPOCH_HEADER, REPORT_FILES, flatten_history
from src.wpp_training import TrainingError, evaluate, evaluate_model, export_report, fit_train_statistics, train
import test_wpp_dataset as _dataset_tests


class WPPReportingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _dataset_tests.WPPDatasetTests(); self.fixture.setUp(); self.root = self.fixture.root
        specs = [
            ("a.mp4", "train", "alpha", "g-a", [[0, 2, 3]]),
            ("b.mp4", "train", "beta", "g-b", [[5, 6]]),
            ("c.mp4", "validation", "alpha", "g-c", [[7, 8]]),
            ("d.mp4", "test", "beta", "g-d", [[9, 10]]),
        ]
        import numpy as np
        self.manifest, self.index = self.fixture.build_complete_native(
            [(name, split, label, group, np.asarray(values, dtype=np.int64)) for name, split, label, group, values in specs])
        from src.wpp_dataset import pack_wpp_dataset
        self.bundle = self.root / "bundle"; pack_wpp_dataset(self.manifest, self.index, self.bundle)

    def tearDown(self):
        self.fixture.tearDown()

    def config(self, epochs=2):
        raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
        raw["run"]["output_dir"] = str(self.root / "run")
        raw["data"].update({"packed_bundle": str(self.bundle), "source_manifest": str(self.manifest), "index": str(self.index)})
        raw["model"].update({"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0})
        raw["training"].update({"max_epochs": epochs, "batch_size": 1, "device": "cpu"})
        return raw

    def test_future_run_writes_exact_compatibility_set_and_online_history(self):
        run = train(self.config(), default_path=ROOT / "configs" / "default.yaml")
        evaluations = run / "evaluations"
        self.assertEqual({item.name for item in evaluations.iterdir() if item.is_file()}, REPORT_FILES)
        with (evaluations / "per_epoch_metrics.csv").open(newline="") as handle:
            self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), EPOCH_HEADER)
        records = json.loads((evaluations / "per_epoch_metrics.json").read_text())
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["train_accuracy"] is not None and record["duration_seconds"] is not None for record in records))
        summary = json.loads((evaluations / "summarized_metrics.json").read_text())
        self.assertEqual(set(summary), {
            "accuracy", "best_observed_per_epoch_metrics", "checkpoint", "class_coverage", "created_at",
            "data_checksum_sha256", "data_path", "evaluation_duration_seconds", "evaluation_kind",
            "label_checksum_sha256", "label_path", "macro_f1", "macro_precision", "macro_recall",
            "parameter_count", "sample_count", "samples_per_second", "test_loss", "weighted_f1",
            "weighted_precision", "weighted_recall", "zero_division_handling",
        })
        self.assertEqual(summary["evaluation_kind"], "original_test")
        self.assertEqual(len(json.loads((evaluations / "test_per_class_metrics.json").read_text())), 2)
        # Private retained prediction lists must never leak into backward-compatible test.json.
        self.assertNotIn("_truths", json.loads((run / "native_test_metrics.json").read_text()))
        standalone = self.root / "standalone-success"
        evaluate(run / "checkpoints" / "best.pt", self.config(), default_path=ROOT / "configs" / "default.yaml",
                 partition="test", output_dir=standalone)
        self.assertEqual({item.name for item in (standalone / "evaluations").iterdir() if item.is_file()}, REPORT_FILES)

    def test_fresh_export_reconstructs_only_missing_train_metrics_and_does_not_refit(self):
        run = train(self.config(), default_path=ROOT / "configs" / "default.yaml")
        history_path, test_path = run / "history.json", run / "native_test_metrics.json"
        source_history, source_test = history_path.read_bytes(), test_path.read_bytes()
        # Simulate the completed historical schema: CE/validation are genuine;
        # online classification/LR/duration were never persisted.
        history = json.loads(source_history)
        for row in history["epochs"]:
            for key in list(row):
                if key.startswith("train_") and key != "train_cross_entropy" or key in ("learning_rate", "duration_seconds"):
                    row.pop(key, None)
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
        historical_bytes = history_path.read_bytes()
        output = self.root / "exported"
        with mock.patch("src.wpp_training.fit_train_statistics", side_effect=AssertionError("must not refit")):
            exported = export_report(run, output, default_path=ROOT / "configs" / "default.yaml",
                                     reconstruct_train_metrics=True, device_spec="cpu")
        self.assertEqual(exported, output)
        self.assertEqual(history_path.read_bytes(), historical_bytes)
        self.assertEqual(test_path.read_bytes(), source_test)
        self.assertEqual({item.name for item in (output / "evaluations").iterdir() if item.is_file()}, REPORT_FILES)
        records = json.loads((output / "evaluations" / "per_epoch_metrics.json").read_text())
        self.assertTrue(all(row["train_macro_f1"] is not None for row in records))
        self.assertTrue(all(row["duration_seconds"] is None for row in records))
        provenance = json.loads((output / "report_provenance.json").read_text())
        self.assertEqual(provenance["reconstructed_train_metric_epochs"], [1, 2])
        self.assertEqual(provenance["train_metric_source"], "frozen_post_epoch_checkpoint_snapshot_inference")

    def test_export_collision_and_checkpoint_mismatch_leave_no_output(self):
        run = train(self.config(epochs=3), default_path=ROOT / "configs" / "default.yaml")
        occupied = self.root / "occupied"; occupied.mkdir()
        with self.assertRaisesRegex(TrainingError, "fresh"):
            export_report(run, occupied, default_path=ROOT / "configs" / "default.yaml", reconstruct_train_metrics=False)
        # An incompatible middle epoch checkpoint cannot be used to manufacture a report.
        bad = run / "checkpoints" / "epoch_0002.pt"
        payload = wpp_training._safe_checkpoint(bad)
        payload["provenance"] = copy.deepcopy(payload["provenance"]); payload["provenance"]["native_ledger_sha256"] = "0" * 64
        wpp_training._atomic_torch(bad, payload)
        destination = self.root / "must-not-publish"
        # All checkpoint lineage validation happens before any reconstruction
        # inference, so a corrupt middle checkpoint consumes no dataset pass.
        with mock.patch("src.wpp_training.evaluate_model", side_effect=AssertionError("no inference")):
            with self.assertRaisesRegex(TrainingError, "incompatible"):
                export_report(run, destination, default_path=ROOT / "configs" / "default.yaml", reconstruct_train_metrics=True)
        self.assertFalse(destination.exists())

    def test_compatibility_publication_is_all_or_nothing_and_finite_history_is_enforced(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        output = self.root / "standalone"
        before_history = (run / "history.json").read_bytes()
        before_checkpoint = (run / "checkpoints" / "best.pt").read_bytes()
        with mock.patch("src.wpp_reporting._plot_curves", side_effect=RuntimeError("synthetic plot failure")):
            with self.assertRaisesRegex(RuntimeError, "plot failure"):
                evaluate(run / "checkpoints" / "best.pt", self.config(epochs=1),
                         default_path=ROOT / "configs" / "default.yaml", partition="test", output_dir=output)
        self.assertTrue((output / "metrics.json").is_file())
        self.assertFalse((output / "evaluations").exists())
        self.assertEqual((run / "history.json").read_bytes(), before_history)
        self.assertEqual((run / "checkpoints" / "best.pt").read_bytes(), before_checkpoint)
        with self.assertRaisesRegex(ValueError, "finite"):
            flatten_history([{"epoch": 1, "train_cross_entropy": float("nan"), "validation_metrics": {}}])

    def test_foreign_report_rejects_resume_before_history_or_checkpoint_mutation(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        history, checkpoint = (run / "history.json").read_bytes(), (run / "checkpoints" / "last.pt").read_bytes()
        (run / "evaluations" / "foreign.txt").write_text("not owned", encoding="utf-8")
        resumed = self.config(epochs=2)
        with self.assertRaisesRegex(TrainingError, "ownership|foreign"):
            train(resumed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")
        self.assertEqual((run / "history.json").read_bytes(), history)
        self.assertEqual((run / "checkpoints" / "last.pt").read_bytes(), checkpoint)

    def test_archived_report_recovers_before_resume_and_preserves_generation(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        evaluations, archives = run / "evaluations", run / wpp_training._REPORT_ARCHIVES
        archives.mkdir(); archived = archives / "generation-0000"
        import os
        os.replace(evaluations, archived)
        stage = run / ".synthetic-report-stage"; stage.mkdir()
        wpp_training._atomic_json(run / wpp_training._REPORT_JOURNAL,
                                  {"phase": "archived", "archive": archived.name, "stage": stage.name})
        resumed = self.config(epochs=2)
        train(resumed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")
        self.assertFalse((run / wpp_training._REPORT_JOURNAL).exists())
        self.assertFalse(stage.exists())
        self.assertEqual({item.name for item in (run / "evaluations").iterdir() if item.is_file()}, REPORT_FILES)
        self.assertTrue((archives / "generation-0001").is_dir())

    def test_injected_post_archive_failure_recovers_without_duplicate_epoch(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        import os
        original_replace, failed = os.replace, {"value": False}
        def fail_publish(source, destination):
            source_path, destination_path = Path(source), Path(destination)
            if (not failed["value"] and source_path.name == "evaluations" and destination_path == run / "evaluations"
                    and source_path.parent.name.startswith(".native-wpp-evaluations-stage-")):
                failed["value"] = True
                raise OSError("injected post-archive publish failure")
            return original_replace(source, destination)
        resumed = self.config(epochs=2)
        with mock.patch("src.wpp_training.os.replace", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "post-archive"):
                train(resumed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")
        self.assertTrue(failed["value"])
        self.assertTrue((run / wpp_training._REPORT_JOURNAL).is_file())
        # Resume from epoch two: recovery restores generation one before the
        # next optimizer epoch, so no completed epoch is replayed.
        resumed["training"]["max_epochs"] = 3
        train(resumed, default_path=ROOT / "configs" / "default.yaml", resume_checkpoint=run / "checkpoints" / "last.pt")
        self.assertEqual(len(json.loads((run / "history.json").read_text())["epochs"]), 3)
        self.assertFalse((run / wpp_training._REPORT_JOURNAL).exists())
        self.assertEqual({item.name for item in (run / "evaluations").iterdir() if item.is_file()}, REPORT_FILES)

    def test_evaluation_loop_timer_excludes_post_loop_metrics_and_epoch_timer_has_reference_bounds(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        payload = wpp_training._safe_checkpoint(run / "checkpoints" / "best.pt")
        bundle = wpp_training._bundle(self.config(epochs=1))
        model = wpp_training.make_model(payload["model_config"], 2)
        model.load_state_dict(payload["model_state"])
        with mock.patch("src.wpp_training.time.perf_counter", side_effect=[10.0, 13.5]):
            result = evaluate_model(model, bundle.partition_view("test"), batch_size=1,
                                    device=wpp_training.torch.device("cpu"), statistics=payload["normalization"],
                                    class_names=["alpha", "beta"], detailed=False)
        self.assertEqual(result["_inference_duration_seconds"], 3.5)
        self.assertNotIn("per_class", result)
        self.assertNotIn("confusion_matrix", result)

    def test_export_report_uses_retained_inference_timer_for_summary(self):
        run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        sentinel = 123.456
        original_evaluate = wpp_training.evaluate_model

        def retained_timer(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            result["_inference_duration_seconds"] = sentinel
            return result

        output = self.root / "timer-export"
        with mock.patch("src.wpp_training.evaluate_model", side_effect=retained_timer):
            export_report(run, output, default_path=ROOT / "configs" / "default.yaml",
                          reconstruct_train_metrics=False, device_spec="cpu")
        summary = json.loads((output / "evaluations" / "summarized_metrics.json").read_text())
        self.assertEqual(summary["evaluation_duration_seconds"], sentinel)
        self.assertEqual(summary["samples_per_second"], summary["sample_count"] / sentinel)

    def test_epoch_and_test_timers_have_documented_boundaries(self):
        # Calls are: epoch start; validation-loop start/end; epoch end;
        # final test-loop start/end. Report/JSON/checkpoint IO occurs later.
        ticks = iter([0., 1., 2., 3., 4., 5.]); original_clock = time.perf_counter
        def clock():
            try:
                return next(ticks)
            except StopIteration:
                return original_clock()
        with mock.patch("src.wpp_training.time.perf_counter", side_effect=clock):
            run = train(self.config(epochs=1), default_path=ROOT / "configs" / "default.yaml")
        history = json.loads((run / "history.json").read_text())
        self.assertEqual(history["epochs"][0]["duration_seconds"], 3.)
        summary = json.loads((run / "evaluations" / "summarized_metrics.json").read_text())
        self.assertEqual(summary["evaluation_duration_seconds"], 1.)


if __name__ == "__main__":
    unittest.main()
