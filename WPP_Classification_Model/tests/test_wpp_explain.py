from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from src.wpp_explain import _attention_rollout, _compute, _integrated_gradients, _trace_attention, export_explanation
from src.wpp_normalization import POLICY
from src.wpp_explain_types import (AttentionSummary, DataRoots, MediaTiming, Prediction, VideoExplanation, VideoSummary)
from src.wpp_transformer import TIME_STEPS, WPPTransformer
import test_wpp_dataset as _dataset_tests
from src.wpp_dataset import pack_wpp_dataset
from src.wpp_training import train
import yaml


class _LinearMaskedLogit(torch.nn.Module):
    def forward(self, values, rows, times, *, normalized=False):
        valid = rows[:, :, None] & times[:, None, :]
        return (values * valid).sum(dim=(1, 2), keepdim=False).unsqueeze(1)


class _TwoLogitMaskedModel(torch.nn.Module):
    def forward(self, values, rows, times, *, normalized=False):
        valid = rows[:, :, None] & times[:, None, :]
        total = (values * valid).sum(dim=(1, 2), keepdim=False)
        return torch.stack((total, total + 1.0), dim=1)


class WPPExplainTests(unittest.TestCase):
    def test_trace_does_not_change_logits_or_leave_hooks(self):
        torch.manual_seed(5)
        model = WPPTransformer(3, d_model=8, nhead=2, num_layers=2, dim_feedforward=16, dropout=0).eval()
        values = torch.randn(1, 2, TIME_STEPS)
        rows = torch.tensor([[True, True]])
        times = torch.zeros((1, TIME_STEPS), dtype=torch.bool); times[:, :17] = True
        ordinary = model(values, rows, times, normalized=True)
        traced, inputs = model.forward_with_attention_inputs(values, rows, times, normalized=True)
        self.assertTrue(torch.equal(ordinary, traced))
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all(value.shape == (1, 2 * 188, 8) for value in inputs))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))
        logits, incoming, rollout = _trace_attention(model, values, rows, times, chunk_queries=2, callback=None)
        self.assertTrue(torch.allclose(ordinary, logits, rtol=1e-5, atol=1e-6))
        self.assertEqual([tuple(value.shape) for value in incoming], [(2, 4), (2, 4)])
        self.assertAlmostEqual(float(rollout.sum()), 1.0, places=6)

    def test_chunked_rollout_and_incoming_match_dense_reference(self):
        queries = torch.tensor([[[1.0], [2.0], [0.0]]])
        keys = torch.tensor([[[0.0], [1.0], [2.0]]])
        relevance, incoming = _attention_rollout([(queries, keys)], chunk_queries=1, callback=None)
        matrix = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)), dim=-1)[0]
        expected_incoming = matrix.mean(dim=0)
        expected_rollout = torch.full((3,), 1 / 3) @ ((torch.eye(3) + matrix) / 2)
        self.assertTrue(torch.allclose(incoming[0][0], expected_incoming, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(relevance, expected_rollout, atol=1e-7, rtol=1e-7))

    def test_relocated_roots_open_but_tampered_bundle_is_rejected(self):
        fixture = _dataset_tests.WPPDatasetTests(); fixture.setUp()
        try:
            specs = [("a.mp4", "train", "alpha", "g-a", np.array([[0, 2]], dtype=np.int64)),
                     ("b.mp4", "train", "beta", "g-b", np.array([[3, 4]], dtype=np.int64)),
                     ("c.mp4", "validation", "alpha", "g-c", np.array([[5, 6]], dtype=np.int64))]
            manifest, index = fixture.build_complete_native(specs)
            bundle = fixture.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
            raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
            run = fixture.root / "run"
            raw["run"]["output_dir"] = str(run)
            raw["data"].update({"packed_bundle": str(bundle), "source_manifest": str(manifest), "index": str(index)})
            raw["model"].update({"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0})
            raw["training"].update({"max_epochs": 1, "batch_size": 1, "device": "cpu"})
            train(raw, default_path=ROOT / "configs" / "default.yaml")
            relocated = fixture.root / "relocated"; relocated.mkdir()
            copied = []
            for name, original in (("bundle", bundle), ("manifest", manifest), ("index", index)):
                target = relocated / name; shutil.copytree(original, target); copied.append(target)
            from src.wpp_explain import ExplanationError, open_explainer
            roots = DataRoots(
                packed_bundle=copied[0], source_manifest=copied[1], index=copied[2], source_video_root=fixture.root)
            session = open_explainer(run_dir=run, roots=roots, device="cpu")
            self.assertEqual(session.checkpoint_path.name, "best.pt")
            with (copied[0] / "train-data.csv").open("ab") as handle:
                handle.write(b"0\n")
            with self.assertRaisesRegex(ExplanationError, "cannot open"):
                open_explainer(run_dir=run, roots=roots, device="cpu")
        finally:
            fixture.tearDown()

    def test_export_writes_separate_data_figures_and_metadata(self):
        video = VideoSummary("sha256:x", "test", "truth", 0, "a.mp4", 2, 2, 1, False)
        prediction = Prediction(0, "truth", 1.0, 0.8, 0, "truth", True)
        media = MediaTiming(Path("/source/a.mp4"), None, 64, 64, "1:1", 0,
                            np.array([1.0, 1.2]), np.array([0.0, 0.2]), 1.4, 0.4, True, "test")
        attention = AttentionSummary(np.array([[0.4]], dtype=np.float32), np.array([[0.4, 0.4]], dtype=np.float32),
                                     np.array([0.4, 0.4], dtype=np.float32), np.ones((1, 1, 1, 1), dtype=np.float32), 16, 1, 1)
        explanation = VideoExplanation(video, prediction, media, np.array([[1.0, -1.0]], dtype=np.float32),
                                       np.array([1.0, -1.0], dtype=np.float32), attention, {"ig_diagnostics": {}}, "positive")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "export"
            export_explanation(explanation, destination)
            self.assertTrue((destination / "metadata.json").is_file())
            self.assertTrue((destination / "data" / "arrays.npz").is_file())
            self.assertTrue((destination / "figures" / "overview.png").is_file())
            with self.assertRaisesRegex(Exception, "fresh"):
                export_explanation(explanation, destination)

    def test_explicit_ig_target_does_not_change_predicted_label(self):
        item = {"values": np.pad(np.array([[2.0, 3.0]], dtype=np.float32), ((0, 0), (0, 2998))),
                "row_count": 1, "used_time_length": 2, "class_id": 0, "video_id": "target-test"}
        statistics = {**POLICY, "count": 2, "mean": 0.0, "std": 1.0}
        session = SimpleNamespace(model=_TwoLogitMaskedModel().eval(), device=torch.device("cpu"), normalization=statistics,
                                  class_names=("truth", "predicted"))
        def traced(model, normalized, rows, times, **_kwargs):
            return model(normalized, rows, times, normalized=True), object()
        with patch("src.wpp_explain._attention_summary", side_effect=traced):
            prediction, ig, diagnostics, _ = _compute(session, item, ig_steps=5, chunk_queries=1,
                                                        callback=None, target_class_id=0)
        self.assertEqual(prediction.predicted_class_id, 1)
        self.assertEqual(prediction.predicted_label, "predicted")
        self.assertFalse(prediction.correct)
        self.assertEqual(diagnostics["target_class_id"], 0.0)
        self.assertAlmostEqual(float(ig.sum()), 5.0, places=6)

    def test_trapezoid_ig_uses_fixed_zero_baseline_and_preserves_mask(self):
        model = _LinearMaskedLogit().eval()
        values = torch.zeros((1, 2, 4), dtype=torch.float32)
        values[0, 0, :2] = torch.tensor([2.0, -3.0])
        values[0, 1, :2] = torch.tensor([4.0, 5.0])
        rows = torch.tensor([[True, True]])
        times = torch.tensor([[True, True, False, False]])
        attribution, diagnostics = _integrated_gradients(model, values, rows, times, 0, 5, None)
        self.assertTrue(torch.equal(attribution, values))
        self.assertTrue(torch.equal(attribution[..., 2:], torch.zeros_like(attribution[..., 2:])))
        self.assertAlmostEqual(diagnostics["completeness_residual"], 0.0, places=7)
        with self.assertRaisesRegex(Exception, "at least 2"):
            _integrated_gradients(model, values, rows, times, 0, 1, None)


if __name__ == "__main__":
    unittest.main()
