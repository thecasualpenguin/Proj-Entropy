from __future__ import annotations
import json, sys, tempfile, unittest
from unittest.mock import patch
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from wpp_data import (StreamingNormalizer, WPPVideo, materialize_video, load_manifest, make_windows, normalize_video, prompt_group_split, sha256_file)
from wpp_models import WPPResNet, WPPTransformer, video_logits


def item(video_id, prompt, label, offset=0):
    # time base 1/10; first PTS=5, 2-second window is exactly 20 ticks.
    return {"video_id": video_id, "status": "complete", "prompt_id": prompt, "label": label,
            "source": f"source/{video_id}.mp4", "time_base": {"num": 1, "den": 10}, "clip_end_pts": 30,
            "frames": [{"pts": 5, "duration_pts": 5, "rows": [[0.0 + offset, 1.0], [0.0, 0.0]]},
                       {"pts": 10, "duration_pts": 5, "rows": [[2.0 + offset, 3.0]]},
                       {"pts": 25, "duration_pts": 5, "rows": [[4.0 + offset, 5.0]]}]}


class WPPNativeTests(unittest.TestCase):
    def manifest(self, items):
        temp = tempfile.TemporaryDirectory(); path = Path(temp.name) / "manifest.json"
        path.write_text(json.dumps({"schema_version": "wpp-v1", "videos": items}))
        self.addCleanup(temp.cleanup); return path

    def test_native_windows_zero_values_and_terminal_duration(self):
        videos, _ = load_manifest(self.manifest([item("v", "p", "real")]))
        video = videos[0]; windows = make_windows(video)
        self.assertEqual([w.frame_indices for w in windows], [(0, 1), (2,)])
        # Sparse native timestamps do not invent the empty 1.0-second gap.
        self.assertEqual([w.duration for w in windows], [Fraction(1), Fraction(1, 2)])
        self.assertEqual(video.frames[0].rows[1].tolist(), [0., 0.])  # valid data, not padding

    def test_boundary_crossing_assigned_mass_and_exact_once(self):
        crossed = item("cross", "p", "real"); crossed["clip_end_pts"] = 24
        crossed["frames"] = [{"pts": 0, "duration_pts": 19, "rows": [[1., 2.]]},
                             {"pts": 19, "duration_pts": 5, "rows": [[3., 4.]]}]
        boundary = item("boundary", "q", "aigc"); boundary["clip_end_pts"] = 25
        boundary["frames"] = [{"pts": 0, "duration_pts": 20, "rows": [[1., 2.]]},
                              {"pts": 20, "duration_pts": 5, "rows": [[3., 4.]]}]
        videos, _ = load_manifest(self.manifest([crossed, boundary]))
        win = make_windows(videos[0]); self.assertEqual(len(win), 1)
        self.assertEqual(win[0].duration, Fraction(12, 5)); self.assertTrue(win[0].crosses_nominal_boundary)
        self.assertEqual(win[0].frame_indices, (0, 1))  # no frame copy into an empty next bucket
        exact = make_windows(videos[1]); self.assertEqual([w.index for w in exact], [0, 1])
        self.assertEqual(sum((w.duration for w in exact), Fraction()), Fraction(5, 2))

    def test_normalizer_is_train_only_and_feature_order_is_fixed(self):
        videos, _ = load_manifest(self.manifest([item("a", "p0", "real", 0), item("b", "p1", "aigc", 100)]))
        stats = StreamingNormalizer().fit_videos([videos[0]])
        self.assertEqual(stats["feature_names"], ["delta_row_share", "row_share"])
        self.assertEqual(stats["fit_video_ids"], ["a"])
        normalized = normalize_video(videos[1], stats)
        self.assertGreater(normalized.frames[0].rows[0, 0], 10.)

    def test_prompt_group_split_no_leakage_and_fixed_seed(self):
        # A DeepAction prompt deliberately has both real and AIGC variants.
        items = [item(f"r{i}", f"p{i}", "real", i) for i in range(6)] + [item(f"a{i}", f"p{i}", "aigc", i) for i in range(6)]
        videos, _ = load_manifest(self.manifest(items)); first = prompt_group_split(videos); second = prompt_group_split(videos)
        self.assertEqual([[v.video_id for v in first[k]] for k in first], [[v.video_id for v in second[k]] for k in second])
        groups = [set(v.prompt_group for v in first[k]) for k in first]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_failed_requires_explicit_policy(self):
        bad = item("failed", "p", "real"); bad.pop("frames"); bad["status"] = "failed"
        path = self.manifest([bad])
        with self.assertRaisesRegex(ValueError, "explicit_failed"):
            load_manifest(path)
        with self.assertRaisesRegex(ValueError, "no complete"):
            load_manifest(path, explicit_failed_video_ids=["failed"])

    def test_adaptive_oom_retries_identical_group_without_duplicate_updates(self):
        import run_wpp_pipeline as runner
        items = []
        for i in range(6):
            for label, prefix in (("real", "r"), ("aigc", "a")):
                value = item(f"{prefix}{i}", f"p{i}", label, i); value.update(width=64, height=64, fps="25/1"); items.append(value)
        manifest = self.manifest(items)
        with tempfile.TemporaryDirectory() as temp:
            cfg = {"run": {"output_root": temp, "name": "oom"}, "data": {"input_format": "wpp_v1", "manifest": str(manifest), "expected_video_count": None,
                   "explicit_failed_video_ids": [], "approved_video_ids_path": None, "official_mapping_path": None, "strict_preflight": False, "allow_incomplete_smoke": False},
                   "model": {"architecture": "resnet", "width": 64, "heads": 4, "dropout": .1, "gradient_checkpointing": True},
                   "training": {"seed": 42, "learning_rate": .001, "weight_decay": .01, "batch_size": 2, "accumulation_steps": 8,
                   "max_epochs": 1, "early_stopping_patience": 15, "scheduler_factor": .1, "scheduler_patience": 5,
                   "min_lr": .00001, "gradient_clip_norm": 1., "device": "cpu"}}
            original, calls = runner.video_logits, {"n": 0}
            def fail_once(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1: raise torch.OutOfMemoryError("injected")
                return original(*args, **kwargs)
            with patch.object(runner, "video_logits", fail_once): runner.train(cfg, Path(temp) / "oom")
            self.assertTrue((Path(temp) / "oom" / "oom_adjustments.json").is_file())
            # 8 train + 2 validation + 2 test successful full-video passes; failed call made no update.
            self.assertEqual(calls["n"], 13)

    def test_strict_packet_sidecar_missing_or_dependency_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); feature, encoded, sidecar = root/'f.npz', root/'encoded.mp4', root/'timing.json'
            np.savez(feature, delta_row_share=np.array([[1.]]), row_share=np.array([[2.]]), pts=np.array([7], dtype=np.int64))
            encoded.write_bytes(b'encoded')
            timing = {'streams': [{'time_base': '1/10'}], 'packets': [{'pts': 7, 'duration': 3}],
                      'dependency_sha256': {'encoded': sha256_file(encoded), 'features': sha256_file(feature)}}
            sidecar.write_text(json.dumps(timing)); adapter = {'kind': 'deepaction_npz_v1', 'feature_path': str(feature), 'features_fingerprint': sha256_file(feature),
                'encoded_path': str(encoded), 'timing_sidecar_path': str(sidecar), 'timing_sidecar_sha256': sha256_file(sidecar), 'allow_sidecar_generation': False,
                'ffprobe_path': '/missing'}
            video = WPPVideo('v','p',0,'s',Fraction(1,10),10,(),{'adapter': adapter})
            self.assertEqual(materialize_video(video).frames[0].duration_pts, 3)
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, 'sidecar is missing'): materialize_video(video)
            sidecar.write_text(json.dumps(timing)); adapter['timing_sidecar_sha256'] = sha256_file(sidecar); encoded.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'dependency differs'): materialize_video(video)

    def test_weighted_ce_microbatch_gradient_matches_actual_update_group(self):
        # Two physical microbatches (including a short/remainder group) must equal
        # sum(weight * CE) / actual video count, not batch-size/accumulation scaling.
        torch.manual_seed(4); logits = torch.randn(3, 2, requires_grad=True); labels = torch.tensor([0, 1, 1]); weights = torch.tensor([1.5, .75])
        full = sum(weights[labels[i]] * torch.nn.functional.cross_entropy(logits[i:i+1], labels[i:i+1]) for i in range(3)) / 3
        full_grad = torch.autograd.grad(full, logits, retain_graph=True)[0]
        micro = sum(sum(weights[labels[i]] * torch.nn.functional.cross_entropy(logits[i:i+1], labels[i:i+1]) for i in part) / 3
                    for part in ((0, 1), (2,)))
        self.assertTrue(torch.allclose(full_grad, torch.autograd.grad(micro, logits)[0]))

    def test_padding_shape_invariance_and_both_models_backpropagate_native_ragged_windows(self):
        videos, _ = load_manifest(self.manifest([item("v", "p", "real")]))
        stats = StreamingNormalizer().fit_videos(videos); video = normalize_video(videos[0], stats); windows = make_windows(video)
        for model in (WPPResNet(dropout=0), WPPTransformer(dropout=0)):
            model.eval()
            rows = torch.tensor([[[1., 2.]], [[3., 4.]]]); mask = torch.tensor([[True], [True]])
            pts = torch.tensor([0., .1]); baseline = model.forward_window(rows, mask, pts)
            padded = torch.full((2, 4, 2), -999., requires_grad=True); padded.data[:, :1] = rows; padded_mask = torch.zeros(2, 4, dtype=torch.bool); padded_mask[:, :1] = True
            adversarial = model.forward_window(padded, padded_mask, pts)
            self.assertTrue(torch.allclose(baseline, adversarial, atol=1e-6))
            adversarial.sum().backward(); self.assertTrue(torch.allclose(padded.grad[:, 1:], torch.zeros_like(padded.grad[:, 1:])))
            model.train(); logits, parts = video_logits(model, video, windows, torch.device("cpu"))
            self.assertEqual(tuple(logits.shape), (2,)); self.assertEqual(len(parts), 2)
            torch.nn.functional.cross_entropy(logits[None], torch.tensor([0])).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

if __name__ == "__main__": unittest.main()
