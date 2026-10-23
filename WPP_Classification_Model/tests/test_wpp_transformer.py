from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wpp_transformer import TIME_STEPS, WPPTransformer, two_dimensional_sincos_positions


class WPPTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def inputs(*, batch=1, rows=1, used=17, fill=1.0):
        values = torch.full((batch, rows, TIME_STEPS), fill, dtype=torch.float32)
        row_mask = torch.ones((batch, rows), dtype=torch.bool)
        time_mask = torch.zeros((batch, TIME_STEPS), dtype=torch.bool)
        time_mask[:, :used] = True
        return values, row_mask, time_mask

    def test_original_style_defaults_and_one_flattened_attention_sequence(self):
        model = WPPTransformer(3)
        self.assertEqual((model.patch_size, model.d_model, model.patches_per_row), (16, 32, 188))
        self.assertEqual(len(model.layers), 2)
        for layer in model.layers:
            self.assertTrue(layer.norm_first)
            self.assertEqual(layer.activation, torch.nn.functional.gelu)
            self.assertEqual(layer.linear1.out_features, 64)
            self.assertEqual(layer.dropout.p, 0.2)
        values, rows, times = self.inputs(rows=2, used=17)
        seen = []
        handle = model.layers[0].register_forward_pre_hook(lambda _module, args: seen.append(args[0].shape))
        try:
            logits = model(values, rows, times)
        finally:
            handle.remove()
        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(seen, [torch.Size((1, 2 * 188, 32))])
        self.assertEqual(model._token_mask(rows, times).shape, (1, 2 * 188))

    def test_final_partial_patch_and_mask_lengths(self):
        model = WPPTransformer(2, dropout=0).eval()
        for used in (1, 16, 17, 2999, 3000):
            with self.subTest(used=used):
                values, rows, times = self.inputs(used=used)
                mask = model._token_mask(rows, times)
                self.assertEqual(int(mask.sum()), math.ceil(used / 16))
                self.assertTrue(torch.isfinite(model(values, rows, times)).all())

    def test_rows_and_final_time_patch_do_not_cross_boundaries(self):
        model = WPPTransformer(2, dropout=0).eval()
        values, rows, times = self.inputs(rows=2, used=17)
        values[:, 0] = 1
        values[:, 1] = 2
        seen = []
        handle = model.input_projection.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            model(values, rows, times)
        finally:
            handle.remove()
        patches = seen[0]
        self.assertEqual(patches.shape, (1, 2, 188, 16))
        self.assertTrue(torch.equal(patches[0, 0, 0], torch.ones(16)))
        self.assertTrue(torch.equal(patches[0, 0, 1], torch.tensor([1.] + [0.] * 15)))
        self.assertTrue(torch.equal(patches[0, 1, 0], torch.full((16,), 2.0)))
        self.assertTrue(torch.equal(patches[0, 1, 1], torch.tensor([2.] + [0.] * 15)))
        mask = model._token_mask(rows, times).reshape(1, 2, 188)
        self.assertTrue(mask[0, :, :2].all())
        self.assertFalse(mask[0, :, 2:].any())

    def test_invalid_padding_including_nonfinite_values_cannot_change_eval_logits(self):
        model = WPPTransformer(3, dropout=0).eval()
        values, rows, times = self.inputs(rows=3, used=17)
        rows[0, 2] = False
        valid = rows[:, :, None] & times[:, None, :]
        baseline = model(values, rows, times)
        poisoned = values.clone()
        poisoned[~valid] = float("nan")
        poisoned[0, 2, 0] = float("inf")
        self.assertTrue(torch.allclose(baseline, model(poisoned, rows, times), atol=0, rtol=0))

    def test_batch_row_padding_and_observed_zeros_are_explicitly_masked(self):
        model = WPPTransformer(2, dropout=0).eval()
        alone, one_row, times = self.inputs(rows=1, used=1, fill=0.0)
        self.assertTrue(torch.isfinite(model(alone, one_row, times)).all())  # a valid zero is not padding
        padded = torch.zeros((2, 3, TIME_STEPS))
        padded[0, 0] = alone[0, 0]
        padded[1, :3] = 2
        row_mask = torch.tensor([[True, False, False], [True, True, True]])
        time_mask = torch.zeros((2, TIME_STEPS), dtype=torch.bool)
        time_mask[0, :1] = True
        time_mask[1, :17] = True
        batched = model(padded, row_mask, time_mask)
        self.assertTrue(torch.allclose(model(alone, one_row, times)[0], batched[0], atol=1e-6, rtol=1e-6))

    def test_2d_positions_distinguish_axes_and_row_order_has_an_effect(self):
        positions = two_dimensional_sincos_positions(2, 2, 32, device=torch.device("cpu"), dtype=torch.float32)
        self.assertFalse(torch.equal(positions[0, 1], positions[1, 0]))
        model = WPPTransformer(2, dropout=0).eval()
        values, rows, times = self.inputs(rows=2, used=17)
        values[0, 0] = 0
        values[0, 1] = 9
        swapped = values[:, [1, 0]].clone()
        self.assertFalse(torch.allclose(model(values, rows, times), model(swapped, rows, times)))

    def test_invalid_inputs_and_hyperparameters_are_rejected(self):
        for kwargs in ({"num_classes": 0}, {"num_classes": 2, "patch_size": 0},
                       {"num_classes": 2, "d_model": 30, "nhead": 4},
                       {"num_classes": 2, "d_model": 32, "nhead": 3},
                       {"num_classes": 2, "dropout": 1.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    WPPTransformer(**kwargs)
        model = WPPTransformer(2)
        values, rows, times = self.inputs()
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            model(values, torch.zeros_like(rows), times)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            model(values, rows, torch.zeros_like(times))
        with self.assertRaisesRegex(ValueError, "bool"):
            model(values, rows.to(torch.int64), times)
        with self.assertRaisesRegex(ValueError, "floating"):
            model(values.to(torch.int64), rows, times)
        with self.assertRaisesRegex(ValueError, "row_mask"):
            model(values, torch.ones((1, 2), dtype=torch.bool), times)
        with self.assertRaisesRegex(ValueError, "time_mask"):
            model(values, rows, times[:, :-1])
        with self.assertRaisesRegex(ValueError, "prefix"):
            model(values, rows, torch.tensor([[False, True] + [False] * (TIME_STEPS - 2)]))
        with self.assertRaisesRegex(ValueError, "3000"):
            model(values[..., :-1], rows, times)
        with self.assertRaisesRegex(ValueError, "finite"):
            values[0, 0, 0] = float("nan")
            model(values, rows, times)

    def test_synthetic_backward_has_finite_gradients(self):
        model = WPPTransformer(4, dropout=0.2).train()
        values, rows, times = self.inputs(batch=2, rows=2, used=17)
        logits = model(values, rows, times)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
        loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
