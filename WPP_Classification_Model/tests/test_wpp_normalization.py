from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wpp_normalization import (NormalizationError, WelfordPopulation, normalize_numpy,
                                   normalize_torch, validate_statistics)


class WPPNormalizationTests(unittest.TestCase):
    def test_valid_zeros_are_fitted_and_padding_is_excluded_and_zeroed(self):
        values = np.array([[[0.0, 2.0, np.nan], [4.0, 6.0, np.inf]]])
        rows = np.array([[True, True]])
        times = np.array([[True, True, False]])
        moments = WelfordPopulation(); moments.update(values, rows, times)
        stats = moments.freeze(expected_count=4)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["std"], np.sqrt(5.0))
        result = normalize_numpy(values, rows, times, stats)
        self.assertTrue(np.all(result[:, :, 2] == 0))
        self.assertAlmostEqual(float(result[0, 0, 0]), -3 / np.sqrt(5))
        torch_result = normalize_torch(torch.from_numpy(values.astype(np.float32)), torch.from_numpy(rows),
                                       torch.from_numpy(times), stats)
        self.assertTrue(torch.isfinite(torch_result).all())
        self.assertEqual(float(torch_result[0, 1, 2]), 0.0)

    def test_merge_is_batch_invariant_and_stable_at_large_offset(self):
        values = (1e12 + np.array([[[0., 1., 2., 3.]], [[4., 5., 6., 7.]]]))
        rows = np.ones((2, 1), dtype=bool); times = np.ones((2, 4), dtype=bool)
        together = WelfordPopulation(); together.update(values, rows, times)
        split = WelfordPopulation()
        for index in range(2): split.update(values[index:index + 1], rows[index:index + 1], times[index:index + 1])
        first, second = together.freeze(), split.freeze()
        self.assertAlmostEqual(first["mean"], second["mean"], places=6)
        self.assertAlmostEqual(first["std"], second["std"], places=10)
        self.assertAlmostEqual(first["std"], np.sqrt(5.25), places=10)

    def test_constant_or_nonfinite_valid_data_and_invalid_stats_fail(self):
        values = np.ones((1, 1, 2)); rows = np.array([[True]]); times = np.array([[True, True]])
        moments = WelfordPopulation(); moments.update(values, rows, times)
        with self.assertRaisesRegex(NormalizationError, "nonzero"):
            moments.freeze()
        bad = {"count": 1, "mean": 0., "std": 0.}
        with self.assertRaises(NormalizationError): validate_statistics(bad)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(NormalizationError, "finite"):
            WelfordPopulation().update(values, rows, times)


if __name__ == "__main__":
    unittest.main()
