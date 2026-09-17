from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from transformer_tsc import TransformerTSC
from supervised_data import infer_lengths, normalize_length_aware
from model_factory import make_model
import linear_probe_resnet as probe
from run_pipeline import validate_config


class TransformerTests(unittest.TestCase):
    def test_forward_padding_partial_patch_and_odd_width(self):
        torch.manual_seed(1)
        model = TransformerTSC((1, 3000), 3, d_model=33, nhead=3, dropout=0).eval()
        x = torch.zeros(2, 1, 3000)
        x[0, 0, :17] = torch.arange(1, 18)  # one full and one partial patch
        x[1, 0, :1] = 5                     # one-frame valid sequence
        lengths = torch.tensor([17, 1])
        before = model(x, lengths)
        x[:, :, 20:] = 999
        self.assertEqual(tuple(before.shape), (2, 3))
        self.assertTrue(torch.allclose(before, model(x, lengths)))

    def test_length_and_shape_validation(self):
        model = TransformerTSC((1, 3000), 2, dropout=0)
        x = torch.zeros(1, 1, 3001)
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            model(x, torch.tensor([3001]))
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            TransformerTSC((1, 3000), 2, patch_size=True)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            TransformerTSC((1, 3000), 2, d_model=True)

    def test_lengths_and_train_only_normalization(self):
        raw = np.zeros((2, 3000)); raw[0, :2] = [2, 4]; raw[1, :1] = 100
        lengths = infer_lengths(raw)
        _, mean, std = normalize_length_aware(raw[:1], lengths[:1])
        other, _, _ = normalize_length_aware(raw[1:], lengths[1:], mean, std)
        self.assertEqual(lengths.tolist(), [2, 1]); self.assertEqual(float(other[0, 1]), 0.)
        self.assertEqual((mean, std), (3., 1.))
        raw[0, 1] = 0; raw[0, 2] = 2
        with self.assertRaisesRegex(ValueError, 'zero followed'): infer_lengths(raw)

    def test_probe_rejects_transformer_checkpoint(self):
        import tempfile
        cfg = {'architecture': 'transformer', 'patch_size': 16, 'd_model': 32, 'nhead': 4,
               'num_layers': 2, 'dim_feedforward': 64, 'dropout': .2, 'length_aware': True}
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            torch.save({'format_version': 2, 'model_state': {}, 'model_config': cfg,
                        'preprocessing_config': {}, 'label_mapping': {'a': 0},
                        'split_information': {}}, f.name)
            with self.assertRaisesRegex(ValueError, 'only supports ResNet'):
                probe._load_checkpoint(Path(f.name), torch.device('cpu'))

    def test_transformer_config_defaults(self):
        cfg = validate_config({'data': {'mode': 'auto', 'data_path': 'data.csv',
                                        'label_path': 'labels.csv'},
                               'model': {'architecture': 'transformer'}})
        self.assertEqual(cfg['model'], {'architecture': 'transformer', 'patch_size': 16,
                         'd_model': 32, 'nhead': 4, 'num_layers': 2,
                         'dim_feedforward': 64, 'dropout': .2, 'length_aware': True})
        self.assertEqual(cfg['preprocessing']['method'], 'train_global_valid_zscore')

    def test_factory_and_config_errors(self):
        cfg = {'architecture': 'transformer', 'patch_size': 16, 'd_model': 32, 'nhead': 4,
               'num_layers': 2, 'dim_feedforward': 64, 'dropout': .2, 'length_aware': True}
        self.assertIsInstance(make_model(cfg, 2), TransformerTSC)
        cfg['nhead'] = 3
        with self.assertRaisesRegex(ValueError, 'divisible'): make_model(cfg, 2)
        cfg['nhead'] = True
        with self.assertRaisesRegex(ValueError, 'positive integer'): make_model(cfg, 2)


if __name__ == '__main__': unittest.main()
