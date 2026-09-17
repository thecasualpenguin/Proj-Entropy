"""Run with: ../../.venv/bin/python -m unittest discover -s tests -v"""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from resnet_tsc import ResNet
from transformer_tsc import TransformerTSC


torch.set_num_threads(2)


class LengthTests(unittest.TestCase):
    def test_real_only_normalization(self):
        x = np.array([[2, 4, 99, 99], [6, 99, 99, 99]], dtype=np.float32)
        lengths = np.array([2, 1])
        mean, std = train.fit_normalization(x, lengths)
        np.testing.assert_allclose(mean, 4)
        np.testing.assert_allclose(std, np.std([2, 4, 6]), rtol=1e-6)
        out = train.normalize_inputs(x, mean, std, lengths)
        self.assertEqual(out[1, 1:].tolist(), [0, 0, 0])
        self.assertEqual(out[0, 2:].tolist(), [0, 0])
        mean_old, std_old = train.fit_normalization(x)
        np.testing.assert_allclose(train.normalize_inputs(x, mean_old, std_old),
                                   (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0)))

    def test_padding_does_not_change_logits_or_gradients(self):
        for kind in ('transformer', 'resnet'):
            with self.subTest(model=kind):
                model = (TransformerTSC((1, 32), 3, patch_size=4, dropout=0, length_aware=True)
                         if kind == 'transformer' else ResNet((1, 32), 3, 4, length_aware=True))
                lengths = torch.tensor([1, 5, 13])
                x = torch.randn(3, 32)
                valid = torch.arange(32)[None, :] < lengths[:, None]
                changed = x.masked_fill(~valid, 1e5)
                model.eval()
                with torch.no_grad():
                    expected = model(x, lengths)
                    torch.testing.assert_close(expected, model(changed, lengths))
                    torch.testing.assert_close(expected, model(x[:, :13], lengths))
                    # Short sample alone must ignore extra tokens in a longer peer.
                    torch.testing.assert_close(expected[:1], model(x[:1], lengths[:1]), atol=1e-6, rtol=1e-5)
                model.train()
                clone = copy.deepcopy(model)
                a = model(x, lengths); b = clone(changed, lengths)
                torch.testing.assert_close(a, b)
                a.square().sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                with self.assertRaises(ValueError):
                    model(x)
                with self.assertRaises(ValueError):
                    model(x, torch.tensor([0, 5, 13]))

    def test_manifest_rejects_bad_padding_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'examples.csv'
            manifest.write_text('row,subject,original_frames,valid_frames\n0,001,2,2\n')
            x = np.array([[8, 16, 0]], dtype=np.float32)
            np.testing.assert_equal(train.load_lengths(root, x, ['001']), [2])
            with self.assertRaises(ValueError):
                train.load_lengths(root, x, ['002'])
            x[0, -1] = 8
            with self.assertRaises(ValueError):
                train.load_lengths(root, x, ['001'])

    def test_single_real_frame_resnet_backward(self):
        model = ResNet((1, 10), 2, 4, length_aware=True).train()
        out = model(torch.randn(1, 10), torch.tensor([1]))
        out.sum().backward()
        self.assertTrue(torch.isfinite(out).all())


class RunnerTests(unittest.TestCase):
    def test_checkpoints_and_both_modes(self):
        real, labels = train.load_training_data(Path('outputs/matrix'))
        labels = np.asarray(labels)
        indices = np.concatenate([np.flatnonzero(labels == label)[:3] for label in sorted(set(labels))[:2]])
        with Path('outputs/matrix/examples.csv').open() as handle:
            manifest = list(csv.DictReader(handle))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir()
            # Actual full-width CASIA-B rows for all four integration paths.
            np.save(data / 'X.npy', real[indices]); np.save(data / 'Y.npy', labels[indices])
            with (data / 'examples.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest[0].keys()); writer.writeheader()
                for i, index in enumerate(indices):
                    writer.writerow({**manifest[index], 'row': i})
            for kind in ('resnet', 'transformer'):
                for aware in (False, True):
                    with self.subTest(model=kind, length_aware=aware):
                        run = root / f'{kind}-{aware}'; run.mkdir()
                        # First exercise real train/evaluate, plotting and all checkpoint outputs.
                        with patch('train.summary'), patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=False):
                            train._run(kind, epochs=1, run_dir=run, data_dir=data, length_aware=aware)
                        self.assertEqual({p.name for p in run.glob('*.pt')}, {'best_train.pt', 'best_validation.pt', 'last.pt'})
                        for name in ('best_train', 'best_validation', 'last'):
                            model, checkpoint = train.load_checkpoint(run / f'{name}.pt')
                            self.assertEqual(checkpoint['epoch'], 1)
                            self.assertEqual(checkpoint['config']['length_aware'], aware)
                            self.assertEqual(checkpoint['config']['optimizer'], 'adamw')
                            self.assertEqual(checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay'], 0.01)
                            lengths = train.load_lengths(data, real[indices], labels[indices]) if aware else None
                            normalized = train.normalize_inputs(real[indices], checkpoint['mean'], checkpoint['std'], lengths)
                            model.eval()
                            with torch.no_grad():
                                logits = model(torch.from_numpy(normalized), *([torch.from_numpy(lengths)] if aware else []))
                            self.assertTrue(torch.isfinite(logits).all())
                        self.assertTrue((run / 'loss-accuracy.png').exists())
            # Scripted metrics verify selection, including keeping the first tied best.
            run = root / 'selection'; run.mkdir()
            scores = [(1., .8), (2., .4), (1., .7), (2., .6), (1., .8), (2., .5)]
            with patch('train.evaluate', side_effect=scores), patch('train.summary'), patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=False):
                train._run('transformer', epochs=3, run_dir=run, data_dir=data, length_aware=True)
            for name, epoch in [('best_train', 1), ('best_validation', 2), ('last', 3)]:
                _, ckpt = train.load_checkpoint(run / f'{name}.pt')
                self.assertEqual(ckpt['epoch'], epoch)
            self.assertEqual(len(json.loads((run / 'loss.json').read_text())['epoch']), 3)


if __name__ == '__main__':
    unittest.main()
