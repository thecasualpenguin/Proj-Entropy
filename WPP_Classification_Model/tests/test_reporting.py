import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wpp_classifier.pipeline import _load_history
from wpp_classifier.reporting import CLASSIFICATION_METRICS, best_observed_history


class ReportingTest(unittest.TestCase):
    def test_independent_maxima_and_earliest_ties(self):
        history = []
        for epoch in (3, 1, 2):
            row = {'epoch': epoch}
            for split in ('train', 'validation'):
                for index, metric in enumerate(CLASSIFICATION_METRICS):
                    # Different metrics peak at different epochs. Epochs 1/2 tie
                    # for even metrics, and input ordering is intentionally mixed.
                    row[f'{split}_{metric}'] = (
                        .9 if (index % 2 == 0 and epoch in (1, 2)) or
                        (index % 2 == 1 and epoch == 3) else .2
                    )
            history.append(row)
        result = best_observed_history(history)
        self.assertEqual(set(result), {'train', 'validation'})
        for split in result:
            self.assertEqual(set(result[split]), set(CLASSIFICATION_METRICS))
            for index, metric in enumerate(CLASSIFICATION_METRICS):
                self.assertEqual(result[split][metric], {
                    'epoch': 1 if index % 2 == 0 else 3, 'value': .9,
                })

    def test_nonfinite_and_missing_values_are_not_maxima(self):
        result = best_observed_history([
            {'epoch': 1, 'train_accuracy': float('nan')},
            {'epoch': 2, 'train_accuracy': float('inf')},
            {'epoch': 3, 'train_accuracy': .5},
            {'epoch': 4, 'train_accuracy': True},
        ])
        self.assertEqual(result['train'], {'accuracy': {'epoch': 3, 'value': .5}})
        self.assertEqual(result['validation'], {})

    def test_resume_trims_history_after_checkpoint_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'evaluations').mkdir()
            rows = [{'epoch': epoch, 'train_accuracy': epoch / 10} for epoch in (1, 2, 3)]
            (run / 'evaluations' / 'per_epoch_metrics.json').write_text(json.dumps(rows))
            self.assertEqual(_load_history(run, {'epoch': 2, 'history': rows[:2]}), rows[:2])
            conflicting = [dict(row) for row in rows[:2]]
            conflicting[0]['train_accuracy'] = .9
            with self.assertRaisesRegex(ValueError, 'history differ'):
                _load_history(run, {'epoch': 2, 'history': conflicting})

    def test_missing_or_noncontiguous_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for rows in (None, [], [{'epoch': 2}], [{'epoch': True}]):
                with self.assertRaisesRegex(ValueError, 'history is missing'):
                    _load_history(run, {'epoch': 2, 'history': rows})


if __name__ == '__main__':
    unittest.main()
