from __future__ import annotations
import csv, hashlib, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from wpp_metrics import video_metrics
from wpp_reporting import LEGACY_COLUMNS, regenerate

class WPPReportingTests(unittest.TestCase):
    def test_legacy_metric_definitions(self):
        # predictions [real, aigc, aigc, aigc] against [real, real, aigc, aigc]
        metrics=video_metrics([0,0,1,1],[[.9,.1],[.1,.9],[.1,.9],[.1,.9]])
        self.assertEqual(metrics['accuracy'], .75)
        self.assertAlmostEqual(metrics['macro_precision'], (1 + 2/3)/2)
        self.assertAlmostEqual(metrics['macro_recall'], (.5 + 1)/2)
        self.assertAlmostEqual(metrics['weighted_f1'], (2*(2/3) + 2*(4/5))/4)

    def test_readonly_regeneration_matches_reference_layout_and_null_history(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            history=[{'epoch':1,'train_weighted_objective':.8,'learning_rate':.001,'duration_seconds':1791190618.0,
                      'validation':{'weighted_ce':.3,'accuracy':.75,'per_class':{
                          'real':{'precision':1.,'recall':.5,'f1':2/3,'support':2},
                          'aigc':{'precision':2/3,'recall':1.,'f1':.8,'support':2}}}}]
            canonical={'weighted_ce':.25,'per_video':[{'video_id':'a','label':0,'probabilities':[.9,.1]},
                                                        {'video_id':'b','label':0,'probabilities':[.1,.9]},
                                                        {'video_id':'c','label':1,'probabilities':[.1,.9]},
                                                        {'video_id':'d','label':1,'probabilities':[.1,.9]}]}
            (run/'history.json').write_text(json.dumps(history)); (run/'test_once.json').write_text(json.dumps(canonical))
            (run/'config.yaml').write_text('model: test\n'); (run/'status.json').write_text(json.dumps({'best':{'epoch':1,'macro_f1':.7,'weighted_ce':.3}}))
            (run/'checkpoints').mkdir(); (run/'checkpoints'/'best.pt').write_bytes(b'best'); (run/'checkpoints'/'latest.pt').write_bytes(b'latest')
            before=hashlib.sha256((run/'test_once.json').read_bytes()).hexdigest(); out=regenerate(run)
            self.assertEqual(before,hashlib.sha256((run/'test_once.json').read_bytes()).hexdigest())
            expected={'per_epoch_metrics.json','per_epoch_metrics.csv','summarized_metrics.json','test_per_class_metrics.json','test_per_class_metrics.csv','test_confusion_matrix_raw.csv','test_confusion_matrix_raw.png','test_confusion_matrix_normalized.csv','test_confusion_matrix_normalized.png','loss_curves.png','accuracy_curves.png','precision_curves.png','recall_curves.png','f1_curves.png'}
            self.assertTrue(expected <= {p.name for p in out.iterdir()})
            with (out/'per_epoch_metrics.csv').open() as f: self.assertEqual(next(csv.reader(f)),list(LEGACY_COLUMNS))
            row=json.loads((out/'per_epoch_metrics.json').read_text())[0]
            self.assertIsNone(row['duration_seconds']); self.assertIsNone(row['train_accuracy']); self.assertEqual(row['train_loss'],.8)
            summary=json.loads((out/'summarized_metrics.json').read_text())
            self.assertNotIn('best_validation_loss',summary); self.assertAlmostEqual(summary['accuracy'],.75); self.assertEqual(summary['test_loss'],summary['test_weighted_ce'])
            meta=summary['wpp_reporting_metadata']; self.assertEqual(meta['schema_version'],'wpp-legacy-report-v1'); self.assertEqual(meta['selected_checkpoint']['epoch'],1); self.assertEqual(meta['selected_checkpoint']['sha256'],hashlib.sha256(b'best').hexdigest()); self.assertEqual(meta['input_sha256']['canonical_test_once'],before)
            limitations=json.loads((run/'wpp_diagnostics'/'reporting_limitations.json').read_text())
            self.assertIn('validation macro-F1',limitations['selection_criterion'])

if __name__=='__main__': unittest.main()
