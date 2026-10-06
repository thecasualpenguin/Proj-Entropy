import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from wpp_classifier.reporting import metrics
class MetricsTest(unittest.TestCase):
 def test_configured_absent_class_kept_with_zero_support_and_null_auc(self):
  m,per,_=metrics([0,1],[0,1],np.array([[.8,.1,.1],[.1,.8,.1]]),['a','b','absent'])
  self.assertEqual(len(per),3); self.assertEqual(per[2]['support'],0); self.assertIsNone(per[2]['roc_auc']); self.assertTrue(per[2]['diagnostic_reason'])
 def test_single_class_vocabulary(self):
  m,per,_=metrics([0],[0],np.array([[1.]]),['only']); self.assertEqual(m['accuracy'],1.); self.assertEqual(per[0]['support'],1);self.assertIsNone(per[0]['roc_auc'])
if __name__=='__main__': unittest.main()
