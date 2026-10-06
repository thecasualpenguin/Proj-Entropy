import csv,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from wpp_classifier.config import exact_seconds
from wpp_classifier.data import load_split,make_windows,fit_normalizer,Video

class ContractTest(unittest.TestCase):
 def test_csv_join_and_windows(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); np.savez(d/'a.npz',row_share=np.array([[0.,1.],[.2,.8],[.5,.5]],np.float32),delta_row_share=np.zeros((3,2),np.float32),pts=np.array([10,25,40],np.int64),frame_duration=np.array([15,15,20],np.int64),time_base=np.array([1,10],np.int64))
   (d/'d.csv').write_text('video_id,features_path\na,a.npz\n');(d/'l.csv').write_text('video_id,label\na,1\n')
   v,_=load_split(d/'d.csv',d/'l.csv',3); self.assertEqual([x.tolist() for x,_ in make_windows(v[0],'1.5')],[[0],[1],[2]])
   norm=fit_normalizer(v);self.assertEqual(norm['count'],[6,6])
 def test_extreme_pts_and_finite_nonunit_row_share_are_accepted(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); np.savez(d/'x.npz',row_share=np.array([[-5.,7.],[3.,-1.]],np.float32),delta_row_share=np.zeros((2,2),np.float32),pts=np.array([-9223372036854775807,9223372036854775807],np.int64),frame_duration=np.array([9223372036854775807,9223372036854775807],np.int64),time_base=np.array([1,1],np.int64))
   (d/'d.csv').write_text('video_id,features_path\nx,x.npz\n');(d/'l.csv').write_text('video_id,label\nx,0\n')
   videos,_=load_split(d/'d.csv',d/'l.csv',1); self.assertEqual(len(make_windows(videos[0],'1')),2)
   low=Video('low',0,videos[0].features,np.array([0,1],np.int64),videos[0].duration,(1,30),'')
   high=Video('high',0,videos[0].features,np.array([9223372036854775804,9223372036854775805],np.int64),videos[0].duration,(1,30),'')
   self.assertEqual([x.tolist() for x,_ in make_windows(low,'1/30')],[x.tolist() for x,_ in make_windows(high,'1/30')])
 def test_exact_rejects_float(self):
  with self.assertRaises(ValueError): exact_seconds(1.5)
if __name__=='__main__':unittest.main()
