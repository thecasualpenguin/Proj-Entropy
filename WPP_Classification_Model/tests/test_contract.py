import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from wpp_classifier.config import exact_seconds,load_config
from wpp_classifier.data import load_split,make_windows,fit_normalizer
from wpp_classifier.pipeline import _mapping_from_checkpoint

CONTRACT={'input_contract_version':np.asarray(1,np.int64),'display_width':np.asarray(512,np.int64),'display_height':np.asarray(512,np.int64),'fps_num':np.asarray(15,np.int64),'fps_den':np.asarray(1,np.int64)}
def save_npz(path,**arrays): np.savez(path,**arrays,**CONTRACT)

class ContractTest(unittest.TestCase):
 def _csvs(self,d,name='a'):
  (d/'d.csv').write_text(f'video_id,features_path\n{name},{name}.npz\n');(d/'l.csv').write_text(f'video_id,label\n{name},1\n')
 def test_csv_join_windows_and_contract(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp); save_npz(d/'a.npz',row_share=np.array([[0.,1.],[.2,.8],[.5,.5]],np.float32),delta_row_share=np.zeros((3,2),np.float32),pts=np.array([10,11,12],np.int64),frame_duration=np.ones(3,np.int64),time_base=np.array([1,15],np.int64));self._csvs(d)
   v,_=load_split(d/'d.csv',d/'l.csv',3); self.assertEqual([x.tolist() for x,_ in make_windows(v[0],'2/15')],[[0,1],[2]])
   self.assertEqual(fit_normalizer(v)['count'],[6,6])
 def test_rejects_legacy_and_bad_media_metadata_or_timing(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp); base=dict(row_share=np.ones((2,1),np.float32),delta_row_share=np.zeros((2,1),np.float32),pts=np.array([0,1],np.int64),frame_duration=np.ones(2,np.int64),time_base=np.array([1,15],np.int64))
   np.savez(d/'a.npz',**base);self._csvs(d)
   with self.assertRaisesRegex(ValueError,'input-contract'): load_split(d/'d.csv',d/'l.csv',2)
   bad=dict(CONTRACT);bad['fps_num']=np.asarray(30,np.int64);np.savez(d/'a.npz',**base,**bad)
   with self.assertRaisesRegex(ValueError,'fps_num'): load_split(d/'d.csv',d/'l.csv',2)
   save_npz(d/'a.npz',**{**base,'pts':np.array([0,2],np.int64)})
   with self.assertRaisesRegex(ValueError,'pts must advance'): load_split(d/'d.csv',d/'l.csv',2)
   save_npz(d/'a.npz',**{**base,'frame_duration':np.array([1,2],np.int64)})
   with self.assertRaisesRegex(ValueError,'frame_duration'): load_split(d/'d.csv',d/'l.csv',2)
 def test_convtran_settings_are_ignored_by_existing_architecture_validation(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'config.yaml'; paths='\n'.join(f'  {key}: {key}.csv' for key in ('train_data_path','train_label_path','val_data_path','val_label_path','test_data_path','test_label_path'))
   path.write_text('data:\n'+paths+'\nmodel:\n  architecture: resnet\n  d_model: 6\n  heads: 2\n  convtran_heads: 4\n  convtran_layers: 0\n  convtran_ff_multiplier: 0\n  convtran_kernel_size: 2\n')
   cfg=load_config(path);self.assertNotIn('convtran_max_frames',cfg['model'])
   path.write_text('data:\n'+paths+'\nmodel:\n  architecture: convtran\n  d_model: 6\n  heads: 2\n  convtran_heads: 4\n')
   with self.assertRaisesRegex(ValueError,'convtran_heads'): load_config(path)
 def test_exact_rejects_float(self):
  with self.assertRaises(ValueError): exact_seconds(1.5)
 def test_pre_contract_checkpoint_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'predates the strict input-media contract'): _mapping_from_checkpoint({'version':3})
if __name__=='__main__':unittest.main()
