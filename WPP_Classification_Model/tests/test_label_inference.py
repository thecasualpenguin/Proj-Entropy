import copy,csv,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from wpp_classifier.config import DEFAULT,load_config
from wpp_classifier.data import infer_label_mapping,load_split
from wpp_classifier.pipeline import train,eval_checkpoint

class LabelInferenceTest(unittest.TestCase):
 def _npz(self,path,value):
  np.savez(path,row_share=np.array([[1.]],np.float32),delta_row_share=np.array([[value]],np.float32),pts=np.array([0],np.int64),frame_duration=np.array([1],np.int64),time_base=np.array([1,15],np.int64),input_contract_version=np.array(1,np.int64),display_width=np.array(512,np.int64),display_height=np.array(512,np.int64),fps_num=np.array(15,np.int64),fps_den=np.array(1,np.int64))
 def _csv(self,path,header,rows):
  with path.open('w',newline='',encoding='utf8') as f: w=csv.writer(f);w.writerow(header);w.writerows(rows)
 def test_train_only_raw_vocabulary_checkpoint_and_unknown_rejection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   labels=[' 001 ','016','124']
   for split in ('train','val','test'):
    data=[]; out=[]
    use=labels if split != 'val' else labels[:1]
    for i,label in enumerate(use):
     vid=f'{split}-{i}'; self._npz(root/f'{vid}.npz',i);data.append([vid,f'{vid}.npz']);out.append([vid,label])
    self._csv(root/f'{split}_data.csv',['video_id','features_path'],data)
    self._csv(root/f'{split}_labels.csv',['video_id','label'],out[::-1])
   cfg=copy.deepcopy(DEFAULT);cfg['run']={'output_root':str(root/'runs'),'name':'raw'}
   cfg['data']={f'{s}_{kind}_path':str(root/f'{s}_{"data" if kind=="data" else "labels"}.csv') for s in ('train','val','test') for kind in ('data','label')}
   cfg['classes']={};cfg['model'].update({'architecture':'transformer','window_seconds':1,'d_model':8,'heads':2,'layers':1,'dropout':0})
   cfg['training'].update({'max_epochs':1,'patience':1,'device':'cpu','batch_size':1,'seed':42})
   run=train(cfg)
   mapping=__import__('json').loads((run/'label_mapping.json').read_text())
   self.assertEqual(mapping['class_names'],['001','016','124'])
   self.assertEqual(mapping['mapping'],{'001':0,'016':1,'124':2})
   checkpoint=torch.load(run/'checkpoints'/'best.pt',map_location='cpu',weights_only=False)
   self.assertEqual(checkpoint['label_mapping'],mapping)
   self.assertTrue(eval_checkpoint(cfg,run/'checkpoints'/'best.pt',root/'manual').is_dir())
   self._csv(root/'val_labels.csv',['video_id','label'],[['val-0','999']])
   with self.assertRaisesRegex(ValueError,"not in the training vocabulary"):
    eval_checkpoint(cfg,run/'checkpoints'/'best.pt',root/'unknown')
 def test_canonicalization_subset_and_legacy_indexed_mode(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); labels=root/'labels.csv'; data=root/'data.csv';self._npz(root/'x.npz',0)
   self._csv(labels,['video_id','label'],[['x',' PersonA '],['y','persona'],['z','016']])
   inferred=infer_label_mapping(labels)
   self.assertEqual(inferred['class_names'],['016','persona'])
   self.assertEqual(inferred['mapping']['016'],0) # non-contiguous identity is not used as an index
   self._csv(data,['video_id','features_path'],[['x','x.npz']]);self._csv(labels,['video_id','label'],[['x','1']])
   self.assertEqual(load_split(data,labels,2)[0][0].label,1)
   self._csv(labels,['video_id','label'],[['x','2']])
   with self.assertRaisesRegex(ValueError,'outside 0..1'): load_split(data,labels,2)
 def test_config_allows_absent_or_empty_classes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); paths='\n'.join(f'  {p}: {root}/x.csv' for p in ('train_data_path','train_label_path','val_data_path','val_label_path','test_data_path','test_label_path'))
   for classes in ('','classes:\n  class_names: []\n'):
    config=root/('config'+str(len(classes))+'.yaml');config.write_text('data:\n'+paths+'\n'+classes)
    self.assertEqual(load_config(config)['classes']['label_mode'],'raw_inferred')
if __name__=='__main__':unittest.main()
