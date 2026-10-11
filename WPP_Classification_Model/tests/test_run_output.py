import copy,csv,json,os,re,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from wpp_classifier.config import DEFAULT,load_config,validate_run_name
from wpp_classifier.pipeline import materialize_run_config,train

class RunOutputTest(unittest.TestCase):
 def _dataset(self,root):
  for split in ('train','val','test'):
   vid=f'{split}-0'
   np.savez(root/f'{vid}.npz',row_share=np.array([[1.]],np.float32),delta_row_share=np.array([[0.]],np.float32),pts=np.array([0],np.int64),frame_duration=np.array([1],np.int64),time_base=np.array([1,15],np.int64),input_contract_version=np.array(1,np.int64),display_width=np.array(512,np.int64),display_height=np.array(512,np.int64),fps_num=np.array(15,np.int64),fps_den=np.array(1,np.int64))
   (root/f'{split}_data.csv').write_text(f'video_id,features_path\n{vid},{vid}.npz\n',encoding='utf8')
   (root/f'{split}_labels.csv').write_text(f'video_id,label\n{vid},0\n',encoding='utf8')
 def _config(self,root):
  cfg=copy.deepcopy(DEFAULT)
  cfg['run']={'output_root':str(root/'runs'),'name':None}
  cfg['data']={f'{split}_{kind}_path':str(root/f'{split}_{"data" if kind == "data" else "labels"}.csv')
               for split in ('train','val','test') for kind in ('data','label')}
  cfg['classes']={'class_names':['zero']}
  cfg['model'].update({'architecture':'transformer','window_seconds':1,'d_model':4,'heads':1,'layers':1,'dropout':0})
  cfg['training'].update({'max_epochs':1,'patience':1,'device':'cpu','batch_size':1})
  return cfg
 def test_yaml_data_paths_stay_yaml_relative_while_output_root_uses_cwd(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); config_dir=root/'config';cwd=root/'launch';config_dir.mkdir();cwd.mkdir()
   paths='\n'.join(f'  {key}: input.csv' for key in ('train_data_path','train_label_path','val_data_path','val_label_path','test_data_path','test_label_path'))
   config=config_dir/'run.yaml';config.write_text(f'run:\n  output_root: runs\ndata:\n{paths}\n',encoding='utf8')
   original=Path.cwd();os.chdir(cwd)
   try: cfg=load_config(config)
   finally: os.chdir(original)
   self.assertEqual(cfg['run']['output_root'],str((cwd/'runs').resolve()))
   self.assertEqual(cfg['data']['train_data_path'],str((config_dir/'input.csv').resolve()))
 def test_name_validation_precedence_and_timestamp_fallback(self):
  cfg=copy.deepcopy(DEFAULT)
  cfg['run']={'output_root':'runs','name':'yaml_name'}
  concrete=materialize_run_config(copy.deepcopy(cfg),run_name='cli_name')
  self.assertEqual(concrete['run']['name'],'cli_name')
  self.assertEqual(concrete['run']['output_root'],str((Path.cwd()/'runs').resolve()))
  cfg['run']['name']=None
  generated=materialize_run_config(copy.deepcopy(cfg))['run']['name']
  self.assertRegex(generated,r'^run_\d{8}_\d{6}$')
  for name in ('','.', '..','nested/name',42):
   with self.assertRaises(ValueError): validate_run_name(name)
 def test_generated_name_is_persisted_and_unnamed_resume_reuses_it(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self._dataset(root);cfg=self._config(root);source_before=copy.deepcopy(cfg)
   run=train(cfg);self.assertRegex(run.name,r'^run_\d{8}_\d{6}$')
   saved=json.loads((run/'config.json').read_text(encoding='utf8'))
   latest=torch.load(run/'checkpoints'/'latest.pt',map_location='cpu',weights_only=False)
   self.assertEqual(saved['run']['name'],run.name)
   self.assertEqual(latest['config']['run']['name'],run.name)
   self.assertEqual(train(cfg,run/'checkpoints'/'latest.pt'),run)
   self.assertEqual(cfg,source_before)
   with self.assertRaisesRegex(ValueError,'run name'):
    train(cfg,run/'checkpoints'/'latest.pt','different')
   conflicting=copy.deepcopy(cfg);conflicting['run']['name']='different'
   with self.assertRaisesRegex(ValueError,'config run name'):
    train(conflicting,run/'checkpoints'/'latest.pt')
   self.assertEqual(train(conflicting,run/'checkpoints'/'latest.pt',run.name),run)
   collision=copy.deepcopy(cfg);collision['run']['name']='taken';(root/'runs'/'taken').mkdir()
   with self.assertRaisesRegex(ValueError,'already exists'):
    train(collision)

 def test_cli_override_can_be_used_again_on_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self._dataset(root);cfg=self._config(root)
   cfg['run']['name']='yaml_name';source_before=copy.deepcopy(cfg)
   run=train(cfg,run_name='cli_name')
   self.assertEqual(run.name,'cli_name')
   self.assertEqual(train(cfg,run/'checkpoints'/'latest.pt',run_name='cli_name'),run)
   self.assertEqual(cfg,source_before)

if __name__=='__main__': unittest.main()
