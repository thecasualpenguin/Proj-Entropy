import copy,csv,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from wpp_classifier.config import DEFAULT
from wpp_classifier.pipeline import train,eval_checkpoint

class RawLabelSmokeTest(unittest.TestCase):
 def _write_csv(self,path,headers,rows):
  with path.open('w',newline='',encoding='utf8') as f:
   writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
 def _npz(self,path,index):
  t,r=2+index,1+(index%2)
  np.savez(path,row_share=np.full((t,r),1/r,np.float32),delta_row_share=np.full((t,r),index*.1,np.float32),
           pts=np.arange(t,dtype=np.int64)*7-3,frame_duration=np.full(t,7,np.int64),time_base=np.array([1,10],np.int64))
 def _dataset(self,root,labels=('001','016','124')):
  for split in ('train','val','test'):
   use=labels if split != 'val' else labels[:1]
   data=[];targets=[]
   for index,label in enumerate(use):
    video_id=f'{split}-{index}';self._npz(root/f'{video_id}.npz',index)
    data.append([video_id,f'{video_id}.npz']);targets.append([video_id,label])
   self._write_csv(root/f'{split}_data.csv',['video_id','features_path'],data)
   self._write_csv(root/f'{split}_labels.csv',['video_id','label'],targets[::-1])
 def _config(self,root,name,architecture,device):
  cfg=copy.deepcopy(DEFAULT);cfg['run']={'output_root':str(root/'runs'),'name':name}
  cfg['data']={f'{split}_{kind}_path':str(root/f'{split}_{"data" if kind=="data" else "labels"}.csv')
               for split in ('train','val','test') for kind in ('data','label')}
  cfg['classes']={}  # Deliberately absent: raw text labels define the train vocabulary.
  cfg['model'].update({'architecture':architecture,'window_seconds':'1.5','width':8,'d_model':8,'heads':2,'layers':1,'resnet_blocks':1,'dropout':0})
  cfg['training'].update({'seed':42,'max_epochs':2,'patience':2,'batch_size':1,'device':device,'amp':False})
  return cfg
 def _run_two_architectures(self,device):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self._dataset(root)
   for architecture in ('resnet','transformer'):
    cfg=self._config(root,f'{device}_{architecture}',architecture,device); source_before=copy.deepcopy(cfg)
    run=train(cfg); latest=run/'checkpoints'/'latest.pt'; canonical=run/'test_evaluation'/'test_metrics.json'
    mapping=json.loads((run/'label_mapping.json').read_text(encoding='utf8'))
    self.assertEqual(mapping['class_names'],['001','016','124'])
    self.assertEqual(cfg,source_before)  # Runtime snapshotting never mutates source config.
    report=json.loads(canonical.read_text(encoding='utf8'))
    self.assertEqual([row['class_name'] for row in report['per_class']],['001','016','124'])
    self.assertEqual(report['per_class'][1]['support'],1)
    # Latest checkpoint can be verified/resumed after completion without changing the canonical test.
    canonical_before=canonical.read_bytes();self.assertEqual(train(cfg,latest),run)
    self.assertEqual(canonical.read_bytes(),canonical_before)
    manual=root/f'manual_{architecture}';self.assertTrue(eval_checkpoint(cfg,latest,manual).is_dir())
    with self.assertRaisesRegex(ValueError,'canonical test_evaluation'):
     eval_checkpoint(cfg,latest,run/'test_evaluation')
    # The run-side mapping and checkpoint-side mapping are both tamper guards before weight use.
    mapping_path=run/'label_mapping.json'; mapping_before=mapping_path.read_bytes();mapping_path.write_text('{not-json',encoding='utf8')
    with self.assertRaisesRegex(ValueError,'unreadable run label_mapping'):
     train(cfg,latest)
    mapping_path.write_bytes(mapping_before)
    run_mapping=json.loads(mapping_before);run_mapping['mapping_sha256']='tampered'
    mapping_path.write_text(json.dumps(run_mapping),encoding='utf8')
    with self.assertRaisesRegex(ValueError,'run label_mapping.json differs'):
     train(cfg,latest)
    mapping_path.write_bytes(mapping_before)
    altered=torch.load(latest,map_location='cpu',weights_only=False);altered['label_mapping']=copy.deepcopy(altered['label_mapping']);altered['label_mapping']['mapping']['016']=0
    tampered_checkpoint=root/f'{architecture}_mapping_tampered.pt';torch.save(altered,tampered_checkpoint)
    with self.assertRaisesRegex(ValueError,'regenerated training label mapping differs'):
     train(cfg,tampered_checkpoint)
    with self.assertRaisesRegex(ValueError,'checkpoint configuration differs'):
     eval_checkpoint(cfg,tampered_checkpoint,root/f'mapping_tampered_{architecture}')
    # Valid-but-different feature bytes trigger full supplied-input provenance checks for resume and evaluate.
    npz=root/'test-0.npz';original=npz.read_bytes();self._npz(npz,2)
    with self.assertRaisesRegex(ValueError,'content hashes changed'):
     train(cfg,latest)
    with self.assertRaisesRegex(ValueError,'input hashes differ'):
     eval_checkpoint(cfg,latest,root/f'tampered_{architecture}')
    npz.write_bytes(original)
 def test_two_epoch_auto_raw_cpu_both_architectures(self): self._run_two_architectures('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_two_epoch_auto_raw_cuda_both_architectures(self): self._run_two_architectures('cuda')
 def test_single_raw_class_is_supported_and_reports_null_auc(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self._dataset(root,('016',));cfg=self._config(root,'single','transformer','cpu');cfg['training'].update(max_epochs=1,patience=1)
   run=train(cfg);report=json.loads((run/'test_evaluation'/'test_metrics.json').read_text(encoding='utf8'))
   self.assertEqual(report['per_class'][0]['class_name'],'016')
   self.assertIsNone(report['per_class'][0]['roc_auc'])
   self.assertIsNone(report['per_class'][0]['average_precision'])
if __name__=='__main__':unittest.main()
