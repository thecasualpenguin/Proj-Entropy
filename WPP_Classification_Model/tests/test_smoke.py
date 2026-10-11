import csv,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from wpp_classifier.pipeline import train,eval_checkpoint,_load_resume,_load_history,split_load
from wpp_classifier.models import make_model
import torch
from wpp_classifier.config import DEFAULT
import copy
class SmokeTest(unittest.TestCase):
 def test_current_schema_resume_history_uses_checkpoint_fallback(self):
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);(run/'evaluations').mkdir()
   disk=[{'epoch':1,'train_accuracy':.1}];checkpoint_history=[*disk,{'epoch':2,'train_accuracy':.2}]
   (run/'evaluations'/'per_epoch_metrics.json').write_text(json.dumps(disk))
   self.assertEqual(_load_history(run,{'epoch':2,'history':checkpoint_history}),checkpoint_history)
 def test_two_epoch_three_class_both_architectures(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for split in ('train','val','test'):
    data=[];labels=[]
    for c in range(3):
     vid=f'{split}-{c}';t,r=2+c,1+c
     np.savez(root/f'{vid}.npz',row_share=np.full((t,r),1/r,np.float32),delta_row_share=np.full((t,r),c*.1,np.float32),pts=np.arange(t,dtype=np.int64)*7-4,frame_duration=np.full(t,7,np.int64),time_base=np.array([1,10],np.int64))
     data.append([vid,f'{vid}.npz']);labels.append([vid,c])
    for suffix,rows,head in (('data',data,['video_id','features_path']),('labels',labels[::-1],['video_id','label'])):
     with (root/f'{split}_{suffix}.csv').open('w',newline='') as f: w=csv.writer(f);w.writerow(head);w.writerows(rows)
   for architecture in ('resnet','transformer'):
    cfg=copy.deepcopy(DEFAULT);cfg['run']={'output_root':str(root/'runs'),'name':architecture};cfg['data']={f'{s}_{kind}_path':str(root/f'{s}_{"data" if kind=="data" else "labels"}.csv') for s in ('train','val','test') for kind in ('data','label')};cfg['classes']={'class_names':['zero','one','two']};cfg['model'].update({'architecture':architecture,'window_seconds':'1.5','width':8,'d_model':8,'heads':2,'layers':1,'resnet_blocks':1,'dropout':0});cfg['training'].update({'max_epochs':2,'patience':2,'device':'cpu','batch_size':1})
    run=train(cfg);self.assertTrue((run/'evaluations'/'summarized_metrics.json').is_file())
    for name in ('per_epoch_metrics.csv','per_epoch_metrics.json','summarized_metrics.json','test_per_class_metrics.csv','test_confusion_matrix_raw.csv','test_confusion_matrix_normalized.png','loss_curves.png','accuracy_curves.png','precision_curves.png','recall_curves.png','f1_curves.png'): self.assertTrue((run/'evaluations'/name).is_file(),name)
    history=json.loads((run/'evaluations'/'per_epoch_metrics.json').read_text())
    summary=json.loads((run/'evaluations'/'summarized_metrics.json').read_text())
    self.assertEqual(summary['best_observed_per_epoch_metrics']['train']['accuracy']['epoch'],min(row['epoch'] for row in history if row['train_accuracy']==max(item['train_accuracy'] for item in history)))
    self.assertEqual(summary['checkpoint_selection']['selected_epoch'],summary['selected_epoch'])
    self.assertNotIn('weighted_ce',summary);self.assertIsInstance(summary['test_loss'],float)
    self.assertEqual(json.loads((run/'status.json').read_text())['state'],'completed')
    manual=root/('manual_'+architecture); self.assertTrue(eval_checkpoint(cfg,run/'checkpoints'/'best.pt',manual).is_dir())
    manual_metadata=json.loads((manual/'evaluation_metadata.json').read_text())
    self.assertIsInstance(manual_metadata['metrics']['weighted_loss'],float)
    with self.assertRaises(ValueError): eval_checkpoint(cfg,run/'checkpoints'/'best.pt',run/'evaluations')
    # NPZ-content provenance, not just CSV paths, protects independent evaluation.
    if architecture=='transformer':
     # Resume rejects a changed config before model loading, and exact source hashes
     # reject both NPZ and CSV content changes.
     _, metas=split_load(cfg); model=make_model(cfg,3); opt=torch.optim.AdamW(model.parameters()); scaler=torch.amp.GradScaler('cuda',enabled=False)
     changed=copy.deepcopy(cfg); changed['training']['seed']=99
     with self.assertRaisesRegex(ValueError,'configuration changed'): _load_resume(run/'checkpoints'/'latest.pt',changed,metas,model,opt,scaler)
     # A semantically equivalent relative path still changes the supplied CSV source hash.
     data_path=root/'test_data.csv'; original_data=data_path.read_bytes(); data_path.write_bytes(original_data.replace(b'test-1.npz',b'./test-1.npz'))
     _, changed_metas=split_load(cfg)
     with self.assertRaisesRegex(ValueError,'content hashes changed'): _load_resume(run/'checkpoints'/'latest.pt',cfg,changed_metas,model,opt,scaler)
     # Restore exact bytes to isolate the feature-content tamper.
     data_path.write_bytes(original_data)
     np.savez(root/'test-0.npz',row_share=np.array([[1.]],np.float32),delta_row_share=np.array([[0.]],np.float32),pts=np.array([0],np.int64),frame_duration=np.array([1],np.int64),time_base=np.array([1,1],np.int64))
     with self.assertRaisesRegex(ValueError,'hashes differ'): eval_checkpoint(cfg,run/'checkpoints'/'best.pt',root/'tampered')
if __name__=='__main__':unittest.main()
