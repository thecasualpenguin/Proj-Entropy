from __future__ import annotations
import copy, json, os, random, time, shutil
from contextlib import nullcontext
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from .data import load_split,fit_normalizer,transform,make_windows,provenance,infer_label_mapping,sha256,mapping_sha256
from .models import make_model,aggregate_logits
from .reporting import atomic_json,save_evaluation,save_curves,metrics

def now(): return datetime.now(timezone.utc).isoformat()
def seed_all(n): random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n) if torch.cuda.is_available() else None
def device_for(x):
 if x=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 d=torch.device(x)
 if d.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
 return d
def rng(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
 if s.get('cuda') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def save_torch(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');torch.save(obj,q);q.replace(p)

def label_mode(cfg):
 classes=cfg.get('classes',{})
 return classes.get('label_mode') or ('indexed' if classes.get('class_names') else 'raw_inferred')

def indexed_label_mapping(cfg):
 names=cfg.get('classes',{}).get('class_names')
 if not isinstance(names,list) or not names: raise ValueError('classes.class_names must be a unique nonempty list in indexed mode')
 if any(not isinstance(x,str) or not x for x in names) or len(set(names))!=len(names): raise ValueError('classes.class_names must be a unique nonempty list')
 mapping={'version':1,'mode':'indexed','normalizer':None,'class_names':list(names),
          'mapping':{str(i):i for i in range(len(names))},'train_label_sha256':sha256(cfg['data']['train_label_path'])}
 mapping['mapping_sha256']=mapping_sha256(mapping)
 return mapping

def training_label_mapping(cfg):
 return indexed_label_mapping(cfg) if label_mode(cfg)=='indexed' else infer_label_mapping(cfg['data']['train_label_path'])

def materialize_cfg(cfg,mapping):
 """Return a snapshot config without mutating the caller's parsed source config."""
 out=copy.deepcopy(cfg); out['classes']={'label_mode':mapping['mode'],'class_names':list(mapping['class_names']),
                                        'label_mapping':copy.deepcopy(mapping)}
 weights=out['training']['class_weights']
 if isinstance(weights,list) and len(weights)!=len(mapping['class_names']): raise ValueError('training.class_weights vector must have one value per class')
 return out

def split_load(cfg,mapping=None):
 mapping=mapping or training_label_mapping(cfg)
 result={}; metas={}
 for k in ('train','val','test'):
  result[k],metas[k]=load_split(cfg['data'][f'{k}_data_path'],cfg['data'][f'{k}_label_path'],mapping)
 return result,metas

def class_weights(videos,cfg,device):
 q=cfg['training']['class_weights']; C=len(cfg['classes']['class_names'])
 if q=='none':return None
 if isinstance(q,list):return torch.tensor(q,dtype=torch.float32,device=device)
 counts=np.bincount([v.label for v in videos],minlength=C)
 if (counts==0).any():raise ValueError('inverse_frequency class weights require every configured class represented in training')
 return torch.tensor(len(videos)/(C*counts),dtype=torch.float32,device=device)

def video_logits(model,video,cfg,dev):
 logits=[]; weights=[]
 for ix,duration in make_windows(video,cfg['model']['window_seconds']):
  x=torch.from_numpy(video.features[ix])[None].to(dev); T,R,_=x.shape[1:]; mask=torch.ones((1,T,R),dtype=torch.bool,device=dev); pts=torch.from_numpy(video.pts[ix])[None].to(dev)
  autocast=torch.autocast(device_type=dev.type, enabled=bool(cfg['training']['amp']) and dev.type=='cuda') if dev.type in ('cuda','cpu') else nullcontext()
  with autocast: logits.append(model(x,frame_mask=mask.any(-1),row_mask=mask,pts=pts)[0])
  weights.append(float(duration))
 total=sum(weights)
 return aggregate_logits(torch.stack(logits),[float(w/total) for w in weights])

def _is_oom(exc): return 'out of memory' in str(exc).lower()
def _cpu_snapshot(value):
 if torch.is_tensor(value): return value.detach().cpu().clone()
 if isinstance(value,dict): return {k:_cpu_snapshot(v) for k,v in value.items()}
 if isinstance(value,list): return [_cpu_snapshot(v) for v in value]
 if isinstance(value,tuple): return tuple(_cpu_snapshot(v) for v in value)
 return copy.deepcopy(value)
def optimizer_group_step(model, group, cfg, dev, weight, opt, scaler, logit_fn=video_logits, inject_oom=None):
 """One exact video-mean update; an OOM retry restores all mutable state first."""
 snapshot=(_cpu_snapshot(model.state_dict()),_cpu_snapshot(opt.state_dict()),_cpu_snapshot(scaler.state_dict()),rng())
 def run(attempt, microbatch):
  opt.zero_grad(); total_loss=0.; correct=0; seen=0; outcomes=[]
  for begin in range(0,len(group),microbatch):
   terms=[]
   for g in group[begin:begin+microbatch]:
    z=logit_fn(model,g,cfg,dev); y=torch.tensor([g.label],device=dev); term=F.cross_entropy(z[None],y,reduction='none')[0]; term=term if weight is None else term*weight[g.label]; terms.append(term); total_loss+=float(term.detach()); correct+=int(z.argmax()==g.label); outcomes.append((g.label,int(z.argmax().detach()),torch.softmax(z.detach(),0).cpu().numpy())); seen+=1
   scaler.scale(sum(terms)/len(group)).backward()
   if inject_oom is not None: inject_oom(attempt,seen)
  scaler.step(opt); scaler.update(); opt.zero_grad(); return total_loss,correct,outcomes
 try:
  loss,correct,outcomes=run(0,int(cfg.get('_oom_test_microbatch',len(group)))); return loss,correct,None,outcomes
 except RuntimeError as exc:
  if not _is_oom(exc): raise
  model.load_state_dict(snapshot[0]);opt.load_state_dict(snapshot[1]);scaler.load_state_dict(snapshot[2]);restore(snapshot[3]);opt.zero_grad()
  if dev.type=='cuda': torch.cuda.empty_cache()
  microbatch=max(1,len(group)//2)
  loss,correct,outcomes=run(1,microbatch)
  return loss,correct,{'event':'oom_retry','videos':len(group),'retry_microbatch_size':microbatch},outcomes

def evaluate(model,videos,cfg,dev,weight=None):
 model.eval();ys=[];ps=[];probs=[];losses=[]
 with torch.no_grad():
  for v in videos:
   z=video_logits(model,v,cfg,dev); y=torch.tensor([v.label],device=dev); term=F.cross_entropy(z[None],y,reduction='none')[0]; losses.append(float(term if weight is None else term*weight[v.label])); prob=torch.softmax(z,0).cpu().numpy();ys.append(v.label);ps.append(int(prob.argmax()));probs.append(prob)
 m,_,_=metrics(ys,ps,np.asarray(probs),cfg['classes']['class_names']);m['weighted_ce']=float(np.mean(losses));return m,ys,ps,np.asarray(probs)

def checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping):
 return {'version':2,'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng(),'config':cfg,'label_mapping':label_mapping,'provenance':provenance(metas),'inputs':metas,'normalizer':norm,'best':best,'bad_epochs':bad}

def _load_resume(path,cfg,metas,model,opt,scaler,label_mapping=None):
 ck=torch.load(path,map_location='cpu',weights_only=False)
 saved_mapping=ck.get('label_mapping') or indexed_label_mapping(_checkpoint_config(ck.get('config',{})))
 if label_mapping is not None and saved_mapping != label_mapping: raise ValueError('resume rejected: regenerated training label mapping differs from checkpoint')
 if ck.get('provenance')!=provenance(metas):raise ValueError('resume rejected: supplied CSV or NPZ input content hashes changed')
 if _checkpoint_config(ck.get('config',{}))!=cfg:raise ValueError('resume rejected: configuration changed')
 model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck.get('scaler',{}));restore(ck['rng']);return ck

def _checkpoint_config(cfg):
 """Normalize pre-v2 indexed checkpoints so existing binary runs remain readable."""
 out=copy.deepcopy(cfg)
 if isinstance(out.get('classes'),dict) and 'label_mode' not in out['classes']:
  out['classes']['label_mode']='indexed'
 if isinstance(out.get('classes'),dict) and 'label_mapping' not in out['classes']:
  out['classes']['label_mapping']=indexed_label_mapping(out)
 return out

def train(source_cfg,resume=None):
 label_mapping=training_label_mapping(source_cfg);cfg=materialize_cfg(source_cfg,label_mapping)
 seed_all(cfg['training']['seed']);dev=device_for(cfg['training']['device']); splits,metas=split_load(cfg,label_mapping)
 # In inferred mode this is tautological; retain the legacy indexed guard unchanged.
 missing=set(range(len(cfg['classes']['class_names'])))-{v.label for v in splits['train']}
 if missing: raise ValueError(f'training split is missing configured class IDs {sorted(missing)}')
 norm=fit_normalizer(splits['train']); raw_splits=splits; splits={k:[transform(v,norm) for v in vs] for k,vs in raw_splits.items()}
 run=Path(cfg['run']['output_root'])/cfg['run']['name']; canonical_exists=(run/'test_evaluation').exists()
 if canonical_exists and not resume: raise ValueError('canonical test_evaluation already exists; choose a new run.name instead of overwriting it')
 run.mkdir(parents=True,exist_ok=True)
 if resume and (run/'label_mapping.json').exists():
  try: persisted_mapping=json.loads((run/'label_mapping.json').read_text(encoding='utf8'))
  except (OSError,json.JSONDecodeError) as exc: raise ValueError(f'resume rejected: unreadable run label_mapping.json: {exc}')
  if persisted_mapping != label_mapping: raise ValueError('resume rejected: run label_mapping.json differs from regenerated training mapping')
 if resume and label_mapping['mode']=='raw_inferred' and not (run/'label_mapping.json').exists():
  raise ValueError('resume rejected: raw-label run is missing label_mapping.json')
 model=make_model(cfg,len(cfg['classes']['class_names'])).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate']);scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['training']['amp']) and dev.type=='cuda');weight=class_weights(splits['train'],cfg,dev)
 history=[];start=1;best={'score':-float('inf'),'epoch':None,'weighted_ce':None};bad=0
 if resume:
  ck=_load_resume(resume,cfg,metas,model,opt,scaler,label_mapping); persisted=ck['normalizer']
  if not (np.allclose(norm['mean'],persisted['mean']) and np.allclose(norm['std'],persisted['std']) and norm['count']==persisted['count']): raise ValueError('resume rejected: current train normalization differs from checkpoint')
  norm=persisted; splits={k:[transform(v,norm) for v in vs] for k,vs in raw_splits.items()}; start=ck['epoch']+1;best=ck['best'];bad=ck['bad_epochs']
 # A completed canonical test is immutable. A no-op latest-checkpoint resume is
 # permitted for verification; further epochs require an interrupted pre-test run.
 if canonical_exists:
  if start<=cfg['training']['max_epochs']: raise ValueError('resume would require rerunning canonical test_evaluation; resume an interrupted pre-test run instead')
  return run
 atomic_json(run/'config.json',cfg);atomic_json(run/'label_mapping.json',label_mapping);atomic_json(run/'input_provenance.json',{'splits':metas,'hash':provenance(metas),'created_at':now()});atomic_json(run/'normalizer.json',norm)
 for epoch in range(start,cfg['training']['max_epochs']+1):
  model.train(); order=list(splits['train']);random.shuffle(order); loss_sum=0; correct=0; opt.zero_grad(); pending=0
  group=[]; oom_events=[]; online_y=[]; online_p=[]; online_prob=[]
  effective_group=cfg['training']['batch_size']*cfg['training']['gradient_accumulation']
  for index,v in enumerate(order):
   group.append(v)
   if len(group)<effective_group and index != len(order)-1: continue
   group_loss,group_correct,event,outcomes=optimizer_group_step(model,group,cfg,dev,weight,opt,scaler)
   loss_sum += group_loss; correct += group_correct
   for y,p,prob in outcomes: online_y.append(y); online_p.append(p); online_prob.append(prob)
   if event: oom_events.append(event)
   group=[]
  vm,_,_,_=evaluate(model,splits['val'],cfg,dev,weight); tm,_,_=metrics(online_y,online_p,np.asarray(online_prob),cfg['classes']['class_names'])
  row={'epoch':epoch,'train_loss':loss_sum/len(order),'val_loss':vm['weighted_ce'],'train_accuracy':tm['accuracy'],'val_accuracy':vm['accuracy'],'train_macro_precision':tm['macro_precision'],'val_macro_precision':vm['macro_precision'],'train_weighted_precision':tm['weighted_precision'],'val_weighted_precision':vm['weighted_precision'],'train_macro_recall':tm['macro_recall'],'val_macro_recall':vm['macro_recall'],'train_weighted_recall':tm['weighted_recall'],'val_weighted_recall':vm['weighted_recall'],'train_macro_f1':tm['macro_f1'],'val_macro_f1':vm['macro_f1'],'train_weighted_f1':tm['weighted_f1'],'val_weighted_f1':vm['weighted_f1'],'elapsed_seconds':time.perf_counter()};history.append(row)
  score=vm['macro_f1'] if cfg['training']['monitor']=='macro_f1' else -vm['weighted_ce']; improved=score>best['score'] or (score==best['score'] and (best['weighted_ce'] is None or vm['weighted_ce']<best['weighted_ce']))
  if improved:
   best={'score':score,'epoch':epoch,'weighted_ce':vm['weighted_ce']};bad=0;save_torch(run/'checkpoints'/'best.pt',checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping))
  else:bad+=1
  save_torch(run/'checkpoints'/'latest.pt',checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping));save_curves(run,history)
  if oom_events: atomic_json(run/'oom_retries.json',oom_events)
  if bad>=cfg['training']['patience']:break
 ck=torch.load(run/'checkpoints'/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);m,y,p,prob=evaluate(model,splits['test'],cfg,dev,weight);m['weighted_loss']=m['weighted_ce'];testdir=run/'test_evaluation';testdir.mkdir(exist_ok=True);save_evaluation(testdir,y,p,prob,cfg['classes']['class_names'],'test')
 for artifact in testdir.iterdir():
  if artifact.is_file(): shutil.copy2(artifact,run/'evaluations'/artifact.name)
 summary={'selected_checkpoint':'checkpoints/best.pt','selected_epoch':best['epoch'],'selection':cfg['training']['monitor'],'best_validation':best,'canonical_test_metrics':m,'schema_version':2,'generator':'standalone-wpp-classifier','input_hash':provenance(metas),'label_mapping_sha256':label_mapping['mapping_sha256'],'finished_at':now()};atomic_json(run/'summary.json',summary);atomic_json(run/'summarized_metrics.json',summary);atomic_json(run/'evaluations'/'summarized_metrics.json',summary);return run

def _mapping_from_checkpoint(ck):
 mapping=ck.get('label_mapping')
 if mapping is not None:return mapping
 # Old indexed checkpoints did not serialize this object but retain their class order.
 return indexed_label_mapping(_checkpoint_config(ck['config']))

def eval_checkpoint(source_cfg,checkpoint,output):
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);mapping=_mapping_from_checkpoint(ck)
 if label_mode(source_cfg)!=mapping['mode']: raise ValueError('evaluation rejected: label mode differs from checkpoint')
 if mapping['mode']=='indexed' and source_cfg.get('classes',{}).get('class_names')!=mapping['class_names']: raise ValueError('evaluation rejected: checkpoint class_names differ')
 cfg=materialize_cfg(source_cfg,mapping)
 expected=copy.deepcopy(cfg); saved=_checkpoint_config(ck.get('config',{})); expected.pop('run',None); saved.pop('run',None)
 if expected != saved: raise ValueError('evaluation rejected: checkpoint configuration differs (model/classes/window/training)')
 splits,metas=split_load(cfg,mapping)
 if ck.get('provenance')!=provenance(metas):raise ValueError('evaluation rejected: checkpoint input hashes differ')
 norm=ck['normalizer']; videos=[transform(v,norm) for v in splits['test']];dev=device_for(cfg['training']['device']);model=make_model(cfg,len(cfg['classes']['class_names'])).to(dev);model.load_state_dict(ck['model']);m,y,p,prob=evaluate(model,videos,cfg,dev);out=Path(output)
 if out.name=='test_evaluation':raise ValueError('refusing to overwrite canonical test_evaluation; choose a distinct --output directory')
 m['weighted_loss']=m['weighted_ce'];save_evaluation(out,y,p,prob,cfg['classes']['class_names'],'test');atomic_json(out/'evaluation_metadata.json',{'checkpoint':str(checkpoint),'metrics':m,'input_hash':provenance(metas),'label_mapping':mapping});return out
