from __future__ import annotations
import copy, json, random, time
from contextlib import nullcontext
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from .data import load_split,fit_normalizer,transform,make_windows,provenance,infer_label_mapping,sha256,mapping_sha256
from .models import make_model,aggregate_logits
from .reporting import atomic_json,save_evaluation,save_curves,metrics,best_observed_history
from .config import validate_run_name

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

def checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping,history):
 return {'version':3,'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rng(),'config':cfg,'label_mapping':label_mapping,'provenance':provenance(metas),'inputs':metas,'normalizer':norm,'best':best,'bad_epochs':bad,'history':copy.deepcopy(history)}

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

def _history_rows(value):
 """Accept only contiguous current-schema epoch rows, never silently truncate."""
 if not isinstance(value,list): return []
 rows=[dict(row) for row in value if isinstance(row,dict)]
 if len(rows)!=len(value) or any(not isinstance(row.get('epoch'),int) or isinstance(row['epoch'],bool) for row in rows): return []
 epochs=[row['epoch'] for row in rows]
 return rows if epochs==list(range(1,len(rows)+1)) else []
def _load_history(run,checkpoint):
 disk=run/'evaluations'/'per_epoch_metrics.json'
 try: disk_rows=_history_rows(json.loads(disk.read_text(encoding='utf8')))
 except (OSError,json.JSONDecodeError): disk_rows=[]
 checkpoint_rows=_history_rows(checkpoint.get('history'))
 completed=checkpoint['epoch']
 candidates=[rows for rows in (disk_rows,checkpoint_rows) if len(rows)>=completed and rows[completed-1]['epoch']==completed]
 if not candidates:
  raise ValueError('resume rejected: current-schema per-epoch history is missing or does not reach the checkpoint epoch')
 history=candidates[0][:completed]
 for other in candidates[1:]:
  if other[:completed] != history: raise ValueError('resume rejected: checkpoint and evaluation history differ')
 return history
def _status(run):
 try: value=json.loads((run/'status.json').read_text(encoding='utf8'))
 except (OSError,json.JSONDecodeError): return {}
 return value if isinstance(value,dict) else {}
def _canonical_evaluation_exists(run):
 try: summary=json.loads((run/'evaluations'/'summarized_metrics.json').read_text(encoding='utf8'))
 except (OSError,json.JSONDecodeError): return False
 return isinstance(summary,dict) and summary.get('evaluation_kind')=='original_test'

def _resume_run_name(resume):
 """Read the concrete run name from a current checkpoint before comparing configs."""
 ck=torch.load(resume,map_location='cpu',weights_only=False)
 if ck.get('version') != 3: raise ValueError('resume rejected: only current-schema checkpoints are supported')
 saved=ck.get('config')
 name=saved.get('run',{}).get('name') if isinstance(saved,dict) else None
 try: validate_run_name(name)
 except ValueError as exc: raise ValueError('resume rejected: checkpoint has no valid persisted run.name') from exc
 if name is None: raise ValueError('resume rejected: checkpoint has no persisted concrete run.name')
 return name

def materialize_run_config(cfg,resume=None,run_name=None):
 """Copy-independent concrete output location/name used by one train invocation."""
 configured=cfg.get('run',{}).get('name')
 validate_run_name(configured)
 if run_name is not None: validate_run_name(run_name)
 requested=run_name if run_name is not None else configured
 if resume:
  saved_name=_resume_run_name(resume)
  if requested is not None and requested != saved_name:
   source='--run-name' if run_name is not None else 'config'
   raise ValueError(f'resume rejected: {source} run name {requested!r} differs from checkpoint run name {saved_name!r}')
  name=saved_name
 else:
  name=requested if requested is not None else datetime.now().strftime('run_%Y%m%d_%H%M%S')
 validate_run_name(name)
 cfg['run']['output_root']=str(Path(str(cfg['run']['output_root'])).expanduser().resolve())
 cfg['run']['name']=name
 return cfg

def train(source_cfg,resume=None,run_name=None):
 label_mapping=training_label_mapping(source_cfg);cfg=materialize_cfg(source_cfg,label_mapping)
 cfg=materialize_run_config(cfg,resume,run_name)
 run=Path(cfg['run']['output_root'])/cfg['run']['name']
 if resume:
  if not run.is_dir(): raise ValueError('resume rejected: run directory does not exist')
 else:
  try: run.mkdir(parents=True,exist_ok=False)
  except FileExistsError as exc: raise ValueError('run directory already exists; choose a new run.name or resume its latest checkpoint') from exc
 status=_status(run)
 seed_all(cfg['training']['seed']);dev=device_for(cfg['training']['device']);splits,metas=split_load(cfg,label_mapping)
 missing=set(range(len(cfg['classes']['class_names'])))-{v.label for v in splits['train']}
 if missing: raise ValueError(f'training split is missing configured class IDs {sorted(missing)}')
 norm=fit_normalizer(splits['train']);raw_splits=splits;splits={k:[transform(v,norm) for v in vs] for k,vs in raw_splits.items()}
 if resume and (run/'label_mapping.json').exists():
  try: persisted_mapping=json.loads((run/'label_mapping.json').read_text(encoding='utf8'))
  except (OSError,json.JSONDecodeError) as exc: raise ValueError(f'resume rejected: unreadable run label_mapping.json: {exc}')
  if persisted_mapping != label_mapping: raise ValueError('resume rejected: run label_mapping.json differs from regenerated training mapping')
 if resume and label_mapping['mode']=='raw_inferred' and not (run/'label_mapping.json').exists(): raise ValueError('resume rejected: raw-label run is missing label_mapping.json')
 model=make_model(cfg,len(cfg['classes']['class_names'])).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate']);scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['training']['amp']) and dev.type=='cuda');weight=class_weights(splits['train'],cfg,dev)
 history=[];start=1;best={'score':-float('inf'),'epoch':None,'weighted_ce':None};bad=0
 if resume:
  ck=_load_resume(resume,cfg,metas,model,opt,scaler,label_mapping);persisted=ck['normalizer']
  if not (np.allclose(norm['mean'],persisted['mean']) and np.allclose(norm['std'],persisted['std']) and norm['count']==persisted['count']): raise ValueError('resume rejected: current train normalization differs from checkpoint')
  norm=persisted;splits={k:[transform(v,norm) for v in vs] for k,vs in raw_splits.items()};history=_load_history(run,ck);start=ck['epoch']+1;best=ck['best'];bad=ck['bad_epochs']
 if status.get('state') in ('completed','early_stopped') or _canonical_evaluation_exists(run):
  if start<=cfg['training']['max_epochs']: raise ValueError('resume would rerun the immutable canonical test evaluation')
  return run
 started_at=status.get('started_at',now())
 atomic_json(run/'config.json',cfg);atomic_json(run/'label_mapping.json',label_mapping);atomic_json(run/'input_provenance.json',{'splits':metas,'hash':provenance(metas),'created_at':now()});atomic_json(run/'normalizer.json',norm);atomic_json(run/'status.json',{'schema_version':1,'state':'running','started_at':started_at})
 stop_reason='early_stopping_patience_reached' if bad>=cfg['training']['patience'] else 'max_epochs_reached'
 for epoch in (range(start,cfg['training']['max_epochs']+1) if bad<cfg['training']['patience'] else ()):
  epoch_started=time.perf_counter();model.train();order=list(splits['train']);random.shuffle(order);loss_sum=0
  group=[];oom_events=[];online_y=[];online_p=[];online_prob=[];effective_group=cfg['training']['batch_size']*cfg['training']['gradient_accumulation']
  for index,v in enumerate(order):
   group.append(v)
   if len(group)<effective_group and index != len(order)-1: continue
   group_loss,_,event,outcomes=optimizer_group_step(model,group,cfg,dev,weight,opt,scaler);loss_sum += group_loss
   for y,p,prob in outcomes: online_y.append(y);online_p.append(p);online_prob.append(prob)
   if event: oom_events.append(event)
   group=[]
  vm,_,_,_=evaluate(model,splits['val'],cfg,dev,weight);tm,_,_=metrics(online_y,online_p,np.asarray(online_prob),cfg['classes']['class_names'])
  row={'epoch':epoch,'train_loss':loss_sum/len(order),'validation_loss':vm['weighted_ce'],'train_accuracy':tm['accuracy'],'validation_accuracy':vm['accuracy'],'train_macro_precision':tm['macro_precision'],'validation_macro_precision':vm['macro_precision'],'train_weighted_precision':tm['weighted_precision'],'validation_weighted_precision':vm['weighted_precision'],'train_macro_recall':tm['macro_recall'],'validation_macro_recall':vm['macro_recall'],'train_weighted_recall':tm['weighted_recall'],'validation_weighted_recall':vm['weighted_recall'],'train_macro_f1':tm['macro_f1'],'validation_macro_f1':vm['macro_f1'],'train_weighted_f1':tm['weighted_f1'],'validation_weighted_f1':vm['weighted_f1'],'learning_rate':opt.param_groups[0]['lr'],'duration_seconds':time.perf_counter()-epoch_started};history.append(row)
  score=vm['macro_f1'] if cfg['training']['monitor']=='macro_f1' else -vm['weighted_ce'];improved=score>best['score'] or (score==best['score'] and (best['weighted_ce'] is None or vm['weighted_ce']<best['weighted_ce']))
  if improved: best={'score':score,'epoch':epoch,'weighted_ce':vm['weighted_ce']};bad=0
  else: bad+=1
  if improved: save_torch(run/'checkpoints'/'best.pt',checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping,history))
  save_torch(run/'checkpoints'/'latest.pt',checkpoint_payload(model,opt,scaler,epoch,cfg,metas,norm,best,bad,label_mapping,history));save_curves(run/'evaluations',history)
  if oom_events: atomic_json(run/'oom_retries.json',oom_events)
  if bad>=cfg['training']['patience']: stop_reason='early_stopping_patience_reached';break
 ck=torch.load(run/'checkpoints'/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);evaluation_started=time.perf_counter();evaluated,y,p,prob=evaluate(model,splits['test'],cfg,dev,weight);test_duration=time.perf_counter()-evaluation_started;testdir=run/'evaluations';m=save_evaluation(testdir,y,p,prob,cfg['classes']['class_names'],'test');m.pop('weighted_ce');m.update({'test_loss':evaluated['weighted_ce'],'parameter_count':sum(parameter.numel() for parameter in model.parameters()),'evaluation_duration_seconds':test_duration,'samples_per_second':len(y)/test_duration if test_duration else None})
 selected=next(row for row in history if row['epoch']==best['epoch']);selection={'monitor':cfg['training']['monitor'],'selected_epoch':best['epoch'],'selected_score':best['score'],'selected_validation_weighted_ce':best['weighted_ce'],'train_metrics':{key[6:]:value for key,value in selected.items() if key.startswith('train_')},'validation_metrics':{key[11:]:value for key,value in selected.items() if key.startswith('validation_')}}
 terminal_state='early_stopped' if stop_reason=='early_stopping_patience_reached' else 'completed'
 summary={'schema_version':3,'generator':'standalone-wpp-classifier','created_at':started_at,'finished_at':now(),'checkpoint':'checkpoints/best.pt','selected_checkpoint':'checkpoints/best.pt','selected_epoch':best['epoch'],'checkpoint_selection':selection,'best_observed_per_epoch_metrics':best_observed_history(history),'metric_semantics':{'aggregate_class_set':'ground-truth-present classes','zero_division':0,'train_epoch_metrics':'online predictions during model.train() optimizer updates','validation_epoch_metrics':'post-epoch model.eval() inference','best_observed_per_epoch_metrics':'independent per-metric maxima; not a jointly observed checkpoint'},**m,'evaluation_kind':'original_test','config':{'path':'../config.json'},'input_provenance':{'path':'../input_provenance.json','hash':provenance(metas),'test':metas['test']},'label_mapping':{'path':'../label_mapping.json','sha256':label_mapping['mapping_sha256']},'normalizer':{'path':'../normalizer.json','fit_split':'train','count':norm['count']},'checkpoints':{'best':'../checkpoints/best.pt','latest':'../checkpoints/latest.pt'},'status':{'path':'../status.json','state':terminal_state}}
 atomic_json(testdir/'summarized_metrics.json',summary);atomic_json(run/'status.json',{'schema_version':1,'state':terminal_state,'started_at':started_at,'completed_at':now(),'completed_epochs':len(history),'best_epoch':best['epoch'],'selection':cfg['training']['monitor'],'best_validation_score':best['score'],'best_validation_weighted_ce':best['weighted_ce'],'stop_reason':stop_reason});return run

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
 norm=ck['normalizer']; videos=[transform(v,norm) for v in splits['test']];dev=device_for(cfg['training']['device']);weight=class_weights(splits['train'],cfg,dev);model=make_model(cfg,len(cfg['classes']['class_names'])).to(dev);model.load_state_dict(ck['model']);evaluated,y,p,prob=evaluate(model,videos,cfg,dev,weight);out=Path(output)
 if out.resolve()==(Path(checkpoint).resolve().parent.parent/'evaluations').resolve():raise ValueError('refusing to overwrite canonical evaluations directory; choose a distinct --output directory')
 save_evaluation(out,y,p,prob,cfg['classes']['class_names'],'test');evaluated['weighted_loss']=evaluated['weighted_ce'];atomic_json(out/'evaluation_metadata.json',{'checkpoint':str(checkpoint),'metrics':evaluated,'input_hash':provenance(metas),'label_mapping':mapping});return out
