from __future__ import annotations
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score

def atomic_json(path,obj):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=_json)+'\n'); tmp.replace(p)
def _json(x): return x.item() if isinstance(x,np.generic) else str(x)
def csv_write(path,rows):
 rows=list(rows); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p,prob,names):
 labels=list(range(len(names))); pr,re,f1,s=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
 macro=precision_recall_fscore_support(y,p,labels=labels,average='macro',zero_division=0); weighted=precision_recall_fscore_support(y,p,labels=labels,average='weighted',zero_division=0)
 out={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(macro[0]),'weighted_precision':float(weighted[0]),'macro_recall':float(macro[1]),'weighted_recall':float(weighted[1]),'macro_f1':float(macro[2]),'weighted_f1':float(weighted[2]),'weighted_ce':None}
 per=[]
 for i,n in enumerate(names):
  diag={'roc_auc':None,'average_precision':None,'diagnostic_reason':None}
  truth=np.asarray(y)==i
  if truth.any() and (~truth).any():
   diag['roc_auc']=float(roc_auc_score(truth,prob[:,i])); diag['average_precision']=float(average_precision_score(truth,prob[:,i]))
  else: diag['diagnostic_reason']='class has no positives or no negatives in split'
  per.append({'class_id':i,'class_name':n,'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(s[i]),**diag})
 return out,per,confusion_matrix(y,p,labels=labels)
def save_evaluation(directory,y,p,prob,names,prefix='test'):
 d=Path(directory); m,per,cm=metrics(y,p,prob,names); m['weighted_loss']=m['weighted_ce']; atomic_json(d/f'{prefix}_metrics.json',{**m,'per_class':per}); csv_write(d/f'{prefix}_per_class_metrics.csv',per); atomic_json(d/f'{prefix}_per_class_metrics.json',per)
 np.savetxt(d/f'{prefix}_confusion_matrix_raw.csv',cm,fmt='%d',delimiter=','); norm=cm/cm.sum(1,keepdims=True).clip(1); np.savetxt(d/f'{prefix}_confusion_matrix_normalized.csv',norm,fmt='%.8g',delimiter=',')
 for array,name in ((cm,'raw'),(norm,'normalized')):
  fig,ax=plt.subplots(); im=ax.imshow(array,cmap='Blues');fig.colorbar(im,ax=ax);ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_yticks(range(len(names)),names);ax.set_xlabel('predicted');ax.set_ylabel('true');fig.tight_layout();fig.savefig(d/f'{prefix}_confusion_matrix_{name}.png');plt.close(fig)
 return m

def save_curves(directory,history):
 d=Path(directory); csv_write(d/'per_epoch_metrics.csv',history); atomic_json(d/'per_epoch_metrics.json',history)
 for keys,name in ((('train_loss','val_loss'),'loss'),(('train_accuracy','val_accuracy'),'accuracy'),(('train_macro_precision','val_macro_precision'),'precision'),(('train_macro_recall','val_macro_recall'),'recall'),(('train_macro_f1','val_macro_f1'),'f1')):
  fig,ax=plt.subplots(); x=[r['epoch'] for r in history]
  for k in keys:ax.plot(x,[r.get(k) for r in history],label=k)
  ax.legend();ax.set_xlabel('epoch');fig.tight_layout();fig.savefig(d/f'{name}_curve.png');plt.close(fig)
 # Legacy-style aliases retain the canonical root files while exposing an evaluations layout.
 e=d/'evaluations'; e.mkdir(exist_ok=True); csv_write(e/'per_epoch_metrics.csv',history); atomic_json(e/'per_epoch_metrics.json',history)
 for name in ('loss','accuracy','precision','recall','f1'): shutil.copy2(d/f'{name}_curve.png',e/f'{name}_curve.png')
