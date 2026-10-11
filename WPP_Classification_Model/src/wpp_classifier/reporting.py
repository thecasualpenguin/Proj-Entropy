from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score

CLASSIFICATION_METRICS=('accuracy','macro_precision','weighted_precision','macro_recall','weighted_recall','macro_f1','weighted_f1')

def atomic_json(path,obj):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=_json)+'\n'); tmp.replace(p)
def _json(x): return x.item() if isinstance(x,np.generic) else str(x)
def csv_write(path,rows):
 rows=list(rows); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(key for row in rows for key in row))
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def _coverage(y,names):
 counts=Counter(int(value) for value in y)
 present=[index for index in range(len(names)) if counts[index]]
 return present,{'configured_classes':list(names),'present_classes':[names[index] for index in present],
                 'absent_classes':[names[index] for index in range(len(names)) if not counts[index]],
                 'class_distribution':{names[index]:counts[index] for index in present},'unexpected_classes':[]}
def metrics(y,p,prob,names):
 labels=list(range(len(names))); present,coverage=_coverage(y,names)
 if not present: raise ValueError('classification metrics require at least one ground-truth label')
 pr,re,f1,s=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
 macro=precision_recall_fscore_support(y,p,labels=present,average='macro',zero_division=0); weighted=precision_recall_fscore_support(y,p,labels=present,average='weighted',zero_division=0)
 predicted=Counter(int(value) for value in p)
 out={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(macro[0]),'weighted_precision':float(weighted[0]),'macro_recall':float(macro[1]),'weighted_recall':float(weighted[1]),'macro_f1':float(macro[2]),'weighted_f1':float(weighted[2]),'weighted_ce':None,'sample_count':len(y),'class_coverage':coverage,'zero_division_handling':{'value_used':0,'classes_with_no_predictions':[names[index] for index in labels if not predicted[index]],'occurred':any(not predicted[index] for index in labels)}}
 per=[]
 for i,n in enumerate(names):
  diag={'roc_auc':None,'average_precision':None,'diagnostic_reason':None}
  truth=np.asarray(y)==i
  if truth.any() and (~truth).any():
   diag['roc_auc']=float(roc_auc_score(truth,prob[:,i])); diag['average_precision']=float(average_precision_score(truth,prob[:,i]))
  else: diag['diagnostic_reason']='class has no positives or no negatives in split'
  per.append({'class':n,'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'sample_count':int(s[i]),**diag})
 return out,per,confusion_matrix(y,p,labels=labels)
def _save_matrix(path,matrix,names):
 with Path(path).open('w',newline='',encoding='utf8') as f:
  writer=csv.writer(f);writer.writerow(['true\\pred',*names]);writer.writerows(([name,*row] for name,row in zip(names,matrix)))
def save_evaluation(directory,y,p,prob,names,prefix='test'):
 d=Path(directory);d.mkdir(parents=True,exist_ok=True); m,per,cm=metrics(y,p,prob,names)
 atomic_json(d/f'{prefix}_per_class_metrics.json',per); csv_write(d/f'{prefix}_per_class_metrics.csv',per)
 _save_matrix(d/f'{prefix}_confusion_matrix_raw.csv',cm,names); norm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)!=0); _save_matrix(d/f'{prefix}_confusion_matrix_normalized.csv',norm,names)
 for array,name in ((cm,'raw'),(norm,'normalized')):
  fig,ax=plt.subplots(figsize=(max(7,len(names)*.65),max(6,len(names)*.6))); im=ax.imshow(array,cmap='Blues');fig.colorbar(im,ax=ax);ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_yticks(range(len(names)),names);ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(d/f'{prefix}_confusion_matrix_{name}.png',dpi=150);plt.close(fig)
 return m
def best_observed_history(history):
 """Independently maximize classification metrics, retaining the earliest tie."""
 result={'train':{},'validation':{}}
 for split in result:
  for metric in CLASSIFICATION_METRICS:
   key=f'{split}_{metric}'
   for row in history:
    value,epoch=row.get(key),row.get('epoch')
    if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and isinstance(epoch,int) and not isinstance(epoch,bool):
     current=result[split].get(metric)
     if current is None or value>current['value'] or (value==current['value'] and epoch<current['epoch']): result[split][metric]={'value':float(value),'epoch':epoch}
 return result
def save_curves(directory,history):
 """Write the current history and Video-pipeline-compatible curve artifacts."""
 d=Path(directory);d.mkdir(parents=True,exist_ok=True); csv_write(d/'per_epoch_metrics.csv',history); atomic_json(d/'per_epoch_metrics.json',history)
 specs=(('loss_curves',('train_loss','validation_loss'),'loss'),('accuracy_curves',('train_accuracy','validation_accuracy'),'accuracy'),('precision_curves',('train_macro_precision','train_weighted_precision','validation_macro_precision','validation_weighted_precision'),'precision'),('recall_curves',('train_macro_recall','train_weighted_recall','validation_macro_recall','validation_weighted_recall'),'recall'),('f1_curves',('train_macro_f1','train_weighted_f1','validation_macro_f1','validation_weighted_f1'),'F1'))
 for stem,keys,ylabel in specs:
  if not all(any(key in row for row in history) for key in keys): continue
  fig,ax=plt.subplots()
  for key in keys:
   points=[(row['epoch'],row[key]) for row in history if key in row]
   ax.plot(*zip(*points),label=key)
  ax.set_xlabel('epoch');ax.set_ylabel(ylabel);ax.legend();fig.tight_layout();fig.savefig(d/f'{stem}.png',dpi=140);plt.close(fig)
