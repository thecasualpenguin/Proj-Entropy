"""Strict contract readers, native windowing, and train-only normalization."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import numpy as np
from .config import exact_seconds

@dataclass
class Video:
    video_id:str; label:int; features:np.ndarray; pts:np.ndarray; duration:np.ndarray; time_base:tuple; path:str

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def _csv(path, expected):
 path=Path(path)
 with path.open(newline='',encoding='utf8') as f:
  r=csv.DictReader(f)
  if r.fieldnames!=expected: raise ValueError(f"{path}: headers must be exactly {expected}, in that order")
  rows=list(r)
 if not rows: raise ValueError(f"{path}: CSV has no rows")
 ids=[]
 for row in rows:
  # IDs deliberately remain exact CSV strings: they are never normalized or case-folded.
  if set(row)!=set(expected) or not row['video_id'].strip(): raise ValueError(f"{path}: empty video_id or malformed row")
  ids.append(row['video_id'])
 if len(set(ids))!=len(ids): raise ValueError(f"{path}: duplicate video_id")
 return rows

def canonical_label(value, path=None, video_id=None):
 """Legacy-compatible label normalization for inferred text vocabularies."""
 value=str(value).strip().casefold()
 if not value:
  where=f" for {video_id!r}" if video_id is not None else ''
  raise ValueError(f"{path}: label{where} must be nonempty text")
 return value

def mapping_sha256(mapping):
 return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def infer_label_mapping(label_path):
 """Build the deterministic train-only raw-label vocabulary."""
 rows=_csv(label_path,['video_id','label'])
 names=sorted({canonical_label(r['label'],label_path,r['video_id']) for r in rows})
 mapping={'version':1,'mode':'raw_inferred','normalizer':'str.strip.casefold',
          'class_names':names,'mapping':{name:index for index,name in enumerate(names)},
          'train_label_sha256':sha256(label_path)}
 mapping['mapping_sha256']=mapping_sha256(mapping)
 return mapping

def _indexed_label(value,path,video_id,class_count):
 try:
  # reject e.g. 1.0 / whitespace variants, preserving the legacy indexed mode.
  if str(int(value)) != value.strip(): raise ValueError
  x=int(value)
 except ValueError: raise ValueError(f"{path}: label for {video_id!r} is not an exact integer")
 if not 0<=x<class_count: raise ValueError(f"{path}: label {x} outside 0..{class_count-1}")
 return x

def load_split(data_path,label_path,label_mapping):
 """Load a supplied split using either legacy indexed or inferred raw labels.

 ``label_mapping`` is an integer class count for the legacy public helper API, or
 a mapping descriptor produced by :func:`infer_label_mapping`.
 """
 data_path,label_path=Path(data_path),Path(label_path)
 drows=_csv(data_path,['video_id','features_path']); lrows=_csv(label_path,['video_id','label'])
 labels={}
 if isinstance(label_mapping,int):
  mode,class_count='indexed',label_mapping
  mapping=None
 else:
  mode=label_mapping['mode']; mapping=label_mapping['mapping']; class_count=len(label_mapping['class_names'])
 for r in lrows:
  if mode=='indexed': x=_indexed_label(r['label'],label_path,r['video_id'],class_count)
  else:
   raw=canonical_label(r['label'],label_path,r['video_id'])
   if raw not in mapping:
    raise ValueError(f"{label_path}: label {raw!r} for video_id {r['video_id']!r} is not in the training vocabulary")
   x=mapping[raw]
  labels[r['video_id']]=x
 dids={r['video_id'] for r in drows}
 if dids!=set(labels): raise ValueError(f"data/label ID mismatch: only data={sorted(dids-set(labels))[:3]}, only labels={sorted(set(labels)-dids)[:3]}")
 videos=[]
 for r in drows:
  fp=Path(r['features_path']).expanduser(); fp=fp if fp.is_absolute() else data_path.parent/fp
  videos.append(_npz(fp.resolve(),r['video_id'],labels[r['video_id']]))
 return videos, {'data_path':str(data_path.resolve()),'label_path':str(label_path.resolve()),'data_sha256':sha256(data_path),'label_sha256':sha256(label_path),'video_ids':[v.video_id for v in videos], 'feature_sha256':{v.video_id:sha256(v.path) for v in videos}}

def _npz(path,vid,label):
 if path.suffix.lower()!='.npz': raise ValueError(f"{vid}: features_path must name an NPZ")
 if not path.is_file(): raise ValueError(f"{vid}: NPZ not found: {path}")
 try:
  with np.load(path,allow_pickle=False) as z:
   need={'row_share','delta_row_share','pts','frame_duration','time_base'}
   miss=need-set(z.files)
   if miss: raise ValueError(f"{vid}: missing NPZ arrays {sorted(miss)}")
   a,b,pts,dur,tb=(z[x].copy() for x in ('row_share','delta_row_share','pts','frame_duration','time_base'))
 except Exception as e:
  if isinstance(e,ValueError): raise
  raise ValueError(f"{vid}: unreadable NPZ: {e}")
 if a.dtype not in (np.dtype('float32'),np.dtype('float64')) or b.dtype not in (np.dtype('float32'),np.dtype('float64')): raise ValueError(f"{vid}: feature arrays must be float32 or float64")
 if pts.dtype!=np.int64 or dur.dtype!=np.int64 or tb.dtype!=np.int64: raise ValueError(f"{vid}: pts, frame_duration, time_base must be int64")
 if a.ndim!=2 or a.shape!=b.shape or not all(x>0 for x in a.shape): raise ValueError(f"{vid}: features must share nonempty [T,R] shape")
 t,r=a.shape
 if pts.shape!=(t,) or dur.shape!=(t,) or tb.shape!=(2,): raise ValueError(f"{vid}: timing shapes must be pts[T], frame_duration[T], time_base[2]")
 if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError(f"{vid}: features must be finite")
 # Avoid int64 subtraction overflow for e.g. [-INT64_MAX, INT64_MAX].
 if any(int(pts[i+1]) <= int(pts[i]) for i in range(t-1)) or not np.all(dur>0) or not np.all(tb>0): raise ValueError(f"{vid}: pts must strictly increase; duration and time_base must be positive")
 return Video(vid,label,np.stack([b,a],-1).astype(np.float32),pts,dur,(int(tb[0]),int(tb[1])),str(path))

def make_windows(video, seconds):
 seconds=exact_seconds(seconds); tick=Fraction(video.time_base[0],video.time_base[1]); width=seconds/tick
 starts=[]; grouped={}
 t0=int(video.pts[0])
 for i,p in enumerate(video.pts):
  bucket=(Fraction(int(p)-t0,1)//width)
  grouped.setdefault(int(bucket),[]).append(i)
 for bucket, ix in grouped.items():
  duration=sum((Fraction(int(video.duration[i])*video.time_base[0],video.time_base[1]) for i in ix),Fraction())
  starts.append((np.asarray(ix),duration))
 return starts

def fit_normalizer(videos):
 n=np.zeros(2,dtype=np.int64); total=np.zeros(2); squared=np.zeros(2)
 for v in videos:
  x=v.features.reshape(-1,2).astype(np.float64)
  n += len(x); total += x.sum(0); squared += (x*x).sum(0)
 mean=total/n; std=np.sqrt(np.maximum(squared/n-mean*mean,0)); std[std==0]=1.
 return {'mean':mean.tolist(),'std':std.tolist(),'count':n.tolist()}

def transform(video,norm):
 x=(video.features-np.asarray(norm['mean'],dtype=np.float32))/np.asarray(norm['std'],dtype=np.float32)
 return Video(video.video_id,video.label,x,video.pts,video.duration,video.time_base,video.path)

def provenance(metas): return hashlib.sha256(json.dumps(metas,sort_keys=True).encode()).hexdigest()
