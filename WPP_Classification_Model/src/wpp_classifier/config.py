"""Configuration parsing for the standalone WPP classifier."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import yaml

DEFAULT = {"run":{"output_root":"runs", "name":"wpp_run"},
 "model":{"architecture":"transformer", "window_seconds":2, "width":64, "d_model":64, "heads":4, "layers":2, "dropout":0.1, "resnet_blocks":3},
 "training":{"seed":0,"learning_rate":0.001,"batch_size":1,"max_epochs":20,"patience":5,"device":"auto","amp":False,"gradient_accumulation":1,"class_weights":"none","monitor":"macro_f1"}}
REQUIRED_PATHS=("train_data_path","train_label_path","val_data_path","val_label_path","test_data_path","test_label_path")

def exact_seconds(value):
    if isinstance(value, bool) or isinstance(value, float): raise ValueError("model.window_seconds must be a YAML integer or quoted exact decimal/fraction string (not float/bool)")
    if isinstance(value, int): f=Fraction(value,1)
    elif isinstance(value,str):
        try: f=Fraction(value)
        except (ValueError,ZeroDivisionError): raise ValueError("model.window_seconds must be a positive exact decimal or fraction")
    else: raise ValueError("model.window_seconds must be a YAML integer or quoted exact decimal/fraction string")
    if f <= 0: raise ValueError("model.window_seconds must be positive")
    return f

def load_config(path):
    path=Path(path).expanduser().resolve()
    with path.open(encoding='utf8') as f: raw=yaml.safe_load(f)
    if not isinstance(raw,dict): raise ValueError("configuration root must be a mapping")
    unknown=set(raw)-set(DEFAULT)-{'data','classes'}
    if unknown: raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    if not isinstance(raw.get('data'),dict) or not isinstance(raw.get('classes'),dict): raise ValueError("data and classes mappings are required")
    cfg=deepcopy(DEFAULT)
    for section in ('run','model','training'):
        supplied=raw.get(section,{})
        if not isinstance(supplied,dict): raise ValueError(f"{section} must be a mapping")
        bad=set(supplied)-set(cfg[section])
        if bad: raise ValueError(f"unknown keys in {section}: {sorted(bad)}")
        cfg[section].update(supplied)
    data=raw['data']; missing=[x for x in REQUIRED_PATHS if not data.get(x)]
    if missing: raise ValueError(f"missing required data paths: {missing}")
    cfg['data']={k:str(resolve_path(path.parent,v)) for k,v in data.items() if k in REQUIRED_PATHS}
    names=raw['classes'].get('class_names')
    if not isinstance(names,list) or len(names)<1 or any(not isinstance(x,str) or not x for x in names) or len(set(names))!=len(names): raise ValueError("classes.class_names must be a unique nonempty list")
    cfg['classes']={'class_names':names}
    m,t=cfg['model'],cfg['training']; exact_seconds(m['window_seconds'])
    if m['architecture'] not in ('resnet','transformer'): raise ValueError("model.architecture must be resnet or transformer")
    for k in ('width','d_model','heads','layers','resnet_blocks'):
        if not isinstance(m[k],int) or isinstance(m[k],bool) or m[k]<=0: raise ValueError(f"model.{k} must be a positive integer")
    if m['d_model']%m['heads']: raise ValueError("model.d_model must be divisible by model.heads")
    if not isinstance(m['dropout'],(int,float)) or not 0<=m['dropout']<1: raise ValueError("model.dropout must be in [0,1)")
    for k in ('seed','batch_size','max_epochs','patience','gradient_accumulation'):
        if not isinstance(t[k],int) or isinstance(t[k],bool) or t[k]< (0 if k=='seed' else 1): raise ValueError(f"training.{k} is invalid")
    if not isinstance(t['learning_rate'],(int,float)) or t['learning_rate']<=0: raise ValueError("training.learning_rate must be positive")
    if t['class_weights'] not in ('none','inverse_frequency') and not (isinstance(t['class_weights'],list) and len(t['class_weights'])==len(names) and all(isinstance(x,(int,float)) and x>0 for x in t['class_weights'])): raise ValueError("training.class_weights must be none, inverse_frequency, or a positive vector of C values")
    if not isinstance(t['amp'],bool): raise ValueError("training.amp must be a boolean")
    if t['monitor'] not in ('macro_f1','weighted_ce'): raise ValueError("training.monitor must be macro_f1 or weighted_ce")
    cfg['_config_path']=str(path); cfg['run']['output_root']=str(resolve_path(path.parent,cfg['run']['output_root']))
    return cfg

def resolve_path(base,value):
    p=Path(str(value)).expanduser()
    return p if p.is_absolute() else (base/p).resolve()
