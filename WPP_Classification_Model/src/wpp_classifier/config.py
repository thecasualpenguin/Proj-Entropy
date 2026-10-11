"""Configuration parsing for the standalone WPP classifier."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction

INPUT_CONTRACT={'input_contract_version':1,'display_width':512,'display_height':512,'fps_num':15,'fps_den':1}
from pathlib import Path
import yaml

DEFAULT = {"run":{"output_root":"runs", "name":None},
 "model":{"architecture":"transformer", "window_seconds":2, "width":64, "d_model":64, "heads":4, "layers":2, "dropout":0.1, "resnet_blocks":3, "convtran_layers":2, "convtran_heads":4, "convtran_ff_multiplier":2, "convtran_kernel_size":3},
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
    if not isinstance(raw.get('data'),dict): raise ValueError("data mapping is required")
    if 'classes' in raw and not isinstance(raw['classes'],dict): raise ValueError("classes must be a mapping when supplied")
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
    supplied_classes=raw.get('classes',{})
    bad=set(supplied_classes)-{'class_names'}
    if bad: raise ValueError(f"unknown keys in classes: {sorted(bad)}")
    names=supplied_classes.get('class_names')
    # A missing, null, or empty list opts into raw-label vocabulary inference.
    if names in (None,[]):
        cfg['classes']={'label_mode':'raw_inferred','class_names':None}
    else:
        if not isinstance(names,list) or any(not isinstance(x,str) or not x for x in names) or len(set(names))!=len(names): raise ValueError("classes.class_names must be a unique nonempty list, or be omitted for raw-label inference")
        cfg['classes']={'label_mode':'indexed','class_names':list(names)}
    m,t=cfg['model'],cfg['training']; exact_seconds(m['window_seconds'])
    if m['architecture'] not in ('resnet','transformer','convtran'): raise ValueError("model.architecture must be resnet, transformer, or convtran")
    for k in ('width','d_model','heads','layers','resnet_blocks'):
        if not isinstance(m[k],int) or isinstance(m[k],bool) or m[k]<=0: raise ValueError(f"model.{k} must be a positive integer")
    if m['d_model']%m['heads']: raise ValueError("model.d_model must be divisible by model.heads")
    if m['architecture']=='convtran':
        for k in ('convtran_layers','convtran_heads','convtran_ff_multiplier'):
            if not isinstance(m[k],int) or isinstance(m[k],bool) or m[k]<=0: raise ValueError(f"model.{k} must be a positive integer")
        if m['d_model']%m['convtran_heads']: raise ValueError("model.d_model must be divisible by model.convtran_heads")
        if not isinstance(m['convtran_kernel_size'],int) or isinstance(m['convtran_kernel_size'],bool) or m['convtran_kernel_size'] <= 0 or not m['convtran_kernel_size'] % 2: raise ValueError("model.convtran_kernel_size must be a positive odd integer")
        # ConvTran tables are deliberately finite: derive their bound exactly from the
        # fixed 15 fps media contract rather than accepting a truncating frame cap.
        seconds=exact_seconds(m['window_seconds']); m['convtran_max_frames']=(seconds*15).__ceil__()
        if m['convtran_max_frames'] <= 0: raise ValueError("model.window_seconds gives no ConvTran frames")
    if not isinstance(m['dropout'],(int,float)) or not 0<=m['dropout']<1: raise ValueError("model.dropout must be in [0,1)")
    for k in ('seed','batch_size','max_epochs','patience','gradient_accumulation'):
        if not isinstance(t[k],int) or isinstance(t[k],bool) or t[k]< (0 if k=='seed' else 1): raise ValueError(f"training.{k} is invalid")
    if not isinstance(t['learning_rate'],(int,float)) or t['learning_rate']<=0: raise ValueError("training.learning_rate must be positive")
    if t['class_weights'] not in ('none','inverse_frequency') and not (isinstance(t['class_weights'],list) and all(isinstance(x,(int,float)) and x>0 for x in t['class_weights'])): raise ValueError("training.class_weights must be none, inverse_frequency, or a positive vector of C values")
    if isinstance(t['class_weights'],list) and cfg['classes']['class_names'] is not None and len(t['class_weights'])!=len(cfg['classes']['class_names']): raise ValueError("training.class_weights vector must have one value per class")
    if not isinstance(t['amp'],bool): raise ValueError("training.amp must be a boolean")
    if t['monitor'] not in ('macro_f1','weighted_ce'): raise ValueError("training.monitor must be macro_f1 or weighted_ce")
    # Persist this fixed external media contract into run config/checkpoints.
    cfg['input_contract']=deepcopy(INPUT_CONTRACT)
    cfg['_config_path']=str(path)
    cfg['run']['output_root']=str(Path(str(cfg['run']['output_root'])).expanduser().resolve())
    validate_run_name(cfg['run']['name'])
    return cfg

def validate_run_name(name):
    """Accept an optional, single native path component for a run directory."""
    if name is None: return
    if not isinstance(name,str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('run name must be a single safe path component')

def resolve_path(base,value):
    p=Path(str(value)).expanduser()
    return p if p.is_absolute() else (base/p).resolve()
