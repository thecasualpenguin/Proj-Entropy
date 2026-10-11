#!/usr/bin/env python3
"""Standalone generalized whole-video WPP multiclass classifier."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from wpp_classifier.config import load_config
from wpp_classifier.pipeline import train,eval_checkpoint

def main():
 p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
 t=sub.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--resume',help='latest/best checkpoint to resume');t.add_argument('--run-name',help='single-component output run name (overrides config)')
 e=sub.add_parser('evaluate');e.add_argument('--config',required=True);e.add_argument('--checkpoint',required=True);e.add_argument('--output',required=True,help='new non-canonical evaluation directory')
 a=p.parse_args();cfg=load_config(a.config)
 try:
  output=train(cfg,a.resume,a.run_name) if a.command=='train' else eval_checkpoint(cfg,a.checkpoint,a.output)
  print(output)
 except (ValueError,OSError) as exc:
  p.error(str(exc))
if __name__=='__main__':main()
