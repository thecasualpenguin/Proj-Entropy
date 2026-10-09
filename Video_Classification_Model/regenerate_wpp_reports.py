#!/usr/bin/env python3
"""Regenerate legacy-layout WPP reports solely from canonical saved artifacts."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from wpp_reporting import regenerate

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--run",required=True,type=Path);a=p.parse_args(argv)
 print(regenerate(a.run));return 0
if __name__=='__main__': raise SystemExit(main())
