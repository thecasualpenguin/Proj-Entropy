#!/usr/bin/env python3
"""Compatibility entry point for the supervised PyTorch pipeline.

The video extraction/collation utilities remain under ``src``; this entry point now
runs only the reliable, reproducible PyTorch training/evaluation workflow.
"""
from supervised import main

if __name__ == "__main__":
    raise SystemExit(main())
