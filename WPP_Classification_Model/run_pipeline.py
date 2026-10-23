#!/usr/bin/env python3
"""Train, resume, or evaluate the native flat WPP Transformer only."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.wpp_training import TrainingError, evaluate, export_report, train, validate_config

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "default.yaml"


def _config(path: str | None) -> dict:
    chosen = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    # Validation happens here as well as in the runner so malformed configs fail
    # before any run directory is created.
    return validate_config(yaml.safe_load(chosen.read_text(encoding="utf-8")), DEFAULT_CONFIG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fresh = commands.add_parser("train", help="start a fresh native WPP run")
    fresh.add_argument("--config", required=True, help="native WPP YAML configuration")
    resume = commands.add_parser("resume", help="resume a native WPP epoch-boundary checkpoint")
    resume.add_argument("--run", required=True, help="existing native run directory")
    resume.add_argument("--checkpoint", default=None, help="checkpoint inside --run (default: checkpoints/last.pt)")
    resume.add_argument("--config", default=None, help="configuration; defaults to RUN/config.json")
    evaluation = commands.add_parser("evaluate", help="evaluate a native checkpoint without optimization")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    evaluation.add_argument("--output-dir", required=True, help="fresh directory for JSON evaluation output")
    export = commands.add_parser("export-report", help="publish a fresh Video-format report from a completed native run")
    export.add_argument("--run", required=True, help="completed native run directory (read only)")
    export.add_argument("--output-dir", required=True, help="fresh report root outside --run")
    export.add_argument("--reconstruct-train-metrics", action="store_true",
                        help="frozen eval of each epoch checkpoint on train data; never optimization")
    export.add_argument("--device", default=None, help="cpu, cuda[:N], or auto; defaults to saved config")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            print(train(_config(args.config), default_path=DEFAULT_CONFIG))
        elif args.command == "resume":
            run = Path(args.run).expanduser().resolve()
            config_path = Path(args.config).expanduser().resolve() if args.config else run / "config.json"
            # config.json is valid YAML (JSON is a YAML subset).
            print(train(_config(str(config_path)), default_path=DEFAULT_CONFIG,
                        resume_checkpoint=args.checkpoint or run / "checkpoints" / "last.pt"))
        elif args.command == "evaluate":
            print(evaluate(args.checkpoint, _config(args.config), default_path=DEFAULT_CONFIG,
                           partition=args.partition, output_dir=args.output_dir))
        else:
            print(export_report(args.run, args.output_dir, default_path=DEFAULT_CONFIG,
                                reconstruct_train_metrics=args.reconstruct_train_metrics,
                                device_spec=args.device))
    except (OSError, yaml.YAMLError, TrainingError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
