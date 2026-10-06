# Standalone WPP multiclass video classifier

A self-contained PyTorch classifier for **one integer class per whole video** from native WPP NPZs. It is independent of the encoder/extractor and of `Video_Classification_Model`; it does not create splits, infer corpus/source/prompt metadata, rebalance data, or read MP4/sidecar files.

`INPUT_CONTRACT.md` is authoritative for inputs and exact timing semantics. In particular, callers provide all six explicit CSV paths, CSVs join by `video_id` (not row order), and each feature path is one self-contained NPZ. The included example CSVs are illustrative only: their NPZs are deliberately absent and therefore `configs/example.yaml` is **not runnable**.

## Install

Use Python 3.10+ with PyTorch, NumPy, scikit-learn, PyYAML, and matplotlib (the project environment already supplies these):

```sh
python -m pip install torch numpy scikit-learn PyYAML matplotlib
```

## Run

Paths in YAML resolve relative to its location; `features_path` resolves relative to its data CSV. On a native Windows run, native absolute Windows paths are accepted.

```sh
python run_pipeline.py train --config /absolute/path/to/config.yaml
python run_pipeline.py train --config configs/example.yaml --resume runs/my_run/checkpoints/latest.pt
# An extra evaluation must use a fresh directory; canonical test output is protected.
python run_pipeline.py evaluate --config /absolute/path/to/config.yaml \
  --checkpoint runs/my_run/checkpoints/best.pt --output runs/my_run/manual_evaluation
```

Training uses only supplied train/validation splits to select a checkpoint, then evaluates the supplied test split once into `test_evaluation/`. `evaluate` will not overwrite that canonical directory. Resume rejects any config, CSV, or NPZ-content-hash changes.

## Configuration

`data` contains exactly the six paths shown in `configs/example.yaml`. `classes.class_names` is an ordered vocabulary: its index is the required integer label. It is arbitrary (binary or multiclass); all classes must occur in training. Validation/test classes may be absent and remain reported with support zero.

Defaults: Transformer (`d_model: 64`, `heads: 4`, `layers: 2`, dropout `.1`), or `architecture: resnet` (width 64, three residual temporal blocks). `model.window_seconds` is exact: use an integer (`2`) or a quoted string (`"1.5"`, `"3/2"`), never a YAML float. Frames are bucketed by native PTS, each frame exactly once, and full frame duration is used to aggregate window logits into one video logit.

Useful `training` values are `max_epochs`, `learning_rate`, `batch_size`, `gradient_accumulation`, `patience`, `device` (`auto`, `cpu`, `cuda`), `class_weights` (`none`, `inverse_frequency`, or a positive C-length vector), and `monitor` (`macro_f1` or `weighted_ce`). One optimizer update covers `batch_size * gradient_accumulation` videos; a tail uses its actual video count. On OOM, only retry microbatching is reduced: the full update group and all frames remain unchanged. No balancing is automatic.

## Outputs and metric convention

The run directory stores `config.json`, train-only `normalizer.json`, input/feature content hashes, checkpoints, epoch metrics/curves, and `summary.json`. Canonical test files include per-class support/metrics, raw and row-normalized confusion CSV/PNG, and optional one-vs-rest ROC-AUC/AP diagnostics. The seven headline values are `accuracy`, macro/weighted precision, macro/weighted recall, and macro/weighted F1, using sklearn labels for **all configured classes**. Precision/recall/F1 use `zero_division=0`; absent-class ROC/AP is `null` with a reason, never dropped.

Normalization is a train-only streaming mean/std for `[delta_row_share, row_share]` over valid native cells. Explicit masks—not feature zeros—represent padding. No resizing, interpolation, truncation, or delta reset occurs.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests create temporary synthetic NPZs; they do not train on project data or alter external encoder/extractor artifacts.
