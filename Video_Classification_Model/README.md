# Supervised packet-sequence classification (PyTorch)

This directory contains a reproducible supervised baseline for classifying 3,000-packet video sequences with the existing 1-D ResNet in `src/model_resnet_tsc.py`. The training/evaluation path is PyTorch-only; the legacy TensorFlow implementation and dependencies have been removed.

## Environment and input contract

Use the existing environment:

```bash
conda activate entropy
cd Video_Classification_Model
```

A data CSV has exactly 3,000 columns named `size0` through `size2999`, in that order. Values are packet sizes in bits and must be numeric, finite, nonnegative, present, and nonconstant within each row. A label CSV has exactly one column and the same row count. Labels are whitespace-trimmed and case-folded to lowercase.

Each row is independently transformed as `(row - row.mean()) / row.std(ddof=0)`. No log transform, clipping, extra feature, or epsilon is used. Constant rows are rejected. This definition is stored in `config.yaml` and every checkpoint.

## Interactive menu

Launching without arguments opens the five-action menu:

```bash
python run_pipeline.py
```

The menu asks for an action, configuration, run/path information, and automatic versus manual data selection. Hyperparameters come from YAML rather than a long questionnaire.

## Reproducible CLI

`configs/default.yaml` documents every setting; copy it and fill in data paths. `configs/example.yaml` targets the included example dataset when invoked from this directory.

```bash
# New run (fails rather than overwriting an existing run name)
python run_pipeline.py train --config configs/example.yaml --run-name resnet_001

# Resume from interrupted.pt when present, otherwise latest.pt
python run_pipeline.py resume --run runs/resnet_001

# Increase max_epochs: copy runs/resnet_001/config.yaml, change only max_epochs,
# then supply it. Device and num_workers may also change; incompatible changes fail.
python run_pipeline.py resume --run runs/resnet_001 --config resume.yaml

# Explicit resume checkpoint
python run_pipeline.py resume --run runs/resnet_001 \
  --checkpoint runs/resnet_001/checkpoints/latest.pt

# Repeat evaluation on the exact saved, checksum-verified test split
python run_pipeline.py evaluate-original \
  --checkpoint runs/resnet_001/checkpoints/best.pt

# Evaluate another compatible labeled pair
python run_pipeline.py evaluate-other \
  --checkpoint runs/resnet_001/checkpoints/best.pt \
  --data /path/to/test-data.csv --labels /path/to/test-labels.csv
```

## Splits and training

Automatic mode removes exact sequence duplicates first. Same-label duplicates retain their first source row; conflicting labels stop the run. A seed-controlled, class-stratified 70/15/15 split is then made. Manual mode accepts separate train/validation/test pairs, rejects row-hash overlap, and rejects validation/test classes absent from training. Source paths, SHA-256 checksums, source indices, row hashes, and class distributions are recorded in `split_manifest.json`.

The default model starts with 128 feature maps. Its actual parameter count is printed and saved. Training supports Adam/AdamW, optional training-only inverse-frequency class weights, validation-loss early stopping, and ReduceLROnPlateau. `max_epochs` is always the hard bound. `best.pt` tracks every lower validation loss, while early-stopping patience resets only when improvement reaches `min_improvement`.

Ctrl+C writes `interrupted.pt` from the last completed epoch boundary and marks the run `paused`. `status.json` is atomically replaced at state transitions and after every epoch. States are:

- `initializing`: validating data and writing manifests
- `running`: training is active or an epoch completed
- `paused`: Ctrl+C was handled at an epoch boundary
- `early_stopped`: validation loss exhausted configured patience
- `complete`: maximum epochs completed
- `failed`: initialization or training failed

The untouched test split is evaluated from `best.pt` only after normal completion or early stopping. Smoke-test results are only execution checks, not evidence of model quality.

## Outputs

Runs are unique directories under `run.output_root` and are never silently overwritten. Each contains resolved `config.yaml`, atomic `status.json`, checksums, split and label manifests, `checkpoints/`, epoch CSV/JSON metrics, loss/accuracy/macro-F1 curves, logs, parameter count, and test artifacts. Every evaluation also creates a new timestamped directory under `evaluations/`, preserving earlier results. Metrics include loss, accuracy, balanced accuracy, macro/weighted precision/recall/F1, per-class metrics, raw and row-normalized confusion matrices (CSV and PNG), top-3 accuracy when applicable, timing, throughput, parameter count, and explicit zero-division reporting.

## Tests

The standard-library suite does not require adding pytest:

```bash
conda run -n entropy python -m unittest discover -s tests -v
```

The project contains no TensorFlow/Keras source or dependency declarations.
