# Supervised packet-sequence classification (PyTorch)

This directory contains the maintained supervised PyTorch pipeline for classifying 3,000-packet video sequences with the 1-D ResNet in `src/model_resnet_tsc.py`. `run_pipeline.py` is the training and evaluation entry point; `linear_probe.py` evaluates frozen representations from compatible trained checkpoints.

## Installation

From the repository root:

```sh
python -m pip install -r Video_Classification_Model/requirements.txt
cd Video_Classification_Model
```

The requirements file declares NumPy, pandas, scikit-learn, matplotlib, PyYAML, and PyTorch. Use a Python/PyTorch installation appropriate for the device you intend to select.

## Input and preprocessing contract

The pipeline consumes two CSVs for every data set:

- **Data CSV:** exactly 3,000 columns, named `size0`, `size1`, …, `size2999`, in precisely that order. Every value must be numeric (not Boolean), finite, nonmissing, and nonnegative. Each row is one packet sequence and must have nonzero population standard deviation.
- **Label CSV:** exactly one column (its header may have any name) and exactly the same number of rows as its data CSV. Labels cannot be missing or empty after trimming. They are trimmed and case-folded to lowercase before use.

Each sequence is independently converted to `float32` with population z-scoring:

```text
(row - row.mean()) / row.std(ddof=0)
```

There is no log transform, clipping, added feature, or epsilon. Constant rows are rejected rather than normalized. This fixed preprocessing definition is saved in the run configuration and checkpoints and is checked for compatibility on resume, evaluation, and probing.

## Configuration

Start from `configs/default.yaml`; it documents the accepted configuration structure and defaults.

- `run.output_root` defaults to `runs`; `run.name` defaults to `null` and may be overridden with `--run-name`.
- `data.mode` defaults to `auto`. In `auto`, set `data_path` and `label_path`; in `manual`, set all six `train_*`, `val_*`, and `test_*` data/label paths. All paths default to `null`.
- `model.initial_feature_maps` defaults to `128` and controls ResNet width.
- Training defaults are seed `42`, learning rate `0.001`, batch size `64`, `max_epochs: 300`, optimizer `Adam` (or `AdamW`), class imbalance `none` (or training-only `weighted` inverse-frequency weights), device `auto`, `num_workers: 0`, and `checkpoint_frequency: 1`.
- Early stopping is fixed to monitor `validation_loss`, with patience `25` and `min_improvement: 0.001`. The ReduceLROnPlateau scheduler defaults to factor `0.1`, patience `10`, and `min_lr: 0.00001`.
- `preprocessing` is fixed to `per_sequence_zscore`, length `3000`, population standard deviation, and `epsilon: null`; it must remain exactly as declared.

All supplied paths are expanded and resolved. Unknown sections or keys, invalid settings, and missing paths for the selected data mode are rejected.

## Train

Create a configuration YAML with your input paths, then run:

```sh
python run_pipeline.py train --config /path/to/config.yaml --run-name resnet_001
```

A run name must be a single path component and an existing run directory is never overwritten. If `--run-name` is omitted, the configured name is used, or a timestamp is generated.

### Split behavior

**Automatic mode** loads one labeled pair, removes exact duplicate sequences before splitting (retaining the first same-label occurrence), and rejects duplicate sequences with conflicting labels. It then produces a deterministic, seed-controlled, class-stratified 70/15/15 train/validation/test split.

**Manual mode** accepts independent train, validation, and test pairs. It rejects sequence-hash overlap between any two splits and rejects validation or test labels absent from training. The run records source paths, SHA-256 checksums, source indices, row hashes, and class distributions in `split_manifest.json`; later resume and original-test evaluation verify those inputs.

Training uses validation loss for early stopping and the learning-rate scheduler. `best.pt` is updated whenever validation loss is lower than the previous best; patience resets only for an improvement meeting `min_improvement`. `max_epochs` remains the hard upper bound. On normal completion or early stopping, the untouched saved test split is evaluated using `best.pt`.

### Interrupt and resume

Ctrl+C is handled at the last completed epoch boundary: an `interrupted.pt` checkpoint is written and the run status becomes `paused`.

```sh
# Uses checkpoints/interrupted.pt when it exists, otherwise checkpoints/latest.pt.
python run_pipeline.py resume --run runs/resnet_001

# Resume an explicit checkpoint.
python run_pipeline.py resume --run runs/resnet_001 \
  --checkpoint runs/resnet_001/checkpoints/latest.pt
```

To increase the epoch limit, copy `runs/resnet_001/config.yaml`, change `training.max_epochs` to a value greater than the completed epoch count, and resume with it:

```sh
python run_pipeline.py resume --run runs/resnet_001 --config /path/to/resume.yaml
```

Only `training.max_epochs`, `training.device`, and `training.num_workers` may differ from the saved configuration. The maximum cannot be decreased relative to the checkpoint. Model architecture, preprocessing, label mapping, saved split, and all other training settings must remain compatible.

## Interactive menu and CLI

With no arguments, the program presents a five-action menu:

```sh
python run_pipeline.py
```

It can start a new automatic or manual training run, resume a run, evaluate a checkpoint on its original saved test split, evaluate it on another labeled pair, or exit. It prompts for paths and uses `configs/default.yaml` unless another configuration is selected.

The noninteractive commands are:

```sh
python run_pipeline.py train --config /path/to/config.yaml [--run-name NAME]
python run_pipeline.py resume --run RUN_DIR [--config /path/to/resume.yaml] [--checkpoint /path/to/checkpoint.pt]
python run_pipeline.py evaluate-original --checkpoint /path/to/checkpoint.pt [--device auto|cpu|cuda|cuda:N|mps]
python run_pipeline.py evaluate-other --checkpoint /path/to/checkpoint.pt \
  --data /path/to/data.csv --labels /path/to/labels.csv \
  [--device auto|cpu|cuda|cuda:N|mps]
```

`auto` selects CUDA when available, then MPS when available, otherwise CPU. `evaluate-original` re-creates and checksum-verifies the test split recorded by its parent run. `evaluate-other` loads a separate labeled pair under the same CSV contract and rejects labels not in the checkpoint mapping.

## Runs, status, checkpoints, and artifacts

Every run is a unique directory under `run.output_root`. It includes the resolved `config.yaml`, `status.json`, `split_manifest.json`, `input_checksums.json`, `label_mapping.json`, `logs/training.log`, `parameter_count.txt`, epoch metrics, curves, checkpoints, and test artifacts.

`status.json` is atomically replaced on state transitions and after each epoch. Its states are `initializing`, `running`, `paused`, `early_stopped`, `complete`, and `failed`. It records completed epochs, the best validation result, checkpoint paths, and stop reason.

`checkpoints/` contains `latest.pt` each completed epoch, `best.pt` when validation loss improves, periodic `epoch_####.pt` files according to `checkpoint_frequency`, and `interrupted.pt` after a handled Ctrl+C. Checkpoints record model/optimizer/scheduler state, training and model configuration, preprocessing, label mapping, split information, and random-number-generator state.

Epoch history is written to `metrics.json` and `metrics.csv`; `loss_curves.png`, `accuracy_curves.png`, and `macro_f1_curves.png` plot training and validation history. Each evaluation creates a new timestamped directory in `evaluations/`. Original-test evaluation also creates the run-level `test_*` convenience artifacts if they do not already exist.

Evaluation artifacts include JSON and CSV summary/per-class metrics, raw and row-normalized confusion matrices in CSV and PNG, and provenance (checkpoint, input paths, checksums, and timestamp). Metrics include loss, accuracy, balanced accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1 and support, top-3 accuracy for three or more classes, duration, throughput, parameter count, and explicit reporting when zero-division values of zero were used.

## Frozen linear probes

`linear_probe.py` reads a modern `run_pipeline.py` format-version-1 checkpoint. It reconstructs the frozen ResNet encoder and validates the saved preprocessing, contiguous label mapping, recorded split manifest, source checksums, and split integrity. It does not train or alter the encoder.

```sh
python linear_probe.py --checkpoint runs/resnet_001/checkpoints/best.pt

# Override device and probe settings.
python linear_probe.py --checkpoint runs/resnet_001/checkpoints/best.pt \
  --device cpu --batch-size 64 --c 0.01 0.1 1 10 --max-iter 1000 \
  --output-root /path/to/probe_outputs

# For an automatic-split run whose original pair was relocated, supply a
# byte-identical data/label pair; its SHA-256 checksums must match.
python linear_probe.py --checkpoint runs/resnet_001/checkpoints/best.pt \
  --data-path /new/location/data.csv --label-path /new/location/labels.csv
```

The script uses the temporal mean of each residual block (blocks 1–3; block 3 is the encoder final global-average-pool representation). For every block and each requested positive `C`, it fits a `StandardScaler` and multinomial `LogisticRegression` (`lbfgs`) on training features only, then selects the `C` with highest validation macro-F1 (ties choose the smaller `C`). It then refits the selected probe on train plus validation and evaluates the test split once. Test labels are excluded from selection.

For automatic-split checkpoints, `--data-path` and `--label-path` may relocate the single original pair only when both are supplied and byte-identical by SHA-256. Manual-split checkpoints require their recorded three source paths and do not accept these overrides. Only checkpoints with the required format-version-1 fields and fixed preprocessing are accepted; incompatible checkpoints are rejected.

By default, results are written to `RUN_DIR/linear_probes/TIMESTAMP` (or below `--output-root`). The directory contains `metadata.json`, selection metrics in JSON/CSV, selection plots for accuracy, macro-F1, and log loss, and `final_test_metrics_by_block.png`. Each `block1`, `block2`, and `block3` directory contains final metrics and per-class metrics (JSON/CSV), raw and normalized confusion-matrix CSVs, plus the fitted `scaler.joblib` and `probe.joblib`. When a valid original ResNet test artifact is available, it is recorded and displayed only as a reference; it does not influence selection.

## Tests

Run the standard-library test suite from this directory:

```sh
python -m unittest discover -s tests -v
```

The tests cover strict CSV loading and preprocessing, splitting and duplicate protections, checkpoint/lifecycle/resume behavior, evaluation artifacts, and frozen-probe compatibility, integrity checks, selection, and outputs.

## Planned extraction handoff — not yet implemented

Video-to-CSV extraction is deliberately outside the current pipeline and must remain an independent, hot-swappable producer: it must not import, invoke, or have a hard runtime connection to `run_pipeline.py`. The integration boundary is output-only—the producer writes CSVs, and this pipeline consumes them through the contract above.

A future extraction script should recursively discover videos; obtain FFmpeg packet sizes; emit one 3,000-packet row per usable sequence; derive labels from directory structure or a supplied manifest; and write separate data and label CSVs in **bits** with the exact schema described above. Before declaring output ready, it should validate the loader contract (column names/order, numeric finite nonnegative values, nonconstant rows, and matching one-column labels). It should report short videos/sequences, invalid inputs, and FFmpeg or processing failures concisely and actionably. No extraction CLI is defined or implemented at present.
