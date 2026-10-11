# Standalone WPP multiclass video classifier

A self-contained PyTorch classifier for **one class label per whole video** from native WPP NPZs. It is independent of the encoder/extractor and of `Video_Classification_Model`; it does not create splits, infer corpus/source/prompt metadata, rebalance data, or read MP4/sidecar files.

`INPUT_CONTRACT.md` is authoritative for inputs and exact timing semantics. In particular, callers provide all six explicit CSV paths, CSVs join by `video_id` (not row order), and each feature path is one self-contained NPZ. Every NPZ must verify contract version 1, normalized `512x512` display, and an exact `15/1` fps clock (including every final-frame duration); legacy artifacts are rejected. The included example CSVs are illustrative only: their NPZs are deliberately absent and therefore `configs/example.yaml` is **not runnable**.

## Install

Use Python 3.10+ with PyTorch, NumPy, scikit-learn, PyYAML, and matplotlib (the project environment already supplies these):

```sh
python -m pip install torch numpy scikit-learn PyYAML matplotlib
```

## Run

`data` paths in YAML resolve relative to the YAML file; `features_path` resolves relative to its data CSV. In contrast, `run.output_root` resolves relative to the process working directory (so `runs` is `./runs` where the command is launched). On a native Windows run, native absolute Windows paths are accepted.

```sh
python run_pipeline.py train --config /absolute/path/to/config.yaml
python run_pipeline.py train --config configs/example.yaml --run-name experiment_01
python run_pipeline.py train --config configs/example.yaml --run-name experiment_01 \
  --resume runs/experiment_01/checkpoints/latest.pt
# An extra evaluation must use a fresh directory; canonical test output is protected.
python run_pipeline.py evaluate --config /absolute/path/to/config.yaml \
  --checkpoint runs/my_run/checkpoints/best.pt --output runs/my_run/manual_evaluation
```

Training uses only supplied train/validation splits to select a checkpoint, then evaluates the supplied test split once into `evaluations/`. `evaluate` will not overwrite that canonical directory. Resume rejects any config, CSV, mapping, or NPZ-content-hash changes. A completed canonical test is immutable: resuming its latest checkpoint is allowed only as a no-op verification; additional epochs must resume an interrupted pre-test run.

## Configuration

`data` contains exactly the six paths shown in `configs/example.yaml`. `run.output_root` defaults to `runs`; `run.name` is optional and defaults to a `run_YYYYMMDD_HHMMSS` timestamp. `--run-name` takes precedence over `run.name`. Names must be a single safe path component, and a new run never overwrites an existing directory. When resuming, an omitted name reuses the checkpoint's persisted name; the effective name (CLI override, otherwise YAML) must match the checkpoint. Repeat `--run-name` on resume if it originally overrode a configured name. The minimal raw-label configuration is just those six paths (no `classes` mapping):

```yaml
data:
  train_data_path: train_data.csv
  train_label_path: train_labels.csv
  val_data_path: val_data.csv
  val_label_path: val_labels.csv
  test_data_path: test_data.csv
  test_label_path: test_labels.csv
```

By default `classes.class_names` is omitted: labels are CSV text (for example person IDs `001`, `016`, `124`) and the sorted, `str.strip().casefold()` canonical training labels become the ordered vocabulary. Leading zeroes are retained. Only training labels define classes; unknown validation/test labels are rejected, while known classes absent in either split remain reported with support zero.

For old indexed CSVs, provide a nonempty `classes.class_names` list. This compatibility mode preserves the prior requirement for exact integer labels `0..C-1` and preserves its display order. Do not combine that legacy list with person-ID label CSVs. Each run writes `label_mapping.json`, and each checkpoint embeds the exact mapping; resume verifies a regenerated train mapping before loading weights, and standalone evaluation uses the checkpoint mapping rather than inferring from test labels.

### Audited DeepAction source-label export

`export_deepaction_splits.py` is the one-purpose, read-only packager/exporter for the approved DeepAction WPP cache. It verifies cache/encoded hashes, obtains integer packet durations and time bases from each existing encoded MP4, requires their presentation-sorted PTS to exactly equal the existing NPZ display-order PTS, and writes fresh timing-complete NPZ **copies** plus six CSV tables and audits. Legacy feature arrays are retained exactly; no source cache, metadata, manifest, or encoding is changed. Source names are emitted directly as raw labels; it never uses the legacy binary `label` or `label_id` fields. The output directory must not already exist and may not overlap the source cache. Publication requires atomic `renameat2(RENAME_NOREPLACE)` no-clobber support; unsupported filesystems fail closed rather than risk overwriting a newly created output path.

```sh
/home/naoromi/miniforge3/envs/entropy/bin/python export_deepaction_splits.py
```

The default target is `/mnt/d/projects/proj-entropy/deepaction_wpp_source_splits_transformer_seed42`. It reproduces the adapter's seed-42 prompt grouping (sorted canonical prompt IDs, local shuffle, Python `round`, and clamps), so variants sharing a prompt never cross train/validation/test. Review `split_audit.json`, `prompt_assignments.csv`, and `coverage_counts.csv` before putting the six generated paths into a training config.

Defaults: Transformer (`d_model: 64`, `heads: 4`, `layers: 2`, dropout `.1`), or `architecture: resnet` (width 64, three residual temporal blocks). A third choice, `architecture: convtran`, is an **adapted ConvTran**: it keeps ConvTran tAPE and post-softmax eRPE temporal attention, but replaces the fixed sensor tokenizer with masked local spatial-temporal convolutions over native `[T,R,2]` WPP rows followed by learned masked spatial reduction. It accepts variable row counts directly; it never pre-averages rows. `convtran_layers`, `convtran_heads`, `convtran_ff_multiplier`, and odd `convtran_kernel_size` configure only this architecture. Its finite tAPE/eRPE tables are derived exactly as `ceil(window_seconds * 15)`, so over-bound windows fail rather than truncate; short terminal windows are valid. See `configs/example_convtran.yaml`.

`model.window_seconds` is exact: use an integer (`2`) or a quoted string (`"1.5"`, `"3/2"`), never a YAML float. Frames are bucketed by native PTS, each frame exactly once, and full frame duration is used to aggregate window logits into one video logit.

Useful `training` values are `max_epochs`, `learning_rate`, `batch_size`, `gradient_accumulation`, `patience`, `device` (`auto`, `cpu`, `cuda`), `class_weights` (`none`, `inverse_frequency`, or a positive C-length vector), and `monitor` (`macro_f1` or `weighted_ce`). One optimizer update covers `batch_size * gradient_accumulation` videos; a tail uses its actual video count. On OOM, only retry microbatching is reduced: the full update group and all frames remain unchanged. No balancing is automatic.

## Outputs and metric convention

The run directory stores `config.json` (including the concrete generated or overridden run name, strict media contract, and derived ConvTran temporal bound), train-only `normalizer.json`, input/feature content hashes, `checkpoints/{best,latest}.pt`, and `status.json`. Checkpoints from before this strict media contract are deliberately incompatible and rejected. Canonical reporting is written only to `evaluations/`: `per_epoch_metrics.csv/json`, `loss_curves.png`, `accuracy_curves.png`, `precision_curves.png`, `recall_curves.png`, `f1_curves.png`, `summarized_metrics.json`, and test per-class/confusion-matrix CSV/PNG files. `status.json` records completion, epoch count, checkpoint selection, and whether training stopped by patience or max epochs.

`summarized_metrics.json` contains the canonical test metrics plus provenance, normalizer and checkpoint references, selection details, and `best_observed_per_epoch_metrics`. The latter independently records the maximum value and earliest epoch on ties for every one of the seven train and validation classification metrics; it is not checkpoint selection. `checkpoint_selection` separately records the selected checkpoint epoch and its train/validation metrics. The selected checkpoint continues to use configurable validation `macro_f1` maximization or validation weighted cross-entropy minimization.

The seven headline values are `accuracy`, macro/weighted precision, macro/weighted recall, and macro/weighted F1. Macro and weighted averages use only classes present in that split's ground truth (while per-class files retain every configured class); predictions to absent configured classes remain errors. Precision/recall/F1 use `zero_division=0`; absent-class ROC/AP is `null` with a reason, never dropped. Epoch train metrics are **online** predictions made while shuffled training examples are in `model.train()` and optimizer updates occur; validation and standalone checkpoint evaluation use `model.eval()`. Thus an epoch train value is intentionally not a second, post-epoch inference pass.

Normalization is a train-only streaming mean/std for `[delta_row_share, row_share]` over valid native cells. Explicit masks—not feature zeros—represent padding. No resizing, interpolation, truncation, or delta reset occurs.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests create temporary synthetic NPZs; they do not train on project data or alter external encoder/extractor artifacts.

## ConvTran attribution

The ConvTran attention design is adapted from [Navid Fouladi's ConvTran](https://github.com/Navidfoumani/ConvTran), licensed MIT. This repository contains a clean, minimal WPP-specific implementation rather than copied upstream source; its native-row tokenizer and mask handling are local code. See `THIRD_PARTY_NOTICES.md`.
