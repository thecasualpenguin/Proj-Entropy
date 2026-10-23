# Native flat-WPP classification

This directory trains the native `src.wpp_transformer.WPPTransformer` on a
ledger-verified bounded WPP bundle. It is not the packet-CSV pipeline: copied
ResNet, packet extraction, linear-probe, CSV-split, and legacy Transformer
entrypoints were removed. The maintained alternative for that workflow remains
unchanged in `Video_Classification_Model/`.

## Required inputs

Training accepts only three explicit, trusted inputs:

1. a bundle published by `pack_wpp_dataset.py`;
2. its complete approved source manifest; and
3. its native extraction index.

`WPPDatasetBundle` validates the live source/index contracts, the index-bound
ledger, all six bundle CSV checksums, source coverage, partitions, group rules,
and the exact training-derived class mapping before training begins. There is no
automatic split, relabeling, deduplication, or fallback to self-declared bundle
metadata. Train and validation must be nonempty; test may be empty, but an
explicit evaluation of an empty partition fails.

A complete DeepAction extraction/packed bundle is **not** included or claimed.
Do not run the dataset-wide extraction, packing, or training commands without
separate approval.

## Native upstream chain

`build_manifest_contract.py` and `src/{manifest_contract,deepaction_manifest}.py`
publish original-source contracts. `build_native_wpp_extraction.py` and
`src/native_wpp_workflow.py` produce a ledger-bound extraction index;
`verify_native_wpp_extraction.py` verifies it. `pack_wpp_dataset.py` and
`src/wpp_dataset.py` publish/load the bounded video-aware input. These stages
remain separately documented in:

- `docs/manifest-contracts.md`
- `docs/bounded-wpp-dataset.md`
- `docs/wpp-flat-transformer.md`

Native extraction is fresh by default. Resume only an owned interrupted build with its exact same manifest/profile/runtime fingerprint:

```sh
python build_native_wpp_extraction.py --source-manifest /path/manifest --source-root /path/originals \
  --output-dir /outside/native-build --resume --producer-dir /path/entropy-cabac-wpp
# Terminal extraction failures remain terminal unless explicitly retried:
python build_native_wpp_extraction.py --source-manifest /path/manifest --source-root /path/originals \
  --output-dir /outside/native-build --resume --retry-failed --producer-dir /path/entropy-cabac-wpp
```

`--resume` is a verified no-op for a compatible finalized build (no journal mtime change); terminal failures remain terminal unless `--retry-failed` is explicit. Do not point resume at a different manifest (including the 21-video pilot versus a full retained manifest), alter producer scripts/runtime, or remove a build lock while a writer may be alive. Linux descriptor pinning protects child input/output pathname publication, but not a same-UID actor that can rewrite all owned files. See `docs/manifest-contracts.md` for recovery, receipt inventory, and finalization behavior.

## Training and normalization

Copy `configs/example.yaml`, set all paths to fresh explicit locations, then:

```sh
cd WPP_Classification_Model
python run_pipeline.py train --config /path/to/native-wpp.yaml
```

The default model is the existing flat WPP Transformer: patch 16, width 32,
four heads, two pre-LN GELU layers, feed-forward width 64, dropout 0.2. Batch
size defaults to one (the only measured model smoke was batch one); training is
FP32 with no AMP or attention-kernel override. `auto` selects CUDA when
available, otherwise CPU; `cpu` or `cuda[:N]` may be explicit. The runner owns
dictionary batches and currently requires `num_workers: 0`.

Normalization is fixed and checkpointed: a mergeable Welford float64 population
mean/std is fitted once from **valid train cells only**. Explicit row/time masks,
not values, determine validity; observed zeros participate, all padding is
excluded and written back as zero. Validation/test use the frozen train
statistics. Nonfinite valid values, no valid training cells, or zero/invalid
population variance fail. There is no epsilon, log transform, clipping, or
per-row normalization.

The upstream partitions are used exactly. Each video is one cross-entropy
example regardless of its WPP-row count; masks retain variable row counts and
frame lengths. Validation loss alone drives scheduler/early stopping and
`best.pt`; test is never used for selection or statistics.

## Resume and evaluate

```sh
python run_pipeline.py resume --run /fresh/outside-checkout/native-wpp-run
python run_pipeline.py evaluate \
  --checkpoint /fresh/outside-checkout/native-wpp-run/checkpoints/best.pt \
  --config /path/to/native-wpp.yaml --partition test \
  --output-dir /fresh/outside-checkout/native-wpp-evaluation
```

Runs require a fresh `run.output_dir`; evaluation requires a fresh output
directory. Resume accepts only an explicit epoch-boundary native checkpoint in
that run. It revalidates bundle/source/index/ledger provenance, class mapping,
model configuration, immutable normalization policy/statistics, and all
hyperparameters except a larger `max_epochs`, device, or `num_workers` (which
remains zero). It restores optimizer/scheduler, Python/NumPy/Torch CPU/CUDA RNG
and shuffle-generator state, and never refits normalization.

Checkpoints are `native_wpp_checkpoint` v1.0 payloads containing only primitive
containers and tensors and load through `torch.load(weights_only=True)`;
legacy checkpoints are rejected. `last.pt`, `best.pt`, and periodic epoch files
are atomically published. JSON status, provenance, history, and sample-weighted
cross-entropy/classification metrics are written under the fresh run. Completion
may write one post-selection test report; standalone evaluation never optimizes
or changes saved statistics.

## Evaluation artifacts and historical exports

On completion, the selected `best.pt` atomically publishes the compatible
fourteen-file `evaluations/` set (and nothing else in that directory):
per-epoch JSON/CSV; loss, accuracy, precision, recall, and F1 curves; a
canonical summary; per-class JSON/CSV; and raw/row-normalized confusion-matrix
CSV/PNG files. Native backward-compatible test data is written separately as
run-root `native_test_metrics.json`; older completed runs retaining
`evaluations/test.json` are only read by export compatibility code. Managed run
reports record checksums/ownership in root `.native-wpp-evaluations-state.json`.
A legitimate larger-epoch resume archives its prior owned generation under
`.native-wpp-evaluations-archives/` and journal-publishes a new one; foreign,
corrupt, or symlinked report trees are rejected before optimizer/history changes. The seven aggregate metrics are accuracy plus
macro/weighted precision, recall, and F1. Aggregates use classes present in the
truth split; per-class and confusion reports retain the complete configured
class order. Test reporting is only post-selection `best.pt` inference.

Future epoch records contain online train metrics from the same pre-update
training forwards (no extra train pass), post-scheduler LR, and the measured
train/validation/scheduler duration. A standalone `evaluate` output retains its
native `metrics.json` and also contains a compatible `evaluations/` directory.

To report an older completed run without changing any source-run byte, publish
into a new sibling directory:

```sh
python run_pipeline.py export-report \
  --run /outside/native-wpp-run \
  --output-dir /outside/native-wpp-run-video-format \
  --reconstruct-train-metrics --device cuda
```

With `--reconstruct-train-metrics`, export checks every required epoch
checkpoint with the safe native contract and reuses its embedded frozen train
statistics. Without that flag it validates only the selected `best.pt`.
Reconstructed train metrics are post-epoch `eval()` snapshots, not the
unavailable original online train metrics; this distinction, unchanged
historical train CE, unavailable duration, and checkpoint-derived LR are
recorded in root `report_provenance.json`. The source run, checkpoints, history,
and existing native test JSON are read only. Omitting
`--reconstruct-train-metrics` leaves unavailable historical train classification
values null. Report numeric values must be finite; only explicitly unavailable
historic fields are represented as JSON null.

## Tests

```sh
cd WPP_Classification_Model
python -m unittest discover -s tests -v
```

Tests use small synthetic contract/index/bundle fixtures and tiny CPU model
smokes. They do not encode video, pack an authoritative dataset, or train a full
dataset.
