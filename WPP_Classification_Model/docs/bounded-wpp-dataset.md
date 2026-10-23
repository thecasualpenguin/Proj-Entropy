# Bounded video-aware WPP dataset (v1)

`pack_wpp_dataset.py` is the data layer for the native WPP workflow. It initializes no model, performs no normalization or encoding, and does not extract a dataset-wide DeepAction build. The native `run_pipeline.py` consumes this bundle only through `WPPDatasetBundle` and its explicit trusted parents; the removed legacy CSV pipeline is not an alternative input path.

## Pack

The source manifest and index must be the exact complete pair. Before opening any artifact, the packer loads both contracts and calls `verify_native_build`; this checks the index-bound ledger plus encoded, artifact, and command-log hashes. Any failed index row, source/index mismatch, unsafe path, group spanning partitions, missing partition, or integrity error fails closed.

```sh
cd WPP_Classification_Model
/home/naoromi/miniforge3/envs/entropy/bin/python pack_wpp_dataset.py \
  --source-manifest /path/to/source-manifest \
  --index /path/to/native-build/extraction-index \
  --output-dir /fresh/outside-checkout/wpp-bounded-v1
```

The destination must be fresh, is published with the existing staged atomic publication helper, and is rejected if it is inside this checkout or either input directory. Before creating output staging, the packer rejects symlinked ancestor components and any symlinked required source/index CSV or metadata leaf; it also runs the strict bundle provenance/label validation on staging before atomic publication. There is no automatic splitting, deduplication, relabeling, or failed-record skipping. Source manifest order is retained within each `train`, `validation`, and `test` output. Nonempty `group_id` values may be absent generally, but a supplied value must occur in exactly one partition globally.

A nonempty training partition is required. Class IDs are the deterministic lexicographic mapping of **exact training labels**; labels are not case-folded, trimmed, or otherwise canonicalized. Validation/test labels not present in training fail explicitly.

## Bundle format

The published directory contains six checksummed CSVs:

- `train-data.csv`, `validation-data.csv`, `test-data.csv`
- `train-label-manifest.csv`, `validation-label-manifest.csv`, `test-label-manifest.csv`
- `datasetmetadata.json`

A numeric CSV always has exactly `size0` through `size2999`. A video with producer `row_bytes` shape `R_v x T` emits `R_v` consecutive numeric rows, preserving WPP top-to-bottom row order and producer display/presentation time order. It uses only frames `0..min(T,3000)-1`; short videos are right-zero-padded and long videos are explicitly marked as truncated. There is no frame resampling/reordering and no spatial padding to a global `R`.

Each label-manifest row is **one video**, with a zero-based `stacked_row_start` and `stacked_row_count` locating its contiguous data block. It also records the canonical source ID/digest/path, exact source label/group/partition, `R_v`, original and used frame counts, truncation flag, and extraction artifact relative path. `stacked_row_count` is the sole row-count field (rather than emitting a redundant alias). Empty source partitions produce header-only pairs; they remain valid provided training exists.

Values are raw nonnegative integer byte counts written in CSV and returned by the loader as `float32`. A value of zero is a valid observation. Padding is determined only by the recorded explicit length, never by examining values.

`datasetmetadata.json` records feature axes/dtype/padding semantics, source/index metadata and records digests, verified ledger digest, exact class map, counts, output checksums, and the resource strategy.

## NumPy loader and batch collator

```python
from src.wpp_dataset import WPPDatasetBundle, collate_wpp_videos

# Parent contracts are required trusted context; a bundle's own digest strings
# are not treated as proof of ancestry. Reuse this object for all split views.
validated = WPPDatasetBundle(
    "/fresh/outside-checkout/wpp-bounded-v1",
    source_manifest="/path/to/source-manifest",
    index="/path/to/native-build/extraction-index",
)
train = validated.partition_view("train")
example = train[0]
# example["values"] is float32 [R_v, 3000]
# explicit: used_time_length, row_count, label, class_id, video_id, provenance
batch = collate_wpp_videos([train[0], train[1]])
# values:     float32 [B, R_batch_max, 3000]
# row_mask:   bool [B, R_batch_max]
# time_mask:  bool [B, 3000]
# valid_mask: bool [B, R_batch_max, 3000] (row_mask AND time_mask)
# labels:     int64 [B]
```

`WPPDatasetBundle` requires explicit trusted `source_manifest` and `index` parent paths; it checks their live CSV/metadata digests against the packed provenance, validates the source/index binding, then validates **all three** label manifests against the trusted source records/index rows. This proves source IDs, labels, groups, partitions, paths, dimensions, crop/truncation metadata, artifact references, global coverage, and the exact training-derived class mapping rather than trusting self-declared bundle digests. It also rejects symlinked bundle roots/files/parent components.

It checksum-verifies the immutable six-CSV bundle once per reusable bundle object. That protects accidental mutation; it is not a signature, so an attacker able to rewrite both a packed CSV and its metadata needs the explicit trusted parents and the structural/provenance checks above. Numeric CSV payloads are scanned only for a requested split, with byte offsets retained instead of an eager float64 split copy; `__getitem__` reads only that video's contiguous lines. No native artifact tree is rehashed per item. The collator pads only to the maximum `R_v` in that batch, zeroes every invalid outer-mask cell, and never treats valid zeros as padding.

This v1 loader intentionally does not expose a PyTorch Dataset subclass. The native runner explicitly converts its NumPy arrays/masks and applies its documented train-only masked normalization; no normalization or model behavior is hidden in this loader.

## Scope and limits

The only available native input presently is the bounded synthetic VFR smoke build at `/tmp/native-wpp-smoke-final-review.CJGFKC`. Its complete source/index/ledger is integrity-verifiable, but its source-manifest partition field is empty, so this packer correctly rejects it rather than inventing a training partition. It is not a real DeepAction dataset. No full 2,583-video packed dataset has been extracted or claimed by this change.
