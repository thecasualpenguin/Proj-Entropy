# Canonical input contract: whole-video native WPP NPZ

This is the input contract for the standalone multiclass classifier in this folder. It is not a conversion utility.

## 1. Explicit caller-owned splits

A YAML configuration supplies six paths:

- `data.train_data_path`, `data.train_label_path`
- `data.val_data_path`, `data.val_label_path`
- `data.test_data_path`, `data.test_label_path`

The pipeline will validate each supplied pair and use it as given. It will **not** create or reassign splits, require `group_id`, infer groups, detect leakage policy beyond duplicate IDs inside an individual CSV, rebalance classes, oversample, or discard examples.

YAML relative paths resolve relative to the YAML file. A `features_path` in a data CSV resolves relative to that data CSV's parent directory. Absolute paths are allowed, including native Windows paths when executed on Windows.

## 2. Data CSV

Each data CSV has exactly these headers, in this order:

```csv
video_id,features_path
```

- `video_id`: nonempty unique identifier for one whole video within that CSV.
- `features_path`: nonempty path to one self-contained NPZ; no row ordering relationship to the label CSV is assumed.

No additional data columns are part of this initial contract.

## 3. Label CSV

Each paired label CSV has exactly these headers, in this order:

```csv
video_id,label
```

- `video_id` is nonempty and unique.
- `label` is one nonempty text identity/class value for that video, not a multilabel list. It is read with Python's CSV reader as text, so values such as `016` retain leading zeroes.
- Data and label IDs must match as sets; matching is by `video_id`, never CSV row position. `video_id` values are exact strings and are never case-folded.

By default no class list is needed. The classifier infers its vocabulary **from the supplied training label CSV only**, canonicalizing every label with `str(value).strip().casefold()` and lexically sorting the unique canonical values (so numeric-looking text follows lexical order, e.g. `0`, `1`, `10`). It maps that ordered vocabulary to internal IDs `0..C-1` solely for cross-entropy. Thus training labels `001`, `002`, `016` map to IDs `0`, `1`, `2`; a short training subset `001`, `016` maps to `0`, `1` (never to identity-derived numeric IDs). Whitespace-only labels are rejected. Case/space variants such as `PersonA` and ` persona ` intentionally collapse to the legacy-compatible canonical `persona`.

Validation and test labels must canonicalize to a training vocabulary value. An unknown label is rejected with its `video_id`; validation/test never enlarge the vocabulary. Metrics, reports, checkpoint metadata, and prediction class names use the full training vocabulary, so classes absent from validation/test remain present with zero support.

For compatibility with old trained runs, supplying a nonempty `classes.class_names` list selects **indexed mode**: labels must again be exact integer IDs `0..C-1`, and the list is the display vocabulary. Do not retain a legacy class list when new CSV labels are person IDs such as `016`; omit `classes` (or set `class_names: []`) to use raw-label inference. There is exactly one class per video.

The Gait exporter now writes its `labels.csv` identity values directly. Its optional `class_mapping.csv` and `classes.yaml` remain legacy indexed-mode aids, not inputs required by inferred mode: a short caller-owned training split has its own persisted mapping.

## 4. Required NPZ arrays

Each NPZ must be self-contained and contain exactly these required logical fields (additional fields are ignored):

| key | dtype | shape | meaning |
|---|---|---|---|
| `row_share` | finite `float32` or `float64` | `[T, R]` | native per-frame CTU-row share |
| `delta_row_share` | finite `float32` or `float64` | `[T, R]` | saved native row-share difference; it is not reset at window boundaries |
| `pts` | `int64` | `[T]` | presentation timestamps in `time_base` ticks |
| `frame_duration` | positive `int64` | `[T]` | actual native presentation duration in ticks for each frame |
| `time_base` | `int64` | `[2]` | `[numerator, denominator]`, representing seconds per tick |

Requirements:

- `T > 0`, `R > 0`; both geometry dimensions may differ between videos.
- `row_share.shape == delta_row_share.shape == (T, R)`.
- `pts` must be strictly increasing in presentation order. Negative, positive, and nonzero initial PTS are valid.
- `time_base[0] > 0`, `time_base[1] > 0`; every `frame_duration > 0`.
- Both floating arrays must be finite. Numeric zero is valid feature data, never a padding marker.
- Extra NPZ keys are ignored. No MP4, `metadata.json`, FFprobe, packet timing sidecar, source/generator/prompt ID, encoder provenance, or external corpus metadata is required.

These required timing fields are a **new external packaging/extractor responsibility**. Existing feature artifacts are not assumed compatible and this folder provides no converter.

## 5. Internal windowing semantics

The configurable `model.window_seconds` defaults to integer `2`. It must be either a positive YAML integer or a quoted exact decimal/rational string, for example `2`, `"1.5"`, or `"3/2"`. YAML floating-point values and booleans are forbidden so that bucket boundaries remain reproducible exact rationals. Zero, negative, non-finite, malformed, and non-rational values are validation errors.

Let `t0` be a video's first PTS and let `w = window_seconds / time_base` in exact rational PTS units.

For every frame, assign exactly one nominal bucket:

```text
bucket = floor((pts - t0) / w)
```

- Buckets are anchored at first PTS; frame order is preserved.
- Each frame occurs in exactly one bucket, including frames beginning exactly on a boundary.
- No empty token window is created; every nonempty terminal bucket is included.
- A frame whose full interval crosses a nominal boundary remains wholly in its start-PTS bucket. It is never copied into another window.
- `assigned_duration(window)` is the sum of full `frame_duration * time_base` for frames assigned to that window, even across a boundary.
- Window logits are duration-weighted to one video logit, then exactly one multiclass cross-entropy loss is applied to the video label.

No resampling, capping, truncation, interpolation, padding-based length inference, or geometry normalization is allowed.

## 6. Normalization and masks

The implementation fits mean/std separately for `[delta_row_share, row_share]`, using valid native cells from the training files only. The training statistics are reused unchanged for validation and test. Ragged batch padding is represented by explicit masks; transformed padded positions are set to zero afterwards. Valid zero-valued native entries remain valid and must not affect masks.

## 7. Minimal illustrative NPZ creation

This snippet only illustrates the required arrays; it is not a packaging tool:

```python
import numpy as np
np.savez(
    "video_example.npz",
    row_share=np.asarray([[0.4, 0.6], [0.5, 0.5]], dtype=np.float32),
    delta_row_share=np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float32),
    pts=np.asarray([100, 140], dtype=np.int64),
    frame_duration=np.asarray([40, 40], dtype=np.int64),
    time_base=np.asarray([1, 1000], dtype=np.int64),
)
```
