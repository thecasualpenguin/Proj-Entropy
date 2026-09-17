# CASIA-B video preparation

Run from this directory:

```sh
python run.py --dry-run
python run.py
# Or call the converter directly:
python data.py --source /path/to/CASIA-B --output /path/to/CASIA-B-video
```

The repository launcher also supports `uv run sandbox run entropy-gait`.
Defaults are `datasets/casia-b` and `datasets/casia-b-video` under the repository
root (or `SANDBOX_ROOT`). Requires macOS and FFmpeg with the `hevc_videotoolbox` encoder;
no additional Python packages are needed. Encoding uses the Apple Silicon media
engine on the M3 Pro through VideoToolbox. Software fallback is disabled
(`-allow_sw 0`), so hardware initialization failure stops the conversion. PNG
decoding and padding still run on the CPU.

Each `001/bg-01/000/*.png` sequence becomes `001/bg-01/000.mp4`, with a
`000.json` sidecar recording its labels, source frame order, frame count, and
encoding settings. Subjects, sequence types/repetitions, and views stay separate.
Frames are sorted by their numeric filename suffix; gaps are compacted without
duplicating frames. Every available image contributes one frame at 30 fps.

Defaults: hardware H.265/HEVC, MP4, 4 Mbps target bitrate (lossy), no audio. Images are
not resized; odd dimensions are padded to even dimensions for YUV 4:2:0 encoding.
Use `--fps` or `--bitrate-mbps` to adjust encoding. Existing videos are skipped;
use `--overwrite` to regenerate them, including when changing encoding settings.
The original PNG dataset is retained. Interrupted encodes do not leave a partial
destination MP4.

To regenerate videos previously encoded with the software encoder:

```sh
python run.py --overwrite
```

Extract compressed frame sizes and build the training matrix from existing videos:

```sh
python run.py --extract
# Rebuild only the matrix from saved covers:
python run.py --collate
# Inspect options / preview input discovery:
python run.py --extract --help
python run.py --extract --dry-run
```

Extraction requires `ffprobe`; collation uses NumPy and extraction progress uses
`tqdm` (available in the repository environment). From this directory you can use
`../../.venv/bin/python` in place of `python`.

Defaults read `datasets/casia-b-video` and write `outputs/covers` and
`outputs/matrix` inside this experiment. Override with `--source`, `--covers`,
`--output`, `--workers`, or `--frames` (default 3000). Existing covers are reused;
pass `--overwrite` after changing the source videos.

Extraction accepts video datasets rooted at either
`subject/sequence/view.mp4` (the legacy converter output) or
`subject/condition-sequence/view-angle/video.mp4`, such as
`casiasilhouette/004/nm-02/090/090.mp4` and
`casiabrgb/011/nm-02/162/162.mp4`. Covers mirror the source hierarchy, so the
latter becomes `subject/condition-sequence/view-angle/video.csv`. You may also
point `--source` at a parent containing `casiasilhouette` and `casiabrgb`; that
leading dataset directory is retained in cover paths but is not part of the label.
Legacy `subject/sequence/view.mp4` exports beneath either named dataset root are
also labeled by their subject, sequence, and view rather than by the root name.
In all cases, labels remain subject IDs; `examples.csv` records the condition
sequence and view angle from their directory names.

Each cover contains headerless `source video, zero-based packet index, size in
bits` rows. Sizes come directly from compressed video packets, including codec
headers, in coding order; no decoding, re-encoding, or time aggregation is
performed. Both silhouette and RGB inputs are handled as videos through this same
packet-size extraction path. For these HEVC MP4s, each video packet represents a
compressed frame.

Each matrix row is one video: **n examples × 3000 frame sizes**. Short videos are
zero-padded on the right by default; longer videos use their first 3000 packets.
`--short-policy error|pad|repeat|drop` controls short-video handling. No videos
are concatenated. Outputs:

- `X.npy` / `X.csv`: integer frame sizes in bits, with zero padding.
- `Y.npy` / `Y.csv`: aligned subject IDs (NumPy preserves IDs like `001` as strings).
- `examples.csv`: row index, subject, sequence, view, cover/video paths, original
frame count and valid frame count, allowing padding masks and later splits.
- `summary.json`: shape, units, short-video policy and counts.

CSV files have headers; `X.csv` has no index column. Load `examples.csv` and
`Y.csv` with string subject dtype to retain leading zeros. Examples are sorted
by relative cover path. No train/test split is imposed.

## Training

`train.py` adapts the runner from `entropy-with-transformer`. ResNet remains the
model default. Both ResNet and transformer support optional length-aware training.

Compare padded and length-aware inputs using identical optimizer settings:

```sh
../../.venv/bin/python train.py --model transformer --epochs 200 --no-length-aware --seed 42
../../.venv/bin/python train.py --model transformer --epochs 200 --length-aware --seed 42
# Use --model resnet for the same comparison with the CNN.
```

Both commands use AdamW with learning rate 0.001 and weight decay 0.01. Set
`--weight-decay 0` to disable decay, or use `--optimizer adam --weight-decay 0`
for the previous optimizer recipe. Batch size remains 32; the epoch default is 50.
Transformer patch size defaults to 16 and can be changed with `--patch-size`.
Use `--dropout 0.2` to set dropout (valid range: `0 <= p < 1`; `0` disables it).
Defaults preserve the current settings: transformer 0.2, ResNet 0.0. Transformer
uses the rate for embedding and encoder dropout; ResNet applies it after pooling,
before the classifier. The selected rate is recorded in run metadata and checkpoints.

```sh
../../.venv/bin/python train.py --model transformer --length-aware --dropout 0.2
```
CUDA, then MPS, then CPU is selected automatically.

Inputs default to `outputs/matrix/X.npy` and `Y.npy`; use `--data-dir` to select
another matrix. Length-aware mode also requires the matching `examples.csv`.
The current matrix contains 4,018 videos, 3,000 features, and 37 subject classes.
The saved matrix is not modified.

### Length handling

`--no-length-aware` is the default: retain the previous per-column training
mean/std normalization, including padding, and process every position.

`--length-aware` uses `original_frames`, capped at the stored matrix width and
checked against `valid_frames`. It verifies manifest row/subject alignment and
zero padding. It then:

- Fits one global mean/std over **real training frames only**, preserving relative
size information across clips. Validation uses those same training statistics.
A global scale avoids unreliable estimates at sparsely occupied tail positions.
- Normalizes real values and sets padding back to zero.
- Trims each batch to its longest real sequence inside the model.
- For the transformer, masks fully padded patches as attention keys/values and
excludes them from final pooling. Partially valid patches retain their real
samples, zero-fill missing samples, and count as one valid token in pooling.
- For ResNet, excludes padding from BatchNorm statistics, resets invalid positions
between convolutions, and averages only real output positions. A training batch
with just one real frame uses running BatchNorm statistics for that step.

This comparison changes both normalization and model padding handling; it tests
the combined length-aware pipeline, not attention masking in isolation.

### Splits and repeatability

Each subject's videos are shuffled with split seed 42, with about 10% held out
for validation. This is video-level subject classification with the same identities
in both splits, not a held-out-subject or official CASIA-B evaluation protocol.
`--seed` controls model initialization and training shuffle (default 42), while
the split stays fixed. Matching seeds help comparisons, but do not guarantee
bit-identical execution across devices or identical dropout draws across modes.

### Checkpoints and artifacts

Every run creates `runs/YYYY-MM-DD-NNN/` (override the root with `--runs-dir`).
There are three checkpoint files:

- `best_validation.pt`: highest validation accuracy so far.
- `best_train.pt`: highest training accuracy, evaluated over the full training
split in evaluation mode after the epoch.
- `last.pt`: most recently completed epoch; the final epoch after a normal run.

Ties keep the earliest best epoch. All three include model and optimizer state,
epoch metrics, class labels, normalization statistics, length mode, optimizer
settings and seed. Saves replace files atomically. Checkpoints and `loss.json`
are updated after each completed epoch; `loss-accuracy.png` is saved at the end.
There is no automatic early stopping: the requested epoch count still runs.

`run.log`, `split.npz`, `data.json`, and a copy of `examples.csv` with a `split`
column preserve run settings and row provenance. Older `checkpoint.pt` files
remain loadable. For inference, apply saved normalization and pass real lengths
when the checkpoint used length-aware training:

```python
from pathlib import Path
import torch
from train import load_checkpoint, load_training_data, load_lengths, normalize_inputs

model, checkpoint = load_checkpoint(Path("runs/YOUR-RUN/best_validation.pt"))
data_dir = Path("outputs/matrix")
x, labels = load_training_data(data_dir)
lengths = load_lengths(data_dir, x, labels) if checkpoint["config"].get("length_aware", False) else None
x = normalize_inputs(x, checkpoint["mean"], checkpoint["std"], lengths)
model.eval()
with torch.no_grad():
    # Small inference batch; repeat over remaining rows as needed.
    args = [torch.from_numpy(lengths[:32])] if lengths is not None else []
    logits = model(torch.from_numpy(x[:32]), *args)
```

Validation: `../../.venv/bin/python -m unittest discover -s tests -v` checks
padding invariance, normalization, gradients, manifest errors, and both models
in both modes on a small real-data subset, including checkpoint reload/selection.