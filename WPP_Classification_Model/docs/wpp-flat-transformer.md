# Flat WPP Transformer (model-only API, v1)

`src.wpp_transformer.WPPTransformer` is the native model for the bounded WPP
bundle. It is registered as the sole `wpp_transformer` model in
`src/model_factory.py` and is trained/evaluated by `run_pipeline.py` using
explicit row/time masks from `WPPDatasetBundle`. Legacy CSV architectures are
not registered or supported.

## Input API

The WPP loader and collator remain the source of the masks.  Convert their NumPy
arrays without deriving masks from values:

```python
import torch
from src.wpp_dataset import collate_wpp_videos
from src.wpp_transformer import WPPTransformer

batch = collate_wpp_videos(video_items)
values = torch.from_numpy(batch["values"])       # float32 [B, R, 3000]
row_mask = torch.from_numpy(batch["row_mask"])   # bool    [B, R]
time_mask = torch.from_numpy(batch["time_mask"]) # bool    [B, 3000]
model = WPPTransformer(num_classes=class_count)
logits = model(values, row_mask, time_mask)       # [B, class_count] (raw API)
```

`row_mask` and `time_mask` must be contiguous true prefixes, matching
`collate_wpp_videos`.  `values` has a fixed 3,000-frame axis; valid values must
be finite, nonnegative raw `row_bytes`.  At least one valid row and one valid
frame are required for every video.  Zeros in valid cells are ordinary observed
values.  Invalid cells can contain arbitrary values (including non-finite
padding): they are replaced with zero by `torch.where` before projection, so
multiplication-by-zero NaN propagation cannot occur.

The model itself performs no log transform, clipping, normalization, encoding, or
full-data packing. The runner calls its explicit `normalized=True` mode after
applying the approved float64 population normalization fitted only on valid
training cells (including observed zeros),
reuses frozen statistics for validation/test, and forces masked padding to zero.

## Exact architecture

The defaults preserve the original patch-Transformer style:

- temporal patch size 16, so every WPP row has `ceil(3000/16) = 188` patches;
  the final eight cells are zero padded;
- linear `16 -> 32` patch projection, dropout 0.2;
- two pre-layer-norm `TransformerEncoderLayer`s, four heads, GELU activation,
  feed-forward width 64;
- final layer norm and linear classification head.

Each row is patched separately, so no patch crosses a vertical WPP row boundary.
All `R * 188` row/time patch tokens then form **one** attention sequence per
video.  A token is valid exactly when its row is valid and its temporal patch
contains any valid frame.  Invalid tokens are key-padded and explicitly zeroed
after each layer.  There is no CLS token, per-row classifier, row pooling before
attention, hierarchy, or learned maximum-row table.  Valid tokens are
masked-mean pooled into one video representation.

Positions are generated dynamically as 2-D sinusoidal encodings: the first half
of channels encodes the row number and the second half encodes the temporal
patch number.  Concatenation makes swapped `(row, time)` coordinates distinct
and supports the actual batch row count without a configured R cap.  Consequently
`d_model` must be divisible by four (and by `nhead`).

## Bounded runtime smoke

This is a local FP32 forward/backward smoke only, not a training run or
dataset read.  It uses one batch item and all 3,000 input cells with no token
truncation, optimizer, or update.  GPU allocation/reservation values are
PyTorch peak allocator counters reset after model/input allocation and before
timing.  The selected scaled-dot-product-attention kernel is recorded only when
profiler measurement identifies it; enabled backend flags alone are not proof.

### Reproducible measured run (2026-10-17)

The model was explicitly in `train()` mode, so effective dropout was the default
`0.2`; the loss was `torch.nn.functional.cross_entropy`.  This is still not a
training experiment: it makes exactly one forward/backward pass for each row
count, with no optimizer.  It seeds both CPU and CUDA with `20261017`, uses
`float32` tensors with autocast disabled, and asserts that **every** model
parameter has a present, finite gradient after backward.  CUDA is synchronized
immediately before the timer and immediately after backward.  There is no model
forward/backward warmup: CUDA is initialized by the headroom preflight only, so
R=4 is cold with respect to model/kernel first use and R=17 follows R=4.  The
first timing must therefore not be used as a warm throughput figure.

```sh
cd WPP_Classification_Model
/home/naoromi/miniforge3/envs/entropy/bin/python - <<'PY'
import time
import torch
from src.wpp_transformer import TIME_STEPS, WPPTransformer

SEED, device = 20261017, torch.device("cuda:0")
if not torch.cuda.is_available():
    raise SystemExit("ABORT: CUDA unavailable")
free, total = torch.cuda.mem_get_info(device)
if free < 1024 ** 3:
    raise SystemExit(f"ABORT: suspicious CUDA headroom: {free / 2**20:.2f} MiB free (<1024 MiB)")
torch.cuda.synchronize(device)  # initializes CUDA only; no model warmup
for rows in (4, 17):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    model = WPPTransformer(3).to(device).train()  # dropout=0.2
    values = torch.rand((1, rows, TIME_STEPS), device=device, dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(SEED))
    row_mask = torch.ones((1, rows), device=device, dtype=torch.bool)
    time_mask = torch.ones((1, TIME_STEPS), device=device, dtype=torch.bool)
    target = torch.tensor([1], device=device, dtype=torch.long)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(model(values, row_mask, time_mask), target)
    loss.backward()
    torch.cuda.synchronize(device)
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
               for p in model.parameters())
    print(rows, rows * model.patches_per_row, time.perf_counter() - start,
          torch.cuda.max_memory_allocated(device) / 2**20,
          torch.cuda.max_memory_reserved(device) / 2**20)
PY
```

Result: PyTorch `2.11.0+cu128`, CUDA `12.8`, NVIDIA GeForce RTX 3070; CUDA
preflight reported 7,097.00 MiB free / 8,191.50 MiB total (and aborted below
1 GiB).  FP32 matmul precision was `highest`, CUDA matmul TF32 was disabled,
cudnn TF32 was enabled, and autocast was disabled.  The measurements were:

| rows | tokens | forward/backward elapsed | peak allocated | peak reserved | finite/present parameter gradients |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 752 | 0.974154841 s | 20.482421875 MiB | 26.0 MiB | yes |
| 17 | 3,196 | 0.025741679 s | 30.893554688 MiB | 52.0 MiB | yes |

A preliminary table previously recorded 1.228 s (R=4) and 0.027 s (R=17).
Its recorded invocation is verified as `model.train()` with default dropout
`0.2` and cross-entropy loss, but it did not fix a seed or assert all parameter
gradients; the table above is the reproducible record.  Neither table is a
throughput or training-memory claim.  Flash, efficient, and math SDPA feature
flags were enabled, but a profiler pass did not report a named SDPA kernel, so
the actual attention backend is **not determined** and no Flash claim is made.
R=34 and batches larger than one were intentionally not run; no dataset was
read.
