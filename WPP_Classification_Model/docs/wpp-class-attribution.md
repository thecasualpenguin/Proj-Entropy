# Class-average WPP attribution notebook

`notebooks/wpp_class_attribution.ipynb` displays cache-backed class-average WPP attributions. Notebook support is optional and is not installed by the base runtime. Install it in the intended environment (do not add it to production/model dependencies):

```sh
conda run -n entropy python -m pip install -r requirements-notebook.txt
conda run -n entropy python -m ipykernel install --user --name entropy --display-name "Python (entropy)"
conda run -n entropy jupyter lab
```

`requirements-notebook.txt` explicitly supplies `ipywidgets`, `IPython`, `matplotlib`, kernel execution, and notebook format/client dependencies. If these are absent, constructing the interactive controls raises an actionable request to install that file; headless aggregation/export does not require notebook packages.

The notebook defaults to the seven verified DeepAction labels, ground-truth grouping, **separate train and test** columns, all outcomes, predicted-logit IG as magnitude, per-video unit-total aggregation, shared scales within each method, and no pooled-misclassified extras. That default is 7 classes × 2 splits × 2 methods = **28 maps**. Validation (388 videos) is always excluded.

## Explicit inputs and cache lifecycle

The notebook defaults to the supplied packed bundle, source manifest, extraction index, source-video root, and `best.pt` below `/mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1-training`. Its cache is exactly `class_attributions/` under that training root.

### Windows and WSL local paths

The approved defaults are written in WSL form (`/mnt/d/projects/...`). On native Windows, the notebook/UI and cache-build CLI convert only that `/mnt/<drive>/...` configuration form to the same local drive before resolving it: for this dataset, `/mnt/d/projects/...` becomes `D:\projects\...`. WSL/Linux retain `/mnt/d/projects/...`. Explicit Windows drive paths, UNC paths, relative paths, and other user-supplied locations are left unchanged. This conversion is limited to runtime configuration; it does **not** alter cache `manifest.json`, checkpoint provenance/fingerprints, or move/regenerate a cache.

After updating an already-open Windows notebook, restart its kernel and rerun the initial controls cell. Its exact local variables are:

```python
LOCAL_ROOTS = {name: local_config_path(path) for name, path in DEFAULT_ROOTS.items()}
ui = ClassAttributionUI(training_root=LOCAL_ROOTS["run_dir"], roots=LOCAL_ROOTS,
                        cache_dir=LOCAL_ROOTS["cache_dir"], device="auto",
                        ig_steps=32, attention_chunk_queries=128)
```

The cell prints the configured local cache path; it should start `D:\projects\...` on native Windows (and `/mnt/d/...` in WSL). The controls' initial status repeats that path, so an unreadable cache error can be matched to the intended location. Native Windows can use the existing cache read-only. Full cache building still uses the existing POSIX `fcntl` writer lock, so run the builder under WSL/POSIX rather than concurrently attempting a native-Windows build.

Opening/importing the notebook does not open a model, start GPU work, or build a cache. **Generate/resume full train + test cache** is the only expensive action. It calls `build_class_attribution_cache()` with both splits, `best.pt`, default 32 IG steps, 128 attention query chunk size, and selectable device. The backend writes resumable native per-video results and a fingerprinted `manifest.json`; it always computes predicted IG and computes the extra truth-logit IG only for misclassified videos. Thus IG target is a redraw choice, never a second full-dataset job.

A full session is 1,810 train plus 385 test videos (2,195). Backend aggregation rejects a partial, stale, mixed, or missing cache by default. The notebook makes partial display an explicit preview checkbox and labels cache available/expected/missing counts. It never replaces missing groups or records with invented zeros. Changing controls performs backend cache aggregation only, never model inference.

Cache geometry uses retained presentation-frame index/progress rather than elapsed time. It strictly validates the original-source SHA-256, decoded frame count, complete normalized presentation-PTS timeline, rotation/SAR, and decoded display geometry, but does not require a decoded duration for the final frame. Browser overlays remain stricter: they require that final endpoint duration and never guess an FPS or prior-frame duration. Existing cache-record geometry identity remains the same vertical intervals plus CTU size, so compatible records remain resumable.

## CLI cache build/resume

For a long-running build outside the notebook, run the supplied CLI from this repository. Its defaults are the approved roots above, the training root's `checkpoints/best.pt`, train/test-only scope, `class_attributions/` cache, 32 IG steps, and attention chunk size 128:

```sh
# Approved one-video pilot: still declares the complete 2,195-video expected set.
conda run -n entropy python scripts/build_wpp_class_attribution_cache.py --max-new-videos 1

# Full build, or resume the pilot/interrupted build in the same cache.
conda run -n entropy python scripts/build_wpp_class_attribution_cache.py
```

The cache is per-video resumable; rerun the same command after an interruption. A pilot remains incomplete and should only be viewed through the explicit preview option. The CLI and notebook use the same cache writer lock: do **not** run them simultaneously against one cache. Full attribution duration depends on hardware and source data, so no fixed ETA is implied.

## Single cached-video view

The **Single cached video** selector is a read-only companion to class averages. It starts with an explicit **Refresh cached records** action, listing only backend-validated completed video records with video ID, split, truth/prediction, and correctness. It can read a completed individual record while the overall cache remains partial; missing, publishing, or invalid records are reported as unavailable rather than rendered as zeros. Refreshing, filtering, loading, and downloading this view never opens a model, acquires the writer lock, launches GPU work, or resumes/rebuilds the cache.

Single panels use the native `[WPP row, retained frame]` arrays exactly as cached: x is the 0-based retained presentation-frame index, y is the 0-based WPP row index with row 0 at top. They do **not** coordinate-align, resize, rebin, or average videos. Select predicted-logit (default) or truth-logit IG, absolute IG (default) or signed IG, and existing-explorer display-normalized values (default) or raw values. Display-normalized means division by each native map's finite maximum absolute IG (signed/absolute as applicable) or maximum rollout value, with zeros remaining zeros; it is separate per method and is not the class-average L1 normalization. Raw mode applies no division. Rollout is shown as positive, patch-token relevance repeated across its valid frames for display; use the separately exported native token-mass CSV for mass interpretation, and do not sum repeated frame values as a probability.

Single downloads provide PNG, native IG/frame-rollout/token-mass CSVs, metadata JSON recording target/sign/value mode and separate IG/rollout scales, plus a ZIP bundle. The class-average controls are hidden in this view, including pooled maps.

## Controls and semantics

Class selection supports all/subsets; grouping is ground truth or prediction; splits can be separate or combined and either train/test can be selected. Correctness filters are all/correct/incorrect. The optional pooled-misclassified switch adds two method summaries per displayed split; it is off by default.

IG target is predicted (default) or truth. Every figure title and UI rendering metadata states the target/cohort interpretation explicitly (for example, `Predicted-logit IG within ground-truth class cohort` or `Ground-truth-logit IG within predicted-class cohort`), so group labels cannot imply a fixed class target for errors. IG display is magnitude (default) or signed. Magnitude applies `abs` to native signed IG cells **before** aggregation/rebinning. Per-video unit-total is backend `per_video_normalized`: each map is divided by its native L1 mass before overlap rebinning. It is unrelated to the model's frozen train-population input normalization. Attention rollout is model routing, is not class-specific, and is computed/stored exactly once as native token mass; target/sign controls do not change it or expand it into repeated 16-frame values. Rollout relevance is already unit-total per video, so raw and per-video-unit-total modes are effectively identical for attention; that selector changes IG interpretation, not an attention "strength" or confidence.

The default adjustable display grid is **32 normalized physical WPP vertical bins × 192 normalized retained-presentation-frame progress bins**. The backend call receives canonical `(32 physical band bins, 192 frame-progress bins)`.  Grid changes only re-aggregate native cached values; they do not invalidate or recompute explanations, and display resampling adds no native resolution. The horizontal coordinate is relative retained presentation-frame index/progress (0–100%), not elapsed seconds. The vertical coordinate is physical CTU/WPP-band geometry normalized to 0–100%, including a shorter final partial band. These coordinates align geometry/progress across videos, not semantic scenes/actions.

Signed IG uses a diverging scale centered at zero; magnitude IG and rollout use sequential scales. Scale sharing is allowed within a method only; IG is never scaled with rollout. Empty matching groups are visibly blank (`No matching videos`), never zero heatmaps.

## Downloads and APIs

Each completed redraw provides notebook-local PNG, backend numeric grid CSVs, backend metadata/settings/count JSON, UI target/cohort rendering JSON, and a ZIP containing them. No application server is started. For scripting, `export_rendered_aggregate()` delegates numeric export to `export_class_attributions()` and adds the UI PNG/ZIP.

The UI is intentionally only a renderer/controller over backend dataclasses in `src.wpp_class_attribution`: `CacheBuildResult`, `AttributionGroup`, and `AttributionAggregate`. It calls:

```python
build_class_attribution_cache(training_root, roots=..., splits=("train", "test"),
                              ig_steps=32, attention_chunk_queries=128, device="auto")
aggregate_class_attributions(cache_dir, split=..., group_by=..., correctness=...,
                             classes=..., ig_target=..., sign=...,
                             magnitude="per_video_normalized", grid=(32, 192))
```

No competing UI cache schema or attribution implementation is used.
