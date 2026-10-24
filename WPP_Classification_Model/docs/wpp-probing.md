# WPP decision probing in a notebook

`notebooks/wpp_explorer.ipynb` is an interactive reader for an already-trained
native WPP run. It does **not** retrain, rewrite a checkpoint, copy media, or
modify a dataset. It opens a selected checkpoint only after the **Open** button
is pressed, lists the requested partition, and computes a selected video's
integrated gradients (IG) and attention only after **Compute explanation** is
pressed.

## Start it

Install the small notebook extra in the environment which already has the
project's normal requirements:

```sh
python -m pip install -r requirements-notebook.txt
jupyter lab  # start from a directory which intentionally contains any media served by /files
```

Open the notebook and use its controls:

1. Supply the four explicit `DataRoots` JSON fields: `packed_bundle`,
   `source_manifest`, `index`, and `source_video_root`. `source_manifest` and
   `index` are published **directories**, not individual JSON files.
   Roots/checkpoints are deliberately explicit; the default checkpoint selector
   is `best` and the partition selector defaults to `test`.
2. Choose device (`auto`, CUDA, MPS, or CPU), run directory, then press
   **Open**. The video dropdown is populated without computing attribution.
3. Select a video and press **Compute explanation**. IG steps and attention
   query chunking are editable before this operation. Progress and exceptions
   appear in the status area.
4. Set either a direct browser URL for that one video or a `/files` prefix,
   then press **Show viewer**. Changing overlay mode, opacity, scrub position,
   playback, or frame step is native browser work: it does not invoke Python
   or recompute IG/attention.
5. Choose a new, empty output directory and press **Export figures** to call
   the backend exporter. The notebook refuses an existing/nonempty output path
   before it asks the backend to write it.

The decision summary shows the whole-video prediction, ground truth, correctness,
logit and probability. IG uses the frozen train-population mean baseline.
Attention describes model routing and is **not class-specific**.

## Media URLs are intentional

A notebook output runs in a browser, not in the Python kernel. It cannot
reliably load arbitrary `file://` paths (and must not try to work around that
browser security boundary). The UI therefore never invents a `file://` URL and
never copies, re-encodes, or embeds a source video.

For a local Jupyter server, `/files/relative/path.mp4` works only when the file
is below the server's configured root directory. Put the relative directory
under that root in the notebook's **/files media prefix** field. For example,
with a server root `/data` and a source `/data/originals/a/b.mp4`, enter
`originals`; a source-relative path `a/b.mp4` resolves to
`/files/originals/a/b.mp4`. The control rejects absolute prefixes, backslashes,
and `.`/`..` components rather than normalizing a path outside this mapping.

For a source root outside the Jupyter server root, or a remote Jupyter server,
serve the media with an authenticated/reverse-proxy URL and paste that explicit
browser-reachable URL into **Video URL override**. Do not expose a broad local
file server to an untrusted network merely for this viewer. A browser decode or
network error is shown in the player with this guidance.

If a browser does not support the source codec, convert a *separate explicit
copy* yourself and point the override at it. Do not silently convert originals.
For FFmpeg, preserve variable-frame-rate presentation timestamps (avoid forcing
an FPS); inspect both inputs and result with `ffprobe`. A typical starting point
is `ffmpeg -i input -map 0 -c:v libx264 -vsync vfr -c:a aac output.mp4`, but
validate it for the source/container and your browser before using it.

## Timing, coverage, and overlays

The viewer consumes actual per-frame presentation timestamps (PTS), performs a
preceding-frame/floor lookup, and accepts an explicit media-time offset where
demuxed PTS do not start at browser time zero. The payload convention is
`raw_pts = browser_media_seconds + media_time_offset`, so the first browser
frame at time zero maps exactly to the first raw PTS (including a negative one).
It never guesses a frame rate.
It caps model display to the first 3,000 real, valid frames and clears/stops the
overlay at the supplied final covered-frame end even if the source video is
longer.

The canvas is fitted to the HTML video content box, rather than the surrounding
player rectangle. Backend display-aspect metadata is used to accommodate sample
aspect ratio and rotation safely. Positive IG, signed IG, attention rollout,
and each selectable layer/head attention use independently fixed per-video
scales: no IG scale is shared with attention units. Row-strip visualizations are
clipped to 64 pixels at the bottom of the content box. Attention is retained at
its native patch resolution (usually one map per 16 frames), not expanded into
thousands of duplicate images.

## Programmatic viewer contract

`src.wpp_notebook_ui.py` has no Jupyter/Torch/Pillow dependency. Its
`render_explanation_viewer()` accepts an explanation object exposing
`viewer_payload` or an explicit mapping with `video.url`, raw PTS
`timestamps`, `covered_frame_count`, optional `media_time_offset` and
`covered_end`, `summary`, and `overlays`. Overlay frames may be PNG data URLs
or numeric 2-D maps. Numeric overlays are turned into PNGs with a separate
fixed scale per overlay; attention maps can declare `stride: 16`.

The backend's `export_explanation()` remains responsible for durable PNG/SVG
figures, numeric arrays, and metadata. The HTML viewer is a responsive playback
UI, not an alternate export format.
