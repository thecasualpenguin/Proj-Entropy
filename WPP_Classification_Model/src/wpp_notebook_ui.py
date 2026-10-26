"""Small, dependency-light browser viewer for WPP explanation notebooks.

This module deliberately has no dependency on Jupyter, Torch, Pillow, or a video
server.  ``render_explanation_viewer`` returns self-contained HTML (apart from
the actual video URL) suitable for ``IPython.display.HTML``.  The backend may
supply PNG data URLs directly, or numeric heatmaps which are encoded here using
the standard library.

The stable payload format is documented in :func:`build_viewer_payload`.  It is
also intentionally permissive: explanation objects can expose ``viewer_payload``
or the common fields used below.  This keeps the browser implementation separate
from expensive attribution computation.
"""
from __future__ import annotations

import base64
import html
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

MAX_MODEL_FRAMES = 3000


@dataclass
class ExplorerState:
    """Small testable state holder used by the interactive notebook controls."""

    session: Any = None
    videos: dict[str, Any] = field(default_factory=dict)
    explanation: Any = None
    explanation_video_id: str | None = None
    explanation_partition: str | None = None

    def invalidate_explanation(self) -> None:
        self.explanation = None
        self.explanation_video_id = None
        self.explanation_partition = None

    def set_explanation(self, explanation: Any, *, video_id: str, partition: str) -> None:
        self.explanation = explanation
        self.explanation_video_id = video_id
        self.explanation_partition = partition

    def matches_selection(self, *, video_id: str | None, partition: str) -> bool:
        return (self.explanation is not None and self.explanation_video_id == video_id
                and self.explanation_partition == partition)


def jupyter_files_url(prefix: str, source_relative_path: str) -> str:
    """Safely form a Jupyter ``/files`` URL from server-root-relative paths.

    The caller must ensure the prefix is actually below the configured Jupyter
    server root.  Dot components, absolute paths, and backslashes are rejected
    rather than normalized into a potentially surprising server path.
    """
    def components(value: str, label: str) -> tuple[str, ...]:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError(f"{label} must be a non-empty relative POSIX path")
        path = PurePosixPath(value)
        if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError(f"{label} must be a safe relative POSIX path (no . or ..)")
        return path.parts

    parts = components(prefix.strip(), "/files prefix") + components(source_relative_path, "source_relative_path")
    return "/files/" + "/".join(quote(part, safe="") for part in parts)


def _plain(value: Any) -> Any:
    """Convert lightweight dataclasses/NumPy-like values without importing NumPy."""
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _finite_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _timestamps(values: Any, frame_count: int) -> list[float]:
    """Keep real presentation timestamps, rejecting only unusable values.

    Timestamps are not made uniform: variable-frame-rate media must retain its
    presentation timing.  A missing or invalid tail is conservatively truncated.
    """
    raw = _plain(values) if values is not None else []
    if not isinstance(raw, list):
        raw = []
    answer: list[float] = []
    previous: float | None = None
    for item in raw[:frame_count]:
        current = _finite_number(item, float("nan"))
        # Raw container PTS may legitimately be negative.  The only timeline
        # invariant needed for preceding-frame lookup is strict increase.
        if not math.isfinite(current) or (previous is not None and current <= previous):
            break
        answer.append(current)
        previous = current
    if not answer:
        raise ValueError("viewer payload requires presentation timestamps; do not infer timing from FPS")
    return answer


def _matrix(value: Any) -> list[list[float]] | None:
    """Return a numeric 2-D matrix, accepting a one-row vector."""
    value = _plain(value)
    if not isinstance(value, list) or not value:
        return None
    if not isinstance(value[0], list):
        value = [value]
    rows: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, list) or not row:
            return None
        converted = [_finite_number(cell) for cell in row]
        if width is None:
            width = len(converted)
        if width != len(converted):
            return None
        rows.append(converted)
    return rows


def _png_rgba(width: int, height: int, pixels: bytes) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = b"".join(b"\0" + pixels[row * width * 4 : (row + 1) * width * 4] for row in range(height))
    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    encoded += chunk(b"IDAT", zlib.compress(scanlines, 6)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _colour(value: float, scale: float, signed: bool) -> tuple[int, int, int, int]:
    if scale <= 0 or not math.isfinite(value):
        return (0, 0, 0, 0)
    if signed:
        fraction = max(-1.0, min(1.0, value / scale))
        # Blue is negative, red is positive, white is neutral.
        if fraction >= 0:
            return (255, int(255 * (1 - fraction)), int(255 * (1 - fraction)), int(230 * fraction))
        fraction = -fraction
        return (int(255 * (1 - fraction)), int(255 * (1 - fraction)), 255, int(230 * fraction))
    fraction = max(0.0, min(1.0, value / scale))
    # Transparent black -> yellow/red perceptual-enough browser overlay.
    return (255, int(230 * (1 - fraction)), 0, int(230 * fraction))


def heatmap_data_url(matrix: Any, *, scale: float, signed: bool = False) -> str:
    """Encode a numeric heatmap to an RGBA PNG data URL using a fixed scale."""
    parsed = _matrix(matrix)
    if parsed is None:
        raise ValueError("heatmap must be a non-empty rectangular numeric 2-D matrix")
    height, width = len(parsed), len(parsed[0])
    rgba = bytearray()
    for row in parsed:
        for value in row:
            rgba.extend(_colour(value, scale, signed))
    return _png_rgba(width, height, bytes(rgba))


def _overlay_scale(frames: Sequence[Any], signed: bool) -> float:
    maximum = 0.0
    for frame in frames:
        matrix = _matrix(frame)
        if matrix is None:
            continue
        for row in matrix:
            for item in row:
                maximum = max(maximum, abs(item) if signed else item)
    return maximum


def _normalise_overlay(name: str, spec: Any, covered_frames: int) -> dict[str, Any] | None:
    """Produce the browser overlay schema and independently scale its units."""
    if spec is None:
        return None
    raw = _plain(spec)
    if isinstance(raw, list):
        raw = {"frames": raw}
    if not isinstance(raw, Mapping):
        return None
    frames = raw.get("frames", raw.get("images", raw.get("maps", [])))
    if not isinstance(frames, list):
        return None
    signed = bool(raw.get("signed", name in {"signed_ig", "ig_signed"}))
    stride = max(1, int(_finite_number(raw.get("stride", raw.get("frame_stride", 1)), 1)))
    # Attention maps normally occur one per 16-frame patch.  Do not cap them as
    # though they were full frame maps; their logical coverage is capped instead.
    maximum_items = math.ceil(covered_frames / stride)
    frames = frames[:maximum_items]
    scale = _finite_number(raw.get("scale"), 0.0)
    if scale <= 0:
        scale = _overlay_scale(frames, signed)
    images: list[str] = []
    for frame in frames:
        if isinstance(frame, str):
            images.append(frame)
        elif isinstance(frame, Mapping) and isinstance(frame.get("data_url"), str):
            images.append(str(frame["data_url"]))
        else:
            images.append(heatmap_data_url(frame, scale=scale, signed=signed))
    if not images:
        return None
    strip = raw.get("strip", raw.get("row_strip"))
    strip_images: list[str] = []
    if strip is not None:
        # A strip may be one image for the video, or a frame-aligned sequence.
        strip_items = strip.get("frames", []) if isinstance(strip, Mapping) else strip
        if isinstance(strip_items, str):
            strip_items = [strip_items]
        if isinstance(strip_items, list) and strip_items and (not isinstance(strip_items[0], list) or (strip_items[0] and not isinstance(strip_items[0][0], list))):
            # A vector or numeric 2-D matrix is one strip, not a frame sequence.
            strip_items = [strip_items]
        if isinstance(strip_items, list):
            for item in strip_items[:maximum_items]:
                if isinstance(item, str):
                    strip_images.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("data_url"), str):
                    strip_images.append(str(item["data_url"]))
                else:
                    strip_images.append(heatmap_data_url(item, scale=scale, signed=signed))
    return {
        "label": str(raw.get("label", name.replace("_", " "))),
        "unit": str(raw.get("unit", "attention" if "attention" in name else "IG")),
        "frames": images,
        "strip_images": strip_images,
        "stride": stride,
        "scale": scale,
        "signed": signed,
        "row_bands": bool(raw.get("row_bands", False)),
    }


def _column_frames(matrix: Any) -> list[list[list[float]]]:
    """Turn [WPP row, time/patch] values into one vertical band map per time."""
    parsed = _matrix(matrix)
    if parsed is None:
        return []
    return [[[row[index]] for row in parsed] for index in range(len(parsed[0]))]


def _positive(matrix: Any) -> list[list[float]]:
    parsed = _matrix(matrix)
    return [[max(0.0, item) for item in row] for row in parsed] if parsed is not None else []


def _native_explanation_payload(explanation: Any) -> dict[str, Any] | None:
    """Project the public ``VideoExplanation`` type without importing NumPy/Torch."""
    media = _field(explanation, "media")
    prediction = _field(explanation, "prediction")
    attention = _field(explanation, "attention")
    signed = _field(explanation, "ig_signed")
    if media is None or prediction is None or attention is None or signed is None:
        return None
    raw_pts = _plain(_field(media, "pts_seconds", default=[]))
    media_times = _plain(_field(media, "media_seconds", default=[]))
    if not isinstance(raw_pts, list) or not isinstance(media_times, list) or not raw_pts or len(raw_pts) != len(media_times):
        return None
    signed_matrix = _matrix(signed)
    if signed_matrix is None:
        return None
    patch_size = max(1, int(_finite_number(_field(attention, "patch_size", default=16), 16)))
    overlays: dict[str, Any] = {
        "positive_ig": {"label": "positive IG", "unit": "predicted-logit IG", "frames": _column_frames(_positive(signed_matrix)),
                          "strip": _positive(signed_matrix), "row_bands": True},
        "signed_ig": {"label": "signed IG", "unit": "predicted-logit IG", "frames": _column_frames(signed_matrix),
                      "strip": signed_matrix, "signed": True, "row_bands": True},
        # Native rollout is already one value per patch.  Keeping stride avoids
        # materializing 16 duplicate browser images for every patch.
        "attention_rollout": {"label": "attention rollout", "unit": "attention routing", "frames": _column_frames(_field(attention, "rollout_tokens")),
                              "strip": _field(attention, "rollout_frames"), "stride": patch_size, "row_bands": True},
    }
    incoming = _plain(_field(attention, "incoming_tokens", default=[]))
    if isinstance(incoming, list):
        for layer, heads in enumerate(incoming):
            if not isinstance(heads, list):
                continue
            for head, tokens in enumerate(heads):
                overlays["attention_layer_%d_head_%d" % (layer + 1, head + 1)] = {
                    "label": "incoming attention L%d H%d" % (layer + 1, head + 1),
                    "unit": "attention routing", "frames": _column_frames(tokens), "stride": patch_size, "row_bands": True,
                }
    return {
        "video": {"url": _field(media, "media_url"),
                  "display_aspect_ratio": _finite_number(_field(media, "width", default=0)) / max(1, _finite_number(_field(media, "height", default=1))),
                  "source_height": _field(media, "height", default=0),
                  "rotation": _field(media, "rotation_degrees", default=0)},
        # Preserve raw source PTS, while the explicit offset maps browser time
        # zero to the first presentation timestamp.
        "timestamps": raw_pts,
        "media_time_offset": _finite_number(raw_pts[0]),
        "covered_end": _field(media, "covered_end_pts_seconds", default=raw_pts[-1]),
        "covered_frame_count": min(len(raw_pts), len(signed_matrix[0])),
        "overlays": overlays,
        "initial_mode": _field(explanation, "display_mode", default="positive"),
        "summary": {
            "predicted_label": _field(prediction, "predicted_label"),
            "ground_truth_label": _field(prediction, "ground_truth_label"),
            "correct": _field(prediction, "correct"),
            "logit": _field(prediction, "predicted_logit"),
            "probability": _field(prediction, "predicted_probability"),
            "baseline": "train-population mean (normalized zero)",
        },
    }


def build_viewer_payload(explanation: Any, *, video_url: str | None = None) -> dict[str, Any]:
    """Create JSON-safe viewer data from an explanation or an explicit payload.

    Preferred backend interchange schema::

        {"video": {"url": "https://...", "display_aspect_ratio": 1.777},
         "timestamps": [presentation_seconds...], "covered_frame_count": 3000,
         "overlays": {"positive_ig": {"frames": [2d arrays or PNG data URLs],
                       "stride": 1, "unit": "IG"},
                      "attention_rollout": {"frames": [...], "stride": 16,
                       "unit": "attention"}},
         "summary": {"predicted_label": ..., "ground_truth_label": ...}}

    Overlay scales are calculated per overlay and are never shared between IG
    and attention.  At most the first 3000 real/model-valid frames are exposed.
    """
    native_payload = _native_explanation_payload(explanation)
    source = native_payload if native_payload is not None else _field(explanation, "viewer_payload", default=explanation)
    source = _plain(source)
    if not isinstance(source, Mapping):
        raise TypeError("explanation must be a mapping, dataclass, or expose viewer_payload")
    video = _plain(source.get("video", {}))
    if not isinstance(video, Mapping):
        video = {}
    url = video_url or video.get("url") or _field(explanation, "video_url", "media_url", "source_url")
    if not url:
        raise ValueError("a browser-reachable video_url is required (file:// is not supported by default)")
    timestamps_source = source.get("timestamps", source.get("frame_timestamps", _field(explanation, "frame_timestamps", "timestamps", default=[])))
    # Browser media time often begins at zero whereas demuxed PTS may not.  Keep
    # the raw timestamps and pass the explicitly supplied offset to JavaScript.
    # Never fabricate an FPS/origin from a frame index.
    media_time_offset = _finite_number(source.get("media_time_offset", video.get("media_time_offset", 0.0)), 0.0)
    requested = _finite_number(source.get("covered_frame_count", source.get("used_time_length", _field(explanation, "used_time_length", "frame_count", default=MAX_MODEL_FRAMES))), MAX_MODEL_FRAMES)
    covered = max(1, min(MAX_MODEL_FRAMES, int(requested)))
    timestamps = _timestamps(timestamps_source, covered)
    covered = min(covered, len(timestamps))
    overlays = source.get("overlays", {})
    if not isinstance(overlays, Mapping):
        overlays = {}
    aliases = {
        "positive_ig": ("positive_ig", "ig_positive", "positive"),
        "signed_ig": ("signed_ig", "ig_signed", "signed"),
        "attention_rollout": ("attention_rollout", "rollout", "attention"),
    }
    result_overlays: dict[str, Any] = {}
    for canonical, names in aliases.items():
        item = next((overlays.get(name) for name in names if name in overlays), None)
        if item is None:
            item = _field(explanation, *names)
        normalised = _normalise_overlay(canonical, item, covered)
        if normalised:
            result_overlays[canonical] = normalised
    # Retain already-named layer/head maps as independent browser modes.
    consumed = {alias for names in aliases.values() for alias in names}
    for key, item in overlays.items():
        key = str(key)
        if key in consumed or key == "attention_heads":
            continue
        normalised = _normalise_overlay(key, item, covered)
        if normalised:
            result_overlays[key] = normalised
    # Per-layer/head attention is deliberately left as separate selectable modes.
    layer_heads = overlays.get("attention_heads", _field(explanation, "attention_heads", "per_layer_head_attention"))
    if isinstance(layer_heads, Mapping):
        for key, item in layer_heads.items():
            normalised = _normalise_overlay("attention_" + str(key), item, covered)
            if normalised:
                result_overlays["attention_" + str(key)] = normalised
    summary = _plain(source.get("summary", {}))
    if not isinstance(summary, Mapping):
        summary = {}
    for target, alternatives in {
        "predicted_label": ("predicted_label", "prediction", "predicted_class"),
        "ground_truth_label": ("ground_truth_label", "true_label", "label"),
        "correct": ("correct", "is_correct"),
        "logit": ("logit", "predicted_logit"),
        "probability": ("probability", "predicted_probability"),
        "baseline": ("baseline", "baseline_description"),
    }.items():
        if target not in summary:
            candidate = _field(explanation, *alternatives)
            if candidate is not None:
                summary[target] = _plain(candidate)
    return {
        "video": {
            "url": str(url),
            "display_aspect_ratio": _finite_number(video.get("display_aspect_ratio", video.get("aspect_ratio")), 0.0),
            "source_height": _finite_number(video.get("source_height", video.get("height", 0)), 0.0),
            "rotation": int(_finite_number(video.get("rotation", 0))),
        },
        "timestamps": timestamps[:covered],
        "covered_frame_count": covered,
        # A backend may know the final frame's presentation end; use it rather
        # than displaying the final attribution over the rest of a longer video.
        "covered_end": _finite_number(source.get("covered_end", source.get("covered_end_timestamp", timestamps[covered - 1] if covered else 0.0)), timestamps[covered - 1] if covered else 0.0),
        "media_time_offset": media_time_offset,
        "overlays": result_overlays,
        "initial_mode": str(source.get("initial_mode", source.get("display_mode", _field(explanation, "display_mode", default="positive")))),
        "summary": summary,
    }


_HTML = r'''<div class="wpp-viewer" data-wpp-payload="__PAYLOAD__">
  <style>
    .wpp-viewer{font:14px system-ui,sans-serif;max-width:960px;color:#202124}.wpp-summary{margin:.4rem 0}.wpp-stage{position:relative;background:#111;line-height:0;width:100%;max-height:70vh;overflow:hidden}.wpp-stage video{display:block;width:100%;max-height:70vh;object-fit:contain}.wpp-stage canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.wpp-controls{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.5rem 0}.wpp-controls input[type=range]{min-width:13rem}.wpp-error{color:#b00020;line-height:1.4}.wpp-caption{font-size:12px;color:#555;line-height:1.4}.wpp-hidden{display:none}
  </style>
  <div class="wpp-summary"></div><div class="wpp-stage"><video controls preload="metadata" playsinline></video><canvas></canvas></div>
  <div class="wpp-controls"><label>Overlay <select class="wpp-mode"></select></label><label>Opacity <input class="wpp-opacity" type="range" min="0" max="1" step=".05" value=".65"></label><button class="wpp-prev" type="button">Frame −</button><button class="wpp-next" type="button">Frame +</button><input class="wpp-scrub" type="range" min="0" step="1"></div>
  <div class="wpp-caption">Overlays stop at the model-covered endpoint. IG is relative to the train-mean baseline; attention is not class-specific. Attention maps have patch (typically 16-frame) resolution.</div><div class="wpp-error wpp-hidden"></div>
</div>
<script>(function(){
const root=document.currentScript.previousElementSibling, data=JSON.parse(root.dataset.wppPayload); const $=s=>root.querySelector(s), video=$('video'), canvas=$('canvas'), ctx=canvas.getContext('2d'), error=$('.wpp-error'), mode=$('.wpp-mode'), opacity=$('.wpp-opacity'), scrub=$('.wpp-scrub');
const names=Object.keys(data.overlays), images={}; let raf=0;
function esc(x){return String(x===undefined||x===null?'':x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
$('.wpp-summary').innerHTML=['predicted_label','ground_truth_label','correct','logit','probability','baseline'].filter(k=>data.summary[k]!==undefined).map(k=>'<b>'+esc(k.replaceAll('_',' '))+':</b> '+esc(data.summary[k])).join(' &nbsp; ');
names.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=data.overlays[n].label+' ('+data.overlays[n].unit+', scale '+Number(data.overlays[n].scale||0).toPrecision(4)+')';mode.append(o);images[n]=data.overlays[n].frames.map(src=>{const i=new Image();i.src=src;return i;});images[n+'_strip']=(data.overlays[n].strip_images||[]).map(src=>{const i=new Image();i.src=src;return i;});});
if(names.includes(data.initial_mode))mode.value=data.initial_mode;if(!names.length){mode.disabled=true;opacity.disabled=true;}
scrub.max=Math.max(0,data.covered_frame_count-1); function indexAt(t){let a=data.timestamps,lo=0,hi=a.length;while(lo<hi){let m=(lo+hi)>>1;if(a[m]<=t)lo=m+1;else hi=m;}return Math.max(0,lo-1)}
function box(){const w=video.clientWidth,h=video.clientHeight,ratio=data.video.display_aspect_ratio||((video.videoWidth||16)/(video.videoHeight||9));let bw=w,bh=w/ratio;if(bh>h){bh=h;bw=h*ratio}return{x:(w-bw)/2,y:(h-bh)/2,w:bw,h:bh}}
function draw(t){cancelAnimationFrame(raf); if(video.readyState<1)return; const dpr=devicePixelRatio||1,w=video.clientWidth,h=video.clientHeight;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const n=mode.value,o=data.overlays[n];if(o){const index=Math.min(images[n].length-1,Math.floor(indexAt(t)/o.stride)),img=images[n][index],b=box();ctx.globalAlpha=Number(opacity.value);if(img&&img.complete){if(o.row_bands&&data.video.source_height){const band=64*b.h/data.video.source_height;ctx.save();ctx.beginPath();ctx.rect(b.x,b.y,b.w,b.h);ctx.clip();for(let row=0;row<img.naturalHeight;row++)ctx.drawImage(img,0,row,img.naturalWidth,1,b.x,b.y+row*band,b.w,band);ctx.restore();}else ctx.drawImage(img,b.x,b.y,b.w,b.h);}const strips=images[n+'_strip'],strip=strips&&strips[Math.min(strips.length-1,index)];if(strip&&strip.complete){const sh=Math.min(64,b.h);ctx.save();ctx.beginPath();ctx.rect(b.x,b.y+b.h-sh,b.w,sh);ctx.clip();ctx.drawImage(strip,b.x,b.y+b.h-sh,b.w,sh);ctx.restore();}ctx.globalAlpha=1;}scrub.value=indexAt(t);}
function rawTime(t){return t+Number(data.media_time_offset||0)} function browserTime(t){return Math.max(0,t-Number(data.media_time_offset||0))}
function update(){const raw=rawTime(video.currentTime);if(raw>=data.covered_end){if(!video.paused){video.pause();video.currentTime=browserTime(data.covered_end)}ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);return;}draw(raw)}
function schedule(){update();if(!video.paused)raf=requestAnimationFrame(schedule)}
video.addEventListener('loadedmetadata',schedule);video.addEventListener('play',schedule);video.addEventListener('pause',update);video.addEventListener('seeked',update);video.addEventListener('timeupdate',update);if(video.requestVideoFrameCallback){(function vf(){video.requestVideoFrameCallback((_,m)=>{updateAt(rawTime(m.mediaTime));vf()})})()}function updateAt(t){if(t>=data.covered_end){update();return;}if(!video.paused)draw(t)}
video.addEventListener('error',()=>{error.classList.remove('wpp-hidden');error.textContent='The browser could not decode or reach this video URL. Use a Jupyter /files/ URL or an explicit browser-reachable URL; check codec support. Do not use file://. If conversion is necessary, do it explicitly and preserve VFR timestamps.'});
video.src=data.video.url;
mode.addEventListener('change',update);opacity.addEventListener('input',update);scrub.addEventListener('input',()=>{video.currentTime=browserTime(data.timestamps[Number(scrub.value)]||0);update()});$('.wpp-prev').onclick=()=>{scrub.value=Math.max(0,Number(scrub.value)-1);scrub.dispatchEvent(new Event('input'))};$('.wpp-next').onclick=()=>{scrub.value=Math.min(Number(scrub.max),Number(scrub.value)+1);scrub.dispatchEvent(new Event('input'))};
})();</script>'''


def render_explanation_viewer(explanation: Any, *, video_url: str | None = None) -> str:
    """Return safe standalone HTML for a responsive, client-side explanation viewer."""
    payload = build_viewer_payload(explanation, video_url=video_url)
    # JSON in an attribute must be both HTML-escaped and script-safe; no user
    # supplied label/URL can terminate the template or inject markup.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _HTML.replace("__PAYLOAD__", html.escape(encoded, quote=True))


def render_explanation_iframe(explanation: Any, *, video_url: str | None = None) -> str:
    """Return a widget-safe iframe wrapper for the standalone explanation viewer.

    The sandbox permits scripts and same-origin video access; it is transport
    isolation for notebook output, not an untrusted-code security boundary.
    """
    srcdoc = html.escape(render_explanation_viewer(explanation, video_url=video_url), quote=True)
    return ('<iframe class="wpp-viewer-frame" title="WPP explanation viewer" '
            'sandbox="allow-scripts allow-same-origin" srcdoc="' + srcdoc + '" '
            'style="display:block;width:100%;max-width:960px;height:760px;border:0"></iframe>')


__all__ = ["ExplorerState", "MAX_MODEL_FRAMES", "build_viewer_payload", "heatmap_data_url", "jupyter_files_url", "render_explanation_iframe", "render_explanation_viewer"]
