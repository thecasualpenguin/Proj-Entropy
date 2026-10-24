"""Node-backed behavior checks for the generated WPP notebook viewer.

Node is optional: these tests exercise the exact JavaScript emitted by the
renderer with a deliberately tiny DOM/video implementation, not a browser or
an additional runtime dependency.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wpp_notebook_ui import MAX_MODEL_FRAMES, render_explanation_viewer


class _ViewerHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.payload: str | None = None
        self.script: str | None = None
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div":
            self.payload = dict(attrs).get("data-wpp-payload", self.payload)
        if tag == "script":
            self._in_script = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.script = (self.script or "") + data


# The viewer uses only this small subset for setup, controls, and timing.  It
# makes behavior assertions deterministic without claiming browser codec/canvas
# coverage or adding jsdom/playwright.
_NODE_HARNESS = r"""
const fs = require('fs');
const script = fs.readFileSync(process.argv[1], 'utf8');
const payload = JSON.parse(process.argv[2]);
const actions = JSON.parse(process.argv[3]);
const snapshots = [], noop = () => {};
function target(value) {
  const listeners = {};
  return Object.assign(value || {}, {
    addEventListener: (name, fn) => { listeners[name] = fn; },
    dispatchEvent: event => { const fn = listeners[event.type || event]; if (fn) fn(event); },
  });
}
const video = target({currentTime: 0, paused: false, readyState: 1, clientWidth: 100,
  clientHeight: 50, videoWidth: 100, videoHeight: 50,
  pause() { this.paused = true; }});
let clearCalls = 0;
const canvas = {clientWidth: 100, clientHeight: 50, width: 0, height: 0,
  getContext: () => ({setTransform: noop, clearRect: () => { clearCalls++; }, drawImage: noop,
    save: noop, beginPath: noop, rect: noop, clip: noop, restore: noop, globalAlpha: 1})};
const options = [];
const mode = target({value: '', disabled: false, append(option) { options.push(option.value); if (!this.value) this.value = option.value; }});
const opacity = target({value: '.65', disabled: false});
const scrub = target({value: 0, max: 0});
const fields = {
  '.wpp-summary': {innerHTML: ''}, video, canvas,
  '.wpp-error': {classList: {remove: noop}, textContent: ''},
  '.wpp-mode': mode, '.wpp-opacity': opacity, '.wpp-scrub': scrub,
  '.wpp-prev': {}, '.wpp-next': {},
};
const root = {dataset: {wppPayload: JSON.stringify(payload)}, querySelector: selector => fields[selector]};
global.document = {currentScript: {previousElementSibling: root}, createElement: () => ({})};
global.Image = class { constructor() { this.complete = true; this.naturalWidth = 1; this.naturalHeight = 1; } set src(value) { this._src = value; } };
global.cancelAnimationFrame = noop; global.requestAnimationFrame = noop; global.devicePixelRatio = 1; global.Event = function(type) { this.type = type; };
eval(script);
function snapshot() { snapshots.push({time: video.currentTime, paused: video.paused, scrub: Number(scrub.value), scrubMax: Number(scrub.max), clearCalls, mode: mode.value}); }
for (const action of actions) {
  if (action.kind === 'time') { video.currentTime = action.value; video.dispatchEvent({type: 'timeupdate'}); }
  if (action.kind === 'scrub') { scrub.value = action.value; scrub.dispatchEvent({type: 'input'}); }
  if (action.kind === 'mode') { mode.value = action.value; mode.dispatchEvent({type: 'change'}); }
  snapshot();
}
console.log(JSON.stringify({snapshots, options, summaryHTML: fields['.wpp-summary'].innerHTML}));
"""


@unittest.skipUnless(shutil.which("node"), "Node is optional for generated-viewer JS checks")
class WPPNotebookJavaScriptTests(unittest.TestCase):
    def _run(self, source: Any, actions: list[dict[str, Any]]) -> dict[str, Any]:
        parser = _ViewerHTML()
        parser.feed(render_explanation_viewer(source))
        self.assertIsNotNone(parser.payload)
        self.assertIsNotNone(parser.script)
        with tempfile.TemporaryDirectory() as directory:
            script_path = Path(directory) / "generated-viewer.js"
            script_path.write_text(parser.script or "", encoding="utf-8")
            completed = subprocess.run(
                [shutil.which("node") or "node", "-e", _NODE_HARNESS, str(script_path), parser.payload or "{}", json.dumps(actions)],
                check=False, text=True, capture_output=True,
            )
        self.assertEqual(0, completed.returncode, completed.stderr)
        return json.loads(completed.stdout)

    @staticmethod
    def _native_explanation(pts: list[float]) -> Any:
        return SimpleNamespace(
            media=SimpleNamespace(media_url="https://example.test/a.mp4", width=64, height=64,
                                  rotation_degrees=0, pts_seconds=pts,
                                  media_seconds=[value - pts[0] for value in pts],
                                  covered_end_pts_seconds=pts[-1] + 0.05),
            prediction=SimpleNamespace(predicted_label="walk", ground_truth_label="walk", correct=True,
                                       predicted_logit=2.0, predicted_probability=0.8),
            ig_signed=[[1.0, -1.0, 0.5]],
            attention=SimpleNamespace(patch_size=16, rollout_tokens=[[0.25]],
                                      rollout_frames=[[0.25, 0.25, 0.25]],
                                      incoming_tokens=[[[[0.5]]]]),
            display_mode="positive",
        )

    def test_native_nonzero_origin_uses_preceding_vfr_frame_and_scrub_seek(self):
        result = self._run(self._native_explanation([7.0, 7.04, 7.20]), [
            {"kind": "time", "value": 0.0},
            {"kind": "time", "value": 0.10},
            {"kind": "scrub", "value": 2},
            {"kind": "time", "value": 0.26},
        ])
        initial, vfr, seek, end = result["snapshots"]
        self.assertEqual(0, initial["scrub"])
        self.assertEqual(1, vfr["scrub"], "0.10 browser seconds must map to raw PTS 7.10")
        self.assertAlmostEqual(0.20, seek["time"], places=6)
        self.assertTrue(end["paused"])
        self.assertAlmostEqual(0.25, end["time"], places=6)
        self.assertGreater(end["clearCalls"], 0)

    def test_native_negative_pts_origin_is_supported(self):
        result = self._run(self._native_explanation([-2.0, -1.90, -1.70]), [
            {"kind": "time", "value": 0.05},
            {"kind": "scrub", "value": 2},
        ])
        self.assertEqual(0, result["snapshots"][0]["scrub"])
        self.assertAlmostEqual(0.30, result["snapshots"][1]["time"], places=6)

    def test_generated_controls_and_3000_frame_cap(self):
        result = self._run(self._native_explanation([7.0, 7.04, 7.20]), [{"kind": "mode", "value": "attention_rollout"}])
        self.assertTrue({"positive_ig", "signed_ig", "attention_rollout", "attention_layer_1_head_1"}.issubset(result["options"]))
        self.assertEqual("attention_rollout", result["snapshots"][0]["mode"])
        capped = self._run({"video": {"url": "https://example.test/a.mp4"},
                            "timestamps": [index / 100.0 for index in range(MAX_MODEL_FRAMES + 5)],
                            "covered_frame_count": MAX_MODEL_FRAMES + 5, "overlays": {}},
                           [{"kind": "time", "value": 0.0}])
        self.assertEqual(MAX_MODEL_FRAMES - 1, capped["snapshots"][0]["scrubMax"])


if __name__ == "__main__":
    unittest.main()
