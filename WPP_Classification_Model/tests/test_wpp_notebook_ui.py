import base64
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from types import SimpleNamespace

from src.wpp_notebook_ui import (ExplorerState, MAX_MODEL_FRAMES, build_viewer_payload,
                                  heatmap_data_url, jupyter_files_url, render_explanation_iframe,
                                  render_explanation_viewer)


class WPPNotebookUITests(unittest.TestCase):
    def payload(self, **changes):
        result = {
            "video": {"url": "/files/media/example.mp4", "display_aspect_ratio": 16 / 9},
            "timestamps": [10.0 + index * 0.031 for index in range(4)],
            "covered_frame_count": 4,
            "overlays": {
                "positive_ig": {"frames": [[[1.0, 2.0]], [[3.0, 4.0]]]},
                "attention_rollout": {"frames": [[[0.25]]], "stride": 16},
            },
        }
        result.update(changes)
        return result

    def test_caps_frames_keeps_raw_pts_and_preserves_attention_stride(self):
        timestamps = [100.25 + index * 0.017 for index in range(MAX_MODEL_FRAMES + 50)]
        output = build_viewer_payload(
            self.payload(timestamps=timestamps, covered_frame_count=len(timestamps), media_time_offset=100.25)
        )
        self.assertEqual(MAX_MODEL_FRAMES, output["covered_frame_count"])
        self.assertEqual(timestamps[:MAX_MODEL_FRAMES], output["timestamps"])
        self.assertEqual(100.25, output["media_time_offset"])
        self.assertEqual(16, output["overlays"]["attention_rollout"]["stride"])
        self.assertEqual(1, len(output["overlays"]["attention_rollout"]["frames"]))

    def test_explorer_state_invalidates_stale_explanations_across_selection_transitions(self):
        state = ExplorerState(session=object(), videos={"first": object(), "second": object()})
        first_explanation = object()
        state.set_explanation(first_explanation, video_id="first", partition="test")
        self.assertTrue(state.matches_selection(video_id="first", partition="test"))
        self.assertFalse(state.matches_selection(video_id="second", partition="test"))
        self.assertFalse(state.matches_selection(video_id="first", partition="validation"))
        state.invalidate_explanation()  # called before open/compute and on video/partition changes
        self.assertIsNone(state.explanation)
        self.assertFalse(state.matches_selection(video_id="first", partition="test"))
        state.set_explanation(object(), video_id="second", partition="validation")
        self.assertTrue(state.matches_selection(video_id="second", partition="validation"))

    def test_jupyter_files_url_rejects_escape_and_quotes_each_component(self):
        self.assertEqual("/files/originals/a%20b%23.mp4", jupyter_files_url("originals", "a b#.mp4"))
        for prefix in (".", "../other", "/absolute", "a/../other", "a\\other"):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "safe relative|relative POSIX"):
                jupyter_files_url(prefix, "clip.mp4")
        with self.assertRaises(ValueError):
            jupyter_files_url("originals", "../clip.mp4")

    def test_overlay_scales_are_separate_and_numeric_frames_become_pngs(self):
        output = build_viewer_payload(self.payload())
        positive = output["overlays"]["positive_ig"]
        attention = output["overlays"]["attention_rollout"]
        self.assertEqual(4.0, positive["scale"])
        self.assertEqual(0.25, attention["scale"])
        encoded = positive["frames"][0].split(",", 1)[1]
        self.assertEqual(b"\x89PNG\r\n\x1a\n", base64.b64decode(encoded)[:8])

    def test_public_explanation_projection_preserves_pts_and_patch_attention(self):
        explanation = SimpleNamespace(
            media=SimpleNamespace(media_url="https://example.test/a.mp4", width=1920, height=1080,
                                  rotation_degrees=0, pts_seconds=[7.0, 7.04], media_seconds=[0.0, 0.04],
                                  covered_end_pts_seconds=7.08),
            prediction=SimpleNamespace(predicted_label="walk", ground_truth_label="walk", correct=True,
                                       predicted_logit=3.0, predicted_probability=0.9),
            ig_signed=[[1.0, -2.0], [3.0, -4.0]],
            attention=SimpleNamespace(patch_size=16, rollout_tokens=[[0.1], [0.2]],
                                      rollout_frames=[[0.1, 0.1], [0.2, 0.2]],
                                      incoming_tokens=[[[[0.3], [0.4]]]]),
        )
        output = build_viewer_payload(explanation)
        self.assertEqual([7.0, 7.04], output["timestamps"])
        self.assertEqual(7.0, output["media_time_offset"])
        self.assertEqual(7.08, output["covered_end"])
        self.assertEqual(16, output["overlays"]["attention_rollout"]["stride"])
        self.assertIn("attention_layer_1_head_1", output["overlays"])
        self.assertEqual(2, len(output["overlays"]["positive_ig"]["frames"]))

    def test_negative_raw_pts_are_valid_with_an_explicit_origin(self):
        output = build_viewer_payload(self.payload(timestamps=[-2.0, -1.96], media_time_offset=-2.0))
        self.assertEqual([-2.0, -1.96], output["timestamps"])
        self.assertEqual(-2.0, output["media_time_offset"])
        self.assertEqual(-1.96, output["covered_end"])

    @unittest.skipUnless(shutil.which("node"), "Node is required to execute emitted JavaScript timing helpers")
    def test_emitted_javascript_maps_nonzero_pts_origin_and_covered_endpoint(self):
        """Execute the timing functions emitted in the HTML, not a Python copy."""
        page = render_explanation_viewer(self.payload(timestamps=[7.0, 7.04, 7.09], media_time_offset=7.0,
                                                    covered_end=7.14))
        script = page.rsplit("<script>", 1)[1].rsplit("</script>", 1)[0]
        index_start = script.index("function indexAt")
        index_end = script.index("\nfunction box", index_start)
        time_start = script.index("function rawTime")
        time_end = script.index("\nfunction update", time_start)
        update_start = time_end + 1
        update_end = script.index("\nfunction schedule", update_start)
        helpers = script[index_start:index_end] + "\n" + script[time_start:time_end] + "\n" + script[update_start:update_end]
        program = "const data=" + json.dumps({"timestamps": [7.0, 7.04, 7.09], "media_time_offset": 7.0, "covered_end": 7.14}) + ";\n" + helpers + "\nlet cleared=false, drawn=false; const video={currentTime:browserTime(data.covered_end),paused:false,pause(){this.paused=true}}; const canvas={clientWidth:4,clientHeight:3}; const ctx={clearRect(){cleared=true}}; function draw(){drawn=true}; update(); console.log(JSON.stringify({atZero:rawTime(0), first:indexAt(rawTime(0)), beforeTransition:indexAt(rawTime(.039)), atTransition:indexAt(rawTime(.04)), seekSecond:browserTime(data.timestamps[1]), endpoint:browserTime(data.covered_end), endpointPaused:video.paused, endpointCleared:cleared, endpointDrawn:drawn}));"
        result = subprocess.run([shutil.which("node"), "-e", program], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(7, observed["atZero"])
        self.assertEqual(0, observed["first"])
        self.assertEqual(0, observed["beforeTransition"])
        self.assertEqual(1, observed["atTransition"])
        self.assertAlmostEqual(0.04, observed["seekSecond"])
        self.assertAlmostEqual(0.14, observed["endpoint"])
        self.assertTrue(observed["endpointPaused"])
        self.assertTrue(observed["endpointCleared"])
        self.assertFalse(observed["endpointDrawn"])
        self.assertIn("if(raw>=data.covered_end)", script)

    def test_requires_actual_presentation_timestamps(self):
        payload = self.payload(timestamps=[])
        with self.assertRaisesRegex(ValueError, "presentation timestamps"):
            build_viewer_payload(payload)

    def test_renderer_is_script_and_html_safe(self):
        attack = '</script><img src=x onerror=alert(1)>'
        payload = self.payload(video={"url": "/files/" + attack})
        payload["summary"] = {"predicted_label": attack}
        page = render_explanation_viewer(payload)
        self.assertNotIn(attack, page)
        self.assertIn("\\u003c/script\\u003e", page)
        self.assertIn("requestVideoFrameCallback", page)
        self.assertIn("indexAt", page)  # preceding-frame/floor lookup is client-side

    def test_iframe_escapes_srcdoc_and_round_trips_standalone_viewer(self):
        payload = self.payload(video={"url": '/files/</iframe><img src=x onerror=alert(1)>'})
        standalone = render_explanation_viewer(payload)
        wrapped = render_explanation_iframe(payload)

        class IframeParser(HTMLParser):
            attrs: dict[str, str | None] = {}

            def handle_starttag(self, tag, attrs):
                if tag == "iframe":
                    self.attrs = dict(attrs)

        parser = IframeParser()
        parser.feed(wrapped)
        self.assertEqual(standalone, parser.attrs["srcdoc"])
        self.assertEqual("allow-scripts allow-same-origin", parser.attrs["sandbox"])
        self.assertEqual("WPP explanation viewer", parser.attrs["title"])
        self.assertNotIn("</iframe><img", wrapped)

    def test_notebook_is_valid_and_code_cells_parse(self):
        notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "wpp_explorer.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(4, notebook["nbformat"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "wpp_explorer.ipynb", "exec")

    def test_notebook_renders_viewer_in_embedded_output_widget(self):
        notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "wpp_explorer.ipynb").read_text(encoding="utf-8"))
        setup, callbacks = ("".join(notebook["cells"][index]["source"]) for index in (4, 5))
        self.assertIn("viewer_output = widgets.Output()", setup)
        self.assertIn("status, viewer_output]))", setup)
        self.assertIn("viewer_output.clear_output(wait=True)", setup)
        self.assertIn("with viewer_output:\n            viewer_output.clear_output(wait=True)\n            display(widgets.HTML", callbacks)


if __name__ == "__main__":
    unittest.main()
