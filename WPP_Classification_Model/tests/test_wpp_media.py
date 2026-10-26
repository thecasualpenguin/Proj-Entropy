from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.native_wpp_workflow import PresentationProbe
from src.wpp_media import (MediaError, _ffprobe_display_metadata, resolve_source_video,
                           source_media_timing, wpp_row_boxes)


class WPPMediaTests(unittest.TestCase):
    def test_ffprobe_supports_modern_and_legacy_frame_durations(self):
        cases = [
            ({"duration": 7}, Fraction(7, 100)),
            ({"pkt_duration": 9}, Fraction(9, 100)),
            ({"duration": 7, "pkt_duration": 9}, Fraction(7, 100)),
            ({"duration": 0, "pkt_duration": 9}, Fraction(9, 100)),
            ({"duration": "N/A", "pkt_duration": 9}, Fraction(9, 100)),
            ({"duration": None, "pkt_duration": 9}, Fraction(9, 100)),
            ({"duration": -1}, None),
            ({"duration": 0}, None),
            ({}, None),
        ]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                payload = {"streams": [{"time_base": "1/100", "sample_aspect_ratio": "1:1"}],
                           "frames": [{"media_type": "video", "best_effort_timestamp": 558, **fields}]}
                with mock.patch("src.wpp_media.shutil.which", return_value="ffprobe"), \
                     mock.patch("src.wpp_media.subprocess.run", return_value=mock.Mock(stdout=json.dumps(payload))) as run:
                    rotation, sar, durations = _ffprobe_display_metadata(Path("source.mp4"))
                self.assertEqual((rotation, sar), (0, "1:1"))
                self.assertEqual(durations, {} if expected is None else {Fraction(558, 100): expected})
                entries = run.call_args.args[0][run.call_args.args[0].index("-show_entries") + 1]
                self.assertIn("best_effort_timestamp,duration,pkt_duration", entries)

    def test_source_timing_keeps_pts_origin_and_next_frame_coverage_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "class" / "a.mp4"; source.parent.mkdir(); source.write_bytes(b"source")
            artifact = root / "artifact"; artifact.mkdir()
            (artifact / "metadata.json").write_text(json.dumps({"time_base": "1/10"}), encoding="utf-8")
            digest = hashlib.sha256(b"source").hexdigest()
            times = (Fraction(10, 10), Fraction(13, 10), Fraction(19, 10))
            probe = PresentationProbe(128, 130, 128, 130, "1:1", Fraction(1, 10), times)
            with mock.patch("src.wpp_media._artifact_presentation_times", return_value=(times, 3)), \
                 mock.patch("src.wpp_media.probe_presentation", return_value=(probe, {})), \
                 mock.patch("src.wpp_media._ffprobe_display_metadata", return_value=(0, "1:1", {times[-1]: Fraction(3, 10)})):
                timing = source_media_timing(source_path=source, expected_sha256=digest, artifact_dir=artifact,
                                             expected_frames=3, retained_frames=2, media_url="/files/a.mp4")
            self.assertEqual(timing.pts_seconds.tolist(), [1.0, 1.3])
            self.assertEqual(timing.media_seconds.tolist(), [0.0, 0.3])
            self.assertAlmostEqual(timing.covered_end_pts_seconds, 1.9)
            self.assertAlmostEqual(timing.covered_end_media_seconds, 0.9)
            self.assertEqual(timing.media_url, "/files/a.mp4")

    def test_3000_frame_cutoff_uses_next_vfr_timestamp_as_coverage_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "a.mp4"; source.write_bytes(b"source")
            artifact = root / "artifact"; artifact.mkdir()
            (artifact / "metadata.json").write_text(json.dumps({"time_base": "1/1000"}), encoding="utf-8")
            # Irregular VFR timeline: the endpoint must be frame 3000, not an FPS estimate.
            times = tuple(Fraction(index * index + 7, 1000) for index in range(3001))
            probe = PresentationProbe(64, 64, 64, 64, "1:1", Fraction(1, 1000), times)
            with mock.patch("src.wpp_media._artifact_presentation_times", return_value=(times, 3001)), \
                 mock.patch("src.wpp_media.probe_presentation", return_value=(probe, {})), \
                 mock.patch("src.wpp_media._ffprobe_display_metadata", return_value=(0, "1:1", {})):
                timing = source_media_timing(source_path=source, expected_sha256=hashlib.sha256(b"source").hexdigest(),
                                             artifact_dir=artifact, expected_frames=3001, retained_frames=3000)
            self.assertEqual(len(timing.pts_seconds), 3000)
            self.assertAlmostEqual(timing.covered_end_pts_seconds, float(times[3000]))
            self.assertAlmostEqual(timing.covered_end_media_seconds, float(times[3000] - times[0]))

    def test_full_vfr_video_uses_last_decoded_duration_not_previous_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "a.mp4"; source.write_bytes(b"source")
            artifact = root / "artifact"; artifact.mkdir()
            (artifact / "metadata.json").write_text(json.dumps({"time_base": "1/100"}), encoding="utf-8")
            digest = hashlib.sha256(b"source").hexdigest()
            # A nonzero origin and irregular gaps make an FPS or prior-gap guess wrong.
            times = (Fraction(500, 100), Fraction(519, 100), Fraction(558, 100))
            probe = PresentationProbe(64, 64, 64, 64, "1:1", Fraction(1, 100), times)
            with mock.patch("src.wpp_media._artifact_presentation_times", return_value=(times, 3)), \
                 mock.patch("src.wpp_media.probe_presentation", return_value=(probe, {})), \
                 mock.patch("src.wpp_media._ffprobe_display_metadata", return_value=(0, "1:1", {times[-1]: Fraction(7, 100)})):
                timing = source_media_timing(source_path=source, expected_sha256=digest, artifact_dir=artifact,
                                             expected_frames=3, retained_frames=3)
            self.assertEqual(timing.pts_seconds.tolist(), [5.0, 5.19, 5.58])
            self.assertAlmostEqual(timing.covered_end_pts_seconds, 5.65)
            self.assertAlmostEqual(timing.covered_end_media_seconds, 0.65)

    def test_negative_pts_preserve_raw_origin_and_media_relative_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "a.mp4"; source.write_bytes(b"source")
            artifact = root / "artifact"; artifact.mkdir()
            (artifact / "metadata.json").write_text(json.dumps({"time_base": "1/10"}), encoding="utf-8")
            times = (Fraction(-1, 10), Fraction(1, 10))
            probe = PresentationProbe(64, 64, 64, 64, "1:1", Fraction(1, 10), times)
            with mock.patch("src.wpp_media._artifact_presentation_times", return_value=(times, 2)), \
                 mock.patch("src.wpp_media.probe_presentation", return_value=(probe, {})), \
                 mock.patch("src.wpp_media._ffprobe_display_metadata", return_value=(0, "1:1", {times[-1]: Fraction(1, 10)})):
                timing = source_media_timing(source_path=source, expected_sha256=hashlib.sha256(b"source").hexdigest(),
                                             artifact_dir=artifact, expected_frames=2, retained_frames=2)
            self.assertEqual(timing.pts_seconds.tolist(), [-0.1, 0.1])
            self.assertAlmostEqual(timing.covered_end_pts_seconds, 0.2)
            self.assertAlmostEqual(timing.covered_end_media_seconds, 0.3)

    def test_row_boxes_clip_final_row_and_root_cannot_escape(self):
        self.assertTrue(np.array_equal(wpp_row_boxes(rows=3, frame_height=130), np.array([[0, 64], [64, 64], [128, 2]])))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "a.mp4").write_bytes(b"a")
            self.assertEqual(resolve_source_video(root, "a.mp4"), (root / "a.mp4").resolve())
            with self.assertRaises(MediaError):
                resolve_source_video(root, "../outside.mp4")
            with self.assertRaises(MediaError):
                wpp_row_boxes(rows=3, frame_height=128)


if __name__ == "__main__":
    unittest.main()
