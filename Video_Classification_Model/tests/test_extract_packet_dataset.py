from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract_packet_dataset as app


class PacketDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input"

    def tearDown(self):
        self.temp.cleanup()

    def video(self, class_label: str, group: str, name: str = "clip.MP4") -> Path:
        path = self.input / class_label / group / "nested" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not inspected in these tests")
        return path

    def test_discovery_hierarchy_case_and_literal_names(self):
        self.video("Class A", "Group 1")
        (self.input / "ignore.txt").write_text("not a video")
        videos = app.discover_videos(self.input)
        self.assertEqual([(v.class_label, v.group, v.source) for v in videos],
                         [("Class A", "Group 1", "Class A/Group 1/nested/clip.MP4")])
        bad = self.input / "orphan.avi"; bad.write_bytes(b"")
        with self.assertRaisesRegex(app.ExtractionError, "lacks required"):
            app.discover_videos(self.input)

    def test_group_splits_are_deterministic_complete_and_nonleaking(self):
        videos = [self.video("literal Label", f"g{i}", f"{i}.mKv") for i in range(25)]
        discovered = app.discover_videos(self.input)
        first = app.assign_splits(discovered, (.68, .16, .16), 42)
        second = app.assign_splits(discovered, (.68, .16, .16), 42)
        self.assertEqual(first, second)
        self.assertEqual([list(first.values()).count(split) for split in app.SPLITS], [17, 4, 4])
        group_splits = {}
        for video, split in first.items():
            group_splits.setdefault(video.group, set()).add(split)
        self.assertTrue(all(len(value) == 1 for value in group_splits.values()))
        self.assertEqual(set(first.values()), set(app.SPLITS))
        self.assertEqual(len(videos), 25)

    def test_default_stream_and_packet_bits(self):
        metadata = '{"streams":[{"codec_type":"video","disposition":{"default":0}}, {"codec_type":"video","disposition":{"default":1}}]}'
        with mock.patch.object(app, "run_ffprobe", side_effect=[metadata, "2\n7\n"]) as probe:
            self.assertEqual(app.packet_sizes_bits(Path("sample.mp4")), [16, 56])
        self.assertIn("v:1", probe.call_args_list[1].args[0])
        no_default = '{"streams":[{"codec_type":"audio"}, {"codec_type":"video","disposition":{"default":0}}]}'
        with mock.patch.object(app, "run_ffprobe", side_effect=[no_default, "3\n"]) as probe:
            self.assertEqual(app.packet_sizes_bits(Path("sample.mp4")), [24])
        self.assertIn("v:0", probe.call_args_list[1].args[0])
        with mock.patch.object(app, "run_ffprobe", side_effect=['{"streams":[] }']):
            with self.assertRaisesRegex(app.ExtractionError, "no video stream"):
                app.packet_sizes_bits(Path("sample.mp4"))

    def test_default_output_is_script_relative_and_output_file_is_rejected(self):
        args = app.parse_args(["--input-dir", "input with spaces"])
        self.assertEqual(args.output_dir, ROOT / "extracted_frames")
        output_file = self.root / "not-a-directory"
        output_file.write_text("occupied")
        with self.assertRaisesRegex(app.ExtractionError, "not a directory"):
            app.build_dataset(self.input, output_file, (.68, .16, .16), 42, False)

    def test_outputs_padding_truncation_manifest_and_collision_protection(self):
        for group in ("g1", "g2", "g3"):
            self.video("literal", group, f"{group}.avi")
        output = self.root / "results"
        with mock.patch.object(app, "packet_sizes_bits", side_effect=[[8, 16], list(range(3002)), []]):
            self.assertEqual(app.build_dataset(self.input, output, (.68, .16, .16), 42, False), 3)
        with (output / "manifest.csv").open(newline="") as handle:
            manifest = list(csv.DictReader(handle))
        self.assertEqual(set(row["split"] for row in manifest), set(app.SPLITS))
        self.assertEqual([row["original_packet_count"] for row in manifest], ["2", "3002", "0"])
        for split in app.SPLITS:
            with (output / split / f"{split}-data.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0][0], "size0"); self.assertEqual(rows[0][-1], "size2999")
            self.assertEqual(len(rows[0]), 3000); self.assertEqual(len(rows[1]), 3000)
        with self.assertRaisesRegex(app.ExtractionError, "managed output already exists"):
            app.build_dataset(self.input, output, (.68, .16, .16), 42, False)
        stale = output / "train" / "stale.csv"
        stale.write_text("stale")
        with mock.patch.object(app, "packet_sizes_bits", return_value=[8]):
            self.assertEqual(app.build_dataset(self.input, output, (.68, .16, .16), 42, True), 3)
        self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
