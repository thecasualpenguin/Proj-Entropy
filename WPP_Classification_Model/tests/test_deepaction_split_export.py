import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deepaction_export", ROOT / "export_deepaction_splits.py")
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DeepActionSplitExportTest(unittest.TestCase):
    def make_manifest(self, root):
        rows = []
        # Every prompt has both labels, plus an extra Pexels take: source counts are imbalanced.
        for prompt_id in ("0", "1", "2"):
            for take, source in enumerate(("Pexels", "BDAnimateDiffLightning", "Pexels")):
                video_id = f"video-{prompt_id}-{take}"
                feature_dir = root / "features" / video_id
                feature_dir.mkdir(parents=True)
                np.savez(feature_dir / "features.npz",
                    row_share=np.array([[.2, .8], [.3, .7]], np.float32),
                    delta_row_share=np.array([[0., 0.], [.1, -.1]], np.float32),
                    pts=np.array([10, 20], np.int64), extra=np.array([take], np.int16))
                encoded = root / "encoded" / f"{video_id}.mp4"; encoded.parent.mkdir(exist_ok=True)
                encoded.write_bytes(f"encoded-{video_id}".encode())
                original = root / "original" / f"{video_id}.mp4"; original.parent.mkdir(exist_ok=True)
                original.write_bytes(f"original-{video_id}".encode())
                metadata = {"source": str(encoded), "deepaction_wpp": {"video_id": video_id, "source": source,
                    "prompt_id": prompt_id, "prompt_identity": prompt_id, "original_source": str(original)}}
                (feature_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf8")
                (feature_dir / "rows.csv").write_text("pts\n10\n20\n", encoding="utf8")
                rows.append({"video_id": video_id, "source": source, "prompt_id": prompt_id, "prompt_text": f"Prompt {prompt_id}",
                    "prompt_identity": prompt_id, "feature_path": str(feature_dir), "encoded_path": str(encoded), "original_path": str(original),
                    "status": "complete", "features_fingerprint": digest(feature_dir / "features.npz"),
                    "metadata_fingerprint": digest(feature_dir / "metadata.json"), "rows_csv_fingerprint": digest(feature_dir / "rows.csv"),
                    "encoded_fingerprint": digest(encoded), "source_fingerprint": digest(original)})
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return manifest, rows

    @staticmethod
    def timing(encoded, _ffprobe, _video_id):
        return np.array([10, 20], np.int64), np.array([10, 10], np.int64), np.array([1, 1000], np.int64)

    def export_fixture(self, root):
        manifest, rows = self.make_manifest(root)
        with patch.object(exporter, "packet_timing", self.timing):
            result = exporter.export_splits(manifest, root, root.parent / "exported", expected_video_count=len(rows),
                expected_source_counts=None, expected_prompt_ids=None)
        return result, root.parent / "exported", rows

    def test_adapter_equivalent_deterministic_prompt_assignment(self):
        prompts = [str(value) for value in range(95)] + [str(value) for value in range(99, 104)]
        actual = exporter.prompt_assignments(prompts, 42)
        self.assertEqual(Counter(actual.values()), {"train": 70, "val": 15, "test": 15})
        self.assertEqual(actual, exporter.prompt_assignments(reversed(prompts), 42))
        self.assertEqual((actual["0"], actual["100"]), ("val", "test"))

    def test_copies_legacy_arrays_adds_authenticated_timing_and_uses_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            result, output, rows = self.export_fixture(Path(temp) / "source")
            self.assertEqual(result["video_count"], 9)
            self.assertEqual(result["source_counts"], {"Pexels": 6, "BDAnimateDiffLightning": 3})
            ids, labels, prompts = set(), [], {}
            for split in ("train", "val", "test"):
                with (output / f"{split}_data.csv").open(newline="", encoding="utf8") as handle: data = list(csv.DictReader(handle))
                with (output / f"{split}_labels.csv").open(newline="", encoding="utf8") as handle: split_labels = list(csv.DictReader(handle))
                self.assertEqual({r["video_id"] for r in data}, {r["video_id"] for r in split_labels})
                for row in data:
                    self.assertFalse(Path(row["features_path"]).is_absolute())
                    copied = output / row["features_path"]
                    self.assertTrue(copied.is_file())
                    source = Path(next(r for r in rows if r["video_id"] == row["video_id"])["feature_path"]) / "features.npz"
                    with np.load(source, allow_pickle=False) as old, np.load(copied, allow_pickle=False) as new:
                        self.assertEqual(set(new.files), set(old.files) | {"frame_duration", "time_base"})
                        for name in old.files: self.assertEqual(new[name].dtype, old[name].dtype); self.assertTrue(np.array_equal(new[name], old[name]))
                        self.assertTrue(np.array_equal(new["pts"], [10, 20]))
                        self.assertTrue(np.array_equal(new["frame_duration"], [10, 10]))
                        self.assertTrue(np.array_equal(new["time_base"], [1, 1000]))
                    prompt = row["video_id"].split("-")[1]
                    self.assertEqual(prompts.setdefault(prompt, split), split)
                    ids.add(row["video_id"])
                labels.extend(r["label"] for r in split_labels)
            self.assertEqual(ids, {r["video_id"] for r in rows})
            self.assertEqual(Counter(labels), {"Pexels": 6, "BDAnimateDiffLightning": 3})
            self.assertTrue((output / "split_audit_manifest.csv").is_file())

    def test_packet_mismatch_or_invalid_duration_leaves_no_published_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"; manifest, rows = self.make_manifest(root); output = root.parent / "exported"
            def mismatch(*_): return np.array([11, 20], np.int64), np.array([10, 10], np.int64), np.array([1, 1000], np.int64)
            with patch.object(exporter, "packet_timing", mismatch), self.assertRaisesRegex(ValueError, "do not exactly match"):
                exporter.export_splits(manifest, root, output, expected_video_count=len(rows), expected_source_counts=None, expected_prompt_ids=None)
            self.assertFalse(os.path.lexists(output))
            def bad_duration(*_): return np.array([10, 20], np.int64), np.array([0, 10], np.int64), np.array([1, 1000], np.int64)
            with patch.object(exporter, "packet_timing", bad_duration), self.assertRaisesRegex(ValueError, "positive"):
                exporter.export_splits(manifest, root, output, expected_video_count=len(rows), expected_source_counts=None, expected_prompt_ids=None)
            self.assertFalse(os.path.lexists(output))

    def test_metadata_source_mismatch_and_output_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"; manifest, rows = self.make_manifest(root)
            meta = Path(rows[0]["feature_path"]) / "metadata.json"
            payload = json.loads(meta.read_text()); payload["deepaction_wpp"]["source"] = "permuted"
            meta.write_text(json.dumps(payload), encoding="utf8"); rows[0]["metadata_fingerprint"] = digest(meta)
            with manifest.open("w", newline="", encoding="utf8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "provenance disagrees"):
                exporter.read_and_validate_manifest(manifest, root, expected_video_count=len(rows), expected_source_counts=None, expected_prompt_ids=None)
            dangling = root.parent / "dangling-output"; os.symlink(root.parent / "missing", dangling)
            with self.assertRaisesRegex(ValueError, "existing output"):
                exporter.ensure_safe_fresh_output(root, dangling)
            existing = root.parent / "existing-output"; existing.mkdir()
            with self.assertRaisesRegex(ValueError, "existing output"):
                exporter.ensure_safe_fresh_output(root, existing)
            with self.assertRaisesRegex(ValueError, "overlap"):
                exporter.ensure_safe_fresh_output(root, root / "output")

    def test_ffprobe_rejects_missing_packet_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); encoded = root / "encoded.mp4"; encoded.write_bytes(b"not-read")
            probe = root / "fake_ffprobe.py"
            probe.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'streams':[{'time_base':'1/1000'}], 'packets':[{'pts':'10'}]}))\n")
            probe.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "missing integer pts/duration"):
                exporter.packet_timing(encoded, str(probe), "video")

    def test_original_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"; manifest, rows = self.make_manifest(root)
            Path(rows[0]["original_path"]).write_bytes(b"changed-after-manifest")
            with self.assertRaisesRegex(ValueError, "source_fingerprint mismatch"):
                exporter.read_and_validate_manifest(manifest, root, expected_video_count=len(rows), expected_source_counts=None, expected_prompt_ids=None)

    def test_no_clobber_publication_fails_closed_without_renameat2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); staging = root / "staging"; final = root / "final"; staging.mkdir()
            class NoRenameAt2: pass
            with patch.object(exporter.ctypes, "CDLL", return_value=NoRenameAt2()):
                with self.assertRaisesRegex(RuntimeError, "no-clobber publication"):
                    exporter._publish_no_clobber(staging, final)
            self.assertTrue(staging.is_dir()); self.assertFalse(final.exists())

    def test_full_inventory_shape_and_split_counts_without_large_fixture(self):
        prompts = [str(value) for value in range(95)] + [str(value) for value in range(99, 104)]
        sources = [source for source, count in exporter.EXPECTED_SOURCE_COUNTS.items() for _ in range(count)]
        records = [exporter.Record(f"v{i}", source, prompts[i % len(prompts)], Path("f"), Path("m"), Path("r"), Path("e"), Path("o"), {})
                   for i, source in enumerate(sources)]
        exporter.validate_inventory(records, 2605, exporter.EXPECTED_SOURCE_COUNTS, set(prompts))
        self.assertEqual(Counter(exporter.prompt_assignments(prompts).values()), {"train": 70, "val": 15, "test": 15})


if __name__ == "__main__": unittest.main()
