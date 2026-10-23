from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.manifest_contract import canonical_video_id, load_source_manifest, sha256_file, write_extraction_index, write_source_manifest
from src.wpp_dataset import WPPDatasetBundle, WPPDatasetError, WPPVideoDataset, collate_wpp_videos, pack_wpp_dataset


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


class WPPDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def reset_root(self):
        self.temp.cleanup(); self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def build_complete_native(self, specs=None):
        # A complete manifest/index/ledger bundle.  The verifier's production
        # boundary is exercised; tests do not mock or bypass it.
        specs = specs or [("a.mp4", "train", "Label A", "g1", np.array([[0, 2, 3, 4], [5, 0, 7, 8]], dtype=np.int64)),
                          ("b.mp4", "validation", "Label A", "g2", np.array([[9, 10]], dtype=np.int64)),
                          ("c.mp4", "test", "Label A", "", np.arange(1, 3002, dtype=np.int64).reshape(1, -1))]
        sources = self.root / "sources"; sources.mkdir()
        source_rows = []
        for name, partition, label, group, values in specs:
            path = sources / name; path.write_bytes(("source-" + name).encode())
            digest = sha256_file(path)
            source_rows.append({"video_id": canonical_video_id(digest), "source_sha256": digest,
                                "source_relative_path": name, "source_bytes": path.stat().st_size,
                                "label": label, "group_id": group, "partition": partition})
        manifest = self.root / "manifest"
        write_source_manifest(manifest, source_rows, dataset_id="synthetic", source_root_ref="sources")
        build = self.root / "build"; (build / "encoded" / "sha").mkdir(parents=True); (build / "artifacts" / "sha").mkdir(parents=True); (build / "logs").mkdir()
        index_rows, ledger_records, logs = [], [], {}
        for source in source_rows:
            values = next(item[4] for item in specs if item[0] == source["source_relative_path"])
            digest = source["source_sha256"]; artifact = build / "artifacts" / "sha" / digest; artifact.mkdir()
            np.savez(artifact / "features.npz", row_bytes=values, display_index=np.arange(values.shape[1], dtype=np.int64))
            (artifact / "metadata.json").write_text(json.dumps({"schema_version": "2.0", "rows": values.shape[0], "frames": values.shape[1]}))
            encoded = build / "encoded" / "sha" / f"{digest}.mp4"; encoded.write_bytes(b"encoded-" + digest.encode())
            log_refs = {}
            for stage in ("encoder", "extractor"):
                rel = f"logs/{digest}.{stage}.log"; path = build / rel; path.write_text(stage)
                logs[rel] = sha256_file(path); log_refs[stage] = {"log_relative_path": rel, "log_sha256": logs[rel]}
            file_hashes = {p.relative_to(artifact).as_posix(): sha256_file(p) for p in sorted(artifact.rglob("*")) if p.is_file()}
            tree = hashlib.sha256(_json_bytes(file_hashes)).hexdigest()
            ledger_records.append({"video_id": source["video_id"], "encoded_sha256": sha256_file(encoded),
                                   "artifact": {"artifact_files_sha256": file_hashes, "artifact_tree_sha256": tree}, **log_refs})
            index_rows.append({"video_id": source["video_id"], "status": "success", "encoded_relative_path": f"sha/{digest}.mp4",
                               "encoded_sha256": sha256_file(encoded), "artifact_relative_path": f"sha/{digest}",
                               "producer_schema_version": "2.0", "row_count": str(values.shape[0]), "frame_count": str(values.shape[1]),
                               "failure_code": "", "failure_message": ""})
        ledger = {"records": ledger_records, "logs": logs}; ledger_path = build / "build_ledger.json"; ledger_path.write_bytes(_json_bytes(ledger))
        index = build / "extraction-index"
        write_extraction_index(index, index_rows, source_manifest_dir=manifest, encoded_root_ref="encoded", artifact_root_ref="artifacts",
                               processing_profile={"name": "synthetic"}, extractor_identity={"name": "synthetic"},
                               additional_metadata={"native_workflow_ledger": {"path_base": "parent_of_extraction_index_directory", "relative_path": "build_ledger.json", "sha256": sha256_file(ledger_path)}})
        return manifest, index

    def test_pack_load_video_blocks_masks_and_time_policy(self):
        manifest, index = self.build_complete_native()
        bundle = self.root / "bundle"; metadata = pack_wpp_dataset(manifest, index, bundle)
        self.assertEqual(metadata["class_mapping"], {"Label A": 0})
        validated = WPPDatasetBundle(bundle, source_manifest=manifest, index=index)
        train, val, test = (validated.partition_view(name) for name in ("train", "validation", "test"))
        self.assertEqual((len(train), len(val), len(test)), (1, 1, 1))
        item = train[0]
        self.assertEqual(item["values"].shape, (2, 3000)); self.assertEqual(item["used_time_length"], 4)
        self.assertEqual(item["values"][0, 0], 0)  # observed zero is retained, not padding
        self.assertTrue(np.all(item["values"][:, 4:] == 0))
        self.assertEqual(test[0]["used_time_length"], 3000); self.assertTrue(test[0]["time_truncated"])
        batch = collate_wpp_videos([item, val[0]])
        self.assertEqual(batch["values"].shape, (2, 2, 3000))
        self.assertTrue(batch["row_mask"].tolist() == [[True, True], [True, False]])
        self.assertTrue(batch["time_mask"][0, :4].all()); self.assertFalse(batch["valid_mask"][1, 1].any())
        self.assertEqual(batch["values"][0, 0, 0], 0)

    def test_tampered_padding_redeclared_checksum_still_rejected(self):
        manifest, index = self.build_complete_native(); bundle = self.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
        data = bundle / "train-data.csv"
        with data.open(newline="", encoding="utf-8") as handle: rows = list(csv.reader(handle))
        rows[1][10] = "9"  # past used_time_length=4
        with data.open("w", newline="", encoding="utf-8") as handle: csv.writer(handle, lineterminator="\n").writerows(rows)
        meta_path = bundle / "datasetmetadata.json"; meta = json.loads(meta_path.read_text()); meta["output_checksums"][data.name] = sha256_file(data)
        meta_path.write_bytes(_json_bytes(meta))
        with self.assertRaisesRegex(WPPDatasetError, "right-padding"):
            WPPVideoDataset(bundle, "train", source_manifest=manifest, index=index)[0]

    def test_packer_rejects_ancestor_and_contract_leaf_symlinks_before_publish(self):
        manifest, index = self.build_complete_native()
        linked_parent = self.root / "linked-parent"; linked_parent.symlink_to(self.root, target_is_directory=True)
        output = self.root / "ancestor-output"
        with self.assertRaisesRegex(WPPDatasetError, "symlink"):
            pack_wpp_dataset(linked_parent / "manifest", linked_parent / "build" / "extraction-index", output)
        self.assertFalse(output.exists())
        for location, filename in (("manifest", "source_manifest.csv"),
                                   ("manifest", "source_manifest.metadata.json"),
                                   ("build/extraction-index", "extraction_index.csv"),
                                   ("build/extraction-index", "extraction_index.metadata.json")):
            self.reset_root(); manifest, index = self.build_complete_native()
            path = self.root / location / filename; real = path.with_name(path.name + ".real")
            path.rename(real); path.symlink_to(real.name)
            output = self.root / "must-not-publish"
            with self.assertRaisesRegex(WPPDatasetError, "symlink"):
                pack_wpp_dataset(manifest, index, output)
            self.assertFalse(output.exists())

    def test_loader_rejects_all_trusted_parent_contract_leaf_symlinks(self):
        for location, filename in (("manifest", "source_manifest.csv"),
                                   ("manifest", "source_manifest.metadata.json"),
                                   ("build/extraction-index", "extraction_index.csv"),
                                   ("build/extraction-index", "extraction_index.metadata.json")):
            manifest, index = self.build_complete_native(); bundle = self.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
            path = self.root / location / filename; real = path.with_name(path.name + ".real")
            path.rename(real); path.symlink_to(real.name)
            with self.assertRaisesRegex(WPPDatasetError, "symlink"):
                WPPDatasetBundle(bundle, source_manifest=manifest, index=index)
            self.reset_root()

    def test_global_label_manifests_reject_redeclared_cross_split_identity(self):
        manifest, index = self.build_complete_native(); bundle = self.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
        train_path, test_path = bundle / "train-label-manifest.csv", bundle / "test-label-manifest.csv"
        with train_path.open(newline="", encoding="utf-8") as handle: train_row = next(csv.DictReader(handle))
        with test_path.open(newline="", encoding="utf-8") as handle:
            fields = csv.DictReader(handle).fieldnames
        forged = dict(train_row); forged["partition"] = "test"; forged["stacked_row_start"] = "0"
        with test_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerow(forged)
        meta_path = bundle / "datasetmetadata.json"; meta = json.loads(meta_path.read_text())
        meta["output_checksums"][test_path.name] = sha256_file(test_path)
        meta["partition_counts"]["test"]["stacked_row_count"] = 2
        meta_path.write_bytes(_json_bytes(meta))
        with self.assertRaisesRegex(WPPDatasetError, "trusted source/index provenance|duplicate"):
            WPPDatasetBundle(bundle, source_manifest=manifest, index=index)

    def test_mapping_and_trusted_parent_context_are_not_self_declared(self):
        manifest, index = self.build_complete_native(); bundle = self.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
        with self.assertRaisesRegex(WPPDatasetError, "required trusted parent"):
            WPPVideoDataset(bundle, "train")
        meta_path = bundle / "datasetmetadata.json"; meta = json.loads(meta_path.read_text())
        meta["class_mapping"] = {"bogus": 0, "Label A": 1}; meta_path.write_bytes(_json_bytes(meta))
        with self.assertRaisesRegex(WPPDatasetError, "class_mapping"):
            WPPDatasetBundle(bundle, source_manifest=manifest, index=index)
        # A syntactically valid but different manifest cannot be substituted for
        # the one to which the extraction index is actually bound.
        records = load_source_manifest(manifest)["records"]
        changed = [dict(row) for row in records]; changed[0]["label"] = "different exact label"
        alternate = self.root / "alternate-manifest"
        write_source_manifest(alternate, changed, dataset_id="synthetic", source_root_ref="sources")
        with self.assertRaisesRegex(WPPDatasetError, "trusted source/index contract"):
            WPPDatasetBundle(bundle, source_manifest=alternate, index=index)

    def test_bundle_symlink_root_and_file_reject(self):
        manifest, index = self.build_complete_native(); bundle = self.root / "bundle"; pack_wpp_dataset(manifest, index, bundle)
        linked = self.root / "linked-bundle"; linked.symlink_to(bundle, target_is_directory=True)
        with self.assertRaisesRegex(WPPDatasetError, "symlink"):
            WPPDatasetBundle(linked, source_manifest=manifest, index=index)
        parent_link = self.root / "linked-parent"; parent_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(WPPDatasetError, "symlink"):
            WPPDatasetBundle(parent_link / "bundle", source_manifest=manifest, index=index)
        data = bundle / "train-data.csv"; actual = bundle / "train-data.real"; data.rename(actual); data.symlink_to(actual.name)
        with self.assertRaisesRegex(WPPDatasetError, "symlink"):
            WPPDatasetBundle(bundle, source_manifest=manifest, index=index)

    def test_group_overlap_rejects(self):
        specs = [("a.mp4", "train", "train-label", "same", np.ones((1, 2), dtype=np.int64)),
                 ("b.mp4", "test", "train-label", "same", np.ones((1, 2), dtype=np.int64))]
        manifest, index = self.build_complete_native(specs)
        with self.assertRaisesRegex(WPPDatasetError, "group_id spans"):
            pack_wpp_dataset(manifest, index, self.root / "bad")

    def test_unknown_evaluation_label_and_output_collision_reject(self):
        specs = [("a.mp4", "train", "train-label", "", np.ones((1, 2), dtype=np.int64)),
                 ("b.mp4", "test", "unseen", "", np.ones((1, 2), dtype=np.int64))]
        manifest, index = self.build_complete_native(specs)
        with self.assertRaisesRegex(WPPDatasetError, "absent from training"):
            pack_wpp_dataset(manifest, index, self.root / "bad")
        # Build a distinct complete fixture in an independent temporary root.
        self.temp.cleanup(); self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        manifest, index = self.build_complete_native()
        output = self.root / "bundle"; pack_wpp_dataset(manifest, index, output)
        with self.assertRaisesRegex(WPPDatasetError, "overwrite"):
            pack_wpp_dataset(manifest, index, output)


if __name__ == "__main__":
    unittest.main()
