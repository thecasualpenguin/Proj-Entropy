from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_manifest_contract import main as manifest_main
from src.deepaction_manifest import (
    AUDIT_FIELDS, DEEPACTION_LABELS, EXCLUSION_REPORT_FILE, EXCLUSION_FIELDS,
    build_deepaction_manifest_inputs, build_deepaction_records, inspect_deepaction_audit,
)
from src.manifest_contract import (
    ContractError, INDEX_FIELDS, SOURCE_FIELDS, canonical_video_id, load_extraction_index,
    load_source_manifest, records_from_discovery, sha256_file, validate_extraction_index_records,
    validate_source_records, write_extraction_index, write_source_manifest,
)


class ManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source_record(self, rel: str, content: bytes, label: str = "label"):
        source = self.root / "originals" / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        digest = sha256_file(source)
        return {"video_id": canonical_video_id(digest), "source_sha256": digest,
                "source_relative_path": rel, "source_bytes": source.stat().st_size, "label": label}

    def test_deterministic_sha_identity_paths_and_atomic_manifest(self):
        first = self.source_record("z/clip.mp4", b"z")
        second = self.source_record("a/clip.mp4", b"a")
        records = validate_source_records([first, second])
        self.assertEqual([row["source_relative_path"] for row in records], ["a/clip.mp4", "z/clip.mp4"])
        self.assertEqual(first["video_id"], "sha256:" + hashlib.sha256(b"z").hexdigest())
        output = self.root / "manifest"
        result = write_source_manifest(output, [first, second], dataset_id="tiny", source_root_ref="originals")
        self.assertEqual(result["metadata"]["record_count"], 2)
        self.assertEqual(load_source_manifest(output)["records"], records)
        with self.assertRaisesRegex(ContractError, "overwrite"):
            write_source_manifest(output, [first, second], dataset_id="tiny", source_root_ref="originals")
        bad = dict(first); bad["source_relative_path"] = "../escape.mp4"
        with self.assertRaisesRegex(ContractError, "safe"):
            validate_source_records([bad])

    def test_cooperating_publication_lock_fails_closed_then_releases(self):
        record = self.source_record("one.mp4", b"one")
        output = self.root / "manifest"
        lock = self.root / ".manifest.publish.lock"
        lock.mkdir()
        with self.assertRaisesRegex(ContractError, "publication lock"):
            write_source_manifest(output, [record], dataset_id="tiny", source_root_ref="originals")
        self.assertFalse(output.exists())
        self.assertTrue(lock.exists())
        lock.rmdir()
        write_source_manifest(output, [record], dataset_id="tiny", source_root_ref="originals")
        self.assertTrue(output.is_dir())
        self.assertFalse(lock.exists())

    def test_content_collision_and_conflicting_duplicate_assignments_rejected(self):
        first = self.source_record("a/one.mp4", b"same")
        second = self.source_record("b/two.mp4", b"same")
        with self.assertRaisesRegex(ContractError, "collision"):
            validate_source_records([first, second])
        conflicting = dict(first); conflicting["partition"] = "test"
        first["partition"] = "train"
        with self.assertRaisesRegex(ContractError, "collision"):
            validate_source_records([first, conflicting])
        wrong_id = dict(first); wrong_id["video_id"] = "legacy"
        with self.assertRaisesRegex(ContractError, "exactly sha256"):
            validate_source_records([wrong_id])

    def _deepaction_fixture(self, *, duplicate: bool = False):
        root = self.root / "deepaction_v1"
        rows = []
        # Each prompt spans all labels, so this tests that group assignment is global, not per class.
        for prompt, split in (("1", "train"), ("2", "val"), ("3", "test")):
            for number, label in enumerate(DEEPACTION_LABELS):
                path = root / label / prompt / "d.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"{label}-{prompt}".encode()
                if duplicate and label == DEEPACTION_LABELS[-1] and prompt == "3":
                    payload = f"{DEEPACTION_LABELS[0]}-1".encode()
                path.write_bytes(payload)
                rows.append({
                    "video_id": f"legacy-{label}-{prompt}", "split": split, "prompt_id": prompt,
                    "source": label, "packaged_features_path": "/tmp/features.npz", "packaged_features_sha256": "a" * 64,
                    "source_features_path": "/tmp/source.npz", "source_features_sha256": "b" * 64,
                    "encoded_path": "/tmp/encoded.mp4", "encoded_sha256": "c" * 64,
                    "original_source_path": str(path), "original_source_sha256": sha256_file(path),
                    "packet_time_base": "1/1", "packet_count": "1",
                })
        audit = self.root / "audit.csv"
        with audit.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return root, audit, rows

    def test_deepaction_class_group_parsing_val_normalization_and_hash_mismatch(self):
        root, audit, _ = self._deepaction_fixture()
        inspection = inspect_deepaction_audit(root, audit)
        self.assertEqual(inspection["audit_rows"], 21)
        self.assertEqual(inspection["inventory_count"], 21)
        self.assertEqual(inspection["split_counts"], {"train": 7, "validation": 7, "test": 7})
        records = build_deepaction_records(root, audit, verify_source_hashes=True)
        self.assertEqual({row["group_id"] for row in records if row["prompt_id"] == "2"}, {"dataset:prompt:2"})
        self.assertEqual({row["partition"] for row in records if row["prompt_id"] == "2"}, {"validation"})
        self.assertTrue(all(row["source_relative_path"].count("/") == 2 for row in records))
        changed = root / DEEPACTION_LABELS[0] / "1" / "d.mp4"
        changed.write_bytes(b"changed")
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            build_deepaction_records(root, audit, verify_source_hashes=True)

    def test_symlink_files_and_directory_components_are_rejected_before_resolution(self):
        originals = self.root / "originals"
        originals.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "clip.mp4").write_bytes(b"outside")
        (originals / "linked-dir").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, "symlink component"):
            records_from_discovery(originals, [{"source_relative_path": "linked-dir/clip.mp4", "label": "a"}])
        (originals / "linked-file.mp4").symlink_to(outside / "clip.mp4")
        with self.assertRaisesRegex(ContractError, "symlink component"):
            records_from_discovery(originals, [{"source_relative_path": "linked-file.mp4", "label": "a"}])

        root, audit, _ = self._deepaction_fixture()
        original = root / DEEPACTION_LABELS[0] / "1" / "d.mp4"
        original.unlink()
        original.symlink_to(outside / "clip.mp4")
        with self.assertRaisesRegex(ContractError, "symlink component"):
            inspect_deepaction_audit(root, audit, check_inventory=False)
        original.unlink()
        original.write_bytes(b"restored")
        prompt_directory = root / DEEPACTION_LABELS[1] / "1"
        moved = self.root / "moved-prompt"
        prompt_directory.rename(moved)
        prompt_directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, "symlink component"):
            inspect_deepaction_audit(root, audit, check_inventory=False)

    def test_deepaction_quarantine_all_excludes_every_duplicate_and_verifies_exclusions(self):
        root, audit, rows = self._deepaction_fixture()
        # One duplicate pair shares a prompt/partition; the other crosses prompt and partition.
        duplicate_pairs = [
            ((DEEPACTION_LABELS[0], "1"), (DEEPACTION_LABELS[1], "1"), b"same-prompt"),
            ((DEEPACTION_LABELS[2], "2"), (DEEPACTION_LABELS[3], "3"), b"cross-prompt-partition"),
        ]
        for first, second, payload in duplicate_pairs:
            for label, prompt in (first, second):
                path = root / label / prompt / "d.mp4"
                path.write_bytes(payload)
                for row in rows:
                    if row["source"] == label and row["prompt_id"] == prompt:
                        row["original_source_sha256"] = sha256_file(path)
        with audit.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader(); writer.writerows(rows)

        with self.assertRaisesRegex(ContractError, "canonical ID collision"):
            inspect_deepaction_audit(root, audit)
        with self.assertRaisesRegex(ContractError, "canonical ID collision"):
            build_deepaction_records(root, audit, verify_source_hashes=False)
        inspection = inspect_deepaction_audit(root, audit, duplicate_policy="quarantine-all")
        self.assertEqual(inspection["audit_rows"], 21)
        self.assertEqual(inspection["duplicate_groups"], 2)
        self.assertEqual(inspection["duplicate_rows"], 4)
        self.assertEqual(inspection["retained_rows"], 17)
        self.assertEqual(inspection["excluded_rows"], 4)

        inputs = build_deepaction_manifest_inputs(root, audit, verify_source_hashes=True,
                                                  duplicate_policy="quarantine-all")
        self.assertEqual(len(inputs["records"]), 17)
        self.assertEqual(len(inputs["exclusion_rows"]), 4)
        self.assertEqual({row["duplicate_group_count"] for row in inputs["exclusion_rows"]}, {"2"})
        self.assertEqual(tuple(inputs["exclusion_rows"][0]), EXCLUSION_FIELDS)
        self.assertTrue(all(row["exclusion_reason"] == "duplicate_original_source_sha256_quarantine_all"
                            for row in inputs["exclusion_rows"]))
        self.assertEqual(inputs["stats"]["retained_rows"] + inputs["stats"]["excluded_rows"], 21)
        self.assertEqual(len({row["video_id"] for row in inputs["records"]}), 17)

        output = self.root / "quarantined-manifest"
        self.assertEqual(manifest_main([
            "build-deepaction", "--source-root", str(root), "--audit", str(audit),
            "--output-dir", str(output), "--dataset-id", "deepaction-test",
            "--source-root-ref", "deepaction_v1", "--duplicate-policy", "quarantine-all",
            "--verify-source-hashes",
        ]), 0)
        published = load_source_manifest(output)
        self.assertEqual(published["metadata"]["record_count"], 17)
        quarantine = published["metadata"]["provenance"]["duplicate_quarantine"]
        self.assertEqual(quarantine["excluded_rows"], 4)
        self.assertEqual(quarantine["duplicate_groups"], 2)
        self.assertEqual(quarantine["report_row_count"], 4)
        self.assertEqual(sha256_file(output / EXCLUSION_REPORT_FILE), quarantine["report_sha256"])

        # This is an excluded source: it must still be rehashed, not silently trusted.
        changed = root / DEEPACTION_LABELS[0] / "1" / "d.mp4"
        changed.write_bytes(b"changed after audit")
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            build_deepaction_manifest_inputs(root, audit, verify_source_hashes=True,
                                              duplicate_policy="quarantine-all")

    def test_deepaction_missing_extra_and_id_collision_fail(self):
        root, audit, _ = self._deepaction_fixture()
        (root / DEEPACTION_LABELS[0] / "1" / "d.mp4").unlink()
        with self.assertRaisesRegex(ContractError, "non-symlink regular"):
            inspect_deepaction_audit(root, audit)
        root, audit, _ = self._deepaction_fixture()
        extra = root / DEEPACTION_LABELS[0] / "99" / "x.mp4"
        extra.parent.mkdir(parents=True); extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ContractError, "inventory mismatch"):
            inspect_deepaction_audit(root, audit)
        root, audit, _ = self._deepaction_fixture(duplicate=True)
        with self.assertRaisesRegex(ContractError, "canonical ID collision"):
            inspect_deepaction_audit(root, audit)

    def test_discovery_helper_and_extraction_index_success_failure_and_membership(self):
        self.source_record("one.mp4", b"one")
        self.source_record("two.mp4", b"two")
        records = records_from_discovery(self.root / "originals", [
            {"source_relative_path": "one.mp4", "label": "a"},
            {"source_relative_path": "two.mp4", "label": "b", "group_id": "g"},
        ])
        manifest_dir = self.root / "source-manifest"
        write_source_manifest(manifest_dir, records, dataset_id="tiny", source_root_ref="originals")
        success, failed = sorted(records, key=lambda item: item["source_relative_path"])
        index_rows = [
            {"video_id": success["video_id"], "status": "success", "encoded_relative_path": "encoded/one.mp4",
             "encoded_sha256": "d" * 64, "artifact_relative_path": "artifacts/one.npz", "producer_schema_version": "2.0",
             "row_count": "2", "frame_count": "3", "failure_code": "", "failure_message": ""},
            {"video_id": failed["video_id"], "status": "failed", "encoded_relative_path": "", "encoded_sha256": "",
             "artifact_relative_path": "", "producer_schema_version": "2.0", "row_count": "", "frame_count": "",
             "failure_code": "decode_error", "failure_message": "bad input"},
        ]
        index_dir = self.root / "index"
        write_extraction_index(index_dir, index_rows, source_manifest_dir=manifest_dir,
                               encoded_root_ref="encoded", artifact_root_ref="artifacts",
                               processing_profile={"name": "native"}, extractor_identity={"name": "future", "version": "2.0"})
        self.assertEqual(len(load_extraction_index(index_dir, manifest_dir)["records"]), 2)
        fabricated = dict(index_rows[1]); fabricated["encoded_relative_path"] = "should-not-exist.mp4"
        with self.assertRaisesRegex(ContractError, "must not fabricate"):
            validate_extraction_index_records([index_rows[0], fabricated], [r["video_id"] for r in records])
        with self.assertRaisesRegex(ContractError, "cover every"):
            validate_extraction_index_records([index_rows[0]], [r["video_id"] for r in records])
        metadata_path = index_dir / "extraction_index.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["contract_version"], "1.0")
        metadata["processing_profile"] = {}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "non-empty objects"):
            load_extraction_index(index_dir, manifest_dir)

    def test_schema_version_and_type_validation(self):
        record = self.source_record("one.mp4", b"one")
        bad = dict(record); bad["source_bytes"] = True
        with self.assertRaisesRegex(ContractError, "integer"):
            validate_source_records([bad])
        manifest = self.root / "manifest"
        write_source_manifest(manifest, [record], dataset_id="tiny", source_root_ref="originals")
        csv_path = manifest / "source_manifest.csv"
        csv_path.write_text("not,the,right,header\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "header"):
            load_source_manifest(manifest)
        metadata_manifest = self.root / "metadata-manifest"
        write_source_manifest(metadata_manifest, [record], dataset_id="tiny", source_root_ref="originals")
        metadata_path = metadata_manifest / "source_manifest.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["identity"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "identity declaration"):
            load_source_manifest(metadata_manifest)
        metadata["identity"] = "video_id is sha256:<full original source SHA-256>"
        metadata["provenance"] = []
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "provenance"):
            load_source_manifest(metadata_manifest)


if __name__ == "__main__":
    unittest.main()
