from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.manifest_contract import canonical_video_id, load_extraction_index, sha256_file, write_source_manifest
from src.native_wpp_workflow import (
    PresentationProbe, WorkflowError, _acquire_build_lock, _release_build_lock, _safe_source_path, compare_native_geometry_timing,
    probe_presentation, run_workflow, validate_artifact, verified_stage_copy, verify_native_build,
)


class NativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"; self.sources.mkdir()
        self.encoder = self.root / "encode_wpp.sh"; self.encoder.write_text("stub", encoding="utf-8")
        self.extractor = self.root / "extract_wpp.py"; self.extractor.write_text("stub", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, names=("a.mp4",)):
        records = []
        for name in names:
            source = self.sources / "nested" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes((name + " bytes").encode())
            digest = sha256_file(source)
            records.append({"video_id": canonical_video_id(digest), "source_sha256": digest,
                            "source_relative_path": f"nested/{name}", "source_bytes": source.stat().st_size,
                            "label": "x"})
        bundle = self.root / "manifest"
        write_source_manifest(bundle, records, dataset_id="tiny", source_root_ref="sources")
        return bundle, records

    @staticmethod
    def _probe_json(path: Path):
        # VFR display timeline: 0, 0.2, 0.5 seconds. Packets intentionally do
        # not have one-per-frame shape, proving source timing is frame-based.
        return json.dumps({"streams": [{"codec_type": "video", "width": 320, "height": 240,
                                           "coded_width": 320, "coded_height": 256,
                                           "sample_aspect_ratio": "1:1", "time_base": "1/10"}],
                           "frames": [{"media_type": "video", "best_effort_timestamp": "0"},
                                      {"media_type": "video", "best_effort_timestamp": "2"},
                                      {"media_type": "video", "best_effort_timestamp": "5"}],
                           "packets": [{"pts": "0"}, {"pts": "0"}, {"pts": "5"}]})

    def _write_artifact(self, out: Path, n=2, t=3):
        out.mkdir(parents=True)
        values = np.arange(1, n * t + 1, dtype=np.int64).reshape(n, t)
        arrays = {name: values.astype(float) for name in (
            "row_bytes", "row_share", "log1p_row_bytes", "row_bits_per_coded_pixel", "delta_row_share")}
        arrays["row_bytes"] = values
        arrays["total_payload_bytes"] = values.sum(axis=0)
        pts = np.array([0, 2, 5]) if t == 3 else np.arange(t)
        fields = {"display_index": np.arange(t), "decode_index": np.arange(t),
                  "pts": pts, "pts_seconds": pts.astype(float) / 10,
                  "slice_type": np.full(t, "P"), "slice_qp": np.full(t, 28),
                  "temporal_id": np.zeros(t, dtype=np.int64), "nal_type": np.full(t, 1), "vcl_nal_bytes": np.full(t, 10),
                  "header_bytes": np.full(t, 2), "payload_bytes": np.full(t, 8),
                  "cu_qp_delta_enabled": np.zeros(t, dtype=np.int64)}
        np.savez_compressed(out / "features.npz", **arrays, **fields)
        (out / "metadata.json").write_text(json.dumps({"schema_version": "2.0", "rows": n, "frames": t,
                                                           "coded_width": 320, "coded_height": 256,
                                                           "ctu_size": 64, "time_base": "1/10"}), encoding="utf-8")
        with (out / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow([f"frame_{i}" for i in range(t)]); writer.writerows(values)
        frame_keys = list(fields)
        with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=frame_keys); writer.writeheader()
            for i in range(t): writer.writerow({key: fields[key][i] for key in frame_keys})

    def test_artifact_validation_retains_more_than_3000_frames(self):
        artifact = self.root / "artifact"
        self._write_artifact(artifact, n=2, t=3001)
        result = validate_artifact(artifact)
        self.assertEqual((result["row_count"], result["frame_count"]), (2, 3001))
        self.assertIn("features.npz", result["artifact_files_sha256"])

    def test_timing_vfr_bframes_quantization_and_rejections(self):
        src = PresentationProbe(320, 240, 320, 240, "1:1", Fraction(1, 10),
                                (Fraction(1), Fraction(6, 5), Fraction(3, 2)))
        # Origin moves; output is one 1/100-second tick from source at the final frame.
        out = PresentationProbe(320, 240, 320, 256, "1:1", Fraction(1, 100),
                                (Fraction(2), Fraction(11, 5), Fraction(251, 100)))
        result = compare_native_geometry_timing(src, out)
        self.assertEqual(result["output_tick_tolerance_seconds"], "1/100")
        with self.assertRaisesRegex(WorkflowError, "timeline"):
            compare_native_geometry_timing(src, PresentationProbe(320, 240, 320, 240, "1:1", Fraction(1, 100),
                (Fraction(2), Fraction(11, 5), Fraction(13, 5))))
        with self.assertRaisesRegex(WorkflowError, "frame count"):
            compare_native_geometry_timing(src, PresentationProbe(320, 240, 320, 240, "1:1", Fraction(1, 10), src.presentation_times[:2]))
        with self.assertRaisesRegex(WorkflowError, "aspect"):
            compare_native_geometry_timing(src, PresentationProbe(320, 240, 320, 240, "4:3", Fraction(1, 10), src.presentation_times))
        with self.assertRaisesRegex(WorkflowError, "width"):
            compare_native_geometry_timing(src, PresentationProbe(640, 240, 640, 256, "1:1", Fraction(1, 10), src.presentation_times))

    def test_source_protection_hash_mismatch_and_mutation_detection(self):
        source = self.sources / "a.mp4"; source.write_bytes(b"original")
        staged = self.root / "stage.mp4"
        details = verified_stage_copy(source, sha256_file(source), source.stat().st_size, staged)
        self.assertEqual(details["source_sha256"], sha256_file(staged))
        with self.assertRaisesRegex(WorkflowError, "SHA-256 mismatch"):
            verified_stage_copy(source, "0" * 64, source.stat().st_size, self.root / "bad.mp4")
        with mock.patch("src.native_wpp_workflow.shutil.copyfile", side_effect=lambda a, b: (Path(b).write_bytes(Path(a).read_bytes()), Path(a).write_bytes(b"changed"))):
            with self.assertRaisesRegex(WorkflowError, "changed while staging"):
                verified_stage_copy(source, sha256_file(source), source.stat().st_size, self.root / "again.mp4")

    def test_probe_rejects_missing_or_duplicate_pts(self):
        video = self.root / "video.mp4"; video.write_bytes(b"x")
        def invoke(payload):
            return lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(payload), "")
        base = {"streams": [{"codec_type": "video", "width": 2, "height": 2,
                              "sample_aspect_ratio": "1:1", "time_base": "1/10"}]}
        with self.assertRaisesRegex(WorkflowError, "duplicate"):
            probe_presentation(video, invoke({**base, "frames": [{"media_type": "video", "best_effort_timestamp": "0"}, {"media_type": "video", "best_effort_timestamp": "0"}]}))
        with self.assertRaisesRegex(WorkflowError, "missing"):
            probe_presentation(video, invoke({**base, "frames": [{"media_type": "video"}]}))
        # Multiple/duplicate packets for a single source frame are deliberately ignored.
        probe, _ = probe_presentation(video, invoke({**base, "frames": [{"media_type": "video", "best_effort_timestamp": "0"}, {"media_type": "video", "best_effort_timestamp": "2"}], "packets": [{"pts": "0"}, {"pts": "0"}, {"pts": "2"}]}))
        self.assertEqual(len(probe.presentation_times), 2)

    def test_artifact_missing_required_npz_field_is_rejected(self):
        artifact = self.root / "missing-field"; self._write_artifact(artifact)
        with np.load(artifact / "features.npz") as archive:
            arrays = {key: archive[key] for key in archive.files if key != "row_share"}
        np.savez_compressed(artifact / "features.npz", **arrays)
        with self.assertRaisesRegex(WorkflowError, "missing arrays"):
            validate_artifact(artifact)

    def test_optional_nested_artifact_links_are_rejected_before_any_external_hash(self):
        import src.native_wpp_workflow as workflow
        outside = self.root / "outside-hash"; outside.write_bytes(b"never hash me")
        for kind in ("file", "directory", "dangling"):
            with self.subTest(kind=kind):
                artifact = self.root / f"nested-{kind}"; self._write_artifact(artifact)
                nested = artifact / "optional"; nested.mkdir()
                if kind == "file":
                    target = outside
                elif kind == "directory":
                    target = self.root / "outside-dir"; target.mkdir(exist_ok=True); (target / "file").write_bytes(b"outside")
                else:
                    target = self.root / "does-not-exist"
                (nested / "link").symlink_to(target, target_is_directory=kind == "directory")
                actual_hash = workflow.sha256_file
                seen: list[Path] = []
                def tracked(path):
                    seen.append(Path(path))
                    if Path(path).resolve(strict=False) == target.resolve(strict=False):
                        raise AssertionError("attempted to hash external link target")
                    return actual_hash(path)
                with mock.patch("src.native_wpp_workflow.sha256_file", side_effect=tracked):
                    with self.assertRaisesRegex(WorkflowError, "symlink"):
                        validate_artifact(artifact)
                self.assertFalse(any(path.resolve(strict=False) == target.resolve(strict=False) for path in seen))

    def test_unsafe_optional_artifact_is_not_published(self):
        import src.native_wpp_workflow as workflow
        manifest, _ = self.manifest(("unsafe-link.mp4",))
        output, outside = self.root / "unsafe-link-output", self.root / "unsafe-link-target"
        outside.write_bytes(b"outside")
        calls = {"encodes": 0, "extracts": 0}
        base = self._resume_fake_run(calls)
        def unsafe_run(cmd, **kwargs):
            result = base(cmd, **kwargs)
            if cmd[0] not in ("ffprobe", "bash"):
                artifact = Path(cmd[cmd.index("--out") + 1]); (artifact / "optional").mkdir()
                (artifact / "optional" / "link").symlink_to(outside)
            return result
        real_publish = workflow._publish_at
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]), \
             mock.patch("src.native_wpp_workflow._publish_at", wraps=real_publish) as publish:
            # Cleanup also refuses the untrusted link, so no failed public index
            # is finalized around an unsafe private artifact tree.
            with self.assertRaisesRegex(WorkflowError, "symlink"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, run=unsafe_run)
        self.assertFalse(any(call.kwargs.get("destination_relative") == "artifacts/sha256" for call in publish.call_args_list))
        self.assertFalse(any((output / "artifacts" / "sha256").iterdir()))
        self.assertFalse((output / "extraction-index").exists())

    def test_artifact_csv_corruption_is_rejected_against_all_npz_values(self):
        artifact = self.root / "csv-corruption"; self._write_artifact(artifact)
        rows_path = artifact / "rows.csv"
        with rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows[1][1] = "999"
        with rows_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        with self.assertRaisesRegex(WorkflowError, "exactly match"):
            validate_artifact(artifact)
        self._write_artifact(artifact := self.root / "frame-corruption")
        frames_path = artifact / "frames.csv"
        with frames_path.open(newline="", encoding="utf-8") as handle:
            frames = list(csv.DictReader(handle))
        fields = list(frames[0])
        frames[0]["slice_type"] = "I"
        with frames_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(frames)
        with self.assertRaisesRegex(WorkflowError, "differs from features.npz"):
            validate_artifact(artifact)
        self._write_artifact(artifact := self.root / "numeric-frame-corruption")
        frames_path = artifact / "frames.csv"
        with frames_path.open(newline="", encoding="utf-8") as handle:
            frames = list(csv.DictReader(handle))
        fields = list(frames[0]); frames[1]["payload_bytes"] = "999"
        with frames_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(frames)
        with self.assertRaisesRegex(WorkflowError, "payload_bytes differs"):
            validate_artifact(artifact)

    def test_symlink_source_path_is_rejected(self):
        outside = self.root / "outside.mp4"; outside.write_bytes(b"x")
        (self.sources / "link.mp4").symlink_to(outside)
        with self.assertRaisesRegex(WorkflowError, "symlink"):
            _safe_source_path(self.sources, "link.mp4")

    def test_workflow_complete_index_success_and_per_video_failure(self):
        manifest, records = self.manifest(("a.mp4", "b.mp4"))
        output = self.root / "derived"
        calls = {"encodes": 0}
        def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
            if cmd[0] == "ffprobe":
                return subprocess.CompletedProcess(cmd, 0, self._probe_json(Path(cmd[-1])), "")
            if cmd[0] == "bash":
                calls["encodes"] += 1
                stage_dir = Path(cmd[2]); source = next(stage_dir.glob("*.mp4"))
                if calls["encodes"] == 2:
                    return subprocess.CompletedProcess(cmd, 1, "", "synthetic encoder failure")
                generated = stage_dir / "hevc-wpp-gop-16-qp-28-b-0" / f"{source.name}.mp4"
                generated.parent.mkdir(); generated.write_bytes(b"encoded")
                return subprocess.CompletedProcess(cmd, 0, "ok", "")
            # extractor command
            self._write_artifact(Path(cmd[cmd.index("--out") + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            index, had_failures = run_workflow(source_manifest_dir=manifest, source_root=self.sources,
                output_dir=output, encoder_script=self.encoder, extractor_script=self.extractor, run=fake_run)
        self.assertTrue(had_failures)
        rows = {row["video_id"]: row for row in index["records"]}
        self.assertEqual(set(rows), {record["video_id"] for record in records})
        self.assertEqual(sorted(row["status"] for row in rows.values()), ["failed", "success"])
        self.assertTrue((output / "build_ledger.json").is_file())
        self.assertTrue((output / "extraction-index" / "extraction_index.csv").is_file())
        self.assertTrue(verify_native_build(output / "extraction-index", manifest)["verified"])

        def assert_tamper(path, expected):
            original = path.read_bytes(); path.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(WorkflowError, expected):
                verify_native_build(output / "extraction-index", manifest)
            path.write_bytes(original)
        success = next(row for row in index["records"] if row["status"] == "success")
        assert_tamper(output / "encoded" / success["encoded_relative_path"], "encoded SHA")
        assert_tamper(output / "artifacts" / success["artifact_relative_path"] / "features.npz", "artifact SHA")
        assert_tamper(next((output / "logs").glob("*.log")), "command log")
        ledger = output / "build_ledger.json"
        assert_tamper(ledger, "ledger SHA")
        metadata_path = output / "extraction-index" / "extraction_index.metadata.json"
        original_metadata_bytes = metadata_path.read_bytes()
        original_ledger_bytes = ledger.read_bytes()
        rebound_ledger = json.loads(original_ledger_bytes)
        rebound_ledger["logs"] = {}
        ledger.write_text(json.dumps(rebound_ledger), encoding="utf-8")
        rebound_metadata = json.loads(original_metadata_bytes)
        rebound_metadata["native_workflow_ledger"]["sha256"] = sha256_file(ledger)
        metadata_path.write_text(json.dumps(rebound_metadata), encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "absent from the log hash inventory"):
            verify_native_build(output / "extraction-index", manifest)
        ledger.write_bytes(original_ledger_bytes)
        metadata_path.write_bytes(original_metadata_bytes)
        metadata = json.loads(original_metadata_bytes)
        metadata["native_workflow_ledger"]["relative_path"] = "../build_ledger.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "safe|relative"):
            verify_native_build(output / "extraction-index", manifest)
        metadata_path.write_bytes(original_metadata_bytes)

    def test_successful_encoder_without_fresh_file_is_not_silently_skipped(self):
        manifest, records = self.manifest()
        output = self.root / "silent-skip"
        def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
            if cmd[0] == "ffprobe":
                return subprocess.CompletedProcess(cmd, 0, self._probe_json(Path(cmd[-1])), "")
            # Mimics the producer's existing-output skip: a zero status but no file.
            return subprocess.CompletedProcess(cmd, 0, "Skipping existing", "")
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            index, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources,
                output_dir=output, encoder_script=self.encoder, extractor_script=self.extractor, run=fake_run)
        self.assertTrue(failed)
        self.assertEqual(index["records"][0]["status"], "failed")
        self.assertIn("silent skip", index["records"][0]["failure_message"])

    def test_existing_output_is_run_level_error(self):
        manifest, _ = self.manifest()
        output = self.root / "already"; output.mkdir()
        with self.assertRaisesRegex(WorkflowError, "overwrite"):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                encoder_script=self.encoder, extractor_script=self.extractor)

    def test_output_inside_code_repository_is_rejected(self):
        manifest, _ = self.manifest()
        output = ROOT / "native-wpp-test-output-must-not-exist"
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(WorkflowError, "outside the code repository"):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                encoder_script=self.encoder, extractor_script=self.extractor)

    def _resume_fake_run(self, calls, *, fail_second=False):
        def fake_run(cmd, check=False, capture_output=False, text=False, **kwargs):
            if cmd[0] == "ffprobe":
                return subprocess.CompletedProcess(cmd, 0, self._probe_json(Path(cmd[-1])), "")
            if cmd[0] == "bash":
                calls["encodes"] += 1
                if fail_second and calls["encodes"] == 2:
                    return subprocess.CompletedProcess(cmd, 1, "", "synthetic encoder failure")
                stage_dir = Path(cmd[2]); source = next(stage_dir.glob("*.mp4"))
                generated = stage_dir / "hevc-wpp-gop-16-qp-28-b-0" / f"{source.name}.mp4"
                generated.parent.mkdir(); generated.write_bytes(b"encoded")
                return subprocess.CompletedProcess(cmd, 0, "ok", "")
            calls["extracts"] += 1
            self._write_artifact(Path(cmd[cmd.index("--out") + 1]))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        return fake_run

    def test_resume_reuses_durable_success_and_finalizes_after_interrupt(self):
        manifest, _ = self.manifest(("a.mp4", "b.mp4"))
        output, calls = self.root / "resume", {"encodes": 0, "extracts": 0}
        stopped = {"value": False}
        def stop_once(stage, item):
            if stage == "after_success_commit" and not stopped["value"]:
                stopped["value"] = True
                raise KeyboardInterrupt("controlled stop")
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop_once)
            self.assertEqual(calls, {"encodes": 1, "extracts": 1})
            self.assertFalse((output / "extraction-index").exists())
            index, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                         run=self._resume_fake_run(calls))
        self.assertFalse(failed)
        self.assertEqual(calls, {"encodes": 2, "extracts": 2})
        self.assertTrue(verify_native_build(output / "extraction-index", manifest)["verified"])
        # A finalized compatible resume is read-only.
        journal = output / ".native-wpp-state" / "journal.json"
        before = journal.stat().st_mtime_ns
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                         run=self._resume_fake_run(calls))
        self.assertEqual(journal.stat().st_mtime_ns, before)
        self.assertEqual(index["metadata"]["record_count"], 2)

    def test_resume_failed_item_requires_explicit_retry_and_tamper_fails_closed(self):
        manifest, _ = self.manifest(("a.mp4", "b.mp4"))
        output, calls = self.root / "retry", {"encodes": 0, "extracts": 0}
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            _, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor,
                                     run=self._resume_fake_run(calls, fail_second=True))
            self.assertTrue(failed)
            _, still_failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                           encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                           run=self._resume_fake_run(calls))
            self.assertTrue(still_failed)
            self.assertEqual(calls["encodes"], 2)
            _, recovered = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                        encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                        retry_failed=True, run=self._resume_fake_run(calls))
        self.assertFalse(recovered)
        self.assertEqual(calls["encodes"], 3)
        self.assertTrue(verify_native_build(output / "extraction-index", manifest)["verified"])
        success = next(row for row in load_extraction_index(output / "extraction-index", manifest)["records"] if row["status"] == "success")
        encoded = output / "encoded" / success["encoded_relative_path"]
        encoded.write_bytes(encoded.read_bytes() + b"tamper")
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaisesRegex(WorkflowError, "committed encoded"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))

    def test_resume_promotes_fsynced_receipt_before_journal_without_reencoding(self):
        manifest, _ = self.manifest(("a.mp4", "b.mp4"))
        output, calls = self.root / "receipt-crash", {"encodes": 0, "extracts": 0}
        stopped = {"value": False}
        def stop(stage, item):
            if stage == "after_receipt_before_journal" and not stopped["value"]:
                stopped["value"] = True; raise KeyboardInterrupt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop)
            self.assertEqual(calls, {"encodes": 1, "extracts": 1})
            _, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                     run=self._resume_fake_run(calls))
        self.assertFalse(failed)
        self.assertEqual(calls, {"encodes": 2, "extracts": 2})

    def test_resume_rejects_forged_terminal_failure_before_finalization(self):
        manifest, _ = self.manifest()
        output, calls = self.root / "forged", {"encodes": 0, "extracts": 0}
        def stop(stage, item):
            if stage == "after_encode":
                raise KeyboardInterrupt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop)
            journal_path = output / ".native-wpp-state" / "journal.json"
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            item = next(iter(journal["items"].values()))
            attempt = item["attempts"][-1]; attempt["outcome"] = "failed"
            failure = {"code": "processing_error", "message": "forged", "stage": "encoder",
                       "ledger_entry": {"video_id": item["video_id"], "attempt_id": attempt["id"],
                                        "failure": {"code": "processing_error", "message": "forged", "stage": "encoder"}}}
            item["state"] = "failed"; item["failure"] = failure
            item["receipt"] = {"relative_path": f".native-wpp-state/receipts/{item['source_sha256']}/{attempt['id']}.json", "sha256": "0" * 64}
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "receipt|identity or fields|failure provenance|latest attempt"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))

    def test_resume_rejects_symlinked_state_and_artifact_required_leaf(self):
        manifest, _ = self.manifest()
        output, external = self.root / "state-link", self.root / "external-state"
        output.mkdir(); external.mkdir(); marker = external / "marker"; marker.write_text("unchanged", encoding="utf-8")
        (output / ".native-wpp-state").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(WorkflowError, "symlink"):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True)
        self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
        artifact, outside = self.root / "linked-artifact", self.root / "outside-metadata"
        self._write_artifact(artifact); outside.write_text("{}", encoding="utf-8")
        (artifact / "metadata.json").unlink(); (artifact / "metadata.json").symlink_to(outside)
        with self.assertRaisesRegex(WorkflowError, "symlink"):
            validate_artifact(artifact)
        self.assertEqual(outside.read_text(encoding="utf-8"), "{}")

    def test_resume_rejects_source_and_script_fingerprint_changes_before_reuse(self):
        manifest, _ = self.manifest()
        output, calls = self.root / "mismatch", {"encodes": 0, "extracts": 0}
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor, run=self._resume_fake_run(calls))
            source = self.sources / "nested" / "a.mp4"; original = source.read_bytes(); source.write_bytes(b"changed")
            with self.assertRaisesRegex(WorkflowError, "source changed"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))
            source.write_bytes(original)
            original_script = self.encoder.read_bytes(); self.encoder.write_bytes(original_script + b"# changed\n")
            with self.assertRaisesRegex(WorkflowError, "fingerprint mismatch"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))
            self.encoder.write_bytes(original_script)

    def test_resume_recovers_each_durable_interruption_boundary(self):
        manifest, _ = self.manifest(("a.mp4", "b.mp4"))
        for stage in ("after_encode", "after_encoded_publish", "after_artifact_validation",
                      "after_success_commit", "before_final_ledger", "after_final_ledger", "after_final_index"):
            with self.subTest(stage=stage):
                output, calls = self.root / f"boundary-{stage}", {"encodes": 0, "extracts": 0}
                stopped = {"value": False}
                def stop(target, item):
                    if target == stage and not stopped["value"]:
                        stopped["value"] = True; raise KeyboardInterrupt(stage)
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaises(KeyboardInterrupt):
                        run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor,
                                     run=self._resume_fake_run(calls), test_hook=stop)
                    index, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                                 encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                                 run=self._resume_fake_run(calls))
                self.assertFalse(failed)
                self.assertEqual(index["metadata"]["record_count"], 2)
                self.assertTrue(verify_native_build(output / "extraction-index", manifest)["verified"])

    def test_fd_pinned_encoded_and_artifact_publication_reject_parent_swaps(self):
        manifest, records = self.manifest()
        for target in ("encoded", "artifacts"):
            with self.subTest(target=target):
                output, outside = self.root / f"swap-{target}", self.root / f"outside-{target}"
                outside.mkdir(); marker = outside / "marker"; marker.write_text("unchanged", encoding="utf-8")
                calls, swapped = {"encodes": 0, "extracts": 0}, {"value": False}
                base = self._resume_fake_run(calls)
                def attack(cmd, **kwargs):
                    result = base(cmd, **kwargs)
                    # Swap only while the child callback owns control.  The child
                    # receives /proc/self/fd paths, so it must not write outside.
                    if not swapped["value"] and ((target == "encoded" and cmd[0] == "bash") or
                                                   (target == "artifacts" and cmd[0] not in ("ffprobe", "bash"))):
                        swapped["value"] = True
                        victim = output / target; victim.rename(output / f"{target}-old")
                        victim.symlink_to(outside, target_is_directory=True)
                    return result
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaisesRegex(WorkflowError, "replaced|unsafe|output"):
                        run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor, run=attack)
                self.assertTrue(swapped["value"])
                self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
                digest = records[0]["source_sha256"]
                self.assertFalse((outside / "sha256" / f"{digest}.mp4").exists())
                self.assertFalse((outside / "sha256" / digest).exists())

    def test_log_fd_survives_parent_swap_without_outside_write(self):
        manifest, _ = self.manifest()
        output, outside = self.root / "swap-logs", self.root / "outside-logs"
        outside.mkdir(); marker = outside / "marker"; marker.write_text("unchanged", encoding="utf-8")
        calls, swapped = {"encodes": 0, "extracts": 0}, {"value": False}
        base = self._resume_fake_run(calls)
        def attack(cmd, **kwargs):
            result = base(cmd, **kwargs)
            if cmd[0] == "bash" and not swapped["value"]:
                swapped["value"] = True
                logs = output / "logs"; logs.rename(output / "logs-old")
                logs.symlink_to(outside, target_is_directory=True)
            return result
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaisesRegex(WorkflowError, "symlink|log|output"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, run=attack)
        self.assertTrue(swapped["value"])
        self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
        self.assertFalse(any(outside.glob("*.log")))

    def test_historical_receipts_are_retained_and_inventory_is_exact(self):
        manifest, _ = self.manifest(("a.mp4", "b.mp4"))
        output, calls = self.root / "receipts", {"encodes": 0, "extracts": 0}
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor,
                         run=self._resume_fake_run(calls, fail_second=True))
            journal = json.loads((output / ".native-wpp-state" / "journal.json").read_text(encoding="utf-8"))
            failed_item = next(item for item in journal["items"].values() if item["state"] == "failed")
            old_receipt = output / failed_item["attempts"][-1]["receipt"]["relative_path"]
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True, retry_failed=True,
                         run=self._resume_fake_run(calls))
            self.assertTrue(old_receipt.exists())
            original_receipt = old_receipt.read_bytes()
            old_receipt.write_bytes(original_receipt + b"tamper")
            with self.assertRaisesRegex(WorkflowError, "receipt"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))
            old_receipt.write_bytes(original_receipt)
        # A receipt not named by an attempt is not tolerated even after completion.
        extra = output / ".native-wpp-state" / "receipts" / "extra" / "replay.json"
        extra.parent.mkdir(parents=True); extra.write_text("{}", encoding="utf-8")
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaisesRegex(WorkflowError, "receipt"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))

    def test_resume_fingerprint_rejects_valid_manifest_csv_metadata_extractor_and_runtime_changes_before_work(self):
        manifest, _ = self.manifest()
        output, calls = self.root / "fingerprint-direct", {"encodes": 0, "extracts": 0}
        def stop(stage, item):
            if stage == "after_encode":
                raise KeyboardInterrupt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop)
            baseline = dict(calls)
            metadata_path = manifest / "source_manifest.metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")); metadata["dataset_id"] = "tiny-changed"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "fingerprint mismatch"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))
            self.assertEqual(calls, baseline)
            metadata["dataset_id"] = "tiny"; metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            csv_path = manifest / "source_manifest.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["label"] = "changed"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            metadata["records_sha256"] = sha256_file(csv_path); metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "fingerprint mismatch"):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                             run=self._resume_fake_run(calls))
            self.assertEqual(calls, baseline)

        # Separate interrupted fixtures isolate script/runtime fingerprint checks.
        for kind in ("extractor", "runtime"):
            with self.subTest(kind=kind):
                m = manifest
                out, counted = self.root / f"fingerprint-{kind}", {"encodes": 0, "extracts": 0}
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaises(KeyboardInterrupt):
                        run_workflow(source_manifest_dir=m, source_root=self.sources, output_dir=out,
                                     encoder_script=self.encoder, extractor_script=self.extractor,
                                     run=self._resume_fake_run(counted), test_hook=stop)
                    before = dict(counted)
                    if kind == "extractor":
                        original = self.extractor.read_bytes(); self.extractor.write_bytes(original + b"# changed\n")
                        context = mock.patch("src.native_wpp_workflow._runtime_version", wraps=__import__("src.native_wpp_workflow", fromlist=["_runtime_version"])._runtime_version)
                    else:
                        original = None
                        context = mock.patch("src.native_wpp_workflow._runtime_version", return_value={"path": "changed", "version_sha256": "0" * 64, "returncode": "0"})
                    with context:
                        with self.assertRaisesRegex(WorkflowError, "fingerprint mismatch"):
                            run_workflow(source_manifest_dir=m, source_root=self.sources, output_dir=out,
                                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                         run=self._resume_fake_run(counted))
                    if original is not None:
                        self.extractor.write_bytes(original)
                    self.assertEqual(counted, before)

    def test_verifier_rejects_active_and_finalizing_journals(self):
        manifest, _ = self.manifest(("active.mp4",))
        for boundary, phase in (("after_success_commit", "active"), ("before_final_ledger", "finalizing")):
            with self.subTest(phase=phase):
                output, calls = self.root / f"verify-{phase}", {"encodes": 0, "extracts": 0}
                def stop(stage, item, boundary=boundary):
                    if stage == boundary:
                        raise KeyboardInterrupt
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaises(KeyboardInterrupt):
                        run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor,
                                     run=self._resume_fake_run(calls), test_hook=stop)
                with self.assertRaisesRegex(WorkflowError, "unfinished|invalid resume journal"):
                    verify_native_build(output / "extraction-index", manifest)

    def test_legacy_no_journal_is_readonly_and_failed_legacy_cannot_retry(self):
        manifest, _ = self.manifest(("legacy-a.mp4", "legacy-b.mp4"))
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second):
                output, calls = self.root / f"legacy-{fail_second}", {"encodes": 0, "extracts": 0}
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                 encoder_script=self.encoder, extractor_script=self.extractor,
                                 run=self._resume_fake_run(calls, fail_second=fail_second))
                state = output / ".native-wpp-state"; shutil.rmtree(state)
                before = {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in output.rglob("*") if path.is_file()}
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    if fail_second:
                        with self.assertRaisesRegex(WorkflowError, "failures and cannot be retried"):
                            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                         retry_failed=True, run=self._resume_fake_run(calls))
                    else:
                        _, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                                 encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                                 run=self._resume_fake_run(calls))
                        self.assertFalse(failed)
                after = {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns)
                         for path in output.rglob("*") if path.is_file()}
                self.assertEqual(after, before)

    def test_cli_workflow_failure_is_one_and_argparse_error_is_two(self):
        import build_native_wpp_extraction as cli
        argv = ["--source-manifest", str(self.root / "m"), "--source-root", str(self.root / "s"),
                "--output-dir", str(self.root / "o")]
        with mock.patch.object(cli, "run_workflow", side_effect=WorkflowError("nope")):
            self.assertEqual(cli.main(argv), 1)
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args([])
        self.assertEqual(raised.exception.code, 2)

    def test_release_lock_preserves_foreign_nonce_and_replacement_inode(self):
        output = self.root / "lock-owner"
        lock = _acquire_build_lock(output)
        owner = lock.path / "owner.json"
        foreign = json.loads(owner.read_text(encoding="utf-8")); foreign["nonce"] = "foreign"
        owner.write_text(json.dumps(foreign), encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "ownership changed"):
            _release_build_lock(lock)
        self.assertTrue(lock.path.exists())
        shutil.rmtree(lock.path)
        lock.path.mkdir(); (lock.path / "owner.json").write_text(json.dumps(foreign), encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "ownership changed"):
            _release_build_lock(lock)
        self.assertTrue(lock.path.exists())

    def test_receipt_missing_replay_wrong_identity_and_symlink_reject_before_finalization(self):
        manifest, _ = self.manifest(("receipt-a.mp4", "receipt-b.mp4"))
        output, calls = self.root / "receipt-variants-base", {"encodes": 0, "extracts": 0}
        def stop(stage, item):
            if stage == "before_final_ledger":
                raise KeyboardInterrupt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop)
        for kind in ("missing", "replay", "wrong_attempt", "wrong_source", "symlink"):
            with self.subTest(kind=kind):
                candidate = self.root / f"receipt-{kind}"; shutil.copytree(output, candidate)
                journal_path = candidate / ".native-wpp-state" / "journal.json"
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                items = list(journal["items"].values()); target, other = items[0], items[1]
                receipt = candidate / target["attempts"][-1]["receipt"]["relative_path"]
                if kind == "missing":
                    receipt.unlink()
                elif kind == "replay":
                    target["attempts"][-1]["receipt"] = dict(other["attempts"][-1]["receipt"]); target["receipt"] = dict(other["receipt"])
                elif kind == "wrong_attempt":
                    ref = dict(target["attempts"][-1]["receipt"]); ref["relative_path"] = ref["relative_path"].replace(target["attempts"][-1]["id"], "0" * 32)
                    target["attempts"][-1]["receipt"] = ref; target["receipt"] = ref
                elif kind == "wrong_source":
                    ref = dict(target["attempts"][-1]["receipt"]); ref["relative_path"] = ref["relative_path"].replace(target["source_sha256"], "0" * 64)
                    target["attempts"][-1]["receipt"] = ref; target["receipt"] = ref
                else:
                    outside = self.root / "receipt-outside"; outside.write_text("unchanged", encoding="utf-8")
                    receipt.unlink(); receipt.symlink_to(outside)
                journal_path.write_text(json.dumps(journal), encoding="utf-8")
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaisesRegex(WorkflowError, "receipt|symlink"):
                        run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=candidate,
                                     encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                     run=self._resume_fake_run({"encodes": 0, "extracts": 0}))

    def test_failed_orphan_receipt_promotes_and_multiple_retry_receipts_remain_bound(self):
        manifest, _ = self.manifest(("receipt-chain.mp4",))
        output, calls = self.root / "failed-orphan", {"encodes": 0, "extracts": 0}
        base = self._resume_fake_run(calls)
        def fail_encode(cmd, **kwargs):
            result = base(cmd, **kwargs)
            if cmd[0] == "bash":
                return subprocess.CompletedProcess(cmd, 1, "", "planned failure")
            return result
        import src.native_wpp_workflow as workflow
        original_write = workflow._write_completion_receipt
        def crash_after_failed_receipt(*args, **kwargs):
            receipt = original_write(*args, **kwargs)
            if kwargs["outcome"] == "failed":
                raise KeyboardInterrupt("after failed receipt")
            return receipt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]), \
             mock.patch("src.native_wpp_workflow._write_completion_receipt", side_effect=crash_after_failed_receipt):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor, run=fail_encode)
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            _, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                                     encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                     run=self._resume_fake_run(calls))
        self.assertTrue(failed)
        journal = json.loads((output / ".native-wpp-state" / "journal.json").read_text(encoding="utf-8"))
        self.assertEqual(next(iter(journal["items"].values()))["state"], "failed")

        # Two failed retries followed by success retain all three immutable facts.
        chained, chain_calls = self.root / "retry-chain", {"encodes": 0, "extracts": 0}
        failures = {"left": 2}; chain_base = self._resume_fake_run(chain_calls)
        def fail_twice(cmd, **kwargs):
            result = chain_base(cmd, **kwargs)
            if cmd[0] == "bash" and failures["left"]:
                failures["left"] -= 1
                return subprocess.CompletedProcess(cmd, 1, "", "planned failure")
            return result
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=chained,
                         encoder_script=self.encoder, extractor_script=self.extractor, run=fail_twice)
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=chained,
                         encoder_script=self.encoder, extractor_script=self.extractor, resume=True, retry_failed=True, run=fail_twice)
            _, failed = run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=chained,
                                     encoder_script=self.encoder, extractor_script=self.extractor, resume=True, retry_failed=True, run=fail_twice)
        self.assertFalse(failed)
        chained_journal = json.loads((chained / ".native-wpp-state" / "journal.json").read_text(encoding="utf-8"))
        attempts = next(iter(chained_journal["items"].values()))["attempts"]
        self.assertEqual([attempt["outcome"] for attempt in attempts], ["failed", "failed", "success"])
        self.assertTrue(all((chained / attempt["receipt"]["relative_path"]).is_file() for attempt in attempts))

    def test_corrupt_orphan_and_unrelated_receipt_inventory_reject(self):
        manifest, _ = self.manifest(("orphan.mp4",))
        output, calls = self.root / "orphan-base", {"encodes": 0, "extracts": 0}
        def stop(stage, item):
            if stage == "after_encode":
                raise KeyboardInterrupt
        with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                             encoder_script=self.encoder, extractor_script=self.extractor,
                             run=self._resume_fake_run(calls), test_hook=stop)
        for kind in ("corrupt_orphan", "unrelated"):
            with self.subTest(kind=kind):
                candidate = self.root / f"orphan-{kind}"; shutil.copytree(output, candidate)
                journal = json.loads((candidate / ".native-wpp-state" / "journal.json").read_text(encoding="utf-8"))
                item = next(iter(journal["items"].values())); attempt = item["attempts"][-1]
                receipts = candidate / ".native-wpp-state" / "receipts"
                if kind == "corrupt_orphan":
                    path = receipts / item["source_sha256"] / f"{attempt['id']}.json"; path.parent.mkdir(parents=True); path.write_text("{}", encoding="utf-8")
                else:
                    path = receipts / "unrelated" / "extra.json"; path.parent.mkdir(parents=True); path.write_text("{}", encoding="utf-8")
                with mock.patch("src.native_wpp_workflow._preflight", return_value=[]):
                    with self.assertRaisesRegex(WorkflowError, "receipt"):
                        run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=candidate,
                                     encoder_script=self.encoder, extractor_script=self.extractor, resume=True,
                                     run=self._resume_fake_run({"encodes": 0, "extracts": 0}))

    def test_resume_lock_fails_closed(self):
        manifest, _ = self.manifest()
        output = self.root / "locked"
        lock = self.root / ".locked.native-wpp-build.lock"; lock.mkdir()
        with self.assertRaisesRegex(WorkflowError, "build lock"):
            run_workflow(source_manifest_dir=manifest, source_root=self.sources, output_dir=output,
                         encoder_script=self.encoder, extractor_script=self.extractor)


if __name__ == "__main__":
    unittest.main()
