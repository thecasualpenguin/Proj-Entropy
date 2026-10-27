from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from scripts import build_wpp_class_attribution_cache as cli
from src.wpp_class_attribution import CacheBuildResult


class ClassAttributionCliTests(unittest.TestCase):
    def test_direct_script_help(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_wpp_class_attribution_cache.py"), "--help"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--max-new-videos", completed.stdout)
        self.assertIn("--attention-chunk-queries", completed.stdout)

    def test_defaults_and_explicit_paths_map_to_cache_builder(self):
        defaults = cli.parse_args([])
        self.assertEqual(defaults.training_root, cli.DEFAULT_TRAINING_ROOT)
        self.assertEqual(defaults.ig_steps, 32)
        self.assertEqual(defaults.attention_chunk_queries, 128)

        result = CacheBuildResult(Path("/tmp/cache"), True, 2, 2, 1, 1, "checkpoint")
        with patch.object(cli, "build_class_attribution_cache", return_value=result) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli.main([
                "--training-root", "/tmp/training",
                "--packed-bundle", "/tmp/packed",
                "--source-manifest", "/tmp/manifest",
                "--extraction-index", "/tmp/index",
                "--source-videos", "/tmp/videos",
                "--cache-dir", "/tmp/cache",
                "--device", "cpu",
                "--ig-steps", "8",
                "--attention-chunk-queries", "64",
                "--max-new-videos", "1",
            ])

        self.assertEqual(exit_code, 0)
        kwargs = build.call_args.kwargs
        self.assertEqual(build.call_args.args[0], "/tmp/training")
        self.assertEqual(kwargs["cache_dir"], "/tmp/cache")
        self.assertEqual(kwargs["splits"], ("train", "test"))
        self.assertEqual(kwargs["ig_steps"], 8)
        self.assertEqual(kwargs["attention_chunk_queries"], 64)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["max_new_videos"], 1)
        self.assertEqual(kwargs["roots"].packed_bundle, "/tmp/packed")
        self.assertEqual(kwargs["roots"].source_manifest, "/tmp/manifest")
        self.assertEqual(kwargs["roots"].index, "/tmp/index")
        self.assertEqual(kwargs["roots"].source_video_root, "/tmp/videos")
        self.assertTrue(callable(kwargs["progress"]))
        self.assertTrue(callable(kwargs["should_cancel"]))


if __name__ == "__main__":
    unittest.main()
