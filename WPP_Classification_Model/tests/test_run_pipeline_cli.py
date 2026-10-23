from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunPipelineCliTests(unittest.TestCase):
    def test_malformed_yaml_is_clean_argparse_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "bad.yaml"
            config.write_text("run: [unterminated\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "run_pipeline.py"), "train", "--config", str(config)],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("error:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_export_report_invalid_source_is_clean_argparse_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "run_pipeline.py"), "export-report", "--run", str(root / "missing"),
                 "--output-dir", str(root / "report"), "--reconstruct-train-metrics"],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("error:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
