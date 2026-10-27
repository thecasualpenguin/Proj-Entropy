"""Structural guards for the inert class-attribution notebook."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import unittest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "wpp_class_attribution.ipynb"
CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class WPPClassAttributionNotebookIntegrityTests(unittest.TestCase):
    def test_notebook_is_utf8_json_with_valid_v4_structure(self):
        raw = NOTEBOOK.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "notebooks must not have a UTF-8 BOM")
        notebook = json.loads(raw.decode("utf-8"))

        self.assertEqual(4, notebook["nbformat"])
        self.assertIsInstance(notebook["nbformat_minor"], int)
        self.assertEqual("python3", notebook["metadata"]["kernelspec"]["name"])
        self.assertEqual("python", notebook["metadata"]["kernelspec"]["language"])
        self.assertEqual("python", notebook["metadata"]["language_info"]["name"])

        cells = notebook["cells"]
        self.assertTrue(cells)
        ids = []
        for cell in cells:
            self.assertIn(cell["cell_type"], {"code", "markdown", "raw"})
            self.assertIsInstance(cell["metadata"], dict)
            self.assertIsInstance(cell["id"], str)
            self.assertRegex(cell["id"], CELL_ID)
            ids.append(cell["id"])
            source = cell["source"]
            self.assertTrue(isinstance(source, str) or (
                isinstance(source, list) and all(isinstance(line, str) for line in source)
            ))
            if cell["cell_type"] == "code":
                self.assertIn("outputs", cell)
                self.assertTrue(cell["execution_count"] is None or isinstance(cell["execution_count"], int))
                compile("".join(source) if isinstance(source, list) else source, str(NOTEBOOK), "exec")
        self.assertEqual(len(ids), len(set(ids)), "Jupyter cell IDs must be unique")

    def test_controls_use_local_path_boundary_before_constructing_ui(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        controls = next(cell for cell in notebook["cells"] if cell["id"] == "class-attribution-controls")
        source = "".join(controls["source"])
        self.assertIn("local_config_path", source)
        self.assertIn("LOCAL_ROOTS = {name: local_config_path(path)", source)
        self.assertIn("training_root=LOCAL_ROOTS['run_dir']", source)
        self.assertIn("cache_dir=LOCAL_ROOTS['cache_dir']", source)
        self.assertIn("Configured local cache path:", source)

    @unittest.skipUnless(importlib.util.find_spec("nbformat"), "nbformat is an optional notebook dependency")
    def test_nbformat_schema_validation_when_notebook_dependencies_are_installed(self):
        import nbformat

        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)


if __name__ == "__main__":
    unittest.main()
