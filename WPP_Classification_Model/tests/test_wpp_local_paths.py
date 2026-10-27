from pathlib import PureWindowsPath
import unittest

from src.wpp_class_attribution_ui import DEFAULT_ROOTS
from src.wpp_local_paths import normalize_local_path_text


class LocalConfigurationPathTests(unittest.TestCase):
    def test_windows_maps_wsl_drive_mount_before_path_resolution(self):
        self.assertEqual(
            r"D:\projects\proj-entropy\derived\native_wpp",
            normalize_local_path_text("/mnt/d/projects/proj-entropy/derived/native_wpp", windows=True),
        )
        # This is how a WSL-form default may arrive after PureWindowsPath has
        # rendered it; the helper remains filesystem-free and deterministic.
        self.assertEqual(
            r"D:\projects\proj-entropy",
            normalize_local_path_text(PureWindowsPath("/mnt/d/projects/proj-entropy"), windows=True),
        )

    def test_windows_maps_every_approved_default_and_keeps_other_inputs_verbatim(self):
        for value in DEFAULT_ROOTS.values():
            self.assertTrue(normalize_local_path_text(value, windows=True).startswith("D:\\projects\\"))
        for value in (r"E:\owned\cache", r"\\server\share\cache", r"\\mnt\d\cache", "relative/cache", "/outside/mnt/d"):
            self.assertEqual(value, normalize_local_path_text(value, windows=True))

    def test_wsl_linux_does_not_rewrite_approved_defaults(self):
        value = "/mnt/d/projects/proj-entropy/derived/native_wpp"
        self.assertEqual(value, normalize_local_path_text(value, windows=False))


if __name__ == "__main__":
    unittest.main()
