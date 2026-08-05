from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rquant.factors.engine import _atomic_replace_directory


class FactorDirectoryPublishTests(unittest.TestCase):
    def test_atomic_replace_directory_publishes_complete_staging_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "combined"
            staging = root / ".combined.staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "manifest.json").write_text("old", encoding="utf-8")
            (staging / "manifest.json").write_text("new", encoding="utf-8")

            _atomic_replace_directory(staging, destination)

            self.assertEqual("new", (destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(staging.exists())
            self.assertEqual([], list(root.glob(".combined.backup.*")))

    def test_atomic_replace_directory_restores_previous_tree_when_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "combined"
            staging = root / ".combined.staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "manifest.json").write_text("old", encoding="utf-8")
            (staging / "manifest.json").write_text("new", encoding="utf-8")
            real_replace = os.replace

            def replace_with_publish_failure(source: str | Path, target: str | Path) -> None:
                if Path(source) == staging:
                    raise OSError("simulated publish failure")
                real_replace(source, target)

            with mock.patch("rquant.factors.engine.os.replace", side_effect=replace_with_publish_failure):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    _atomic_replace_directory(staging, destination)

            self.assertEqual("old", (destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("new", (staging / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".combined.backup.*")))


if __name__ == "__main__":
    unittest.main()
