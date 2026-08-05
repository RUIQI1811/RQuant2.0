from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from rquant.runs import RunContext


class RunManifestTests(unittest.TestCase):
    def test_success_and_failure_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            successful = RunContext(temporary, ["rquant", "doctor"], {"seed": 42})
            successful.add_input("data", "fingerprint")
            successful.add_output("rows", 3)
            successful.complete()
            with successful.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual("complete", manifest["status"])
            self.assertIn("started", successful.log_path.read_text(encoding="utf-8"))
            self.assertIn("completed", successful.log_path.read_text(encoding="utf-8"))

            failed = RunContext(Path(temporary), ["rquant", "data", "sync"], {})
            failed.fail(ValueError("fixture"))
            with failed.manifest_path.open("r", encoding="utf-8") as handle:
                failed_manifest = json.load(handle)
            self.assertEqual("failed", failed_manifest["status"])
            self.assertIn("ValueError: fixture", failed_manifest["error"])

    def test_yaml_dates_are_serialized_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {"data": {"formal_start": date(2010, 1, 1)}}
            run = RunContext(temporary, ["rquant", "data", "sync"], config)
            run.complete()
            with run.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual("2010-01-01", manifest["config"]["data"]["formal_start"])
            self.assertEqual(64, len(manifest["config_fingerprint"]))


if __name__ == "__main__":
    unittest.main()
