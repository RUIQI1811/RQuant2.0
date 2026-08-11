from __future__ import annotations

import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rquant.cli import LOCKED_DEPENDENCIES, build_parser, main


class CliTests(unittest.TestCase):
    def test_required_command_tree_parses(self) -> None:
        parser = build_parser()
        cases = [
            ["doctor", "--skip-permission-check"],
            ["data", "sync", "--through", "2025-01-01"],
            ["data", "sync", "--through", "2025-01-01", "--no-progress"],
            ["data", "build-qlib"],
            ["factors", "catalog"],
            ["factors", "build", "--factor-set", "combined"],
            ["factors", "validate", "--factor-set", "wq_alpha101"],
            ["factors", "evaluate", "--factor-set", "combined", "--horizon", "20d"],
            ["walk-forward", "--model", "lgb", "--horizon", "1d", "--factor-set", "combined"],
            ["report", "run-id"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertTrue(callable(parser.parse_args(argv).handler))

    def test_catalog_json_is_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["factors", "catalog", "--factor-set", "wq_alpha101", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual(101, len(payload["factors"]))
        self.assertEqual("a101_001", payload["factors"][0]["canonical_name"])
        self.assertEqual("a101_101", payload["factors"][-1]["canonical_name"])

    def test_alpha360_catalog_json_is_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["factors", "catalog", "--factor-set", "qlib_alpha360", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual(360, len(payload["factors"]))
        self.assertEqual("a360_001", payload["factors"][0]["canonical_name"])
        self.assertEqual("a360_360", payload["factors"][-1]["canonical_name"])

    def test_doctor_accepts_the_registered_factor_catalog(self) -> None:
        output = io.StringIO()
        clang = subprocess.CompletedProcess(["clang++", "--version"], 0, stdout="Apple clang version fixture\n")
        with (
            patch("rquant.cli._dependency_versions", return_value=LOCKED_DEPENDENCIES),
            patch("rquant.cli.importlib.import_module"),
            patch("rquant.cli.subprocess.run", return_value=clang),
            patch.dict("os.environ", {"TUSHARE_TOKEN": "fixture"}),
            redirect_stdout(output),
        ):
            code = main(["doctor", "--skip-permission-check"])

        payload = json.loads(output.getvalue())
        factor_catalog = next(check for check in payload["checks"] if check["name"] == "factor_catalog")
        self.assertEqual(0, code)
        self.assertEqual("ok", payload["status"])
        self.assertTrue(factor_catalog["ok"])
        self.assertEqual(809, factor_catalog["detail"]["columns"])


if __name__ == "__main__":
    unittest.main()
