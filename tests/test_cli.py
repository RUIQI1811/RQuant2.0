from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from rquant.cli import build_parser, main


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


if __name__ == "__main__":
    unittest.main()
