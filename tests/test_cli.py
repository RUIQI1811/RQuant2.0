from __future__ import annotations

import argparse
import io
import json
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rquant.cli import LOCKED_DEPENDENCIES, _resolve_report_run_directory, _run, build_parser, main
from rquant.config import ProjectPaths
from rquant.errors import DataContractError


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

    def test_only_factor_evaluate_and_walk_forward_have_run_categories(self) -> None:
        parser = build_parser()
        factor_args = parser.parse_args(
            ["factors", "evaluate", "--factor-set", "combined", "--horizon", "20d"]
        )
        walk_args = parser.parse_args(
            ["walk-forward", "--model", "lgb", "--horizon", "1d", "--factor-set", "combined"]
        )
        doctor_args = parser.parse_args(["doctor", "--skip-permission-check"])

        self.assertEqual("factors-evaluate", factor_args.run_category)
        self.assertEqual("walk-forward", walk_args.run_category)
        self.assertFalse(hasattr(doctor_args, "run_category"))

    def test_run_uses_category_beneath_configured_runs_root(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = ProjectPaths.from_root(temporary, runs_root="custom-runs")
            args = argparse.Namespace(run_category="walk-forward")
            output = io.StringIO()
            with (
                patch("rquant.cli._config", return_value=({}, paths)),
                patch("rquant.cli._dependency_versions", return_value={}),
                redirect_stdout(output),
            ):
                code = _run(args, lambda run, config, project_paths: {"ok": True})

            payload = json.loads(output.getvalue())
            run_directory = Path(payload["run_directory"])
            self.assertEqual(0, code)
            self.assertEqual(paths.runs / "walk-forward", run_directory.parent)
            self.assertTrue((run_directory / "run.json").is_file())

    def test_run_without_category_remains_directly_beneath_runs_root(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = ProjectPaths.from_root(temporary, runs_root="custom-runs")
            output = io.StringIO()
            with (
                patch("rquant.cli._config", return_value=({}, paths)),
                patch("rquant.cli._dependency_versions", return_value={}),
                redirect_stdout(output),
            ):
                code = _run(argparse.Namespace(), lambda run, config, project_paths: {"ok": True})

            payload = json.loads(output.getvalue())
            run_directory = Path(payload["run_directory"])
            self.assertEqual(0, code)
            self.assertEqual(paths.runs, run_directory.parent)

    def test_report_run_resolution_prefers_categorized_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            categorized = runs_root / "walk-forward" / "same-id"
            legacy = runs_root / "same-id"
            categorized.mkdir(parents=True)
            legacy.mkdir()

            self.assertEqual(categorized, _resolve_report_run_directory(runs_root, "same-id"))

    def test_report_run_resolution_falls_back_to_legacy_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            legacy = runs_root / "legacy-id"
            legacy.mkdir()

            self.assertEqual(legacy, _resolve_report_run_directory(runs_root, "legacy-id"))

    def test_report_run_resolution_rejects_missing_run_without_creating_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            expected = runs_root / "walk-forward" / "missing-id"

            with self.assertRaises(DataContractError) as caught:
                _resolve_report_run_directory(runs_root, "missing-id")

            self.assertEqual(f"Run does not exist: {expected}", str(caught.exception))
            self.assertFalse(expected.exists())

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
