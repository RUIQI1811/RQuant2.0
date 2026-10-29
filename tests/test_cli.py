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
            ["factors", "build", "--factor-set", "oap_jq_cnc"],
            ["factors", "validate", "--factor-set", "wq_alpha101"],
            ["factors", "evaluate", "--factor-set", "oap_jq_cnc", "--horizon", "20d"],
            ["walk-forward", "--model", "lgb", "--horizon", "1d", "--factor-set", "oap_jq_cnc"],
            ["report", "run-id"],
            ["rdagent", "export", "--mining-end", "2021-12-31"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertTrue(callable(parser.parse_args(argv).handler))

    def test_research_commands_have_run_categories(self) -> None:
        parser = build_parser()
        factor_args = parser.parse_args(
            ["factors", "evaluate", "--factor-set", "oap_jq_cnc", "--horizon", "20d"]
        )
        walk_args = parser.parse_args(
            ["walk-forward", "--model", "lgb", "--horizon", "1d", "--factor-set", "oap_jq_cnc"]
        )
        report_args = parser.parse_args(["report", "run-id"])
        doctor_args = parser.parse_args(["doctor", "--skip-permission-check"])

        self.assertEqual("factors-evaluate", factor_args.run_category)
        self.assertEqual("walk-forward", walk_args.run_category)
        self.assertEqual("report", report_args.run_category)
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

    def test_through_comes_from_arguments_or_config_never_the_clock(self) -> None:
        from datetime import date

        from rquant.cli import _through

        self.assertEqual(
            date(2024, 1, 5), _through(argparse.Namespace(through=date(2024, 1, 5)), {"data": {"through": None}})
        )
        configured = {"data": {"through": "2024-02-01"}}
        self.assertEqual(date(2024, 2, 1), _through(argparse.Namespace(through=None), configured))
        with self.assertRaisesRegex(ValueError, "--through is required"):
            _through(argparse.Namespace(through=None), {"data": {"through": None}})

    def test_universe_is_validated(self) -> None:
        from rquant.cli import _universe

        self.assertEqual("csi300", _universe({}))
        self.assertEqual("all", _universe({"data": {"universe": "all"}}))
        self.assertEqual("main", _universe({"data": {"universe": "main"}}))
        with self.assertRaisesRegex(ValueError, "data.universe"):
            _universe({"data": {"universe": "csi500"}})

    def test_universe_flag_overrides_config_and_separates_factor_stores(self) -> None:
        import json

        from rquant.cli import _config, _require_factor_universe, _walk_forward_universe, build_parser

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "config" / "default.yaml").write_text("data:\n  universe: csi300\n", encoding="utf-8")
            args = build_parser().parse_args(["--root", temporary, "--universe", "main", "doctor"])
            config, paths = _config(args)
            self.assertEqual("main", config["data"]["universe"])
            self.assertEqual(root / "data" / "factors-main", paths.factors)

            manifest = root / "factors" / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({"universe": "csi300"}), encoding="utf-8")
            _require_factor_universe(manifest, "csi300")
            with self.assertRaisesRegex(DataContractError, "built for universe 'csi300', not 'main'"):
                _require_factor_universe(manifest, "main")

            run = root / "run"
            run.mkdir()
            self.assertEqual("csi300", _walk_forward_universe(run, {}))
            (run / "walk_forward.json").write_text(json.dumps({"universe": "main"}), encoding="utf-8")
            self.assertEqual("main", _walk_forward_universe(run, {}))

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
        clang = subprocess.CompletedProcess(
            ["clang++", "--version"], 0, stdout=b"Apple clang version fixture\n", stderr=b""
        )
        # Replacing importlib.import_module affects how later patch targets resolve, so it must be patched last.
        with (
            patch.dict("os.environ", {"TUSHARE_TOKEN": "fixture"}),
            patch("rquant.cli._dependency_versions", return_value=LOCKED_DEPENDENCIES),
            patch("rquant.cli.cpp_compiler_probe", return_value=("clang++", ["clang++", "--version"], None)),
            patch("rquant.cli.subprocess.run", return_value=clang),
            patch("rquant.cli.importlib.import_module"),
            redirect_stdout(output),
        ):
            code = main(["doctor", "--skip-permission-check"])

        payload = json.loads(output.getvalue())
        factor_catalog = next(check for check in payload["checks"] if check["name"] == "factor_catalog")
        self.assertEqual(0, code)
        self.assertEqual("ok", payload["status"])
        self.assertTrue(factor_catalog["ok"])
        self.assertEqual(2063, factor_catalog["detail"]["columns"])

    def test_doctor_reports_missing_clang_instead_of_crashing(self) -> None:
        output = io.StringIO()
        with (
            patch.dict("os.environ", {"TUSHARE_TOKEN": "fixture"}),
            patch("rquant.cli._dependency_versions", return_value=LOCKED_DEPENDENCIES),
            patch("rquant.cli.cpp_compiler_probe", return_value=("clang++", ["clang++", "--version"], None)),
            patch("rquant.cli.subprocess.run", side_effect=FileNotFoundError("clang++")),
            patch("rquant.cli.importlib.import_module"),
            redirect_stdout(output),
        ):
            code = main(["doctor", "--skip-permission-check"])

        payload = json.loads(output.getvalue())
        clang = next(check for check in payload["checks"] if check["name"] == "clang++")
        self.assertEqual(1, code)
        self.assertFalse(clang["ok"])
        self.assertIn("missing", clang["detail"])


if __name__ == "__main__":
    unittest.main()
