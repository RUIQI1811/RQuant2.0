from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rquant.config import ProjectPaths, load_config, resolve_paths
from rquant.data.canonical import CanonicalBuilder
from rquant.data.collector import TushareCollector
from rquant.data.qlib_builder import QlibProviderBuilder
from rquant.data.reference import ReferenceBuilder
from rquant.errors import DataContractError, DependencyError, RQuantError
from rquant.factors.catalog import CATALOG_VERSION, FactorSet, get_catalog
from rquant.factors.engine import (
    FactorBuildConfig,
    cpp_compiler_probe,
    create_factor_engine,
    validate_factor_frame,
)
from rquant.factors.evaluation import FactorEvaluationConfig, FactorEvaluator
from rquant.integrations.rdagent import RDAgentExporter, default_mining_window
from rquant.io import atomic_write_json, sha256_file, stable_hash
from rquant.reporting import build_report
from rquant.runs import RunContext
from rquant.universe import SUPPORTED_UNIVERSES
from rquant.workflow.backtest import PortfolioBacktestRunner, PortfolioConfig
from rquant.workflow.runner import RollingWorkflowRunner

LOCKED_DEPENDENCIES = {"pyqlib": "0.9.7", "KunQuant": "0.1.11", "tushare": "1.4.29", "TA-Lib": "0.8.2"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rquant", description="RQuant A-share daily research framework")
    parser.add_argument("--root", default=".", help="Project root (default: current directory)")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration path relative to project root")
    parser.add_argument(
        "--universe",
        choices=SUPPORTED_UNIVERSES,
        help=(
            "Override data.universe: csi300, main (SH/SZ main board without ST and new listings), sh_main "
            "(Shanghai main board only, same screens) or all"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Validate interpreter, dependencies, token and local datasets")
    doctor.add_argument("--skip-permission-check", action="store_true", help="Do not probe Tushare endpoint access")
    doctor.set_defaults(handler=_doctor)

    data = commands.add_parser("data", help="Collect and normalize market data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    sync = data_commands.add_parser("sync", help="Incrementally fetch Tushare data")
    sync.add_argument("--through", type=_date, help="Inclusive date, YYYY-MM-DD (default: data.through)")
    sync.add_argument("--force", action="store_true", help="Refetch validated partitions")
    sync.add_argument("--no-progress", action="store_true", help="Disable the terminal progress bar")
    sync.set_defaults(handler=_data_sync)
    sync_fundamentals = data_commands.add_parser(
        "sync-fundamentals", help="Fetch per-stock Tushare financial statements for the research universe"
    )
    sync_fundamentals.add_argument("--through", type=_date, help="Inclusive date, YYYY-MM-DD (default: data.through)")
    sync_fundamentals.add_argument("--force", action="store_true", help="Refetch validated partitions")
    sync_fundamentals.add_argument("--no-progress", action="store_true", help="Disable the terminal progress bar")
    sync_fundamentals.set_defaults(handler=_data_sync_fundamentals)
    build_qlib = data_commands.add_parser(
        "build-qlib", help="Build canonical Parquet, reference data and Qlib provider"
    )
    build_qlib.set_defaults(handler=_data_build_qlib)
    build_reference = data_commands.add_parser(
        "build-reference", help="Rebuild only the point-in-time reference data next to the canonical bars"
    )
    build_reference.set_defaults(handler=_data_build_reference)

    factors = commands.add_parser("factors", help="Inspect, build and validate factor stores")
    factor_commands = factors.add_subparsers(dest="factor_command", required=True)
    catalog = factor_commands.add_parser("catalog", help="Show the locked canonical factor catalog")
    catalog.add_argument("--factor-set", choices=_factor_sets(), default="jq240")
    catalog.add_argument("--format", choices=("table", "json", "csv"), default="table")
    catalog.add_argument("--output", help="Export JSON or CSV to this path")
    catalog.set_defaults(handler=_factor_catalog)
    factor_build = factor_commands.add_parser("build", help="Compute a registered factor library and write partitions")
    factor_build.add_argument("--factor-set", choices=_factor_sets(), required=True)
    factor_build.add_argument("--start")
    factor_build.add_argument("--end")
    factor_build.add_argument(
        "--external-input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register a project-local external factor input, for example style_factors=data/reference/style.parquet",
    )
    factor_build.set_defaults(handler=_factor_build)
    factor_validate = factor_commands.add_parser("validate", help="Validate stored factor columns and ordering")
    factor_validate.add_argument("--factor-set", choices=_factor_sets(), required=True)
    factor_validate.set_defaults(handler=_factor_validate)
    factor_evaluate = factor_commands.add_parser(
        "evaluate", help="Compute IC and benchmark-adjusted long/short-side effectiveness"
    )
    factor_evaluate.add_argument("--factor-set", choices=_factor_sets(), required=True)
    factor_evaluate.add_argument("--horizon", choices=("1d", "5d", "20d"), required=True)
    factor_evaluate.add_argument("--start", type=_date)
    factor_evaluate.add_argument("--end", type=_date)
    factor_evaluate.add_argument("--quantile", type=float, default=0.2)
    factor_evaluate.add_argument("--min-cross-section", type=int, default=20)
    factor_evaluate.add_argument("--min-effective-days", type=int, default=20)
    factor_evaluate.set_defaults(handler=_factor_evaluate, run_category="factors-evaluate")

    walk = commands.add_parser("walk-forward", help="Run yearly three-year rolling out-of-sample training")
    walk.add_argument("--model", choices=("lgb", "double-ensemble"), required=True)
    walk.add_argument("--horizon", choices=("1d", "5d"), required=True)
    walk.add_argument("--factor-set", choices=_factor_sets(), required=True)
    walk.add_argument("--first-year", type=int)
    walk.add_argument("--last-year", type=int)
    walk.add_argument("--through", type=_date, help="Inclusive date, YYYY-MM-DD (default: data.through)")
    walk.set_defaults(handler=_walk_forward, run_category="walk-forward")

    report = commands.add_parser("report", help="Run portfolio backtest and build the report for a walk-forward run")
    report.add_argument("run_id")
    report.add_argument("--topk", type=int, help="Override backtest.topk")
    report.add_argument("--n-drop", type=int, help="Override backtest.n_drop")
    report.add_argument("--rebalance-days", type=int, help="Override backtest.rebalance_days")
    report.set_defaults(handler=_report, run_category="report")

    rdagent = commands.add_parser("rdagent", help="Exchange data with Microsoft RD-Agent factor mining")
    rdagent_commands = rdagent.add_subparsers(dest="rdagent_command", required=True)
    rdagent_export = rdagent_commands.add_parser(
        "export", help="Write daily price/volume data and settings for RD-Agent fin_factor, cut at the mining end"
    )
    rdagent_export.add_argument(
        "--mining-end",
        type=_date,
        default=date(2021, 12, 31),
        help="Last date RD-Agent may see, YYYY-MM-DD (default 2021-12-31); later data is the holdout",
    )
    rdagent_export.add_argument("--output", help="Bundle directory (default: <data_root>/rdagent)")
    rdagent_export.add_argument(
        "--debug-instruments", type=int, default=100, help="Instruments in RD-Agent's quick debug data (default 100)"
    )
    rdagent_export.set_defaults(handler=_rdagent_export, run_category="rdagent-export")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (RQuantError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


def _factor_sets() -> tuple[str, ...]:
    return get_catalog().factor_sets()


def _factor_external_inputs(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    root = _root(args)
    parsed: dict[str, str] = {}
    for raw in args.external_input:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"Invalid --external-input {raw!r}; expected NAME=PATH")
        if name in parsed:
            raise ValueError(f"Duplicate --external-input name: {name}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"External factor input escapes the project root: {path}")
        parsed[name] = str(path)
    return tuple(parsed.items())


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root).expanduser().resolve()


def _config(args: argparse.Namespace) -> tuple[dict[str, Any], ProjectPaths]:
    root = _root(args)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    if getattr(args, "universe", None):
        config.setdefault("data", {})["universe"] = args.universe
    paths = resolve_paths(config, root)
    paths.create_runtime_dirs()
    return config, paths


def _run(
    args: argparse.Namespace,
    action: Callable[[RunContext, dict[str, Any], ProjectPaths], dict[str, Any]],
) -> int:
    config, paths = _config(args)
    run_category = getattr(args, "run_category", None)
    runs_root = paths.runs / run_category if run_category else paths.runs
    run = RunContext(runs_root, list(sys.argv if sys.argv else ["rquant"]), config)
    run.add_input("dependencies", _dependency_versions())
    try:
        result = action(run, config, paths)
        run.add_output("result", result)
        run.complete()
    except BaseException as exc:
        run.fail(exc)
        raise
    print(
        json.dumps(
            {"run_id": run.run_id, "run_directory": str(run.directory), "result": result}, ensure_ascii=False, indent=2
        )
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    root = _root(args)
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any, *, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    record("python", sys.version_info[:2] == (3, 11), platform.python_version())
    versions = _dependency_versions()
    for package, expected in LOCKED_DEPENDENCIES.items():
        actual = versions.get(package)
        record(package, actual == expected, {"expected": expected, "actual": actual})
    for module in ("qlib", "KunQuant", "tushare", "lightgbm", "talib"):
        try:
            importlib.import_module(module)
            record(f"import:{module}", True, "ok")
        except Exception as exc:
            record(f"import:{module}", False, f"{type(exc).__name__}: {exc}")
    compiler_name, compiler_command, compiler_env = cpp_compiler_probe()
    try:
        compiler = subprocess.run(compiler_command, capture_output=True, env=compiler_env, check=False)
    except OSError as exc:
        record(compiler_name, False, f"missing: {exc}")
    else:
        # cl.exe writes its version banner to stderr in the console (OEM) code page; clang++ writes to stdout.
        streams = (compiler.stderr, compiler.stdout) if sys.platform == "win32" else (compiler.stdout, compiler.stderr)
        encoding = "oem" if sys.platform == "win32" else "utf-8"
        banner = next((stream for stream in streams if stream), b"").decode(encoding, errors="replace")
        record(
            compiler_name,
            compiler.returncode == 0,
            banner.splitlines()[0] if banner else "missing",
        )
    token = os.environ.get("TUSHARE_TOKEN")
    record("TUSHARE_TOKEN", bool(token), "set" if token else "missing")
    record(
        "factor_catalog",
        len(get_catalog().specs) == 2078,
        {"columns": len(get_catalog().specs), "fingerprint": get_catalog().fingerprint},
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    record("config", config_path.exists(), str(config_path))
    for label, path in {
        "raw_manifest": root / "data/raw/sync_manifest.json",
        "canonical_manifest": root / "data/canonical/manifest.json",
        "qlib_manifest": root / "data/qlib/manifest.json",
    }.items():
        record(label, path.exists(), str(path), required=False)
    if token and versions.get("tushare") == LOCKED_DEPENDENCIES["tushare"] and not args.skip_permission_check:
        ok, detail = _probe_tushare_permissions(token)
        record("tushare_permissions", ok, detail)
    elif args.skip_permission_check:
        record("tushare_permissions", True, "skipped by request", required=False)
    else:
        record("tushare_permissions", False, "not checked because tushare or TUSHARE_TOKEN is missing")
    payload = {"status": "ok" if all(item["ok"] for item in checks if item["required"]) else "failed", "checks": checks}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


def _probe_tushare_permissions(token: str) -> tuple[bool, dict[str, Any]]:
    try:
        import tushare as ts
    except ImportError as exc:
        return False, {"error": str(exc)}
    today = date.today()
    recent = (today - timedelta(days=14)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    probes = {
        "stock_basic": {"exchange": "", "list_status": "L", "limit": 1},
        "trade_cal": {"exchange": "SSE", "start_date": recent, "end_date": end},
        "daily": {"ts_code": "000001.SZ", "start_date": recent, "end_date": end},
        "adj_factor": {"ts_code": "000001.SZ", "start_date": recent, "end_date": end},
        "daily_basic": {"ts_code": "000001.SZ", "start_date": recent, "end_date": end},
        "stk_limit": {"ts_code": "000001.SZ", "start_date": recent, "end_date": end},
        "index_daily": {"ts_code": "399300.SZ", "start_date": recent, "end_date": end},
        "index_weight": {"index_code": "399300.SZ", "start_date": recent, "end_date": end},
        "index_member_all": {"is_new": "N", "limit": 1},
        "namechange": {"ts_code": "000001.SZ", "limit": 1},
    }
    client = ts.pro_api(token)
    failures: dict[str, str] = {}
    for endpoint, query in probes.items():
        try:
            method = getattr(client, endpoint, None)
            frame = method(**query) if method else client.query(endpoint, **query)
            if frame is None:
                failures[endpoint] = "returned None"
        except Exception as exc:
            failures[endpoint] = f"{type(exc).__name__}: {exc}"
    return not failures, {"endpoints": sorted(probes), "failures": failures}


def _data_sync(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        data = config.get("data", {})
        max_requests_per_minute = data.get("max_requests_per_minute", 190)
        sync_workers = data.get("sync_workers", 8)
        through = _through(args, config)
        collector = TushareCollector(paths.raw, max_requests_per_minute=max_requests_per_minute)
        run.add_input("through", through.isoformat())
        run.add_input("max_requests_per_minute", max_requests_per_minute)
        run.add_input("sync_workers", sync_workers)
        result = collector.sync(
            formal_start=_date_value(data.get("formal_start", "2010-01-01")),
            through=through,
            warmup_trading_days=int(data.get("warmup_trading_days", 300)),
            index_code=str(data.get("tushare_index_code", "399300.SZ")),
            force=args.force,
            show_progress=not args.no_progress,
            workers=sync_workers,
        )
        return result

    return _run(args, action)


def _data_sync_fundamentals(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        data = config.get("data", {})
        max_requests_per_minute = data.get("max_requests_per_minute", 190)
        sync_workers = data.get("sync_workers", 8)
        start_year = int(data.get("fundamentals_start_year", 2004))
        window_years = int(data.get("fundamentals_window_years", 6))
        through = _through(args, config)
        collector = TushareCollector(paths.raw, max_requests_per_minute=max_requests_per_minute)
        run.add_input("through", through.isoformat())
        run.add_input("universe", _universe(config))
        run.add_input("fundamentals_start_year", start_year)
        run.add_input("fundamentals_window_years", window_years)
        run.add_input("max_requests_per_minute", max_requests_per_minute)
        run.add_input("sync_workers", sync_workers)
        result = collector.sync_fundamentals(
            through=through,
            start_year=start_year,
            universe=_universe(config),
            window_years=window_years,
            force=args.force,
            show_progress=not args.no_progress,
            workers=sync_workers,
        )
        # The full artifact list stays in data/raw/fundamentals_manifest.json; the run manifest keeps the summary.
        return {key: value for key, value in result.items() if key != "files"}

    return _run(args, action)


def _data_build_qlib(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        canonical = CanonicalBuilder(paths.raw, paths.canonical).build()
        run.add_input("raw_fingerprint", canonical.get("raw_fingerprint"))
        reference = ReferenceBuilder(paths.raw, paths.canonical).build()
        provider = QlibProviderBuilder(paths.canonical, paths.qlib).build()
        return {"canonical": canonical, "reference": reference, "qlib": provider}

    return _run(args, action)


def _data_build_reference(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        canonical_manifest = paths.canonical / "manifest.json"
        run.add_input("canonical_manifest", _fingerprint(canonical_manifest))
        with canonical_manifest.open("r", encoding="utf-8") as handle:
            canonical_raw = json.load(handle).get("raw_fingerprint")
        if canonical_raw != _fingerprint(paths.raw / "sync_manifest.json"):
            raise DataContractError("Raw sync changed after the canonical build; run `rquant data build-qlib` instead")
        return ReferenceBuilder(paths.raw, paths.canonical).build()

    return _run(args, action)


def _factor_catalog(args: argparse.Namespace) -> int:
    catalog = get_catalog()
    rows = catalog.rows(args.factor_set)
    if args.output:
        if args.format == "table":
            raise ValueError("--output requires --format json or csv")
        destination = Path(args.output).expanduser().resolve()
        catalog.export(destination, factor_set=args.factor_set, format=args.format)
        print(
            json.dumps(
                {
                    "output": str(destination),
                    "rows": len(rows),
                    "fingerprint": catalog.fingerprint_for(args.factor_set),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.format == "json":
        print(
            json.dumps(
                {
                    "catalog_version": CATALOG_VERSION,
                    "fingerprint": catalog.fingerprint_for(args.factor_set),
                    "factors": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.format == "csv":
        raise ValueError("CSV output requires --output")
    headings = ("canonical_name", "source_name", "family", "ordinal", "max_lookback", "implementation", "formula")
    print("\t".join(headings))
    for row in rows:
        print("\t".join(str(row[key]) for key in headings))
    return 0


def _factor_build(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        factor_config = config.get("factors", {})
        external_inputs = _factor_external_inputs(args)
        engine = create_factor_engine(
            paths.cache,
            FactorBuildConfig(
                factor_set=args.factor_set,
                dtype=str(factor_config.get("dtype", "double")),
                input_layout=str(factor_config.get("input_layout", "TS")),
                output_layout=str(factor_config.get("output_layout", "TS")),
                workers=int(factor_config.get("workers", 4)),
                graph_outputs=int(factor_config.get("graph_outputs", 20)),
                universe=_universe(config),
                external_inputs=external_inputs,
            ),
        )
        run.add_input("catalog_fingerprint", get_catalog().fingerprint_for(args.factor_set))
        run.add_input("external_inputs", dict(external_inputs))
        return engine.build_from_canonical(paths.canonical, paths.factors, start=args.start, end=args.end)

    return _run(args, action)


def _factor_validate(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required to validate factor Parquet") from exc
        factor_set: FactorSet = args.factor_set
        factor_root = paths.factors / factor_set
        manifest_path = factor_root / "manifest.json"
        if not manifest_path.exists():
            raise DataContractError(f"Factor manifest does not exist: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete" or manifest.get("factor_set") != factor_set:
            raise DataContractError(f"Factor manifest is not complete for {factor_set}")
        fingerprint = manifest.get("fingerprint")
        fingerprint_payload = dict(manifest)
        fingerprint_payload.pop("fingerprint", None)
        if fingerprint != stable_hash(fingerprint_payload):
            raise DataContractError("Factor manifest fingerprint is invalid")
        if manifest.get("catalog_fingerprint") != get_catalog().fingerprint_for(factor_set):
            raise DataContractError("Factor manifest catalog fingerprint is stale")
        canonical_manifest_path = paths.canonical / "manifest.json"
        with canonical_manifest_path.open("r", encoding="utf-8") as handle:
            canonical_manifest = json.load(handle)
        if manifest.get("canonical_fingerprint") != canonical_manifest.get("fingerprint"):
            raise DataContractError("Factor store was not built from the current canonical dataset")

        partitions = sorted(factor_root.glob("year=*/factors.parquet"))
        if not partitions:
            raise DataContractError(f"No factor partitions found for {factor_set}")
        actual_years = {partition.parent.name.removeprefix("year=") for partition in partitions}
        recorded_partitions = manifest.get("partitions", {})
        if actual_years != set(recorded_partitions):
            raise DataContractError("Factor partitions differ from the manifest")

        expected_columns: list[str] | None = None
        rows = 0
        first_datetime = None
        last_datetime = None
        instruments: set[str] = set()
        result: dict[str, Any] | None = None
        for partition in partitions:
            year = partition.parent.name.removeprefix("year=")
            if sha256_file(partition) != recorded_partitions[year]:
                raise DataContractError(f"Factor partition hash differs from the manifest: {partition}")
            frame = pd.read_parquet(partition)
            current = validate_factor_frame(frame, factor_set)
            if result is None:
                result = current
                expected_columns = list(frame.columns)
            elif list(frame.columns) != expected_columns:
                raise DataContractError(f"Factor column order drifted in {partition}")
            rows += len(frame)
            partition_start = frame["datetime"].min()
            partition_end = frame["datetime"].max()
            first_datetime = partition_start if first_datetime is None else min(first_datetime, partition_start)
            last_datetime = partition_end if last_datetime is None else max(last_datetime, partition_end)
            instruments.update(frame["instrument"].unique())

        if rows != manifest.get("rows"):
            raise DataContractError("Factor row count differs from the manifest")
        if len(instruments) != manifest.get("instruments"):
            raise DataContractError("Factor instrument count differs from the manifest")
        if str(first_datetime) != manifest.get("start") or str(last_datetime) != manifest.get("end"):
            raise DataContractError("Factor date coverage differs from the manifest")
        if result is None or result["columns"] != manifest.get("columns"):
            raise DataContractError("Factor column count differs from the manifest")
        result.update({"partitions": len(partitions), "rows": rows, "fingerprint": fingerprint})
        return result

    return _run(args, action)


def _factor_evaluate(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        data_config = config.get("data", {})
        start = args.start or _date_value(data_config.get("formal_start", "2010-01-01"))
        factor_manifest_path = paths.factors / args.factor_set / "manifest.json"
        run.add_input("factor_manifest", _fingerprint(factor_manifest_path))
        with factor_manifest_path.open("r", encoding="utf-8") as handle:
            factor_manifest = json.load(handle)
        end = args.end or _date_value(str(factor_manifest.get("end", "")).split()[0])
        run.add_input("qlib_manifest", _fingerprint(paths.qlib / "manifest.json"))
        evaluation_config = FactorEvaluationConfig(
            factor_set=args.factor_set,
            horizon=args.horizon,
            benchmark=str(data_config.get("benchmark", "SH000300")),
            quantile=args.quantile,
            min_cross_section=args.min_cross_section,
            min_effective_days=args.min_effective_days,
        )
        run.add_input("factor_evaluation", asdict(evaluation_config) | {"start": start, "end": end})
        evaluator = FactorEvaluator(
            qlib_root=paths.qlib,
            factor_root=paths.factors,
            output_directory=run.directory,
            config=evaluation_config,
        )
        return evaluator.run(start=start, end=end)

    return _run(args, action)


def _walk_forward(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        workflow = config.get("workflow", {})
        through = _through(args, config)
        first_year = args.first_year or int(workflow.get("first_prediction_year", 2013))
        last_year = args.last_year or through.year
        run.add_input("qlib_manifest", _fingerprint(paths.qlib / "manifest.json"))
        run.add_input("factor_manifest", _fingerprint(paths.factors / args.factor_set / "manifest.json"))
        _require_factor_universe(paths.factors / args.factor_set / "manifest.json", _universe(config))
        runner = RollingWorkflowRunner(
            qlib_root=paths.qlib,
            factor_root=paths.factors,
            run_directory=run.directory,
            model=args.model,
            horizon=args.horizon,
            factor_set=args.factor_set,
            seed=int(workflow.get("seed", config.get("project", {}).get("seed", 42))),
            train_years=int(workflow.get("train_years", 3)),
            validation_months=int(workflow.get("validation_months", 6)),
            formal_start_year=_date_value(config.get("data", {}).get("formal_start", "2010-01-01")).year,
            universe=_universe(config),
        )
        run.add_input("through", through.isoformat())
        return runner.run(first_year=first_year, last_year=last_year, through=through)

    return _run(args, action)


def _rdagent_export(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        formal_start = _date_value(config.get("data", {}).get("formal_start", "2010-01-01"))
        window = default_mining_window(formal_start, args.mining_end)
        output = Path(args.output).expanduser() if args.output else paths.data / "rdagent"
        if not output.is_absolute():
            output = paths.root / output
        run.add_input("qlib_manifest", _fingerprint(paths.qlib / "manifest.json"))
        exporter = RDAgentExporter(paths.qlib, output, universe=_universe(config))
        manifest = exporter.export(window, debug_instruments=args.debug_instruments)
        return {"output": str(output), **manifest}

    return _run(args, action)


def _resolve_report_run_directory(runs_root: str | Path, run_id: str) -> Path:
    root = Path(runs_root)
    categorized = root / "walk-forward" / run_id
    if categorized.is_dir():
        return categorized
    legacy = root / run_id
    if legacy.is_dir():
        return legacy
    raise DataContractError(f"Run does not exist: {categorized}")


def _report(args: argparse.Namespace) -> int:
    def action(run: RunContext, config: dict[str, Any], paths: ProjectPaths) -> dict[str, Any]:
        run_directory = _resolve_report_run_directory(paths.runs, args.run_id)
        predictions_path = run_directory / "predictions.parquet"
        if not predictions_path.exists():
            raise DataContractError(f"Predictions do not exist: {predictions_path}")
        backtest = dict(config.get("backtest", {}))
        workflow_path = run_directory / "walk_forward.json"
        if workflow_path.exists():
            with workflow_path.open("r", encoding="utf-8") as handle:
                horizon = json.load(handle).get("horizon")
            backtest.update(backtest.get("by_horizon", {}).get(horizon, {}))
        portfolio_config = PortfolioConfig(
            initial_capital=float(backtest.get("initial_capital", 1_000_000)),
            topk=args.topk if args.topk is not None else int(backtest.get("topk", 50)),
            n_drop=args.n_drop if args.n_drop is not None else int(backtest.get("n_drop", 5)),
            benchmark=str(config.get("data", {}).get("benchmark", "SH000300")),
            trade_unit=int(backtest.get("trade_unit", 100)),
            commission_rate=float(backtest.get("commission_rate", 0.0003)),
            minimum_commission=float(backtest.get("minimum_commission", 5.0)),
            stamp_tax_before=float(backtest.get("stamp_tax_before_2023_08_28", 0.001)),
            stamp_tax_after=float(backtest.get("stamp_tax_from_2023_08_28", 0.0005)),
            impact_cost=float(backtest.get("impact_cost", 0.0)),
            universe=_walk_forward_universe(run_directory, config),
            rebalance_days=(
                args.rebalance_days if args.rebalance_days is not None else int(backtest.get("rebalance_days", 1))
            ),
        )
        run.add_input("walk_forward_run_id", args.run_id)
        run.add_input("walk_forward_run_directory", str(run_directory))
        run.add_input("predictions_sha256", sha256_file(predictions_path))
        run.add_input("qlib_manifest", _fingerprint(paths.qlib / "manifest.json"))
        run.add_input("portfolio", asdict(portfolio_config))
        portfolio = PortfolioBacktestRunner(
            qlib_root=paths.qlib,
            run_directory=run_directory,
            config=portfolio_config,
        ).run()
        report = build_report(run_directory)
        manifest_path = run_directory / "run.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                run_manifest = json.load(handle)
            run_manifest.setdefault("outputs", {})["backtest"] = portfolio
            run_manifest["outputs"]["report"] = report
            run_manifest["report_run_id"] = run.run_id
            run_manifest["report_completed_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(manifest_path, run_manifest)
            with (run_directory / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{run_manifest['report_completed_at']} report completed run_id={run.run_id}\n")
        return {"walk_forward_run_id": args.run_id, "backtest": portfolio, "report": report}

    return _run(args, action)


def _dependency_versions() -> dict[str, str | None]:
    packages = [*LOCKED_DEPENDENCIES, "numpy", "pandas", "pyarrow", "lightgbm", "PyYAML"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _fingerprint(path: Path) -> Any:
    if not path.exists():
        raise DataContractError(f"Required manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise DataContractError(f"Manifest is not complete: {path}")
    return payload.get("fingerprint") or payload.get("catalog_fingerprint")


def _through(args: argparse.Namespace, config: dict[str, Any]) -> date:
    """Resolve the inclusive end date from the command line or config; never default to the wall clock."""
    if args.through is not None:
        return args.through
    configured = config.get("data", {}).get("through")
    if configured is None:
        raise ValueError("--through is required unless data.through is set in the configuration")
    return _date_value(configured)


def _require_factor_universe(manifest_path: Path, universe: str) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        built = json.load(handle).get("universe", "csi300")
    if built != universe:
        raise DataContractError(
            f"Factor store {manifest_path.parent} was built for universe {built!r}, not {universe!r}; "
            "rebuild it with `rquant factors build` for this universe"
        )


def _walk_forward_universe(run_directory: Path, config: dict[str, Any]) -> str:
    """The universe a walk-forward run trained on, so its report backtests the same universe."""
    manifest_path = run_directory / "walk_forward.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            universe = json.load(handle).get("universe")
        if universe:
            return str(universe)
    return _universe(config)


def _universe(config: dict[str, Any]) -> str:
    universe = str(config.get("data", {}).get("universe", "csi300"))
    if universe not in SUPPORTED_UNIVERSES:
        raise ValueError(f"data.universe must be one of {SUPPORTED_UNIVERSES}, got {universe!r}")
    return universe


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _date_value(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
