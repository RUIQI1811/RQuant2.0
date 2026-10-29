"""Source-timing China anomaly definitions, independently adapted to Tushare.

Li, Liu, Liu and Wei (2024), Management Science, appendix A.4.18–21.
Source CSMAR values and Tushare values have not been shown equal. Annual
signals form at June close; quarter signals form at the first monthly trading
day using reports known before that day. A missing prior report falls back to
the most recent eligible quarter as a project-specific conservative policy.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rquant.errors import DataContractError
from rquant.factors import formation_dates, formation_grid, statements
from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.io import sha256_file, stable_hash

FIELDS = ("total_assets", "operate_profit", "n_income", "n_cashflow_act")
EQUITY_FIELD = "total_hldr_eqy_inc_min_int"
KEYS = ("instrument", "end_date", "ann_date")


def _check_legacy_calendar(canonical_root: Path, calendar: pd.DataFrame, raw_fingerprint: str) -> str:
    """Verify legacy unpinned calendars against the recorded Tushare partitions."""
    raw = canonical_root.parent / "raw"
    sync_path = raw / "sync_manifest.json"
    if not sync_path.is_file():
        raise DataContractError("china469_pit needs raw trade calendar manifests for a legacy reference")
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    if sync.get("status") != "complete" or sync.get("fingerprint") != raw_fingerprint:
        raise DataContractError("china469_pit raw sync differs from the canonical reference")
    paths = sorted((raw / "trade_cal").glob("**/data.parquet"))
    if not paths:
        raise DataContractError("china469_pit raw trade calendar partitions are missing")
    parts, hashes = [], {}
    for path in paths:
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise DataContractError(f"china469_pit raw trade calendar manifest missing: {manifest_path}")
        info = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if info.get("status") != "complete" or info.get("endpoint") != "trade_cal" or info.get("sha256") != digest:
            raise DataContractError(f"china469_pit raw trade calendar partition failed integrity: {path}")
        if "query" in info and info.get("query_fingerprint") != stable_hash(info["query"]):
            raise DataContractError(f"china469_pit raw trade calendar query fingerprint mismatch: {path}")
        hashes[path.relative_to(raw).as_posix()] = digest
        parts.append(pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"]))
    columns = ["exchange", "cal_date", "is_open"]
    expected = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(columns).reset_index(drop=True)
    observed = calendar[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)
    if not expected.equals(observed):
        raise DataContractError("china469_pit reference differs from verified raw trade calendar")
    return stable_hash(hashes)


def load_reference(canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict[str, str]]:
    from rquant.data.reference import load_reference_manifest

    root = Path(canonical_root) / "reference"
    manifest = load_reference_manifest(canonical_root)
    actual = {}
    for name in ("fundamentals.parquet", "trade_cal.parquet"):
        path = root / name
        if not path.is_file():
            raise DataContractError(f"china469_pit needs reference file: {path}")
        actual[name] = sha256_file(path)
        expected = manifest["files"].get(name)
        if expected is not None and actual[name] != expected:
            raise DataContractError(f"china469_pit reference differs from manifest: {path}")
    if "fundamentals.parquet" not in manifest["files"]:
        raise DataContractError("china469_pit fundamentals must be pinned by the reference manifest")
    fundamentals = pd.read_parquet(root / "fundamentals.parquet", columns=[*KEYS, *FIELDS, EQUITY_FIELD],
                                   filters=[("instrument", "in", list(instruments))])
    calendar = pd.read_parquet(root / "trade_cal.parquet", columns=["exchange", "cal_date", "is_open"])
    raw_hash = None
    if "trade_cal.parquet" not in manifest["files"]:
        raw_hash = _check_legacy_calendar(Path(canonical_root), calendar, manifest["raw_fingerprint"])
    calendar = calendar[calendar.exchange.eq("SSE")].copy()
    if calendar.empty or calendar.cal_date.duplicated().any() or not calendar.is_open.isin([0, 1]).all():
        raise DataContractError("china469_pit requires a complete unique SSE trade calendar")
    calendar["date"] = pd.to_datetime(calendar.cal_date.astype(str), format="%Y%m%d", errors="coerce")
    if calendar.date.isna().any():
        raise DataContractError("china469_pit trade calendar contains invalid dates")
    calendar = calendar.sort_values("date")
    expected_days = pd.date_range(calendar.date.iloc[0], calendar.date.iloc[-1], freq="D")
    if not pd.DatetimeIndex(calendar.date).equals(expected_days):
        raise DataContractError("china469_pit SSE calendar has missing calendar dates")
    output_dates = pd.DatetimeIndex(dates)
    open_dates = pd.DatetimeIndex(calendar.loc[calendar.is_open.eq(1), "date"])
    if len(output_dates) and not (open_dates > output_dates[-1] + pd.offsets.MonthEnd(0)).any():
        raise DataContractError("china469_pit requires a next-month open day after the final output month end")
    if (open_dates.get_indexer(output_dates) < 0).any():
        raise DataContractError("china469_pit output dates disagree with the SSE trading calendar")
    hashes = {
        "china469_reference": manifest["fingerprint"],
        "china469_fundamentals": actual["fundamentals.parquet"],
        "china469_trade_calendar": actual["trade_cal.parquet"],
    }
    if raw_hash is not None:
        hashes["china469_raw_trade_partitions"] = raw_hash
    return {"fundamentals": fundamentals, "formation_calendar": open_dates}, hashes


class _Base(FactorLibrary):
    required_inputs = ("close",)
    statement_fields = FIELDS
    strict_float = True
    mode: str
    source: str
    column: str
    output: str
    ordinal: int

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return (FactorSpec(self.column, self.source, self.family, self.ordinal,
                           f"{self.source}: Tushare report-based anomaly with frozen formation dates",
                           1, f"{__name__}.{self.source}", 1),)

    @property
    def implementation_fingerprint(self) -> str:
        return stable_hash({
            "module": super().implementation_fingerprint,
            "formation_grid": inspect.getsource(formation_grid),
            "formation_dates": inspect.getsource(formation_dates),
            "statements": inspect.getsource(statements),
        })

    def make_statement_grid(self, inputs: dict[str, Any], dates: Any, instruments: Any) -> Any:
        frame = inputs["fundamentals"].loc[:, [*KEYS, *self.statement_fields]]
        return formation_grid.FormationGrid(
            frame, dates, instruments, full_calendar=inputs["formation_calendar"], mode=self.mode
        )

    def build_statements(self, inputs: dict[str, Any]) -> dict[str, Any]:
        profit, income, cash = (inputs[name] for name in ("operate_profit", "n_income", "n_cashflow_act"))
        if self.mode == "monthly_previous":
            quarter = inputs["fiscal_quarter"]
            profit, income, cash = (k.select(k.eq(quarter, 0), value, value - k.delay(value, 1))
                                    for value in (profit, income, cash))
            denominator = k.delay(inputs["total_assets"], 1)
        else:
            denominator = inputs[EQUITY_FIELD] if self.source == "Cbe" else inputs["total_assets"]
            if self.mode == "annual_previous":
                denominator = k.delay(denominator, 4)
        value = k.replace_inf((profit - income + cash) / denominator)
        return {self.output: k.mask(value, k.ne(denominator, 0.0))}

    def build(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}


class China469AnnualCurrentLibrary(_Base):
    family = "china469_cop"
    mode, source, column, output, ordinal = "annual_current", "Cop", "cnax_001", "y_cnax_cop", 1
    statement_outputs = (output,)
    reference_inputs = (output,)

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        return load_reference(canonical_root, dates, instruments)

    def build(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {self.source: k.mask(inputs[self.output], k.valid(k.replace_inf(inputs["close"])))}


class China469AnnualPreviousLibrary(_Base):
    family = "china469_cla"
    mode, source, column, output, ordinal = "annual_previous", "Cla", "cnax_002", "y_cnax_cla", 2
    statement_outputs = (output,)
    statement_factors = {source: output}


class China469QuarterPreviousLibrary(_Base):
    family = "china469_claq"
    mode, source, column, output, ordinal = "monthly_previous", "Claq", "cnax_003", "q_cnax_claq", 3
    statement_outputs = (output,)
    statement_factors = {source: output}


class China469EquityLibrary(_Base):
    family = "china469_cbe"
    mode, source, column, output, ordinal = "annual_current", "Cbe", "cnax_004", "y_cnax_cbe", 4
    statement_fields = (EQUITY_FIELD, "operate_profit", "n_income", "n_cashflow_act")
    statement_outputs = (output,)
    statement_factors = {source: output}
