"""Source-timing China anomaly definitions, independently adapted to Tushare.

Li, Liu, Liu and Wei (2024), Management Science, appendix A.3.17, A.3.22–23, A.4.5–6,
A.4.10, A.4.12–21, A.4.24 and A.4.26.
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
ADDITIONAL_FIELDS = ("income_tax", "total_revenue", "revenue", "oper_cost",
                     "n_cashflow_inv_act", "n_cash_flows_fnc_act")
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
    fundamentals = pd.read_parquet(root / "fundamentals.parquet", columns=[*KEYS, *FIELDS, EQUITY_FIELD,
                                                                            *ADDITIONAL_FIELDS],
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
        return {self.output: _profit_ratio(profit - income + cash, denominator)}

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
    statement_fields = (*FIELDS, EQUITY_FIELD)
    statement_outputs = (output, "y_cnax_opa", "y_cnax_ope")
    statement_factors = {source: output, "Opa": "y_cnax_opa", "Ope": "y_cnax_ope"}

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return (*super().specs, *_profit_specs(self.family, (("Opa", 5), ("Ope", 6))))

    def build_statements(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {**super().build_statements(inputs),
                "y_cnax_opa": _profit_ratio(inputs["operate_profit"], inputs["total_assets"]),
                "y_cnax_ope": _profit_ratio(inputs["operate_profit"], inputs[EQUITY_FIELD])}


def _profit_ratio(numerator: Any, denominator: Any) -> Any:
    finite = k.valid(k.replace_inf(numerator), k.replace_inf(denominator))
    return k.mask(k.replace_inf(numerator / denominator), k.all_of(finite, k.ne(denominator, 0.0)))


def _profit_specs(family: str, names: tuple[tuple[str, int], ...]) -> tuple[FactorSpec, ...]:
    return tuple(FactorSpec(f"cnax_{ordinal:03d}", name, family, ordinal,
                            f"{name}: Tushare operating profit with frozen formation dates", 1,
                            f"{__name__}.{name}", 1) for name, ordinal in names)


class _OperatingLagBase(_Base):
    statement_fields = ("operate_profit", "total_assets", EQUITY_FIELD)
    names: tuple[tuple[str, int], ...]

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return _profit_specs(self.family, self.names)

    def build_statements(self, inputs: dict[str, Any]) -> dict[str, Any]:
        profit = inputs["operate_profit"]
        periods = 4
        if self.mode == "monthly_previous":
            profit = k.select(k.eq(inputs["fiscal_quarter"], 0), profit, profit - k.delay(profit, 1))
            periods = 1
        assets = k.delay(inputs["total_assets"], periods)
        equity = k.delay(inputs[EQUITY_FIELD], periods)
        return {self.statement_outputs[0]: _profit_ratio(profit, assets),
                self.statement_outputs[1]: _profit_ratio(profit, equity)}


class China469OperatingAnnualPreviousLibrary(_OperatingLagBase):
    family = "china469_operating_annual_lag"
    mode, source, column, output, ordinal = "annual_previous", "Ola", "cnax_007", "y_cnax_ola", 7
    names = (("Ola", 7), ("Ole", 8))
    statement_outputs = (output, "y_cnax_ole")
    statement_factors = {"Ola": output, "Ole": "y_cnax_ole"}


class China469OperatingQuarterPreviousLibrary(_OperatingLagBase):
    family = "china469_operating_quarter_lag"
    mode, source, column, output, ordinal = "monthly_previous", "Olaq", "cnax_009", "q_cnax_olaq", 9
    names = (("Olaq", 9), ("Oleq", 10))
    statement_outputs = (output, "q_cnax_oleq")
    statement_factors = {"Olaq": output, "Oleq": "q_cnax_oleq"}


class China469AnnualRatiosLibrary(_Base):
    family = "china469_annual_ratios"
    mode, source, column, output, ordinal = "annual_current", "Pm", "cnax_011", "y_cnax_pm", 11
    statement_fields = ("operate_profit", "total_revenue", "n_income", "income_tax", "n_cashflow_act",
                        "total_assets", EQUITY_FIELD, "n_cashflow_inv_act", "n_cash_flows_fnc_act")
    statement_outputs = (output, "y_cnax_tbi", "y_cnax_bl", "y_cnax_poa", "y_cnax_pta")
    statement_factors = {"Pm": output, "Tbi": "y_cnax_tbi", "Bl": "y_cnax_bl", "Poa": "y_cnax_poa",
                         "Pta": "y_cnax_pta"}

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(FactorSpec(f"cnax_{ordinal:03d}", name, self.family, ordinal, description, 1,
                                f"{__name__}.{name}", 1) for name, ordinal, description in (
            ("Pm", 11, "annual operating profit / total operating revenue, frozen at June formation"),
            ("Tbi", 12, "annual (net income + tax expense) / net income, both amounts positive"),
            ("Bl", 13, "annual total assets / total shareholders' equity, frozen at June formation"),
            ("Poa", 14, "annual (net income - operating cash flow) / absolute net income"),
            ("Pta", 15, "annual total accruals / absolute net income"),
        ))

    def build_statements(self, inputs: dict[str, Any]) -> dict[str, Any]:
        income = inputs["n_income"]
        pretax = income + inputs["income_tax"]
        tbi = k.mask(_profit_ratio(pretax, income), k.all_of(k.gt(pretax, 0.0), k.gt(income, 0.0)))
        return {"y_cnax_pm": _profit_ratio(inputs["operate_profit"], inputs["total_revenue"]),
                "y_cnax_tbi": tbi,
                "y_cnax_bl": _profit_ratio(inputs["total_assets"], inputs[EQUITY_FIELD]),
                "y_cnax_poa": _profit_ratio(income - inputs["n_cashflow_act"], k.absolute(income)),
                "y_cnax_pta": _profit_ratio(income - inputs["n_cashflow_act"] - inputs["n_cashflow_inv_act"]
                                            - inputs["n_cash_flows_fnc_act"], k.absolute(income))}


class China469AnnualCapitalTurnoverLibrary(_Base):
    family = "china469_annual_capital_turnover"
    mode, source, column, output, ordinal = "annual_previous", "Cto", "cnax_016", "y_cnax_cto", 16
    statement_fields = ("total_revenue", "total_assets", "revenue", "oper_cost", "n_income", "n_cashflow_act",
                        "n_cashflow_inv_act", "n_cash_flows_fnc_act")
    statement_outputs = (output, "y_cnax_gla", "y_cnax_ta")
    statement_factors = {source: output, "Gla": "y_cnax_gla", "Ta": "y_cnax_ta"}

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return (FactorSpec(self.column, self.source, self.family, self.ordinal,
                           "annual total operating revenue / previous-year total assets", 1,
                           f"{__name__}.Cto", 1),
                FactorSpec("cnax_017", "Gla", self.family, 17,
                           "annual (operating revenue - operating cost) / previous-year total assets", 1,
                           f"{__name__}.Gla", 1),
                FactorSpec("cnax_018", "Ta", self.family, 18,
                           "annual total accruals / previous-year total assets", 1,
                           f"{__name__}.Ta", 1))

    def build_statements(self, inputs: dict[str, Any]) -> dict[str, Any]:
        assets = k.delay(inputs["total_assets"], 4)
        return {self.output: _profit_ratio(inputs["total_revenue"], assets),
                "y_cnax_gla": _profit_ratio(inputs["revenue"] - inputs["oper_cost"], assets),
                "y_cnax_ta": _profit_ratio(inputs["n_income"] - inputs["n_cashflow_act"]
                                           - inputs["n_cashflow_inv_act"] - inputs["n_cash_flows_fnc_act"], assets)}
