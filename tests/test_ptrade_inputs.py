"""The PTrade data adapter (ptrade/runtime/ptrade_inputs.py) against a fake PTrade API built from the same synthetic
market as the QMT adapter's tests, so both adapters must produce the same jq240 inputs."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))
sys.path.insert(0, str(ROOT / "tests"))

import build_jq240  # noqa: E402
import test_qmt_inputs as qmt  # noqa: E402

PT = {code: code.replace(".SH", ".SS") for code in qmt.LISTED}
RQ = {value: key for key, value in PT.items()}


def _shares_at(changes: dict[str, float], day: pd.Timestamp) -> float:
    known = [value for when, value in sorted(changes.items()) if pd.Timestamp(when) <= day]
    return known[-1] if known else next(iter(sorted(changes.items())))[1]


class FakePTrade:
    """get_* functions with PTrade's signatures and return shapes, over qmt.raw_bars."""

    def __init__(self) -> None:
        self.bars = {PT[code]: qmt.raw_bars(code) for code in qmt.LISTED}
        rng = np.random.default_rng(11)
        quarters = pd.date_range("2005-03-31", "2011-03-31", freq="QE")
        self.calls: dict[str, int] = {}
        fields = build_namespace()["PTRADE_FIELDS"]
        self.statements = {}
        for offset, (endpoint, mapping) in enumerate(fields.items()):
            table = build_namespace()["ENDPOINT_TABLES"][endpoint]
            names = sorted({field for field in mapping.values() if field})
            rows = []
            for code in self.bars:
                for quarter in quarters:
                    row = {"end_date": quarter.strftime("%Y-%m-%d"), "secu_code": code,
                           "publ_date": (quarter + pd.Timedelta(days=30 + offset)).strftime("%Y-%m-%d")}
                    row.update({name: float(rng.uniform(1e6, 1e8)) for name in names})
                    rows.append(row)
            self.statements[table] = pd.DataFrame(rows).set_index(["end_date", "secu_code"])

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def get_trade_days(self, start, end):
        self._count("get_trade_days")
        days = qmt.CALENDAR[(qmt.CALENDAR >= pd.Timestamp(start)) & (qmt.CALENDAR <= pd.Timestamp(end))]
        return np.array(days.strftime("%Y-%m-%d"))

    def get_Ashares(self, date=None):  # noqa: N802 - PTrade's name
        return [code for code in self.bars if not code.endswith(".BJ")]

    def get_price(self, security, start_date=None, end_date=None, frequency="1d", fields=None, fq=None, count=None,
                  is_dict=False):
        assert fq is None and is_dict and frequency == "1d"
        self._count("get_price")
        result = {}
        for code in security:
            frame = self.bars[code]
            frame = frame[(frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))]
            columns = {"open": frame["open"], "high": frame["high"], "low": frame["low"], "close": frame["close"],
                       "volume": frame["volume"] * 100.0, "money": frame["amount"], "preclose": frame["preClose"]}
            dtype = [("datetime", "<i8")] + [(name, "<f8") for name in fields]
            bars = np.zeros(len(frame), dtype=dtype)
            bars["datetime"] = frame.index.strftime("%Y%m%d").astype(np.int64)
            for name in fields:
                bars[name] = np.asarray(columns[name], dtype=float)
            result[code] = bars
        return result

    def get_fundamentals(self, security, table, fields=None, date=None, start_year=None, end_year=None, **_):
        self._count("get_fundamentals." + table)
        if table == "valuation":
            day = pd.Timestamp(date)
            rows = {code: {"total_shares": _shares_at(qmt.TOTAL, day), "a_floats": _shares_at(qmt.FLOAT, day),
                           "trading_day": day.strftime("%Y-%m-%d")} for code in security}
            frame = pd.DataFrame.from_dict(rows, orient="index")
            frame.index.name = "secu_code"
            return frame[["trading_day"] + list(fields)]
        frame = self.statements[table]
        years = frame.index.get_level_values("end_date").str[:4].astype(int)
        codes = frame.index.get_level_values("secu_code")
        keep = (years >= int(start_year)) & (years <= int(end_year)) & codes.isin(security)
        return frame[keep][["publ_date"] + list(fields)]

    def get_stock_info(self, stocks, field=None):
        return {code: {"listed_date": qmt.LISTED[RQ[code]]} for code in stocks}

    def get_stock_status(self, stocks, query_type="ST", query_date=None):
        self._count("get_stock_status." + query_type)
        st = {PT[code] for code, name in qmt.NAMES.items() if "ST" in name}
        return {code: (query_type == "ST" and code in st) for code in stocks}


API = ("get_trade_days", "get_Ashares", "get_price", "get_fundamentals", "get_stock_info", "get_stock_status")


@lru_cache(None)
def build_namespace() -> dict:
    names: dict = {}
    source = build_jq240.build() + "\n" + (ROOT / "ptrade" / "runtime" / "ptrade_inputs.py").read_text()
    exec(compile(source, "ptrade_inputs", "exec"), names)  # noqa: S102 - our own sources
    names["MARKET_MIN_STOCKS"] = 1
    return names


@lru_cache(None)
def built() -> tuple[FakePTrade, dict]:
    names = build_namespace()
    fake = FakePTrade()
    for name in API:
        names[name] = getattr(fake, name)
    return fake, names["build_inputs"](qmt.END, qmt.DAYS, log=lambda *args: None)


def _as_rquant(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.rename(columns=RQ)


@pytest.mark.parametrize("key", ["open", "high", "low", "close", "volume", "amount", "cap", "basic_close",
                                 "basic_total_share", "basic_circ_mv", "basic_turnover_rate", "universe"])
def test_daily_panels_match_the_qmt_adapter(key: str) -> None:
    _, inputs = built()
    _, reference = qmt.built()
    ours = _as_rquant(inputs[key])
    theirs = reference[key].reindex(columns=ours.columns)
    assert list(ours.columns) == ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
    assert (ours.index == theirs.index).all()
    if key == "universe":
        np.testing.assert_array_equal(ours.to_numpy(bool), theirs.to_numpy(bool))
    else:
        np.testing.assert_allclose(ours.to_numpy(float), theirs.to_numpy(float), rtol=1e-12, equal_nan=True)


def test_market_return_matches_the_qmt_adapter() -> None:
    _, inputs = built()
    _, reference = qmt.built()
    np.testing.assert_allclose(inputs["mkt"].to_numpy(float), reference["mkt"].to_numpy(float), rtol=1e-12,
                               equal_nan=True)


def test_fundamentals_rows_and_dates() -> None:
    fake, inputs = built()
    frame = inputs["fundamentals"]
    assert set(frame["instrument"]) == {"600001.SS", "600002.SS", "600003.SS", "600004.SS"}
    row = frame[(frame["instrument"] == "600001.SS") & (frame["end_date"] == pd.Timestamp("2010-06-30"))].iloc[0]
    assert row["ann_date"] == pd.Timestamp("2010-07-30")  # income statement, the earliest of the three tables
    source = fake.statements["income_statement"].loc[("2010-06-30", "600001.SS")]
    assert row["n_income_attr_p"] == source["np_parent_company_owners"]
    assert row["oth_eqt_tools_p_shr"] == 0.0 and row["fin_exp_int_inc"] == 0.0
    # Statements start FUNDAMENTAL_YEARS_BACK years before the first day.
    assert frame["end_date"].min() == pd.Timestamp("2005-03-31")
    pe = inputs["basic_pe_ttm"]
    assert (pe > 0).to_numpy().any() and not (pe <= 0).to_numpy().any()  # only positive TTM profit gives a P/E


def test_universe_uses_point_in_time_status() -> None:
    fake, inputs = built()
    assert not inputs["universe"]["600002.SS"].any()
    assert inputs["quotes"]["st"] == {"600002.SS"}
    assert fake.calls["get_stock_status.ST"] == qmt.DAYS


def test_research_bundle_runs_and_scores(capsys) -> None:
    source = build_jq240.build_ptrade()
    source = source.replace('AS_OF = "20261110"', f'AS_OF = "{qmt.END}"').replace("DAYS = 1100", f"DAYS = {qmt.DAYS}")
    source = source.replace("MARKET_MIN_STOCKS = 100", "MARKET_MIN_STOCKS = 1")
    names: dict = {}
    fake = FakePTrade()
    for name in API:
        names[name] = getattr(fake, name)
    exec(compile(source, "bundle", "exec"), names)  # noqa: S102 - our own generated file
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("[RQ] parity=end") for line in lines)
    assert not any("parity.error" in line for line in lines)
    top = next(line for line in lines if line.startswith("[RQ] score.top30="))
    assert set(top.split("=", 1)[1].split(",")) <= {"600001.SH", "600003.SH", "600004.SH"}
    assert names["RESULT"]["factors"].shape[1] == 233


def test_committed_ptrade_bundle_is_current() -> None:
    source = build_jq240.PTRADE_PARITY_OUTPUT.read_text(encoding="ascii")
    assert source == build_jq240.build_ptrade()
    for word in ("import os", "import sys", "import platform", "import traceback", "builtins", "__import__",
                 "open("):
        assert word not in source
