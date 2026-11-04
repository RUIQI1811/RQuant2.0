"""The QMT data adapter (qmt/runtime/qmt_inputs.py) against a fake QMT context with known answers."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))

import build_jq240  # noqa: E402

CALENDAR = pd.bdate_range("2008-01-02", "2011-06-30")
END = "20110630"
DAYS = 400
EX_DATES = {"600001.SH": ["2009-07-01", "2010-07-01"], "600004.SH": ["2010-05-04"]}
SUSPENDED = {"600004.SH": ["2011-03-01", "2011-03-02"]}
LISTED = {"600001.SH": "2000-01-04", "600002.SH": "2001-01-04", "600003.SH": "2011-03-01", "600004.SH": "2003-01-06",
          "000001.SZ": "1991-04-03", "830001.BJ": "2009-01-05"}
NAMES = {"600002.SH": "*ST Two"}
TOTAL = {"2009-06-01": 1.0e8, "2010-06-01": 1.2e8}
FLOAT = {"2009-06-01": 0.8e8, "2010-06-01": 1.0e8}


def to_ms(day: str | pd.Timestamp) -> int:
    return int((pd.Timestamp(day) - pd.Timedelta(hours=8)).value // 1_000_000)


def local_labels(dates: pd.DatetimeIndex) -> list[str]:
    """QMT's date labels on a machine on US Pacific time: the local date of Beijing midnight, the previous day."""
    return list((dates - pd.Timedelta(days=1)).strftime("%Y%m%d"))


def raw_bars(code: str) -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, code)))
    dates = CALENDAR[CALENDAR >= max(pd.Timestamp(LISTED[code]), CALENDAR[0])]
    close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    volume = rng.uniform(1e4, 2e4, len(dates))
    suspend = np.zeros(len(dates))
    for day in SUSPENDED.get(code, []):
        position = dates.get_loc(pd.Timestamp(day))
        close[position] = close[position - 1]
        volume[position] = 0.0
        suspend[position] = 1.0
    pre_close = np.r_[close[0], close[:-1]]
    for day in EX_DATES.get(code, []):
        position = dates.get_loc(pd.Timestamp(day))
        pre_close[position] = close[position - 1] / 1.1
    return pd.DataFrame(
        {"open": close * 0.99, "high": close * 1.02, "low": close * 0.97, "close": close, "volume": volume,
         "amount": volume * close * 100.0, "preClose": pre_close, "suspendFlag": suspend},
        index=dates,
    )


@lru_cache(None)
def namespace() -> dict:
    source = build_jq240.build_strategy()
    names: dict = {}
    exec(compile(source, str(build_jq240.PARITY_OUTPUT), "exec"), names)  # noqa: S102 - our own generated file
    names["MARKET_MIN_STOCKS"] = 1
    names["AS_OF"] = END
    names["DAYS"] = DAYS
    return names


class FakeQMT:
    def __init__(self) -> None:
        self.bars = {code: raw_bars(code) for code in LISTED}
        rng = np.random.default_rng(7)
        quarters = pd.date_range("2005-03-31", "2011-03-31", freq="QE")
        self.statements: dict[str, dict[str, dict[int, float]]] = {}
        fields = namespace()["STATEMENT_FIELDS"]
        targets = {t for endpoint in fields.values() for t in endpoint.values() if t and not t.startswith("CAPITAL")}
        for code in LISTED:
            table = {t: {to_ms(q): float(rng.uniform(1e6, 1e8)) for q in quarters} for t in targets}
            for offset, name in enumerate(("ASHAREINCOME", "ASHAREBALANCESHEET", "ASHARECASHFLOW")):
                table[f"{name}.m_anntime"] = {to_ms(q): float(to_ms(q + pd.Timedelta(days=30 + offset)))
                                              for q in quarters}
            # A later report re-discloses the 2009 year-end balance sheet; QMT then moves its m_anntime.
            table["ASHAREBALANCESHEET.m_anntime"][to_ms("2009-12-31")] = float(to_ms("2011-04-30"))
            table["CAPITALSTRUCTURE.total_capital"] = {to_ms(d): v for d, v in TOTAL.items()}
            table["CAPITALSTRUCTURE.circulating_capital"] = {to_ms(d): v for d, v in FLOAT.items()}
            self.statements[code] = table

    def get_market_data_ex(self, fields, codes, period="1d", start_time="", end_time="", count=-1,
                           dividend_type="none", fill_data=True, subscribe=True):
        assert dividend_type == "none" and fill_data is False
        result = {}
        for code in codes:
            if code == "000300.SH":
                frame = pd.DataFrame({"close": np.ones(len(CALENDAR))}, index=CALENDAR)
            elif code in self.bars:
                frame = self.bars[code]
            else:
                continue
            start = pd.Timestamp(start_time or "19900101")
            frame = frame[(frame.index >= start) & (frame.index <= pd.Timestamp(end_time))]
            frame = frame.assign(time=[float(to_ms(day)) for day in frame.index])[fields].copy()
            frame.index = local_labels(frame.index)
            result[code] = frame
        return result

    def get_raw_financial_data(self, fields, codes, start, end, report_type="report_time"):
        return {code: {f: self.statements[code].get(f, {}) for f in fields} for code in codes if code in LISTED}

    def get_turnover_rate(self, codes, start, end):
        frame = pd.DataFrame({c: self.bars[c]["volume"] / 1e8 for c in codes if c in LISTED})
        frame = frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        frame.index = [int(label) for label in local_labels(frame.index)]
        return frame

    def get_stock_list_in_sector(self, name):
        return list(LISTED)

    def get_instrumentdetail(self, code):
        if code not in LISTED:
            return {}
        return {"OpenDate": int(LISTED[code].replace("-", "")), "InstrumentName": NAMES.get(code, "Name " + code),
                "FloatVolume": 5.0e7, "TotalVolume": 6.0e7}

    def is_last_bar(self):
        return True


@lru_cache(None)
def built() -> tuple[FakeQMT, dict]:
    context = FakeQMT()
    return context, namespace()["build_inputs"](context, END, DAYS, log=lambda *args: None)


def expected_factor(code: str) -> pd.Series:
    """Tushare-style canonical factor from the known corporate actions, independent of preClose."""
    bars = raw_bars(code)
    adj = pd.Series(1.0, index=bars.index)
    for day in EX_DATES.get(code, []):
        adj[adj.index >= pd.Timestamp(day)] *= 1.1
    traded = bars[(bars.index >= pd.Timestamp("2008-05-12")) & (bars["volume"] > 0)]
    first = traded.index[0]
    return adj / (adj[first] * bars.loc[first, "close"])


def test_canonical_prices_and_volume() -> None:
    context, inputs = built()
    dates = CALENDAR[-DAYS:]
    assert list(inputs["close"].index) == list(dates)
    assert list(inputs["close"].columns) == ["600001.SH", "600002.SH", "600003.SH", "600004.SH"]
    for code in ("600001.SH", "600004.SH"):
        bars = context.bars[code].reindex(dates)
        factor = expected_factor(code).reindex(dates)
        traded = bars["volume"] > 0
        np.testing.assert_allclose(inputs["close"][code][traded], (bars["close"] * factor)[traded], rtol=1e-12)
        np.testing.assert_allclose(inputs["volume"][code][traded], (bars["volume"] * 100 / factor)[traded],
                                   rtol=1e-12)
        np.testing.assert_allclose(inputs["amount"][code][traded], bars["amount"][traded])
        assert inputs["close"][code][~traded].isna().all()
    assert inputs["close"]["600004.SH"][SUSPENDED["600004.SH"]].isna().all()


def test_shares_cap_and_turnover() -> None:
    context, inputs = built()
    raw = context.bars["600001.SH"]["close"]
    for day, shares in (("2010-01-04", 1.0e8), ("2011-01-04", 1.2e8)):
        assert inputs["basic_total_share"].loc[day, "600001.SH"] == shares
        assert inputs["cap"].loc[day, "600001.SH"] == pytest.approx(raw[day] * shares)
    assert inputs["basic_circ_mv"].loc["2011-01-04", "600001.SH"] == pytest.approx(raw["2011-01-04"] * 1.0e8)
    turnover = context.bars["600001.SH"]["volume"]["2011-01-04"] * 100 / 1.0e8 * 100
    assert inputs["basic_turnover_rate"].loc["2011-01-04", "600001.SH"] == pytest.approx(turnover)


def test_fundamentals_rows() -> None:
    _, inputs = built()
    frame = inputs["fundamentals"]
    assert set(frame["instrument"]) == {"600001.SH", "600002.SH", "600003.SH", "600004.SH"}
    row = frame[(frame["instrument"] == "600001.SH") & (frame["end_date"] == pd.Timestamp("2010-06-30"))].iloc[0]
    assert row["ann_date"] == pd.Timestamp("2010-07-30")  # the income statement, the earliest of the three
    assert row["total_share"] == 1.2e8
    early = frame[(frame["instrument"] == "600001.SH") & (frame["end_date"] == pd.Timestamp("2006-03-31"))].iloc[0]
    assert early["total_share"] == 1.0e8  # before QMT's share history: its first value
    assert row["oth_eqt_tools_p_shr"] == 0.0  # QMT has no such line: zero on a published statement
    year_end = frame[(frame["instrument"] == "600001.SH") & (frame["end_date"] == pd.Timestamp("2009-12-31"))].iloc[0]
    assert year_end["ann_date"] == pd.Timestamp("2010-01-30")


def test_universe_is_sh_main() -> None:
    _, inputs = built()
    universe = inputs["universe"]
    assert universe["600001.SH"].all()
    assert not universe["600002.SH"].any()  # ST name
    listed = CALENDAR.get_loc(pd.Timestamp("2011-03-01"))
    open_days = pd.Series(np.arange(len(CALENDAR)) - listed + 1, index=CALENDAR).reindex(universe.index)
    np.testing.assert_array_equal(universe["600003.SH"].to_numpy(), (open_days > 60).to_numpy())
    assert not universe.loc[SUSPENDED["600004.SH"], "600004.SH"].any()


def test_market_return() -> None:
    context, inputs = built()
    day, previous = pd.Timestamp("2011-01-04"), pd.Timestamp("2011-01-03")
    weights, returns = [], []
    for code in ("600001.SH", "600002.SH", "600004.SH", "000001.SZ"):  # 600003 not listed yet, BJ excluded
        bars = context.bars[code]
        weights.append(bars.loc[previous, "close"] * 1.0e8)
        returns.append(bars.loc[day, "close"] / bars.loc[day, "preClose"] - 1)
    assert inputs["mkt"][day] == pytest.approx(np.dot(weights, returns) / np.sum(weights))


def test_pe_ttm_uses_positive_ttm_profit() -> None:
    _, inputs = built()
    pe = inputs["basic_pe_ttm"]
    assert pe.loc["2011-01-04", "600001.SH"] > 0


def test_parity_strategy_prints_every_factor(capsys) -> None:
    context, _ = built()
    names = namespace()
    names["init"](context)
    names["handlebar"](context)
    names["handlebar"](context)  # runs once
    lines = capsys.readouterr().out.splitlines()
    assert "[RQ] parity=end" in lines and not any("error" in line.lower() for line in lines)
    for ordinal in range(1, 234):
        assert any(line.startswith(f"[RQ] parity.jq_{ordinal:03d}.0=") for line in lines)
    codes = next(line for line in lines if line.startswith("[RQ] parity.codes.0="))
    assert codes.split("=", 1)[1].split(",") == ["600001.SH", "600003.SH", "600004.SH"]


def test_committed_strategy_is_current_and_python36() -> None:
    import ast

    source = build_jq240.PARITY_OUTPUT.read_text(encoding="ascii")
    assert source == build_jq240.build_strategy()
    assert source.startswith("#encoding:gbk\n")
    tree = ast.parse(source, feature_version=(3, 6))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"numpy", "pandas", "time", "traceback"}
    assert "open(" not in source and "passorder" not in source and "order_" not in source
