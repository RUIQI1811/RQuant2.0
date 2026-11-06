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
            table = {t: {to_ms(q): float(rng.uniform(1e6, 1e8)) for q in quarters} for t in sorted(targets)}
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


def test_parity_strategy_scores_with_an_exported_model(capsys) -> None:
    context, _ = built()
    names = namespace()
    features = [f"jq_{ordinal:03d}" for ordinal in range(1, 234)]
    # One split on the first factor's z-score: above the mean scores 1, else -1.
    model = {"run": "fake", "prediction_year": 2010, "fit_period": ["2009-01-01", "2009-12-31"],
             "features": features, "mean": [0.0] * 233, "std": [1.0] * 233,
             "trees": [{"feature": [0], "threshold": [0.0], "left": [~0], "right": [~1], "leaf": [-1.0, 1.0]}]}
    saved = names["MODEL"], names["_state"]["done"]
    names["MODEL"], names["_state"]["done"] = model, False
    try:
        names["handlebar"](context)
    finally:
        names["MODEL"], names["_state"]["done"] = saved
    lines = capsys.readouterr().out.splitlines()
    assert not any("error" in line.lower() for line in lines)
    first = next(line for line in lines if line.startswith("[RQ] parity.jq_001.0="))
    factor = [float(value) if value else np.nan for value in first.split("=", 1)[1].split(",")]
    scores = next(line for line in lines if line.startswith("[RQ] score.0=")).split("=", 1)[1].split(",")
    assert [float(value) for value in scores] == [1.0 if value > 0 else -1.0 for value in factor]
    top = next(line for line in lines if line.startswith("[RQ] score.top30=")).split("=", 1)[1].split(",")
    assert sorted(top) == ["600001.SH", "600003.SH", "600004.SH"]
    assert any(line.startswith("[RQ] score.model=fake year=2010") for line in lines)


def paper_strategy(now: str) -> tuple[object, dict]:
    context, _ = built()
    names: dict = {}
    source = build_jq240.build_strategy(build_jq240.PAPER_PARTS)
    exec(compile(source, str(build_jq240.PAPER_OUTPUT), "exec"), names)  # noqa: S102 - our own generated file
    features = [f"jq_{ordinal:03d}" for ordinal in range(1, 234)]
    names.update(
        MARKET_MIN_STOCKS=1, PAPER_DAYS=DAYS, PAPER_START="20110601", PAPER_CAPITAL=100_000.0, PAPER_TOPK=2,
        PAPER_N_DROP=1,
        beijing_now=lambda: now,
        MODEL={"run": "fake", "prediction_year": 2011, "fit_period": ["2010-01-01", "2010-12-31"],
               "features": features, "mean": [0.0] * 233, "std": [1.0] * 233,
               "trees": [{"feature": [0], "threshold": [0.0], "left": [~0], "right": [~1], "leaf": [-1.0, 1.0]}]},
    )
    return context, names


def test_paper_strategy_replays_the_book_and_plans_the_next_open(capsys) -> None:
    context, names = paper_strategy(END + "1600")
    names["init"](context)
    names["handlebar"](context)
    lines = capsys.readouterr().out.splitlines()
    assert "[RQ] paper=end" in lines and not any("error" in line.lower() for line in lines), lines[-5:]
    assert "[RQ] paper.data_through=20110630" in lines
    assert not any(line.startswith("[RQ] paper.skipped_days=") for line in lines)
    trading_days = CALENDAR[(CALENDAR > "2011-06-01") & (CALENDAR <= "2011-06-30")]
    navs = [line for line in lines if line.startswith("[RQ] paper.nav.")]
    assert len(navs) == len(trading_days)
    assert any(line.startswith("[RQ] paper.fill.20110602=buy ") for line in lines)
    holdings = [line for line in lines if line.startswith("[RQ] paper.hold=")]
    assert 1 <= len(holdings) <= 2
    value = float(next(line for line in lines if line.startswith("[RQ] paper.value=")).split("=")[1].split()[0])
    assert 50_000 < value < 200_000
    assert any(line.startswith("[RQ] paper.top2=") for line in lines)


def test_paper_strategy_leaves_out_a_session_still_open(capsys) -> None:
    context, names = paper_strategy(END + "1424")
    names["init"](context)
    names["handlebar"](context)
    lines = capsys.readouterr().out.splitlines()
    assert "[RQ] paper=end" in lines and not any("error" in line.lower() for line in lines), lines[-5:]
    assert "[RQ] paper.data_through=20110629" in lines
    assert any(line.startswith("[RQ] paper.skipped_days=20110630 ") for line in lines)
    assert not any(line.startswith("[RQ] paper.nav.20110630=") for line in lines)
    assert any(line.startswith("[RQ] paper.next.buy=") or line.startswith("[RQ] paper.next.sell=")
               or line.startswith("[RQ] paper.next.backups=") for line in lines)


def test_complete_days_drops_days_without_daily_bars() -> None:
    _, names = paper_strategy(END + "1600")
    dates = pd.DatetimeIndex(["2026-11-03", "2026-11-04", "2026-11-05"])
    close = pd.DataFrame({"600000.SH": [10.0, 10.5, np.nan], "600004.SH": [np.nan, 8.0, np.nan]}, index=dates)
    complete = names["complete_days"]
    # QMT lists today before its daily bars arrive: even after 15:00 a day with no close at all is left out.
    assert list(complete({"close": close}, "202611051600")) == list(dates[:2])
    assert list(complete({"close": close}, "202611060900")) == list(dates[:2])
    full = close.fillna(9.0)
    assert list(complete({"close": full}, "202611051459")) == list(dates[:2])
    assert list(complete({"close": full}, "202611051500")) == list(dates)
    # A download cut short: only some stocks have the latest bar, so that day is not complete yet.
    partial = pd.DataFrame(9.0, index=dates, columns=[f"6000{n:02d}.SH" for n in range(20)])
    partial.iloc[-1, 3:] = np.nan
    assert list(complete({"close": partial}, "202611060900")) == list(dates[:2])
    partial.iloc[-1, 3:-1] = 9.0
    assert list(complete({"close": partial}, "202611060900")) == list(dates)


class Row:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


class LiveQMT(FakeQMT):
    """FakeQMT plus live ticks: the next session opens 1% above the last close and trades 0.5% above it."""

    def get_full_tick(self, codes):
        ticks = {}
        for code in codes:
            if code in self.bars:
                close = float(self.bars[code]["close"].iloc[-1])
                ticks[code] = {"lastClose": close, "open": close * 1.01, "lastPrice": close * 1.005}
        return ticks


def live_strategy(clock: list[str], positions: dict, cash: float = 50_000.0, dry_run: bool = False):
    context = LiveQMT()
    names: dict = {}
    source = build_jq240.build_strategy(build_jq240.LIVE_PARTS)
    exec(compile(source, str(build_jq240.LIVE_OUTPUT), "exec"), names)  # noqa: S102 - our own generated file
    sent: list[tuple] = []
    orders: list[Row] = []

    def get_trade_detail_data(account, kind, data):
        assert account == "1234567890" and kind == "stock"
        if data == "account":
            return [Row(m_dAvailable=cash)]
        if data == "position":
            return [Row(m_strInstrumentID=code[:6], m_strExchangeID=code[7:], m_nVolume=shares, m_nCanUseVolume=free)
                    for code, (shares, free) in positions.items()]
        return orders

    def passorder(op, kind, account, code, price_type, price, volume, name, quick, remark, ctx):
        sent.append((op, code, volume, price))
        orders.append(Row(m_strRemark=remark))

    features = [f"jq_{ordinal:03d}" for ordinal in range(1, 234)]
    names.update(
        MARKET_MIN_STOCKS=1, PAPER_DAYS=DAYS, LIVE_TOPK=2, LIVE_N_DROP=1, LIVE_CAPITAL=100_000.0,
        LIVE_DRY_RUN=dry_run, account="1234567890", beijing_clock=lambda: clock[0],
        get_trade_detail_data=get_trade_detail_data, passorder=passorder,
        MODEL={"run": "fake", "prediction_year": 2011, "fit_period": ["2010-01-01", "2010-12-31"],
               "features": features, "mean": [0.0] * 233, "std": [1.0] * 233,
               "trees": [{"feature": [0], "threshold": [0.0], "left": [~0], "right": [~1], "leaf": [-1.0, 1.0]}]},
    )
    return context, names, sent


def test_live_strategy_sells_at_the_open_then_buys(capsys) -> None:
    clock = ["20110701091500"]
    # 600002 is an ST name, out of the universe, so it goes first; the SZ holding is not the strategy's.
    context, names, sent = live_strategy(clock, {"600002.SH": (300, 300), "000001.SZ": (500, 500)})
    names["init"](context)
    names["handlebar"](context)  # plans, then waits for the open
    assert sent == []
    clock[0] = "20110701093010"
    names["handlebar"](context)
    assert [(op, code, volume) for op, code, volume, _ in sent] == [(24, "600002.SH", 300)]
    clock[0] = "20110701093030"
    names["handlebar"](context)
    buys = [order for order in sent if order[0] == 23]
    assert 1 <= len(buys) <= 2 and all(volume % 100 == 0 and volume > 0 for _, _, volume, _ in buys)
    assert {code for _, code, _, _ in buys} <= {"600001.SH", "600003.SH", "600004.SH"}
    # The buys spend no more than the cash on hand (the fake account does not credit the sale).
    assert sum(volume * price for _, _, volume, price in buys) <= 50_000.0
    count = len(sent)
    names["handlebar"](context)
    assert len(sent) == count
    lines = capsys.readouterr().out.splitlines()
    assert "[RQ] live.data_through=20110630" in lines and not any("error" in line for line in lines), lines[-8:]
    assert not any("1234567890" in line for line in lines)


def test_live_dry_run_prints_orders_and_sends_nothing(capsys) -> None:
    context, names, sent = live_strategy(["20110701200000"], {}, cash=100_000.0, dry_run=True)
    names["init"](context)
    names["handlebar"](context)
    lines = capsys.readouterr().out.splitlines()
    assert sent == []
    assert len([line for line in lines if line.startswith("[RQ] live.dry_buy=")]) == 2, lines[-8:]
    assert "[RQ] live.done=dry run, nothing sent" in lines


def test_live_strategy_refuses_stale_data_late_starts_and_repeats(capsys) -> None:
    context, names, sent = live_strategy(["20110701093010"], {})
    names["complete_days"] = lambda inputs, now: inputs["close"].index[:-1]
    names["handlebar"](context)
    assert sent == [] and "daily bars of 20110630 are missing" in capsys.readouterr().out

    context, names, sent = live_strategy(["20110701100100"], {})
    names["handlebar"](context)
    assert sent == [] and "[RQ] live.skip=past 100000 Beijing time: no orders today" in capsys.readouterr().out

    context, names, sent = live_strategy(["20110701093010"], {"600002.SH": (300, 300)})
    names["get_trade_detail_data"]("1234567890", "stock", "order").append(Row(m_strRemark="RQ20110701"))
    names["handlebar"](context)
    assert sent == [] and "orders tagged RQ20110701 already sent today" in capsys.readouterr().out


def test_start_refreshes_recent_daily_bars(capsys) -> None:
    context, names, _ = live_strategy(["20110701200000"], {}, dry_run=True)
    fetched: list[tuple] = []

    def batch(codes, period, start, end):
        raise RuntimeError("not supported")

    names.update(download_history_data2=batch, download_history_data=lambda *args: fetched.append(args))
    names["download_recent_bars"](context)
    order = [code for code, _, _, _ in fetched]
    assert order[0] == "000300.SH" and order[1:4] == ["600001.SH", "600002.SH", "600003.SH"]
    assert len(fetched) == len(set(context.get_stock_list_in_sector("")) | {"000300.SH"})
    assert all(len(start) == 8 and end == "" for _, _, start, end in fetched)
    out = capsys.readouterr().out
    assert "[RQ] download.batch_error=RuntimeError: not supported" in out and "[RQ] download.bars=from " in out


def test_live_strategy_left_running_plans_again_each_day(capsys) -> None:
    clock = ["20110701200000"]
    context, names, _ = live_strategy(clock, {}, cash=100_000.0, dry_run=True)
    names["handlebar"](context)
    names["handlebar"](context)
    clock[0] = "20110702200000"
    names["handlebar"](context)
    lines = capsys.readouterr().out.splitlines()
    assert lines.count("[RQ] live.done=dry run, nothing sent") == 2
    assert len([line for line in lines if line.startswith("[RQ] live.plan_seconds=")]) == 2


def test_live_buys_are_capped_at_a_slot_and_idle_money_tops_up_small_holdings(capsys) -> None:
    _, names, sent = live_strategy(["20110701093030"], {}, cash=99_000.0)
    names.update(LIVE_TOPK=4)
    day = names["Day"](None, {"A.SH": 10.0, "B.SH": 10.0, "C.SH": 10.0}, {"A.SH": 10.0, "B.SH": 10.0, "C.SH": 10.0},
                       {"A.SH": 1.0, "B.SH": 1.0, "C.SH": 1.0}, {"C.SH": True}, {})
    limits = {code: (11.0, 9.0) for code in ("A.SH", "B.SH", "C.SH")}
    sold = {"day": day, "limits": limits, "buys": [("B.SH", 10_000.0)], "sold": 0.0, "kept": {"A.SH": 100, "C.SH": 100},
            "value_after": 2_000.0, "capital": 100_000.0}
    names["live_buy"](None, sold, "1234567890", "RQ20110701")
    # Slot = (2,000 held + 98,000 budget) / 4 = 25,000: B gets one slot, not the 100,000 the backtest would give it.
    # The idle money then tops A up to a slot; C opened limit-up and is left alone.
    assert [(op, code, volume) for op, code, volume, _ in sent] == [(23, "B.SH", 2400), (23, "A.SH", 2300)]
    assert "[RQ] live.topup=A.SH 2300 shares" in capsys.readouterr().out


def test_live_capital_zero_uses_the_whole_account(capsys) -> None:
    context, names, _ = live_strategy(["20110701200000"], {"600002.SH": (300, 300)}, cash=230_000.0, dry_run=True)
    names.update(LIVE_CAPITAL=0)
    names["handlebar"](context)
    out = capsys.readouterr().out
    account = next(line for line in out.splitlines() if line.startswith("[RQ] live.account="))
    assert account.endswith("budget 230000.00"), account
