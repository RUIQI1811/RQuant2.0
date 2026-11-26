"""The main-board blend bundles (ptrade/build_main_blend.py): jq240 + size_value_china on RQuant's ``main`` universe.

The CH-4 adapter is checked against rquant.data.ch4_factors fed the same synthetic all-A market directly, the bundled
size_value_china reference against the test suite's one, the blend against scripts/analysis/blend_runs.py, and the
research and live bundles run end to end on the fake PTrade market of the jq240 tests."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))
sys.path.insert(0, str(ROOT / "ptrade"))
sys.path.insert(0, str(ROOT / "tests"))

import build_main_blend  # noqa: E402
import test_ptrade_inputs as pti  # noqa: E402
import test_ptrade_live as ptl  # noqa: E402
import test_qmt_inputs as qmt  # noqa: E402
from panel_reference.size_value_china import SizeValueChina  # noqa: E402

from rquant.data.ch4_factors import CH4FactorBuilder  # noqa: E402
from rquant.factors.libraries.size_value_china import SIZE_VALUE_CHINA_NAMES  # noqa: E402


@lru_cache(None)
def namespace() -> dict:
    """The parity bundle's definitions (its run at the end is cut off)."""
    source = build_main_blend.build_parity()
    source = source[: source.index('log("parity", "start "')]
    names: dict = {}
    exec(compile(source, "main_blend_parity", "exec"), names)  # noqa: S102 - our own generated file
    return names


class Market:
    """A synthetic all-A market: Shanghai and Shenzhen main board, ChiNext and STAR codes, staggered listings,
    suspensions, share changes and loss makers, as PTrade's get_* functions return it."""

    def __init__(self, stocks: int = 260, days: int = 560, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.calendar = pd.bdate_range("2019-01-02", periods=days)
        prefixes = ("600", "000", "002", "300", "688")
        self.codes = [f"{prefixes[i % 5]}{i:03d}.{'SS' if prefixes[i % 5][0] == '6' else 'SZ'}" for i in range(stocks)]
        self.listed = {}
        self.bars = {}
        self.shares = {}
        self.pe = {}
        for number, code in enumerate(self.codes):
            first = 0 if number % 7 else int(rng.integers(0, days // 2))
            self.listed[code] = self.calendar[first] - pd.Timedelta(days=int(rng.integers(0, 400)))
            dates = self.calendar[first:]
            close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.025, len(dates))))
            pre_close = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, len(dates)))
            volume = rng.uniform(1e5, 1e7, len(dates)) * (1 + 3 * (rng.random(len(dates)) < 0.05))
            traded = rng.random(len(dates)) > 0.03
            volume = np.where(traded, volume, 0.0)
            self.bars[code] = pd.DataFrame({"close": close, "preclose": pre_close, "volume": volume}, index=dates)
            base = rng.uniform(1e8, 5e9) * (number + 1) / stocks
            self.shares[code] = pd.Series(base * np.where(np.arange(days) > days // 2, 1.2, 1.0), index=self.calendar)
            pe = rng.uniform(5, 80, days)
            pe[rng.random(days) < 0.15 if number % 4 == 0 else np.zeros(days, bool)] = -12.0
            self.pe[code] = pd.Series(pe, index=self.calendar)

    def get_trade_days(self, start, end):
        days = self.calendar[(self.calendar >= pd.Timestamp(start)) & (self.calendar <= pd.Timestamp(end))]
        return np.array(days.strftime("%Y-%m-%d"))

    def get_Ashares(self, date=None):  # noqa: N802 - PTrade's name
        return list(self.codes)

    def get_stock_info(self, stocks, field=None):
        return {code: {"listed_date": self.listed[code].strftime("%Y-%m-%d")} for code in stocks}

    def get_price(self, security, start_date=None, end_date=None, frequency="1d", fields=None, fq=None, count=None,
                  is_dict=False):
        result = {}
        for code in security:
            frame = self.bars[code]
            frame = frame[(frame.index >= pd.Timestamp(start_date)) & (frame.index <= pd.Timestamp(end_date))]
            bars = np.zeros(len(frame), dtype=[("datetime", "<i8")] + [(name, "<f8") for name in fields])
            bars["datetime"] = frame.index.strftime("%Y%m%d").astype(np.int64)
            for name in fields:
                bars[name] = frame[name].to_numpy(float)
            result[code] = bars
        return result

    def get_fundamentals(self, security, table, fields=None, date=None, **_):
        assert table == "valuation"
        day = pd.Timestamp(date)
        rows = {code: {"total_shares": self.shares[code][day], "a_floats": 0.8 * self.shares[code][day],
                       "pe_ttm": self.pe[code][day]} for code in security}
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "secu_code"
        return frame[list(fields)]

    def reference_factors(self, days: pd.DatetimeIndex) -> pd.DataFrame:
        """CH4FactorBuilder fed directly from the market, as rquant.data.reference feeds it from Tushare."""
        builder = CH4FactorBuilder(dict(self.listed))
        last_close: dict[str, float] = {}
        rows = []
        for day in days:
            returns, turnover, state = {}, {}, {}
            for code in self.codes:
                bars = self.bars[code]
                if day in bars.index and bars.at[day, "volume"] > 0:
                    row = bars.loc[day]
                    returns[code] = row["close"] / row["preclose"] - 1.0
                    turnover[code] = row["volume"] / self.shares[code][day]
                    last_close[code] = row["close"]
                if code in last_close:
                    state[code] = {"total_mv": last_close[code] * self.shares[code][day], "pe_ttm": self.pe[code][day]}
            today = pd.Series(returns, dtype=float)
            rows.append(builder.update(day, today, pd.DataFrame.from_dict(state, orient="index"),
                                       pd.Series(turnover, dtype=float).reindex(today.index)))
        return pd.DataFrame(rows).set_index("datetime")


def _run_ch4(names: dict, market: Market, end: pd.Timestamp, cache: dict | None, date=None) -> pd.DataFrame:
    for name in ("get_trade_days", "get_Ashares", "get_stock_info", "get_price", "get_fundamentals"):
        names[name] = getattr(market, name)
    calendar = names["trading_calendar"](end.strftime("%Y%m%d"))
    stub = {"calendar": calendar, "all_a": list(market.codes), "listed": names["listing_dates"](market.codes),
            "close": pd.DataFrame(index=calendar[-300:])}
    return names["ch4_factor_returns"](stub, end.strftime("%Y%m%d"), log=lambda *args: None, cache=cache, date=date)


# Break caught: CH-4 factor returns built from PTrade data that drift from the builder RQuant trains on (a day's
# state shifted, suspended stocks counted as traded, loss makers' P/E filled from other days).
def test_ch4_adapter_matches_the_reference_builder() -> None:
    names = namespace()
    market = Market()
    end = market.calendar[-1]
    ours = _run_ch4(names, market, end, None)
    theirs = market.reference_factors(ours.index)
    assert ours["mkt"].notna().sum() > 200 and ours["pmo"].notna().sum() > 200
    for column in ("mkt", "smb", "vmg", "pmo", "smb4", "ch4_stocks"):
        np.testing.assert_allclose(ours[column].to_numpy(float), theirs[column].to_numpy(float), rtol=1e-10,
                                   atol=1e-14, equal_nan=True, err_msg=column)


# Break caught: a warm run (the day's cache from the evening before) scoring differently from a cold one.
def test_ch4_adapter_reuses_the_cache_and_matches_a_cold_run() -> None:
    names = namespace()
    market = Market()
    cache: dict = {}
    _run_ch4(names, market, market.calendar[-2], cache)
    calls = []
    original = market.get_price

    def counted(security, start_date=None, **kwargs):
        calls.append(start_date)
        return original(security, start_date=start_date, **kwargs)

    market.get_price = counted
    warm = _run_ch4(names, market, market.calendar[-1], cache)
    assert calls and min(calls) >= market.calendar[-5].strftime("%Y%m%d")  # only the last few days are read again
    market.get_price = original
    cold = _run_ch4(names, market, market.calendar[-1], None)
    pd.testing.assert_frame_equal(warm, cold)


# Break caught: the bundle's rewrite of the size_value_china reference changing what it computes.
# Break caught: a live morning (inputs end on today, which has no bars yet) streaming the CH-4 factors up to today, so
# the day it scores is not the stream's last and the stream starts a day late (2026-11-25: ch4_last 20261125, all NaN).
def test_ch4_stops_at_the_scored_day() -> None:
    names = namespace()
    market = Market()
    scored = market.calendar[-2]
    ours = _run_ch4(names, market, market.calendar[-1], None, date=scored)
    pd.testing.assert_frame_equal(ours, _run_ch4(names, market, scored, None))
    assert ours.index[-1] == scored


def test_bundled_size_value_china_matches_the_reference() -> None:
    names = namespace()
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=300)
    codes = [f"60{i:04d}.SS" for i in range(6)]
    close = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (300, 6)), axis=0)), index=dates, columns=codes)
    close.iloc[40:45, 2] = np.nan
    inputs = {"close": close,
              "basic_total_mv": close * 1e5, "basic_pe_ttm": pd.DataFrame(rng.uniform(-5, 50, (300, 6)), dates, codes),
              "basic_turnover_total": pd.DataFrame(rng.uniform(0.001, 0.05, (300, 6)), dates, codes)}
    for name in ("mkt", "smb", "vmg", "pmo", "smb4"):
        inputs["ch4_" + name] = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    ours = names["SizeValueChina"](inputs)
    theirs = SizeValueChina(inputs)
    for name in SIZE_VALUE_CHINA_NAMES:
        pd.testing.assert_frame_equal(ours.calculate(name), theirs.calculate(name), check_exact=False, rtol=1e-12)
    features = names["svc_features"](inputs, dates[-1], ["svc_001", "svc_021", "svc_004"], codes)
    np.testing.assert_allclose(features[:, 1], theirs.calculate(SIZE_VALUE_CHINA_NAMES[20]).iloc[-1].to_numpy())


def test_svc_inputs_use_tushare_daily_basic_units() -> None:
    names = namespace()
    dates = pd.bdate_range("2020-01-01", periods=300)
    panel = pd.DataFrame(np.arange(600.0).reshape(300, 2) + 1, index=dates, columns=["600000.SS", "000001.SZ"])
    factors = pd.DataFrame({name: np.linspace(0, 1, 300) for name in ("mkt", "smb", "vmg", "pmo", "smb4")},
                           index=dates)
    inputs = names["svc_inputs"]({"close": panel, "cap": panel * 1e6, "basic_pe_ttm": panel,
                                  "basic_turnover_total": panel / 100}, factors)
    assert len(inputs["close"]) == names["SVC_DAYS"]
    pd.testing.assert_frame_equal(inputs["basic_total_mv"], panel.iloc[-names["SVC_DAYS"]:] * 100.0)
    assert inputs["ch4_pmo"].index.equals(inputs["close"].index)
    morning = names["svc_inputs"]({"close": panel, "cap": panel * 1e6, "basic_pe_ttm": panel,
                                   "basic_turnover_total": panel / 100}, factors, dates[-2])
    assert morning["close"].index[-1] == dates[-2] and len(morning["close"]) == names["SVC_DAYS"]


# Break caught: a live blend that weighs one model by its score scale, or keeps stocks only one model scores.
def test_blend_matches_blend_runs() -> None:
    spec = importlib.util.spec_from_file_location("blend_runs", ROOT / "scripts" / "analysis" / "blend_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(5)
    codes = [f"{i:06d}" for i in range(40)]
    jq = pd.Series(rng.normal(0, 0.1, 40), index=codes)
    svc = pd.Series(np.round(rng.normal(0, 0.01, 40), 3), index=codes).drop(codes[:3])  # ties, fewer stocks
    ours = namespace()["blend_scores"]([jq, svc])
    day = pd.Timestamp("2026-10-21")
    frames = [pd.DataFrame({"datetime": day, "instrument": series.index, "score": series.values})
              for series in (jq, svc)]
    theirs = module.blend(frames).set_index("instrument")["score"]
    pd.testing.assert_series_equal(ours.sort_index(), theirs.sort_index(), check_names=False, rtol=1e-12)


def test_expected_scores_are_rquant_blend_of_the_day() -> None:
    expected = namespace()["EXPECTED"]
    assert len(expected["blend"]) == len(expected["jq240"]) == len(expected["svc"]) > 3000
    jq, svc = pd.Series(expected["jq240"]), pd.Series(expected["svc"])
    blend = namespace()["blend_scores"]([jq, svc])
    np.testing.assert_allclose(blend.sort_index().to_numpy(), pd.Series(expected["blend"]).sort_index().to_numpy(),
                               atol=1e-4)  # the scores are kept to 7 significant digits
    assert max(expected["blend"], key=expected["blend"].get) == "SH603668"


class ValuationFake(ptl.LiveFake):
    """The jq240 tests' fake PTrade with a pe_ttm column in the valuation table (one loss maker)."""

    def get_fundamentals(self, security, table, fields=None, date=None, **kwargs):
        wanted = [field for field in fields if field != "pe_ttm"] if table == "valuation" else fields
        frame = super().get_fundamentals(security, table, wanted, date=date, **kwargs)
        if table == "valuation" and "pe_ttm" in fields:
            frame["pe_ttm"] = [(-3.0 if code == "600003.SS" else 12.0 + len(code)) for code in frame.index]
        return frame


FAKE_SVC_MODEL = {"run": "fake-svc", "prediction_year": 2011, "fit_period": ["2010-01-01", "2010-12-31"],
                  "features": [f"svc_{ordinal:03d}" for ordinal in range(1, 22)], "mean": [0.0] * 21, "std": [1.0] * 21,
                  "trees": [{"feature": [0], "threshold": [3.0], "left": [~0], "right": [~1], "leaf": [-1.0, 1.0]}]}


def _small_market(names: dict) -> None:
    names.update(MARKET_MIN_STOCKS=1, CH4_MIN_STOCKS=1, SVC_MODEL=FAKE_SVC_MODEL)


def test_research_bundle_runs_on_the_fake_market(capsys) -> None:
    source = build_main_blend.build_parity()
    source = source.replace('AS_OF = "20261021"', f'AS_OF = "{qmt.END}"').replace("DAYS = 1100", f"DAYS = {qmt.DAYS}")
    source = source.replace("MARKET_MIN_STOCKS = 100", "MARKET_MIN_STOCKS = 1").replace(
        "CH4_MIN_STOCKS = 100", "CH4_MIN_STOCKS = 1")
    names: dict = {}
    fake = ValuationFake()
    for name in pti.API:
        names[name] = getattr(fake, name)
    exec(compile(source, "bundle", "exec"), names)  # noqa: S102 - our own generated file
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("[RQ] parity=end") for line in lines), lines[-5:]
    assert not any("parity.error" in line for line in lines)
    assert "[RQ] parity.universe=4" in lines  # 600001, 600003, 600004 and 000001; 600002 is ST
    top = next(line for line in lines if line.startswith("[RQ] score.top50="))
    assert "000001.SZ" in top and "830001" not in top
    stocks = next(line for line in lines if line.startswith("[RQ] svc.ch4_stocks_last="))
    assert int(stocks.split("=")[1]) > 0


class BlendLive(ptl.Live):
    """The main-board blend live bundle on the fake market."""

    def __init__(self, positions: dict, cash: float = 50_000.0, dry_run: bool = False) -> None:
        super().__init__(positions, cash=cash, dry_run=dry_run)
        self.fake = ValuationFake()
        names: dict = {"log": self.logger, "g": self.g}
        exec(compile(build_main_blend.build_live(), str(build_main_blend.LIVE_OUTPUT), "exec"), names)  # noqa: S102
        for name in pti.API:
            names[name] = getattr(self.fake, name)
        kept = {key: self.names[key] for key in ("beijing_clock", "get_snapshot", "order", "get_all_orders",
                                                 "run_daily", "get_user_name", "MODEL", "LIVE_DAYS",
                                                 "LIVE_CAPITAL", "LIVE_DRY_RUN")}
        names.update(kept)
        names.update(LIVE_TOPK=3, LIVE_N_DROP=1, MARKET_MIN_STOCKS=1, CH4_MIN_STOCKS=1, SVC_MODEL=FAKE_SVC_MODEL)
        self.names = names


def test_live_bundle_settings_are_the_backtested_ones() -> None:
    source = build_main_blend.build_live()
    for line in ('LIVE_UNIVERSE = "main"', "LIVE_TOPK = 50", "LIVE_N_DROP = 5", "LIVE_EXIT_RANK_PCT = None",
                 "LIVE_DRY_RUN = True", "LIVE_AUCTION = True", 'LIVE_SNAPSHOT_AT = ("15:10", "15:40", "16:10")'):
        assert f"\n{line}" in source
    assert source.index("\ndef plan_scores(") < source.rindex("\ndef plan_scores(")  # main_live's comes last


def test_live_bundle_trades_shenzhen_main_board_on_the_blend() -> None:
    live = BlendLive({"600002.SS": (300, 300), "300001.SZ": (500, 500)})
    live.run("initialize", "20110701080000")
    live.run("rq_prepare", "20110701080000")
    live.run("rq_auction", "20110701092100")
    lines = live.lines()
    assert not any("error" in line for line in lines), lines[-8:]
    assert any(line.startswith("[RQ] live.top3=") and ".SZ" in line for line in lines)
    assert any(line.startswith("[RQ] live.top10_svc=") for line in lines)
    assert any(line.startswith("[RQ] live=start") and "universe=main" in line for line in lines)
    sells = [code for code, amount, _ in live.sent if amount < 0]
    buys = [code for code, amount, _ in live.sent if amount > 0]
    assert sells == ["600002.SS"]  # the ST name; the ChiNext holding is not the strategy's
    assert buys and set(buys) <= {"600001.SS", "600003.SS", "600004.SS", "000001.SZ"}


# Break caught: a session planned from closing snapshots scoring differently from the same session's daily bars
# (a snapshot field mapped to the wrong bar field, the day added twice, or the canonical prices chained wrongly).
def test_snapshot_session_builds_the_inputs_of_its_daily_bars() -> None:
    names = namespace()
    fake = ValuationFake()
    for name in pti.API:
        names[name] = getattr(fake, name)
    last = pd.Timestamp(qmt.END)
    codes = names["main_codes"](qmt.END)
    full = names["build_inputs"](qmt.END, qmt.DAYS, lambda *args: None, codes=codes)
    fields = ["open", "high", "low", "close", "volume", "money", "preclose"]
    day_bars = fake.get_price(list(fake.bars), start_date=qmt.END, end_date=qmt.END, fields=fields, is_dict=True)
    original = fake.get_price

    def hidden(security, start_date=None, end_date=None, **kwargs):  # PTrade before it publishes the day's bars
        data = original(security, start_date=start_date, end_date=end_date, **kwargs)
        return dict((code, bars[bars["datetime"] != int(qmt.END)]) for code, bars in data.items())

    names["get_price"] = hidden
    snapshot = {code: {field: float(bars[field][0]) for field in fields}
                for code, bars in day_bars.items() if len(bars)}
    names["SESSION_EXTRA"].update(day=last, bars=snapshot)
    try:
        snapped = names["build_inputs"](qmt.END, qmt.DAYS, lambda *args: None, codes=codes)
    finally:
        names["SESSION_EXTRA"].clear()
        names["get_price"] = original
    for key in ("close", "open", "volume", "amount", "cap", "basic_turnover_total", "universe"):
        pd.testing.assert_frame_equal(snapped[key], full[key], check_exact=False, rtol=1e-12, obj=key)
    pd.testing.assert_series_equal(snapped["mkt"], full["mkt"], check_exact=False, rtol=1e-12)
    assert full["close"].loc[last].notna().sum() > 0


class SnapshotBlendLive(BlendLive):
    """BlendLive whose snapshots after the close carry the day's whole bar (high, low, volume in shares, amount)."""

    def get_snapshot(self, codes):
        snaps = super().get_snapshot(codes)
        for code, snap in snaps.items():
            volume = float(self.fake.bars[code]["volume"].iloc[-1]) * 100.0
            snap.update(high_px=snap["last_px"] * 1.01, low_px=snap["last_px"] * 0.98, business_amount=volume,
                        business_balance=volume * snap["last_px"])
        return snaps


# Break caught: the blend waiting for the next morning's daily bars (16 minutes from 09:10, too late for the 09:21
# auction on 2026-11-25 and 11-26) instead of planning the session from its closing snapshots that afternoon.
def test_live_bundle_plans_from_the_closing_snapshots_and_trades_the_auction() -> None:
    live = SnapshotBlendLive({"600002.SS": (300, 300)})
    live.run("initialize", "20110701080000")
    assert ("rq_session_close", "15:10") in live.jobs and ("rq_snapshot_check", "09:45") in live.jobs
    live.run("rq_session_close", "20110701151000")  # 07-01 has no daily bars yet: its snapshots stand in
    lines = live.lines()
    assert not any("error" in line for line in lines), lines[-8:]
    assert "[RQ] live.data_through=20110701" in lines
    assert any(line.startswith("[RQ] live.snapshot_plan=20110701") for line in lines)
    assert live.g.rq_plan["date"] == "20110701" and live.g.rq_snapshot["day"] == "20110701"
    count = len(lines)
    live.run("rq_prepare", "20110704080000")  # the morning has its plan: no waiting for bars, no planning
    live.run("rq_auction", "20110704092100")
    assert not any("live.wait" in line or "inputs.codes" in line for line in live.lines()[count:])
    assert any(amount < 0 for _, amount, _ in live.sent) and any(amount > 0 for _, amount, _ in live.sent)
    assert live.g.rq_done == "20110704"


# Break caught: a trade started (or the 21:00 run) after the close planning the day before from daily bars, so the
# next morning waits for the bars again.
def test_live_bundle_started_in_the_evening_plans_from_the_snapshots() -> None:
    live = SnapshotBlendLive({})
    live.run("initialize", "20110701210000")
    live.run("rq_prepare", "20110701210000")
    lines = live.lines()
    assert not any("error" in line for line in lines), lines[-8:]
    assert live.g.rq_plan["date"] == "20110701"
    assert any(line.startswith("[RQ] live.snapshot_plan=20110701") for line in lines)


def test_committed_main_blend_bundles_are_current() -> None:
    assert build_main_blend.PARITY_OUTPUT.read_text(encoding="ascii") == build_main_blend.build_parity()
    live = build_main_blend.LIVE_OUTPUT.read_text(encoding="ascii")
    assert live == build_main_blend.build_live()
    for word in ("import os", "import sys", "import traceback", "globals(", "__import__", "\ndef log("):
        assert word not in live
