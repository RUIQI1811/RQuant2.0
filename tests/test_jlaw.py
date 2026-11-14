from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rquant.qlib_ext.exchange import AshareCostModel
from rquant.workflow.jlaw import (
    RULES,
    JLawParams,
    compute_signals,
    run_backtest,
    strong_sectors,
    summarize,
)

DAYS = 300
SIGNAL = 270  # row of the breakout day in the fixture

# Every screen open, so only the breakout trigger and the trading rules decide.
OPEN_SCREEN = JLawParams(
    min_cap=0.0,
    min_amount=0.0,
    min_beta=None,
    rs_min_pct=0.0,
    sector_top=1.0,
    market_filter=False,
    initial_capital=1_000_000.0,
    add_fractions=(),
)


def make_panels(close_a: np.ndarray, *, low_a: np.ndarray | None = None, volume_a: np.ndarray | None = None) -> dict:
    """Two stocks: A (the one under test) and B (a flat stock that never breaks out)."""
    dates = pd.bdate_range("2020-01-01", periods=len(close_a))
    close_b = np.full(len(close_a), 10.0)
    close = pd.DataFrame({"SH600000": close_a, "SH600001": close_b}, index=dates)
    high = close * 1.001
    low = close * 0.995
    if low_a is not None:
        low["SH600000"] = low_a
    open_ = close.shift(1).fillna(close)
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    if volume_a is not None:
        volume["SH600000"] = volume_a
    ones = pd.DataFrame(1.0, index=dates, columns=close.columns)
    false = pd.DataFrame(False, index=dates, columns=close.columns)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vwap": (open_ + close) / 2.0,
        "factor": ones,
        "raw_high": high,
        "raw_low": low,
        "raw_volume": volume,
        "raw_amount": volume * close,
        "cap": ones * 1e10,
        "sector": pd.DataFrame({"SH600000": 1.0, "SH600001": 2.0}, index=dates),
        "limit_buy": false.copy(),
        "limit_sell": false.copy(),
        "member": ~false,
    }


def uptrend() -> np.ndarray:
    # Rises 0.3% a day, so the close is above its 20-day high every day: only the volume decides the breakout.
    return 10.0 * 1.003 ** np.arange(DAYS)


def breakout_volume() -> np.ndarray:
    volume = np.full(DAYS, 1_000_000.0)
    volume[SIGNAL] = 3_000_000.0
    return volume


def benchmark(panels: dict) -> pd.Series:
    return pd.Series(np.linspace(100, 120, len(panels["close"])), index=panels["close"].index)


def run(panels: dict, params: JLawParams = OPEN_SCREEN, start_row: int = 260):
    dates = panels["close"].index
    return run_backtest(
        panels,
        benchmark(panels),
        params,
        dates[start_row],
        dates[-1],
        cost_model=AshareCostModel(),
    )


def test_rules_name_their_source() -> None:
    assert len(RULES) >= 10
    assert all(len(rule) == 3 and all(rule) for rule in RULES)


def test_breakout_needs_volume_and_new_high() -> None:
    panels = make_panels(uptrend(), volume_a=breakout_volume())
    signals = compute_signals(panels, benchmark(panels), OPEN_SCREEN)
    flagged = signals["breakout"]["SH600000"]
    assert flagged.iloc[SIGNAL]
    assert flagged.sum() == 1
    assert not signals["breakout"]["SH600001"].any()


def test_entry_fills_next_day_at_vwap_sized_by_risk() -> None:
    panels = make_panels(uptrend(), volume_a=breakout_volume())
    result = run(panels)
    entry_row = SIGNAL + 1
    vwap = panels["vwap"]["SH600000"].iloc[entry_row]
    signal_low = panels["low"]["SH600000"].iloc[SIGNAL]
    # Stop distance (signal low to entry) is below min_stop, so the stop sits min_stop under the entry.
    assert (vwap - signal_low) / vwap < OPEN_SCREEN.min_stop
    risk_budget = OPEN_SCREEN.risk_per_trade * OPEN_SCREEN.initial_capital
    expected_value = risk_budget / OPEN_SCREEN.min_stop
    daily = result.daily.set_index("datetime")
    held_value = daily["equity"].iloc[entry_row - 260] - daily["cash"].iloc[entry_row - 260]
    assert result.open_positions.loc[0, "entry_price"] == pytest.approx(vwap)
    assert result.open_positions.loc[0, "initial_stop"] == pytest.approx(vwap * (1 - OPEN_SCREEN.min_stop))
    # Rounded down to 100 shares.
    assert expected_value - 100 * vwap * 1.01 <= held_value <= expected_value * 1.01
    assert result.open_positions.loc[0, "units"] % 100 == 0


def test_stop_fills_at_stop_or_lower_open_and_respects_t_plus_one() -> None:
    close = uptrend()
    low = close * 0.995
    entry_row = SIGNAL + 1
    low[entry_row] = close[entry_row] * 0.9  # Below the stop on the buy day: T+1 keeps the position.
    low[entry_row + 3] = close[entry_row + 3] * 0.95  # Hits the stop three days later.
    panels = make_panels(close, low_a=low, volume_a=breakout_volume())
    result = run(panels)
    trade = result.trades.iloc[0]
    stop = panels["vwap"]["SH600000"].iloc[entry_row] * (1 - OPEN_SCREEN.min_stop)
    assert trade["reason"] == "stop"
    assert trade["exit_date"] == panels["close"].index[entry_row + 3]
    assert trade["exit_price"] == pytest.approx(stop)
    assert trade["r_multiple"] == pytest.approx(-1.0, abs=0.05)


def test_locked_limit_down_cannot_sell() -> None:
    close = uptrend()
    low = close * 0.995
    entry_row = SIGNAL + 1
    low[entry_row + 2] = close[entry_row + 2] * 0.95
    panels = make_panels(close, low_a=low, volume_a=breakout_volume())
    locked = panels["close"].index[entry_row + 2]
    panels["limit_sell"].loc[locked, "SH600000"] = True
    panels["raw_high"].loc[locked, "SH600000"] = panels["raw_low"].loc[locked, "SH600000"]
    result = run(panels)
    # Locked all day, then sold the next day at the open, which is below the stop.
    trade = result.trades.iloc[0]
    assert trade["exit_date"] == panels["close"].index[entry_row + 3]
    assert trade["exit_price"] == pytest.approx(panels["open"]["SH600000"].iloc[entry_row + 3])


def test_close_below_ema_exits_next_open() -> None:
    close = uptrend()
    entry_row = SIGNAL + 1
    # Ten days after the entry the stock drifts down 0.4% a day until it closes under its 21-day EMA.
    for row in range(entry_row + 10, DAYS):
        close[row] = close[row - 1] * 0.996
    panels = make_panels(close, volume_a=breakout_volume())
    params = JLawParams(**{**OPEN_SCREEN.to_dict(), "min_stop": 0.08, "add_fractions": ()})
    result = run(panels, params)
    trade = result.trades.iloc[0]
    assert trade["reason"] == "trend_exit"
    ema = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().to_numpy()
    first_below = next(row for row in range(entry_row, DAYS) if close[row] < ema[row])
    assert trade["exit_date"] == panels["close"].index[first_below + 1]
    assert trade["exit_price"] == pytest.approx(panels["open"]["SH600000"].iloc[first_below + 1])


def test_equity_accounts_for_costs() -> None:
    panels = make_panels(uptrend(), volume_a=breakout_volume())
    result = run(panels)
    daily = result.daily
    # Before the entry the account is all cash; after it, equity is cash plus holdings and a commission was paid.
    assert daily["equity"].iloc[0] == pytest.approx(OPEN_SCREEN.initial_capital)
    entry = daily.iloc[SIGNAL + 1 - 260]
    assert entry["cash"] < OPEN_SCREEN.initial_capital
    assert entry["positions"] == 1
    assert 0 < entry["exposure"] < OPEN_SCREEN.max_weight + 1e-9
    stats = summarize(daily, result.trades)
    assert math.isfinite(stats["annual"])


def test_pyramid_adds_smaller_lots_at_higher_prices() -> None:
    close = 10.0 * 1.003 ** np.arange(DAYS)
    for row in range(SIGNAL + 1, DAYS):
        close[row] = close[row - 1] * 1.01  # A steep trend after the breakout.
    panels = make_panels(close, volume_a=breakout_volume())
    params = JLawParams(**{**OPEN_SCREEN.to_dict(), "add_fractions": (0.5, 0.25), "max_weight": 1.0})
    result = run(panels, params)
    position = result.open_positions.iloc[0]
    assert position["adds"] == 2
    assert position["stop"] >= position["entry_price"]


def test_market_filter_blocks_new_entries() -> None:
    panels = make_panels(uptrend(), volume_a=breakout_volume())
    params = JLawParams(**{**OPEN_SCREEN.to_dict(), "market_filter": True})
    falling = pd.Series(np.linspace(120, 100, DAYS), index=panels["close"].index)
    result = run_backtest(panels, falling, params, panels["close"].index[260], panels["close"].index[-1])
    assert result.open_positions.empty and result.trades.empty


def test_strong_sectors_keeps_top_share() -> None:
    dates = pd.bdate_range("2021-01-01", periods=2)
    returns = pd.DataFrame({"a": [0.3, 0.1], "b": [0.2, 0.2], "c": [-0.1, 0.3]}, index=dates)
    sector = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [3.0, 3.0]}, index=dates)
    marked = strong_sectors(returns, sector, 1.0 / 3.0)
    assert marked.loc[dates[0]].tolist() == [True, False, False]
    assert marked.loc[dates[1]].tolist() == [False, False, True]


def test_units_hold_the_raw_shares_value_when_prices_are_adjusted() -> None:
    # factor != 1: adjusted price = raw * factor, so a buy of N raw shares holds N / factor adjusted units and is worth
    # N * raw price in money.
    panels = make_panels(uptrend(), volume_a=breakout_volume())
    factor = 0.125
    for name in ("open", "high", "low", "close", "vwap"):
        panels[name] = panels[name] * factor
    panels["factor"] = panels["factor"] * factor
    result = run(panels)
    entry_row = SIGNAL + 1
    raw_vwap = panels["vwap"]["SH600000"].iloc[entry_row] / factor
    units = result.open_positions.loc[0, "units"]
    raw_shares = units * factor
    assert raw_shares == pytest.approx(round(raw_shares))
    assert raw_shares % 100 == pytest.approx(0)
    entry = result.daily.iloc[entry_row - 260]
    held = entry["equity"] - entry["cash"]
    # Same money at risk as the unadjusted fixture: about risk / min_stop of equity.
    expected_value = OPEN_SCREEN.risk_per_trade * OPEN_SCREEN.initial_capital / OPEN_SCREEN.min_stop
    assert expected_value - 100 * raw_vwap * 1.01 <= raw_shares * raw_vwap <= expected_value * 1.01
    assert held == pytest.approx(units * panels["close"]["SH600000"].iloc[entry_row])
    assert held == pytest.approx(raw_shares * panels["close"]["SH600000"].iloc[entry_row] / factor)
