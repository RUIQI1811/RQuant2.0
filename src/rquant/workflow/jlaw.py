"""A rule-based approximation of J Law's momentum swing-trading method, backtested day by day on A-share bars.

J Law (two-time U.S. Investing Championship winner, 2024 and 2025) has not published a mechanical system. What he has
said in public is a set of principles; this module turns each into one explicit, testable rule. ``RULES`` lists every
rule with its source and whether the number is his or an approximation chosen here, so reports can print it.

The strategy is long only (A shares cannot be shorted in an ordinary account) and trades a cash account:

* Each day after the close, stocks in the universe that pass the screen (stage 2 trend, size, liquidity, beta,
  relative strength, sector strength) and break out of their 20-day range on volume become candidates. The strongest
  candidates by relative strength fill the free slots, at most ``max_positions`` names.
* Entries fill on the next day at that day's VWAP (his rule: no trades in the first 15 minutes) unless the stock opens
  at its upper limit or gaps above the breakout by more than ``max_gap``.
* The whole position is bought at once. Its size risks ``risk_per_trade`` of equity down to the stop (the breakout
  day's low, clamped to ``min_stop``..``max_stop`` below the entry), capped at ``max_weight`` of equity and by cash.
* The stop is checked against each day's low; a stop that is hit fills at the stop, or at the open when the stock
  opens below it. A stock locked at its lower limit all day cannot be sold. Shares bought today cannot be sold today
  (T+1). Stops never move down; at ``breakeven_r`` times the initial risk the stop moves to the entry price.
* A close below the ``exit_ema``-day EMA (the trend fading) sells at the next open, as does leaving the universe.
* Winners in a steep trend get up to ``len(add_fractions)`` add-ons, each smaller than the last, every time the close
  is ``add_step`` above the last buy price; after an add the stop is at least the first entry price.
* New entries need the market filter: the benchmark index closing above its ``market_ma``-day average. When the
  account is more than ``drawdown_reduce`` below its peak, the risk per trade drops to ``risk_reduced``.

Prices are adjusted (``raw * factor``); a position holds ``units`` of adjusted price, so dividends and bonus shares are
reinvested in kind. Buys are rounded to 100 raw shares (N raw shares are N / factor units). Costs follow
``AshareCostModel`` (commission with a minimum, date-aware sell stamp tax). Daily bars cannot tell whether a stop or a
target was hit first within a day, so stops are approximated as described above.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from rquant.qlib_ext.exchange import AshareCostModel

# Every rule of the strategy: (rule, source, "J Law" when the number or rule is his own words, else "approximation").
RULES: tuple[tuple[str, str, str], ...] = (
    (
        "Stage 2 trend: close > 200-day SMA",
        "Stage 2 screen attributed to J Law (usstockradar.com summary of his video)",
        "J Law (second hand)",
    ),
    (
        "Size and liquidity: total market cap >= min_cap, 20-day average turnover >= min_amount",
        "US screen: market cap >= USD 2B, about USD 30M a day; scaled to A shares here",
        "approximation",
    ),
    ("Beta to the benchmark over 250 days >= min_beta", "Stage 2 screen: beta >= 1", "J Law (second hand)"),
    (
        "Relative strength: 3/6/9/12-month return score (weights 0.4/0.2/0.2/0.2) in the top rs_min_pct of the "
        "universe",
        "Longs come from the strongest stocks (TraderLion interview); the score is IBD-style",
        "approximation",
    ),
    (
        "Sector strength: Shenwan L1 sector median 63-day return in the top sector_top of sectors",
        "Strongest stocks in the strongest sectors; he uses the 63-day line as the quarter",
        "approximation",
    ),
    (
        "Uptrend: close > 50-day SMA > 200-day SMA and the 200-day SMA higher than 20 days ago",
        "He reads 10/20-day MAs for the short trend and 50/200-day MAs for the broad trend",
        "approximation",
    ),
    (
        "Breakout: close above the highest high of the prior breakout_lookback days on volume >= volume_mult x "
        "its 50-day average",
        "Breakout trigger: price clears recent highs or key resistance on increased volume",
        "J Law (numbers are approximations)",
    ),
    (
        "Pullback (optional): leader in an uptrend touches its 21-day EMA, then closes above the prior day's high",
        "Bull market: buy strong stocks on pullbacks; added pullback entries with EMAs",
        "approximation",
    ),
    (
        "Entry at next day's VWAP; skipped at the upper limit or a gap above max_gap",
        "Avoid trading in the first 15 minutes after the open",
        "approximation",
    ),
    (
        "Full position at once, sized to risk risk_per_trade (0.3%) of equity to the stop",
        "Risk per trade at most 0.5% of the account, usually about 0.3%; enter the full position at once",
        "J Law",
    ),
    (
        "Initial stop at the signal day's low (pullback: 5-day low), kept between min_stop and max_stop below entry",
        "Tight stops, low win rate, high reward-to-risk",
        "approximation",
    ),
    (
        "Add-ons only after the close rises add_step above the last buy and above a rising 10-day EMA; each add "
        "smaller than the last (add_fractions of the first buy)",
        "Add only in strongly trending stocks, only at higher prices, each add smaller than the first",
        "J Law (numbers are approximations)",
    ),
    (
        "Exit on a close below the exit_ema-day EMA; stop to breakeven at breakeven_r x initial risk",
        "Ride the trend until momentum fades; 10/20-day MAs mark the short-term trend",
        "approximation",
    ),
    (
        "No new entries unless the benchmark closes above its market_ma-day SMA",
        "Bull market: buy strong stocks; bear market: short or day-trade (not possible long-only)",
        "approximation",
    ),
    (
        "Risk per trade cut to risk_reduced (0.15%) when equity is more than drawdown_reduce below its peak",
        "In a rough patch he cuts his limit from 0.3% to 0.15%",
        "J Law (trigger is an approximation)",
    ),
    (
        "At most max_positions names, each at most max_weight of equity, no leverage",
        "Focus on a few leadership stocks",
        "approximation",
    ),
)


@dataclass(frozen=True)
class JLawParams:
    # Screen
    min_cap: float = 5e9
    min_amount: float = 1e8
    min_beta: float | None = 1.0
    beta_window: int = 250
    rs_min_pct: float = 0.8
    sector_top: float = 1.0 / 3.0
    # Breakout entry
    breakout_lookback: int = 20
    volume_window: int = 50
    volume_mult: float = 1.5
    # Pullback entry (off by default)
    enable_pullback: bool = False
    pullback_rs_min_pct: float = 0.9
    pullback_window: int = 5
    # Sizing and stops
    risk_per_trade: float = 0.003
    risk_reduced: float = 0.0015
    drawdown_reduce: float = 0.10
    min_stop: float = 0.02
    max_stop: float = 0.08
    max_weight: float = 0.25
    max_positions: int = 10
    # Pyramiding
    add_step: float = 0.10
    add_fractions: tuple[float, ...] = (0.5, 0.25)
    # Exits
    exit_ema: int = 21
    breakeven_r: float = 2.0
    # Market filter
    market_filter: bool = True
    market_ma: int = 50
    # Execution
    entry_price: str = "vwap"
    max_gap: float = 0.05
    initial_capital: float = 250_000.0
    trade_unit: int = 100

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["add_fractions"] = list(self.add_fractions)
        return values


# Panels the backtest reads: dates x instruments, adjusted prices unless named raw_.
PANEL_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "factor",
    "raw_high",
    "raw_low",
    "raw_volume",
    "raw_amount",
    "cap",
    "sector",
    "limit_buy",
    "limit_sell",
    "member",
)


@dataclass
class Position:
    instrument: int
    entry_date: Any
    entry_price: float
    initial_stop: float
    stop: float
    units: float
    first_units: float
    last_buy_price: float
    cost_basis: float
    entry_kind: str
    adds: int = 0
    bought_today: float = 0.0
    pending_exit: str | None = None
    fees: float = 0.0


@dataclass
class BacktestResult:
    daily: Any
    trades: Any
    open_positions: Any
    params: dict[str, Any] = field(default_factory=dict)


def compute_signals(panels: dict[str, Any], benchmark_close: Any, params: JLawParams) -> dict[str, Any]:
    """Return the dates x instruments signal panels (computed from data up to and including each day's close)."""
    import pandas as pd

    close = panels["close"]
    member = panels["member"].fillna(False).astype(bool)
    has_bar = close.notna()
    filled = close.ffill()

    def ema(span: int) -> Any:
        return filled.ewm(span=span, adjust=False, min_periods=span).mean()

    ema10 = ema(10)
    ema21 = ema(params.exit_ema)
    sma50 = filled.rolling(50, min_periods=50).mean()
    sma200 = filled.rolling(200, min_periods=200).mean()

    trend = (filled > sma50) & (sma50 > sma200) & (sma200 > sma200.shift(20))

    # IBD-style relative strength, ranked among the day's universe members.
    rs_raw = sum(
        weight * (filled / filled.shift(days) - 1.0)
        for weight, days in ((0.4, 63), (0.2, 126), (0.2, 189), (0.2, 252))
    )
    rs_raw = rs_raw.where(member & has_bar)
    rs_pct = rs_raw.rank(axis=1, pct=True)

    # Sector strength: each Shenwan L1 sector's median 63-day return among members, ranked across sectors.
    ret63 = (filled / filled.shift(63) - 1.0).where(member & has_bar)
    sector_ok = strong_sectors(ret63, panels["sector"], params.sector_top)

    amount20 = panels["raw_amount"].rolling(20, min_periods=15).mean()
    volume_avg = panels["raw_volume"].rolling(params.volume_window, min_periods=params.volume_window // 2).mean()
    prior_high = panels["high"].rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max().shift(1)

    screen = (
        member
        & has_bar
        & (filled > sma200)
        & trend
        & (panels["cap"] >= params.min_cap)
        & (amount20 >= params.min_amount)
        & (rs_pct >= params.rs_min_pct)
        & sector_ok
    )
    if params.min_beta is not None:
        stock_ret = filled.pct_change(fill_method=None)
        market_ret = benchmark_close.reindex(close.index).ffill().pct_change(fill_method=None)
        window = params.beta_window
        mean_x = stock_ret.rolling(window, min_periods=window // 2).mean()
        mean_m = market_ret.rolling(window, min_periods=window // 2).mean()
        mean_xm = stock_ret.mul(market_ret, axis=0).rolling(window, min_periods=window // 2).mean()
        var_m = market_ret.rolling(window, min_periods=window // 2).var(ddof=0)
        beta = (mean_xm - mean_x.mul(mean_m, axis=0)).div(var_m, axis=0)
        screen &= beta >= params.min_beta

    breakout = (
        screen
        & (close > prior_high)
        & (panels["raw_volume"] >= params.volume_mult * volume_avg.shift(1))
    )
    breakout_stop = panels["low"]

    if params.enable_pullback:
        window = params.pullback_window
        touched = (panels["low"] <= ema21 * 1.01).astype(float).rolling(window, min_periods=1).max().astype(bool)
        confirm = (close > panels["high"].shift(1)) & (close > ema10) & (close > ema21)
        pullback = screen & (rs_pct >= params.pullback_rs_min_pct) & touched & confirm & ~breakout
        pullback_stop = panels["low"].rolling(window, min_periods=1).min()
    else:
        pullback = pd.DataFrame(False, index=close.index, columns=close.columns)
        pullback_stop = breakout_stop

    market = benchmark_close.reindex(close.index).ffill()
    market_ok = market > market.rolling(params.market_ma, min_periods=params.market_ma).mean()
    if not params.market_filter:
        market_ok = pd.Series(True, index=close.index)

    return {
        "breakout": breakout.fillna(False),
        "breakout_stop": breakout_stop,
        "pullback": pullback.fillna(False),
        "pullback_stop": pullback_stop,
        "rs_pct": rs_pct,
        "ema10": ema10,
        "ema21": ema21,
        "market_ok": market_ok.fillna(False).astype(bool),
    }


def strong_sectors(returns: Any, sector: Any, top: float) -> Any:
    """Mark the cells whose sector's median return ranks in the ``top`` share of sectors on that day."""
    import pandas as pd

    long = pd.DataFrame({"ret": returns.stack(future_stack=True), "sector": sector.stack(future_stack=True)}).dropna()
    if long.empty:
        return pd.DataFrame(False, index=returns.index, columns=returns.columns)
    dates = long.index.get_level_values(0)
    medians = long.groupby([dates, long["sector"]])["ret"].median()
    rank = medians.groupby(level=0).rank(ascending=False, method="first")
    count = medians.groupby(level=0).transform("size")
    keep = rank <= (count * top).apply(math.ceil).clip(lower=1)
    strong = set(keep[keep].index)
    flags = [(day, code) in strong for day, code in zip(dates, long["sector"], strict=True)]
    marked = pd.Series(flags, index=long.index, dtype=bool).unstack(fill_value=False)
    return marked.reindex(index=returns.index, columns=returns.columns, fill_value=False).astype(bool)


def run_backtest(
    panels: dict[str, Any],
    benchmark_close: Any,
    params: JLawParams,
    start: Any,
    end: Any,
    *,
    cost_model: AshareCostModel | None = None,
    signals: dict[str, Any] | None = None,
) -> BacktestResult:
    """Simulate the strategy from ``start`` to ``end`` on panels that also cover the indicators' warm-up.

    ``signals`` may be ``compute_signals`` output for the same panels and params, to share it between runs.
    """
    import numpy as np
    import pandas as pd

    cost_model = cost_model or AshareCostModel()
    if signals is None:
        signals = compute_signals(panels, benchmark_close, params)
    dates = panels["close"].index
    instruments = list(panels["close"].columns)
    first = int(dates.searchsorted(pd.Timestamp(start)))
    last = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    if first > last:
        raise ValueError(f"No trading days between {start} and {end}")

    flags = ("limit_buy", "limit_sell", "member")
    arr = {name: panels[name].to_numpy(dtype=float) for name in PANEL_FIELDS if name not in flags}
    limit_buy = panels["limit_buy"].fillna(False).to_numpy(dtype=bool)
    limit_sell = panels["limit_sell"].fillna(False).to_numpy(dtype=bool)
    member = panels["member"].fillna(False).to_numpy(dtype=bool)
    last_close = panels["close"].ffill().to_numpy(dtype=float)
    breakout = signals["breakout"].to_numpy(dtype=bool)
    pullback = signals["pullback"].to_numpy(dtype=bool)
    breakout_stop = signals["breakout_stop"].to_numpy(dtype=float)
    pullback_stop = signals["pullback_stop"].to_numpy(dtype=float)
    rs_pct = signals["rs_pct"].to_numpy(dtype=float)
    ema10 = signals["ema10"].to_numpy(dtype=float)
    ema21 = signals["ema21"].to_numpy(dtype=float)
    market_ok = signals["market_ok"].to_numpy(dtype=bool)
    bench = benchmark_close.reindex(dates).ffill().to_numpy(dtype=float)

    cash = float(params.initial_capital)
    peak = cash
    positions: dict[int, Position] = {}
    pending_entries: list[tuple[int, float, float, str]] = []  # (instrument, signal close, stop, kind)
    pending_adds: list[int] = []
    trades: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    previous_equity = cash

    def equity_at(row: int) -> float:
        return cash + sum(position.units * last_close[row, code] for code, position in positions.items())

    def fee(value: float, day: Any, *, sell: bool) -> float:
        return cost_model.total(value, day.date(), is_sell=sell)

    def sell(position: Position, row: int, price: float, units: float, reason: str) -> None:
        nonlocal cash
        value = units * price
        charge = fee(value, dates[row], sell=True)
        cash += value - charge
        position.fees += charge
        share = units / position.units
        basis = position.cost_basis * share
        position.cost_basis -= basis
        position.units -= units
        risk = (position.entry_price - position.initial_stop) * position.first_units
        trades.append(
            {
                "instrument": instruments[position.instrument],
                "entry_kind": position.entry_kind,
                "entry_date": position.entry_date,
                "exit_date": dates[row],
                "entry_price": position.entry_price,
                "exit_price": price,
                "units": units,
                "adds": position.adds,
                "pnl": value - basis - charge,
                "return": price / (basis / units) - 1.0 if units > 0 else 0.0,
                "r_multiple": (value - basis) / (risk * share) if risk > 0 else float("nan"),
                "holding_days": int(dates.get_loc(dates[row]) - dates.get_loc(position.entry_date)),
                "reason": reason,
            }
        )
        if position.units <= 1e-9:
            del positions[position.instrument]

    def buy_units(row: int, code: int, price: float, budget: float) -> tuple[float, float]:
        """Units and fee for the largest whole-lot buy of ``code`` at adjusted ``price`` that fits ``budget``."""
        raw_price = price / arr["factor"][row, code]
        if not raw_price > 0:
            return 0.0, 0.0
        lots = math.floor(budget / (raw_price * params.trade_unit))
        while lots > 0:
            value = lots * params.trade_unit * raw_price
            charge = fee(value, dates[row], sell=False)
            if value + charge <= cash and value <= budget + 1e-9:
                return lots * params.trade_unit / arr["factor"][row, code], charge
            lots -= 1
        return 0.0, 0.0

    for row in range(first, last + 1):
        day = dates[row]
        for position in positions.values():
            position.bought_today = 0.0
        has = np.isfinite(arr["open"][row])
        locked_down = has & limit_sell[row] & (arr["raw_high"][row] <= arr["raw_low"][row])

        # 1. Pending exits at the open.
        for code, position in list(positions.items()):
            if position.pending_exit and has[code] and not locked_down[code]:
                sell(position, row, arr["open"][row, code], position.units, position.pending_exit)

        # 2. Add-ons, then new entries, at the entry price.
        equity_prev = previous_equity
        drawdown = equity_prev / peak - 1.0 if peak > 0 else 0.0
        risk = params.risk_reduced if drawdown < -params.drawdown_reduce else params.risk_per_trade
        fill_field = "vwap" if params.entry_price == "vwap" else "open"
        for code in pending_adds:
            position = positions.get(code)
            if position is None or not has[code] or limit_buy[row, code]:
                continue
            fraction = params.add_fractions[position.adds]
            price = arr[fill_field][row, code]
            room = params.max_weight * equity_prev - position.units * price
            budget = min(fraction * position.first_units * price, room)
            if budget <= 0:
                continue
            units, charge = buy_units(row, code, price, budget)
            if units <= 0:
                continue
            cash -= units * price + charge
            position.units += units
            position.bought_today += units
            position.cost_basis += units * price + charge
            position.fees += charge
            position.adds += 1
            position.last_buy_price = price
            position.stop = max(position.stop, position.entry_price)
        pending_adds = []

        for code, signal_close, stop, kind in pending_entries:
            if code in positions or len(positions) >= params.max_positions:
                continue
            if not has[code] or limit_buy[row, code]:
                continue
            if arr["open"][row, code] > signal_close * (1.0 + params.max_gap):
                continue
            price = arr[fill_field][row, code]
            if not np.isfinite(price):
                continue
            distance = (price - stop) / price
            distance = min(max(distance, params.min_stop), params.max_stop)
            stop_price = price * (1.0 - distance)
            budget = min(risk * equity_prev / (price - stop_price) * price, params.max_weight * equity_prev)
            units, charge = buy_units(row, code, price, budget)
            if units <= 0:
                continue
            cash -= units * price + charge
            positions[code] = Position(
                instrument=code,
                entry_date=day,
                entry_price=price,
                initial_stop=stop_price,
                stop=stop_price,
                units=units,
                first_units=units,
                last_buy_price=price,
                cost_basis=units * price + charge,
                entry_kind=kind,
                bought_today=units,
                fees=charge,
            )
        pending_entries = []

        # 3. Intraday stops on the shares that may be sold today (T+1).
        for code, position in list(positions.items()):
            if not has[code] or arr["low"][row, code] > position.stop:
                continue
            sellable = position.units - position.bought_today
            if locked_down[code]:
                # Locked at the lower limit all day: the stop order sells at the next open it can.
                position.pending_exit = "stop"
                continue
            if sellable <= 1e-9:
                continue
            price = min(arr["open"][row, code], position.stop)
            remaining = position.units - sellable
            sell(position, row, price, sellable, "stop")
            if remaining > 1e-9 and code in positions:
                positions[code].pending_exit = "stop"

        # 4. Close: move stops, mark exits and add-ons.
        for code, position in positions.items():
            if not has[code] or position.pending_exit:
                continue
            close = arr["close"][row, code]
            initial_risk = position.entry_price - position.initial_stop
            if close >= position.entry_price + params.breakeven_r * initial_risk:
                position.stop = max(position.stop, position.entry_price)
            if not member[row, code]:
                position.pending_exit = "left_universe"
            elif close < ema21[row, code]:
                position.pending_exit = "trend_exit"
            elif (
                position.adds < len(params.add_fractions)
                and close >= position.last_buy_price * (1.0 + params.add_step)
                and close > ema10[row, code]
                and ema10[row, code] > ema10[row - 1, code]
            ):
                pending_adds.append(code)

        # 5. New entry signals for tomorrow.
        free = params.max_positions - len(positions)
        if free > 0 and market_ok[row]:
            candidates = np.flatnonzero(breakout[row] | pullback[row])
            candidates = [code for code in candidates if code not in positions]
            candidates.sort(key=lambda code: -rs_pct[row, code])
            for code in candidates[:free]:
                kind = "breakout" if breakout[row, code] else "pullback"
                stop = breakout_stop[row, code] if kind == "breakout" else pullback_stop[row, code]
                pending_entries.append((code, arr["close"][row, code], stop, kind))

        equity = equity_at(row)
        peak = max(peak, equity)
        invested = equity - cash
        daily.append(
            {
                "datetime": day,
                "equity": equity,
                "return": equity / previous_equity - 1.0,
                "cash": cash,
                "exposure": invested / equity if equity > 0 else 0.0,
                "positions": len(positions),
                "bench": bench[row],
                "market_ok": bool(market_ok[row]),
            }
        )
        previous_equity = equity

    daily_frame = pd.DataFrame(daily)
    daily_frame["bench_return"] = daily_frame["bench"].pct_change(fill_method=None).fillna(0.0)
    trade_frame = pd.DataFrame(trades)
    open_frame = pd.DataFrame(
        [
            {
                "instrument": instruments[code],
                "entry_date": position.entry_date,
                "entry_price": position.entry_price,
                "initial_stop": position.initial_stop,
                "stop": position.stop,
                "units": position.units,
                "value": position.units * last_close[last, code],
                "adds": position.adds,
            }
            for code, position in positions.items()
        ]
    )
    return BacktestResult(daily=daily_frame, trades=trade_frame, open_positions=open_frame, params=params.to_dict())


def summarize(daily: Any, trades: Any, start: Any | None = None, end: Any | None = None) -> dict[str, float]:
    """Annualised (252-day) return, Sharpe, drawdown and trade statistics over a window of a finished backtest."""
    import pandas as pd

    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if start is not None:
        frame = frame[frame["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["datetime"] <= pd.Timestamp(end)]
    returns = frame["return"]
    nav = (1.0 + returns).cumprod()
    days = len(returns)
    annual = float(nav.iloc[-1] ** (252.0 / days) - 1.0) if days else float("nan")
    std = float(returns.std(ddof=1))
    drawdown = float((nav / nav.cummax() - 1.0).min()) if days else float("nan")
    bench = frame["bench_return"]
    excess = returns.to_numpy() - bench.to_numpy()
    result = {
        "annual": annual,
        "sharpe": float(returns.mean() * 252 / (std * math.sqrt(252))) if std > 0 else float("nan"),
        "max_drawdown": drawdown,
        "calmar": annual / abs(drawdown) if drawdown < 0 else float("nan"),
        "cumulative": float(nav.iloc[-1] - 1.0) if days else float("nan"),
        "bench_annual": float((1.0 + bench).prod() ** (252.0 / days) - 1.0) if days else float("nan"),
        "excess_annual": float(excess.mean() * 252) if days else float("nan"),
        "exposure": float(frame["exposure"].mean()),
        "days": float(days),
    }
    if trades is not None and len(trades):
        window = trades.copy()
        window["exit_date"] = pd.to_datetime(window["exit_date"])
        if start is not None:
            window = window[window["exit_date"] >= pd.Timestamp(start)]
        if end is not None:
            window = window[window["exit_date"] <= pd.Timestamp(end)]
        wins = window[window["pnl"] > 0]
        losses = window[window["pnl"] <= 0]
        result.update(
            {
                "trades": float(len(window)),
                "win_rate": float(len(wins) / len(window)) if len(window) else float("nan"),
                "avg_win_r": float(wins["r_multiple"].mean()) if len(wins) else float("nan"),
                "avg_loss_r": float(losses["r_multiple"].mean()) if len(losses) else float("nan"),
                "avg_holding_days": float(window["holding_days"].mean()) if len(window) else float("nan"),
            }
        )
    return result


BAR_COLUMNS = (
    "instrument",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "factor",
    "raw_high",
    "raw_low",
    "raw_volume",
    "raw_amount",
    "cap",
    "sector",
    "limit_buy",
    "limit_sell",
    "is_st",
    "listed_days",
    "in_csi300",
)


def load_panels(canonical_root: Any, universe: str, start: Any, end: Any) -> tuple[dict[str, Any], Any]:
    """Read canonical bars from ``start`` to ``end`` into dates x instruments panels, and the CSI300 close.

    Only instruments that are in ``universe`` on at least one day are kept; ``member`` marks the days they are in it.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from rquant.errors import DataContractError
    from rquant.universe import BOARD_PREFIXES, on_board, universe_mask

    root = Path(canonical_root)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for year in range(first.year, last.year + 1):
        path = root / "bars" / f"year={year}" / "bars.parquet"
        if not path.exists():
            raise DataContractError(f"Canonical bars partition is missing: {path}")
        frame = pd.read_parquet(path, columns=list(BAR_COLUMNS))
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame[(frame["datetime"] >= first) & (frame["datetime"] <= last)]
        if universe in BOARD_PREFIXES:
            frame = frame[on_board(frame["instrument"], universe)]
        frame["member"] = universe_mask(frame, universe).to_numpy(dtype=bool)
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True)
    del frames
    ever = set(bars.loc[bars["member"], "instrument"])
    bars = bars[bars["instrument"].isin(ever)]
    index = pd.DatetimeIndex(sorted(bars["datetime"].unique()))
    columns = pd.Index(sorted(ever))
    rows = index.get_indexer(bars["datetime"])
    cols = columns.get_indexer(bars["instrument"])
    panels = {}
    for name in PANEL_FIELDS:
        if name in ("limit_buy", "limit_sell", "member"):
            values = np.zeros((len(index), len(columns)), dtype=bool)
            values[rows, cols] = bars[name].fillna(False).to_numpy(dtype=bool)
        else:
            values = np.full((len(index), len(columns)), np.nan)
            values[rows, cols] = bars[name].to_numpy(dtype=np.float64)
        panels[name] = pd.DataFrame(values, index=index, columns=columns)
    del bars

    index_path = root / "reference" / "index_daily.parquet"
    if not index_path.exists():
        raise DataContractError(f"CSI300 daily bars are missing: {index_path}")
    benchmark = pd.read_parquet(index_path)
    benchmark = benchmark[benchmark["ts_code"].astype(str).isin(["000300.SH", "399300.SZ"])]
    benchmark = benchmark.sort_values("datetime").drop_duplicates("datetime", keep="last")
    benchmark_close = benchmark.set_index(pd.to_datetime(benchmark["datetime"]))["close"].astype(float)
    return panels, benchmark_close.reindex(index).ffill()
