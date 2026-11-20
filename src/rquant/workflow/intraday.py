"""Backtest a walk-forward run's daily book with intraday execution on minute bars, through Qlib's NestedExecutor.

The outer level is the daily backtest of ``rquant report``: Qlib's ``TopkDropoutStrategy`` on the run's predictions
(a prediction dated on day t trades on day t+1), the same costs, trade unit and terminal liquidation. Each day's
orders then go to an inner minute executor: ``WindowTWAPStrategy`` splits every sell over the sell window and every buy
over the buy window, dealing at each bar's VWAP; a bar at its limit price cannot trade in the blocked direction, and
what a window leaves unfilled stays undone that day. T+1 holds as in the daily backtest: the outer strategy decides
once a day and holds a purchase for at least one day (``hold_thresh=1``), and the inner level only executes those
orders, so nothing bought is sold the same day.

The minute quotes of a whole run do not fit in memory, so the backtest runs one calendar month at a time and carries
cash, holdings, their last prices and holding days into the next month. Orders are sized at the first bar of the day
(the daily backtest sizes them at the open).

With a ``start`` date the book starts from the daily backtest's holdings and cash of the day before (the run's
``positions.parquet`` and ``portfolio.parquet``), so both books hold the same stocks and the minute bars planned from
the daily book cover the intraday one. A stock-day that still has no minute bars trades whole at the daily open and is
valued at the daily close (``MinuteAshareExchange`` fallback bars); ``intraday.json`` counts such days and trades.

Qlib's nested executor books costs and turnover in the inner account only, so the outer daily report carries net
returns with zero cost; ``cost`` and ``turnover`` are refilled from the trades and ``return`` restored to the gross
return, so ``return - cost`` is the net return as in ``rquant report``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file
from rquant.qlib_ext.exchange import AshareCostModel
from rquant.qlib_ext.minute import MINUTE_DEAL_PRICE, MinuteAshareExchange, parse_window


@dataclass(frozen=True)
class IntradayConfig:
    initial_capital: float = 1_000_000.0
    topk: int = 50
    n_drop: int = 5
    benchmark: str = "SH000300"
    trade_unit: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_before: float = 0.001
    stamp_tax_after: float = 0.0005
    # Bar start labels, both ends included; "09:30-09:30" is the first minute (opening auction plus 09:30-09:31).
    buy_window: str = "09:30-09:59"
    sell_window: str = "09:30-09:59"
    start: str | None = None
    end: str | None = None
    # Start from the daily backtest's book on the day before ``start`` (needs ``start`` and the run's report files).
    start_from_daily: bool = True

    def __post_init__(self) -> None:
        if self.topk < 1 or not 1 <= self.n_drop <= self.topk:
            raise ValueError(f"need topk >= 1 and 1 <= n_drop <= topk, got {self.topk}/{self.n_drop}")
        parse_window(self.buy_window)
        parse_window(self.sell_window)

    @property
    def label(self) -> str:
        def compact(window: str) -> str:
            return window.replace(":", "").replace("-", "_")

        return f"top{self.topk}-drop{self.n_drop}-sell{compact(self.sell_window)}-buy{compact(self.buy_window)}"


class IntradayBacktestRunner:
    """Run the month-by-month nested backtest and write ``<output>/{portfolio,positions,trades}.parquet`` and
    ``intraday.json`` (the metrics ``rquant report`` writes, with execution prices against the daily open)."""

    def __init__(
        self,
        *,
        qlib_root: str | Path,
        minute_root: str | Path,
        run_directory: str | Path,
        output_directory: str | Path,
        config: IntradayConfig | None = None,
        log: Any = None,
    ) -> None:
        self.qlib_root = Path(qlib_root)
        self.minute_root = Path(minute_root)
        self.run_directory = Path(run_directory)
        self.output_directory = Path(output_directory)
        self.config = config or IntradayConfig()
        self.log = log or (lambda message: None)
        calendar = self.minute_root / "qlib" / "calendars" / "1min.txt"
        if not calendar.exists():
            raise DataContractError(f"Minute calendar does not exist: {calendar}; build the minute bars first")

    def run(self) -> dict[str, Any]:
        qlib, pd = _dependencies()
        from qlib.backtest import backtest
        from qlib.constant import REG_CN
        from qlib.contrib.evaluate import risk_analysis
        from qlib.data import D

        from rquant.qlib_ext.runtime import qlib_init_kwargs
        from rquant.workflow.backtest import METRICS_DEFINITION, load_predictions, risk_metrics

        qlib.init(
            provider_uri={"day": str(self.qlib_root), "1min": str(self.minute_root / "qlib")},
            region=REG_CN,
            **qlib_init_kwargs(),
        )
        predictions = load_predictions(pd, self.run_directory)
        signal = predictions.set_index(["datetime", "instrument"])["score"].sort_index()
        first_signal = predictions["datetime"].min()
        days = pd.DatetimeIndex(D.calendar(start_time=first_signal, freq="day"))
        last_signal = predictions["datetime"].max()
        days = days[days <= (days[days > last_signal][0] if (days > last_signal).any() else last_signal)]
        if self.config.start is not None:
            days = days[days >= pd.Timestamp(self.config.start)]
        if self.config.end is not None:
            days = days[days <= pd.Timestamp(self.config.end)]
        if len(days) < 2:
            raise DataContractError("The intraday backtest needs at least two trading days")
        months = sorted(set(days.to_period("M")))
        cost_model = AshareCostModel(
            commission_rate=self.config.commission_rate,
            minimum_commission=self.config.minimum_commission,
            stamp_tax_before=self.config.stamp_tax_before,
            stamp_tax_after=self.config.stamp_tax_after,
        )

        account: Any = float(self.config.initial_capital)
        previous_value: float | None = None
        start_book = None
        if self.config.start is not None and self.config.start_from_daily:
            start_book = _daily_book(D, pd, self.run_directory, days[0])
            if start_book is not None:
                account, previous_value = start_book["account"], start_book["value"]
                self.log(
                    f"starting from the daily book of {start_book['date']}: {len(account) - 1} holdings, "
                    f"value {previous_value:,.0f}"
                )
            else:
                self.log("warning: no daily book before the start in the run's report files; starting from cash")
        universe = predictions[["datetime", "instrument"]]
        reports, holdings, trades, uncovered = [], [], [], []
        unfilled = {"buy": 0, "sell": 0}
        for number, month in enumerate(months):
            month_days = days[days.to_period("M") == month]
            start, end = month_days[0], month_days[-1]
            final = number == len(months) - 1
            window = universe[(universe["datetime"] >= start - pd.Timedelta(days=15)) & (universe["datetime"] <= end)]
            fallback_codes = set(window["instrument"]) | (
                set(account) - {"cash"} if isinstance(account, dict) else set()
            )
            exchange = MinuteAshareExchange(
                minute_root=self.minute_root,
                fallback_codes=sorted(fallback_codes),
                start_time=start,
                end_time=end,
                codes=[],
                cost_model=cost_model,
                trade_unit=self.config.trade_unit,
            )
            strategy = {
                "class": "LiquidatingTopkDropoutStrategy" if final else "TopkDropoutStrategy",
                "module_path": "rquant.qlib_ext.strategy" if final else "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": signal,
                    "topk": self.config.topk,
                    "n_drop": self.config.n_drop,
                    "risk_degree": 1.0,
                    "hold_thresh": 1,
                    "only_tradable": True,
                    "forbid_all_trade_at_limit": False,
                },
            }
            # Qlib pops the cash out of an account dict.
            carried_cash = float(account["cash"]) if isinstance(account, dict) else float(account)
            portfolio, indicators = backtest(
                start_time=start,
                end_time=end,
                strategy=strategy,
                executor=self._executor(),
                benchmark=self.config.benchmark,
                account=dict(account) if isinstance(account, dict) else account,
                exchange_kwargs={"exchange": exchange},
            )
            report, positions = portfolio["1day"]
            if previous_value is not None:
                report = _rebase_first_day(report, previous_value, carried_cash)
            previous_value = float(report["account"].iloc[-1])
            reports.append(report)
            holdings.append(_holdings(pd, positions, exchange))
            trades.append(_trades(pd, indicators["1day"][1], exchange))
            for direction, count in _unfilled_orders(pd, indicators["1day"][1]).items():
                unfilled[direction] += count
            uncovered.extend(_uncovered(pd, positions, exchange))
            last = positions[max(positions)]
            account = _carry(last)
            self.log(
                f"{month}: account {previous_value:,.0f}, "
                f"{len(exchange.codes) - len({code for code, _ in exchange.fallback_days})}+ instruments with minute "
                f"bars, {len(exchange.fallback_days)} stock-days on daily fallback bars"
            )

        report = pd.concat(reports).sort_index()
        report = report[~report.index.duplicated(keep="last")]
        trades_frame = _with_open_slippage(D, pd, pd.concat(trades, ignore_index=True))
        initial_value = start_book["value"] if start_book is not None else float(self.config.initial_capital)
        report = _with_trade_costs(pd, report, trades_frame, initial_value)
        holdings_frame = pd.concat(holdings, ignore_index=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        portfolio_path = self.output_directory / "portfolio.parquet"
        report.reset_index().to_parquet(portfolio_path, index=False)
        positions_path = self.output_directory / "positions.parquet"
        holdings_frame.to_parquet(positions_path, index=False)
        trades_path = self.output_directory / "trades.parquet"
        trades_frame.to_parquet(trades_path, index=False)

        manifest = {
            "status": "complete",
            "engine": "qlib.backtest.backtest with NestedExecutor (day -> 1min), one calendar month per call",
            "run_directory": str(self.run_directory),
            "start": str(days[0].date()),
            "end": str(days[-1].date()),
            "config": asdict(self.config),
            "execution": {
                "inner_strategy": "WindowTWAPStrategy",
                "deal_price": f"minute bar {MINUTE_DEAL_PRICE} (amount / volume)",
                "sizing_price": "first minute bar of the day",
                "limits": "bar raw close at the daily up / down limit blocks buys / sells in that bar",
                "t_plus_one": "outer TopkDropoutStrategy hold_thresh=1; the inner level only executes daily orders",
                "unfilled": "a slice that cannot trade rolls into later bars of the window; leftovers lapse",
                "no_minute_bars": "the stock-day trades whole at the daily open and is valued at the daily close",
            },
            "start_book": (
                {"date": start_book["date"], "value": start_book["value"], "holdings": len(start_book["account"]) - 1}
                if start_book is not None
                else None
            ),
            "holding_days": len(holdings_frame),
            "rows": len(report),
            "trade_rows": len(trades_frame),
            "uncovered_holding_days": len(uncovered),
            "uncovered_share": len(uncovered) / len(holdings_frame) if len(holdings_frame) else None,
            "uncovered_examples": uncovered[:20],
            "unfilled_orders": unfilled,
            "fallback_trades": int(trades_frame["fallback"].sum()) if len(trades_frame) else 0,
            "fallback_trade_value_share": _finite(
                trades_frame.loc[trades_frame["fallback"], "value"].sum() / trades_frame["value"].sum()
            )
            if len(trades_frame)
            else None,
            "execution_vs_open_bp": _slippage_summary(trades_frame[~trades_frame["fallback"]]),
            "metrics_definition": METRICS_DEFINITION,
            "metrics": risk_metrics(risk_analysis, report),
            "artifacts": {
                "portfolio": sha256_file(portfolio_path),
                "positions": sha256_file(positions_path),
                "trades": sha256_file(trades_path),
            },
        }
        atomic_write_json(self.output_directory / "intraday.json", manifest)
        return manifest

    def _executor(self) -> dict[str, Any]:
        return {
            "class": "ZeroOrderSafeNestedExecutor",
            "module_path": "rquant.qlib_ext.minute",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "verbose": False,
                "inner_strategy": {
                    "class": "WindowTWAPStrategy",
                    "module_path": "rquant.qlib_ext.minute",
                    "kwargs": {"buy_window": self.config.buy_window, "sell_window": self.config.sell_window},
                },
                "inner_executor": {
                    "class": "ZeroOrderSafeSimulatorExecutor",
                    "module_path": "rquant.qlib_ext.executor",
                    "kwargs": {"time_per_step": "1min", "generate_portfolio_metrics": False, "verbose": False},
                },
            },
        }


def _day_span(pd: Any, timestamp: Any) -> tuple[Any, Any]:
    day = pd.Timestamp(timestamp).normalize()
    return day, day + pd.Timedelta(hours=23, minutes=59)


def _factor(pd: Any, exchange: Any, stock: str, timestamp: Any) -> float:
    start, end = _day_span(pd, timestamp)
    value = exchange.get_factor(stock, start, end)
    return float(value) if value is not None else float("nan")


def _holdings(pd: Any, positions: dict[Any, Any], exchange: Any) -> Any:
    rows = []
    for timestamp, position in sorted(positions.items()):
        for stock in sorted(position.get_stock_list()):
            amount = float(position.get_stock_amount(stock))
            factor = _factor(pd, exchange, stock, timestamp)
            rows.append(
                {
                    "datetime": pd.Timestamp(timestamp).normalize(),
                    "instrument": stock,
                    "adjusted_amount": amount,
                    "shares": amount * factor,
                    "factor": factor,
                }
            )
    return pd.DataFrame(rows, columns=["datetime", "instrument", "adjusted_amount", "shares", "factor"])


def _trades(pd: Any, indicator: Any, exchange: Any) -> Any:
    rows = []
    for timestamp, order_indicator in sorted(indicator.order_indicator_his.items()):
        metrics = {
            name: order_indicator.get_index_data(name).to_dict()
            for name in ("amount", "deal_amount", "trade_price", "trade_value", "trade_cost", "trade_dir")
        }
        for instrument in sorted(set().union(*(metric.keys() for metric in metrics.values()))):
            amount = metrics["deal_amount"].get(instrument)
            if amount is None or pd.isna(amount) or abs(float(amount)) <= 1e-12:
                continue
            ordered = metrics["amount"].get(instrument)
            factor = _factor(pd, exchange, instrument, timestamp)
            day = pd.Timestamp(timestamp).normalize()
            rows.append(
                {
                    "datetime": pd.Timestamp(timestamp).normalize(),
                    "instrument": instrument,
                    "direction": "sell" if int(metrics["trade_dir"].get(instrument)) == 0 else "buy",
                    "adjusted_amount": abs(float(amount)),
                    "fill_rate": abs(float(amount)) / abs(float(ordered)) if ordered else None,
                    "shares": abs(float(amount)) * factor,
                    "price": float(metrics["trade_price"].get(instrument)),
                    "value": abs(float(metrics["trade_value"].get(instrument))),
                    "cost": float(metrics["trade_cost"].get(instrument)),
                    "fallback": (instrument, day) in exchange.fallback_days,
                }
            )
    columns = [
        "datetime",
        "instrument",
        "direction",
        "adjusted_amount",
        "fill_rate",
        "shares",
        "price",
        "value",
        "cost",
        "fallback",
    ]
    return pd.DataFrame(rows, columns=columns).astype({"fallback": bool})


def _unfilled_orders(pd: Any, indicator: Any) -> dict[str, int]:
    """Daily orders that dealt nothing (limit, suspension or cash in every bar of their window), by direction."""
    counts = {"buy": 0, "sell": 0}
    for order_indicator in indicator.order_indicator_his.values():
        ordered = order_indicator.get_index_data("amount").to_dict()
        dealt = order_indicator.get_index_data("deal_amount").to_dict()
        directions = order_indicator.get_index_data("trade_dir").to_dict()
        for stock, amount in ordered.items():
            deal = dealt.get(stock)
            if amount and abs(float(amount)) > 1e-12 and (deal is None or pd.isna(deal) or abs(float(deal)) <= 1e-12):
                counts["sell" if int(directions.get(stock, 1)) == 0 else "buy"] += 1
    return counts


def _uncovered(pd: Any, positions: dict[Any, Any], exchange: Any) -> list[str]:
    """Holdings with no minute bar on a day they were held: on daily fallback bars, or (suspended) on no bar at all."""
    missing = []
    for timestamp, position in sorted(positions.items()):
        start, end = _day_span(pd, timestamp)
        for stock in position.get_stock_list():
            if (stock, start) in exchange.fallback_days:
                missing.append(f"{start:%Y-%m-%d} {stock}")
            elif exchange.quote.get_data(stock, start, end, "$close", method="ts_data_last") is None:
                missing.append(f"{start:%Y-%m-%d} {stock} (no bar)")
    return missing


def _daily_book(D: Any, pd: Any, run_directory: Path, first_day: Any) -> dict[str, Any] | None:
    """The daily backtest's cash and holdings at the close of the trading day before ``first_day``, as a Qlib account
    dict with adjusted prices and holding days, or None when the run has no report files for that day."""
    positions_path = run_directory / "positions.parquet"
    portfolio_path = run_directory / "portfolio.parquet"
    if not positions_path.exists() or not portfolio_path.exists():
        return None
    portfolio = pd.read_parquet(portfolio_path)
    portfolio["datetime"] = pd.to_datetime(portfolio["datetime"])
    before = portfolio[portfolio["datetime"] < pd.Timestamp(first_day)]
    if before.empty or "cash" not in before.columns:
        return None
    row = before.iloc[-1]
    date = row["datetime"]
    positions = pd.read_parquet(positions_path, columns=["datetime", "instrument", "adjusted_amount"])
    positions["datetime"] = pd.to_datetime(positions["datetime"])
    held = positions[positions["datetime"] == date]
    account: dict[str, Any] = {"cash": float(row["cash"])}
    if len(held):
        closes = D.features(sorted(held["instrument"]), ["$close"], start_time=date, end_time=date, freq="day")
        closes = (
            closes.reset_index().set_index("instrument")["$close"] if closes is not None else pd.Series(dtype=float)
        )
        history = positions[positions["datetime"] <= date]
        dates = sorted(history["datetime"].unique())
        index = {day: number for number, day in enumerate(dates)}
        for stock, rows in history[history["instrument"].isin(held["instrument"])].groupby("instrument"):
            numbers = sorted(index[day] for day in rows["datetime"])
            count = 1
            while len(numbers) > count and numbers[-1 - count] == numbers[-1] - count:
                count += 1
            amount = float(held.loc[held["instrument"] == stock, "adjusted_amount"].iloc[0])
            price = float(closes.get(stock, float("nan")))
            if amount <= 1e-12 or price != price:
                continue
            account[stock] = {"amount": amount, "price": price, "count_day": count}
    value = float(row["account"]) if "account" in before.columns else float("nan")
    return {"account": account, "value": value, "date": f"{date:%Y-%m-%d}"}


def _with_trade_costs(pd: Any, report: Any, trades: Any, initial_value: float) -> Any:
    """Refill the daily ``cost`` / ``turnover`` rates (and totals) from the trades and make ``return`` gross again;
    the nested executor's outer report has the costs inside its (net) return and books none."""
    report = report.copy()
    index = pd.DatetimeIndex(report.index)
    if len(trades):
        daily = trades.groupby(pd.to_datetime(trades["datetime"]).dt.normalize())[["cost", "value"]].sum()
    else:
        daily = pd.DataFrame(columns=["cost", "value"], dtype=float)
    daily = daily.reindex(index.normalize()).fillna(0.0)
    previous = report["account"].shift(1).fillna(initial_value).to_numpy()
    cost = daily["cost"].to_numpy() / previous
    turnover = daily["value"].to_numpy() / previous
    report["return"] = report["return"].to_numpy() - report["cost"].to_numpy() + cost
    report["cost"] = cost
    report["turnover"] = turnover
    if "total_cost" in report.columns:
        report["total_cost"] = daily["cost"].cumsum().to_numpy()
    if "total_turnover" in report.columns:
        report["total_turnover"] = daily["value"].cumsum().to_numpy()
    return report


def _rebase_first_day(report: Any, previous_account: float, initial_cash: float) -> Any:
    """Qlib rates a backtest's first day against its initial cash; a month that starts with holdings is rated
    against the previous month's closing account value instead."""
    report = report.copy()
    first = report.index[0]
    cost = float(report.at[first, "cost"]) * initial_cash
    turnover = float(report.at[first, "turnover"]) * initial_cash
    earning = float(report.at[first, "account"]) - previous_account
    report.at[first, "return"] = (earning + cost) / previous_account
    report.at[first, "cost"] = cost / previous_account
    report.at[first, "turnover"] = turnover / previous_account
    return report


def _carry(position: Any) -> dict[str, Any]:
    account: dict[str, Any] = {"cash": float(position.get_cash())}
    for stock in position.get_stock_list():
        amount = float(position.get_stock_amount(stock))
        if abs(amount) <= 1e-12:
            continue
        account[stock] = {
            "amount": amount,
            "price": float(position.get_stock_price(stock)),
            "count_day": int(position.get_stock_count(stock, bar="day")),
        }
    return account


def _with_open_slippage(D: Any, pd: Any, trades: Any) -> Any:
    """Each fill's price against the adjusted daily open of its day, in basis points; positive is worse (a buy
    above the open or a sell below it)."""
    if trades.empty:
        trades["open"] = []
        trades["vs_open_bp"] = []
        return trades
    opens = D.features(
        sorted(trades["instrument"].unique()),
        ["$open"],
        start_time=trades["datetime"].min(),
        end_time=trades["datetime"].max(),
        freq="day",
    )
    opens.columns = ["open"]
    opens = opens.reset_index().rename(columns={"datetime": "datetime"})
    merged = trades.merge(opens, on=["instrument", "datetime"], how="left")
    sign = merged["direction"].map({"buy": 1.0, "sell": -1.0})
    merged["vs_open_bp"] = (merged["price"] / merged["open"] - 1.0) * sign * 10_000.0
    return merged


def _slippage_summary(trades: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for direction in ("buy", "sell"):
        rows = trades[trades["direction"] == direction]
        if rows.empty:
            summary[direction] = None
            continue
        weights = rows["value"]
        valid = rows["vs_open_bp"].notna()
        weighted = (rows.loc[valid, "vs_open_bp"] * weights[valid]).sum() / weights[valid].sum()
        summary[direction] = {
            "trades": len(rows),
            "mean_bp": _finite(rows["vs_open_bp"].mean()),
            "value_weighted_bp": _finite(weighted),
            "mean_fill_rate": _finite(rows["fill_rate"].mean()),
        }
    return summary


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import qlib
    except ImportError as exc:
        raise DependencyError("pyqlib==0.9.7 and pandas are required for the intraday backtest") from exc
    return qlib, pd
