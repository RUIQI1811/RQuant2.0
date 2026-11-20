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
        reports, holdings, trades, uncovered = [], [], [], []
        for number, month in enumerate(months):
            month_days = days[days.to_period("M") == month]
            start, end = month_days[0], month_days[-1]
            final = number == len(months) - 1
            exchange = MinuteAshareExchange(
                minute_root=self.minute_root,
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
            if reports:
                report = _rebase_first_day(report, float(reports[-1]["account"].iloc[-1]), carried_cash)
            reports.append(report)
            holdings.append(_holdings(pd, positions, exchange))
            trades.append(_trades(pd, indicators["1day"][1], exchange))
            uncovered.extend(_uncovered(pd, positions, exchange))
            last = positions[max(positions)]
            account = _carry(last)
            self.log(
                f"{month}: account {float(report['account'].iloc[-1]):,.0f}, "
                f"{len(exchange.codes)} instruments with minute bars"
            )

        report = pd.concat(reports).sort_index()
        report = report[~report.index.duplicated(keep="last")]
        holdings_frame = pd.concat(holdings, ignore_index=True)
        trades_frame = _with_open_slippage(D, pd, pd.concat(trades, ignore_index=True))
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
            },
            "rows": len(report),
            "trade_rows": len(trades_frame),
            "uncovered_holding_days": len(uncovered),
            "uncovered_examples": uncovered[:20],
            "execution_vs_open_bp": _slippage_summary(trades_frame),
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
    ]
    return pd.DataFrame(rows, columns=columns)


def _uncovered(pd: Any, positions: dict[Any, Any], exchange: Any) -> list[str]:
    """Holdings with no minute bar on a day they were held: priced at their last close and impossible to sell."""
    missing = []
    for timestamp, position in sorted(positions.items()):
        start, end = _day_span(pd, timestamp)
        for stock in position.get_stock_list():
            if exchange.quote.get_data(stock, start, end, "$close", method="ts_data_last") is None:
                missing.append(f"{start:%Y-%m-%d} {stock}")
    return missing


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
