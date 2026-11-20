"""Qlib pieces for executing a daily book inside the day on minute bars (``rquant.workflow.intraday``).

- ``MinuteAshareExchange``: ``AshareExchange`` (commission minimum, dated stamp tax) whose quotes are the monthly
  minute files of ``rquant.data.minutes`` instead of Qlib feature bins. A price asked for a span longer than one bar
  (the daily strategy sizing its orders) is the span's first bar, the price known when the day's trading starts,
  where Qlib would take the last one. A stock-day of ``fallback_codes`` with no minute bars gets two bars from the
  daily data instead (the daily open at 09:30, the daily close at 14:59), listed in ``fallback_days``.
- ``WindowTWAPStrategy``: Qlib's ``TWAPStrategy`` with its own time window for buys and for sells; an order on a
  fallback stock-day trades whole at 09:30, the daily open, as the daily backtest does.
- ``ZeroOrderSafeNestedExecutor``: Qlib's ``NestedExecutor`` with the trade indicator that stays defined on a day
  without orders, as ``ZeroOrderSafeSimulatorExecutor`` does for the daily executor.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError
from rquant.qlib_ext.exchange import AshareExchange
from rquant.qlib_ext.executor import ZeroOrderSafeIndicator

try:
    from qlib.backtest.executor import NestedExecutor as _NestedExecutor
    from qlib.contrib.strategy.rule_strategy import TWAPStrategy as _TWAPStrategy
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _NestedExecutor:  # type: ignore[no-redef]
        pass

    class _TWAPStrategy:  # type: ignore[no-redef]
        pass


MINUTE_DEAL_PRICE = "$vwap"
MINUTE_LIMITS = ("$limit_buy", "$limit_sell")


class MinuteAshareExchange(AshareExchange):
    """A-share minute exchange reading ``<minute_root>/month=YYYYMM/bars.parquet``."""

    def __init__(
        self, *args: Any, minute_root: str | Path, fallback_codes: list[str] | None = None, **kwargs: Any
    ) -> None:
        self.minute_root = Path(minute_root)
        self.fallback_codes = sorted(set(fallback_codes or []))
        self.fallback_days: set[tuple[str, Any]] = set()
        kwargs.setdefault("freq", "1min")
        kwargs.setdefault("deal_price", MINUTE_DEAL_PRICE)
        kwargs.setdefault("limit_threshold", MINUTE_LIMITS)
        super().__init__(*args, **kwargs)

    def get_quote_from_qlib(self) -> None:
        import pandas as pd

        from rquant.data.minutes import load_minute_quotes

        start = pd.Timestamp(self.start_time)
        end = pd.Timestamp(self.end_time)
        # A day-dated end means the whole of that day.
        if end == end.normalize():
            end = end + pd.Timedelta(hours=23, minutes=59)
        codes = None if isinstance(self.codes, str) or not len(self.codes) else list(self.codes)
        quotes = load_minute_quotes(self.minute_root, start, end, codes)
        if self.fallback_codes:
            fallback = daily_fallback_bars(self.fallback_codes, start, end, quotes)
            if len(fallback):
                days = pd.to_datetime(fallback["datetime"]).dt.normalize()
                self.fallback_days = set(zip(fallback["instrument"], days, strict=True))
                quotes = (
                    pd.concat([quotes, fallback.astype(quotes.dtypes.to_dict())], ignore_index=True)
                    if len(quotes)
                    else fallback
                )
        if quotes.empty:
            raise DataContractError(f"No minute bars between {start} and {end} under {self.minute_root}")
        self.codes = sorted(quotes["instrument"].unique())
        frame = quotes.set_index(["instrument", "datetime"]).sort_index()
        frame.columns = [f"${column}" for column in frame.columns]
        missing = [field for field in self.all_fields if field not in frame.columns]
        if missing:
            raise DataContractError(f"Minute bars lack the exchange fields {missing}")
        self.quote_df = frame[self.all_fields].astype(
            {field: "float64" for field in self.all_fields if field not in MINUTE_LIMITS}
        )
        self.trade_w_adj_price = False
        self._update_limit(self.limit_threshold)

    def get_deal_price(
        self,
        stock_id: str,
        start_time: Any,
        end_time: Any,
        direction: Any,
        method: str | None = "ts_data_last",
    ) -> Any:
        import pandas as pd

        span = pd.Timestamp(end_time) - pd.Timestamp(start_time)
        if method != "ts_data_last" or span < pd.Timedelta(minutes=1):
            return super().get_deal_price(stock_id, start_time, end_time, direction, method)
        prices = super().get_deal_price(stock_id, start_time, end_time, direction, None)
        if prices is None:
            return super().get_deal_price(stock_id, start_time, end_time, direction, method)
        valid = prices.loc[~prices.isna().data.astype(bool)]
        if len(valid) == 0:
            return super().get_deal_price(stock_id, start_time, end_time, direction, method)
        return valid.iloc[0]


def daily_fallback_bars(codes: list[str], start: Any, end: Any, minute_quotes: Any) -> Any:
    """Two minute bars per stock-day of ``codes`` that trades on the day but has no minute bars in ``minute_quotes``:
    09:30 at the adjusted daily open with the daily open-limit flags (what the daily backtest trades at), and 14:59
    at the adjusted daily close (what the day is valued at), flagged untradeable. Needs ``qlib.init`` with a daily
    provider."""
    import pandas as pd
    from qlib.data import D

    from rquant.data.minutes import MINUTE_FIELDS

    columns = ["instrument", "datetime", *MINUTE_FIELDS]
    daily = D.features(
        list(codes),
        ["$open", "$close", "$volume", "$factor", "$limit_buy", "$limit_sell"],
        start_time=pd.Timestamp(start).normalize(),
        end_time=pd.Timestamp(end).normalize(),
        freq="day",
    )
    if daily is None or daily.empty:
        return pd.DataFrame(columns=columns)
    daily.columns = ["open", "close", "volume", "factor", "limit_buy", "limit_sell"]
    daily = daily.reset_index()
    daily = daily[daily["open"].notna() & daily["close"].notna() & daily["volume"].fillna(0).gt(0)]
    if len(minute_quotes):
        covered = minute_quotes[["instrument", "datetime"]].copy()
        covered["datetime"] = pd.to_datetime(covered["datetime"]).dt.normalize()
        covered = covered.drop_duplicates()
        daily = daily.merge(covered, on=["instrument", "datetime"], how="left", indicator=True)
        daily = daily[daily["_merge"] == "left_only"].drop(columns="_merge")
    if daily.empty:
        return pd.DataFrame(columns=columns)
    day = pd.to_datetime(daily["datetime"]).dt.normalize()
    bars = []
    for label, price, buy_flag, sell_flag in (
        ("09:30", daily["open"], daily["limit_buy"].fillna(1).astype(bool), daily["limit_sell"].fillna(1).astype(bool)),
        ("14:59", daily["close"], True, True),
    ):
        hours, minutes = (int(part) for part in label.split(":"))
        bars.append(
            pd.DataFrame(
                {
                    "instrument": daily["instrument"].to_numpy(),
                    "datetime": (day + pd.Timedelta(hours=hours, minutes=minutes)).to_numpy(),
                    "open": price.to_numpy(),
                    "high": price.to_numpy(),
                    "low": price.to_numpy(),
                    "close": price.to_numpy(),
                    "vwap": price.to_numpy(),
                    "volume": (daily["volume"] / 2).to_numpy(),
                    "amount": (daily["volume"] / 2 * price).to_numpy(),
                    "factor": daily["factor"].to_numpy(),
                    "change": float("nan"),
                    "limit_buy": buy_flag if isinstance(buy_flag, bool) else buy_flag.to_numpy(),
                    "limit_sell": sell_flag if isinstance(sell_flag, bool) else sell_flag.to_numpy(),
                }
            )
        )
    return pd.concat(bars, ignore_index=True)[columns]


def parse_window(text: str) -> tuple[time, time]:
    """``"09:30-10:00"`` to its first and last minute bar labels (both inclusive)."""
    try:
        first, last = (part.strip() for part in text.split("-", 1))
        start = time.fromisoformat(first if len(first) > 4 else f"0{first}")
        end = time.fromisoformat(last if len(last) > 4 else f"0{last}")
    except ValueError as exc:
        raise DataContractError(f"A window is HH:MM-HH:MM, got {text!r}") from exc
    if end < start:
        raise DataContractError(f"Window {text!r} ends before it starts")
    if start < time(9, 30) or end > time(14, 59):
        raise DataContractError(f"Window {text!r} must lie within 09:30-14:59 (bar start labels)")
    return start, end


class WindowTWAPStrategy(_TWAPStrategy):  # type: ignore[misc,valid-type]
    """Split each outer order evenly over the minute bars of its direction's window.

    ``buy_window`` and ``sell_window`` are ``"HH:MM-HH:MM"`` bar labels, both ends included (``"09:30-09:30"`` is the
    first bar alone). As in Qlib's ``TWAPStrategy`` a slice that cannot trade (limit, suspension, cash) stays in the
    remainder, and the last bar of the window sends whatever is left.
    """

    def __init__(self, *, buy_window: str = "09:30-14:59", sell_window: str = "09:30-14:59", **kwargs: Any) -> None:
        self.buy_window = parse_window(buy_window)
        self.sell_window = parse_window(sell_window)
        super().__init__(**kwargs)

    def _window_index(self, direction: Any, stock_id: str = "") -> tuple[int, int]:
        import pandas as pd
        from qlib.backtest.decision import Order

        start, end = self.sell_window if direction == Order.SELL else self.buy_window
        day = pd.Timestamp(self.trade_calendar.get_step_time(0)[0]).normalize()
        if (stock_id, day) in getattr(self.trade_exchange, "fallback_days", ()):
            start = end = time(9, 30)
        first = day + pd.Timedelta(hours=start.hour, minutes=start.minute)
        last = day + pd.Timedelta(hours=end.hour, minutes=end.minute)
        return self.trade_calendar.get_range_idx(first, last)

    def generate_trade_decision(self, execute_result: Any = None) -> Any:
        import numpy as np
        from qlib.backtest.decision import Order, TradeDecisionWO

        if execute_result is not None:
            for order, _, _, _ in execute_result:
                self.trade_amount_remain[order.stock_id] -= order.deal_amount
        if len(self.outer_trade_decision.get_decision()) == 0:
            return TradeDecisionWO(order_list=[], strategy=self)

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        order_list = []
        for order in self.outer_trade_decision.get_decision():
            start_idx, end_idx = self._window_index(order.direction, order.stock_id)
            if trade_step < start_idx or trade_step > end_idx:
                continue
            if self.trade_exchange.check_stock_suspended(
                stock_id=order.stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                continue
            trade_len = end_idx - start_idx + 1
            rel_trade_step = trade_step - start_idx
            amount_remain = self.trade_amount_remain[order.stock_id]
            amount_expect = order.amount / trade_len * (rel_trade_step + 1)
            amount_delta = amount_expect - (order.amount - amount_remain)
            unit = self.trade_exchange.get_amount_of_trade_unit(
                stock_id=order.stock_id, start_time=order.start_time, end_time=order.end_time
            )
            if unit is None:
                target = amount_delta
            else:
                target = min(np.round(amount_delta / unit) * unit, amount_remain)
            if rel_trade_step == trade_len - 1:
                target = amount_remain
            if target > 1e-5:
                order_list.append(
                    Order(
                        stock_id=order.stock_id,
                        amount=target,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=order.direction,
                    )
                )
        return TradeDecisionWO(order_list=order_list, strategy=self)


class ZeroOrderSafeNestedExecutor(_NestedExecutor):  # type: ignore[misc,valid-type]
    def reset_common_infra(self, common_infra: Any, copy_trade_account: bool = False) -> None:
        super().reset_common_infra(common_infra, copy_trade_account)
        self.trade_account.indicator = ZeroOrderSafeIndicator()
