"""Qlib pieces for executing a daily book inside the day on minute bars (``rquant.workflow.intraday``).

- ``MinuteAshareExchange``: ``AshareExchange`` (commission minimum, dated stamp tax) whose quotes are the monthly
  minute files of ``rquant.data.minutes`` instead of Qlib feature bins. A price asked for a span longer than one bar
  (the daily strategy sizing its orders) is the span's first bar, the price known when the day's trading starts,
  where Qlib would take the last one.
- ``WindowTWAPStrategy``: Qlib's ``TWAPStrategy`` with its own time window for buys and for sells.
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

    def __init__(self, *args: Any, minute_root: str | Path, **kwargs: Any) -> None:
        self.minute_root = Path(minute_root)
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

    def _window_index(self, direction: Any) -> tuple[int, int]:
        import pandas as pd
        from qlib.backtest.decision import Order

        start, end = self.sell_window if direction == Order.SELL else self.buy_window
        day = pd.Timestamp(self.trade_calendar.get_step_time(0)[0]).normalize()
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
            start_idx, end_idx = self._window_index(order.direction)
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
