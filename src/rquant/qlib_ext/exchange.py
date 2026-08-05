from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

STAMP_TAX_CHANGE_DATE = date(2023, 8, 28)


@dataclass(frozen=True)
class AshareCostModel:
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_before: float = 0.001
    stamp_tax_after: float = 0.0005

    def commission(self, trade_value: float) -> float:
        if trade_value <= 0:
            return 0.0
        return max(self.minimum_commission, trade_value * self.commission_rate)

    def stamp_tax(self, trade_value: float, trade_date: date, *, is_sell: bool) -> float:
        if not is_sell or trade_value <= 0:
            return 0.0
        rate = self.stamp_tax_after if trade_date >= STAMP_TAX_CHANGE_DATE else self.stamp_tax_before
        return trade_value * rate

    def total(self, trade_value: float, trade_date: date, *, is_sell: bool) -> float:
        return self.commission(trade_value) + self.stamp_tax(trade_value, trade_date, is_sell=is_sell)


try:
    from qlib.backtest.decision import Order
    from qlib.backtest.exchange import Exchange
except ImportError:
    Order = None  # type: ignore[assignment]
    Exchange = object  # type: ignore[assignment,misc]


class AshareExchange(Exchange):  # type: ignore[misc,valid-type]
    """Qlib Exchange with commission minimum and date-aware sell stamp tax."""

    def __init__(self, *args: Any, cost_model: AshareCostModel | None = None, **kwargs: Any) -> None:
        self.cost_model = cost_model or AshareCostModel()
        kwargs.setdefault("open_cost", self.cost_model.commission_rate)
        kwargs.setdefault("close_cost", self.cost_model.commission_rate + self.cost_model.stamp_tax_before)
        kwargs.setdefault("min_cost", self.cost_model.minimum_commission)
        kwargs.setdefault("trade_unit", 100)
        kwargs.setdefault("deal_price", "$open")
        kwargs.setdefault("limit_threshold", ("$limit_buy", "$limit_sell"))
        super().__init__(*args, **kwargs)

    def _calc_trade_info_by_order(self, order: Any, position: Any, dealt_order_amount: dict[Any, Any]) -> Any:
        trade_price, trade_value, _ = super()._calc_trade_info_by_order(order, position, dealt_order_amount)
        trade_date = _as_date(order.start_time)
        is_sell = bool(order.direction == Order.SELL)
        cost = self.cost_model.total(abs(float(trade_value)), trade_date, is_sell=is_sell)
        return trade_price, trade_value, cost


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return value.to_pydatetime().date()
    except AttributeError:
        return datetime.fromisoformat(str(value)).date()


def round_buy_shares(shares: float, trade_unit: int = 100) -> int:
    if shares <= 0:
        return 0
    return int(shares // trade_unit * trade_unit)


def is_open_limit_blocked(raw_open: float, limit_price: float, *, buy: bool) -> bool:
    if raw_open != raw_open or limit_price != limit_price:
        return True
    return round(raw_open, 2) >= round(limit_price, 2) if buy else round(raw_open, 2) <= round(limit_price, 2)
