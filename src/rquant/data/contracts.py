from __future__ import annotations

from dataclasses import dataclass
from datetime import date

RAW_DAILY_ENDPOINTS = ("daily", "adj_factor", "daily_basic", "stk_limit", "suspend_d")

CANONICAL_BAR_COLUMNS = (
    "instrument",
    "datetime",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_pre_close",
    "raw_volume",
    "raw_amount",
    "raw_vwap",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "factor",
    "change",
    "cap",
    "sector",
    "industry",
    "subindustry",
    "suspended",
    "limit_buy",
    "limit_sell",
    "in_csi300",
)


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
