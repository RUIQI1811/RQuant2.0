from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class RollingWindow:
    prediction_year: int
    train_start: date
    inner_train_end: date
    validation_start: date
    validation_end: date
    refit_end: date
    test_start: date
    test_end: date
    purge_trading_days: int


def generate_rolling_windows(
    *,
    first_prediction_year: int,
    last_prediction_year: int,
    horizon: str,
    through: date,
) -> list[RollingWindow]:
    if first_prediction_year < 2013:
        raise ValueError("Three-year rolling evaluation cannot predict before 2013 when formal history starts in 2010")
    if last_prediction_year < first_prediction_year:
        raise ValueError("last_prediction_year must not precede first_prediction_year")
    if horizon not in {"1d", "5d"}:
        raise ValueError(f"Unsupported horizon: {horizon}")
    purge = 1 if horizon == "1d" else 5
    windows = []
    for year in range(first_prediction_year, last_prediction_year + 1):
        test_end = min(date(year, 12, 31), through)
        if test_end < date(year, 1, 1):
            continue
        validation_start = date(year - 1, 7, 1)
        windows.append(
            RollingWindow(
                prediction_year=year,
                train_start=date(year - 3, 1, 1),
                inner_train_end=validation_start - timedelta(days=1),
                validation_start=validation_start,
                validation_end=date(year - 1, 12, 31),
                refit_end=date(year - 1, 12, 31),
                test_start=date(year, 1, 1),
                test_end=test_end,
                purge_trading_days=purge,
            )
        )
    return windows


def is_five_day_anchor(calendar_position: int) -> bool:
    if calendar_position < 0:
        raise ValueError("calendar_position cannot be negative")
    return calendar_position % 5 == 0


def previous_trading_day(calendar: list[date], boundary: date, steps: int) -> date:
    """Return the last usable label date before boundary after a trading-day purge."""
    eligible = [value for value in calendar if value < boundary]
    position = len(eligible) - 1 - steps
    if position < 0:
        raise ValueError(f"Not enough calendar history to purge {steps} trading days before {boundary}")
    return eligible[position]
