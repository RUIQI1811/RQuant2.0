from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

RETRAIN_MONTHS = (1, 2, 3, 4, 6, 12)


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
    retrain_months: int = 12

    @property
    def label(self) -> str:
        """``2013`` for a calendar-year window, ``2013-04`` for one that starts in April."""
        if self.retrain_months == 12:
            return str(self.prediction_year)
        return f"{self.prediction_year}-{self.test_start.month:02d}"


def generate_rolling_windows(
    *,
    first_prediction_year: int,
    last_prediction_year: int,
    horizon: str,
    through: date,
    train_years: int = 3,
    validation_months: int = 6,
    formal_start_year: int = 2010,
    retrain_months: int = 12,
) -> list[RollingWindow]:
    """One window per ``retrain_months`` of every prediction year: the model is trained on the ``train_years`` before
    the window's first day (its last ``validation_months`` are the validation months) and predicts the window."""
    if retrain_months not in RETRAIN_MONTHS:
        raise ValueError(f"retrain_months must be one of {RETRAIN_MONTHS} (a divisor of 12), got {retrain_months}")
    if train_years < 1:
        raise ValueError("train_years must be at least 1")
    if not 1 <= validation_months < train_years * 12:
        raise ValueError("validation_months must be at least 1 and shorter than the training window")
    earliest = formal_start_year + train_years
    if first_prediction_year < earliest:
        raise ValueError(
            f"{train_years}-year rolling evaluation cannot predict before {earliest} "
            f"when formal history starts in {formal_start_year}"
        )
    if last_prediction_year < first_prediction_year:
        raise ValueError("last_prediction_year must not precede first_prediction_year")
    if horizon not in {"1d", "5d"}:
        raise ValueError(f"Unsupported horizon: {horizon}")
    purge = 1 if horizon == "1d" else 5
    windows = []
    for year in range(first_prediction_year, last_prediction_year + 1):
        for month in range(1, 13, retrain_months):
            test_start = date(year, month, 1)
            test_end = min(_months_before(test_start, -retrain_months) - timedelta(days=1), through)
            if test_end < test_start:
                continue
            validation_start = _months_before(test_start, validation_months)
            windows.append(
                RollingWindow(
                    prediction_year=year,
                    train_start=date(year - train_years, month, 1),
                    inner_train_end=validation_start - timedelta(days=1),
                    validation_start=validation_start,
                    validation_end=test_start - timedelta(days=1),
                    refit_end=test_start - timedelta(days=1),
                    test_start=test_start,
                    test_end=test_end,
                    purge_trading_days=purge,
                    retrain_months=retrain_months,
                )
            )
    return windows


def _months_before(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 - months
    return date(index // 12, index % 12 + 1, 1)


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
