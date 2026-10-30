"""First-known annual and monthly formation dates for statement records.

These functions derive eligibility dates only. They neither calculate factors
nor select a report for a given date. Callers must independently prevent a late
annual report from substituting for the previous year's Q4 report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError


def _formation_days(dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Use only calendar months whose beginning/end is known from the input."""
    if not len(dates):
        return dates, dates
    periods = dates.to_period("M")
    starts = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1]
    # If the calendar begins midmonth, its first row might not be the actual
    # first trading day. The calendar day 1 is the sole unambiguous exception.
    if dates[0].day != 1:
        starts = starts[1:]
    monthly = dates[starts]

    june = dates[dates.month == 6]
    annual = []
    for year in june.year.unique():
        last = june[june.year == year][-1]
        month_end = pd.Timestamp(year=int(year), month=6, day=30)
        # A June cutoff before the final calendar day does not establish that
        # there is no later June trading day. A July row proves the month ended.
        if last == month_end or dates[-1] > month_end:
            annual.append(last)
    return monthly, pd.DatetimeIndex(annual)


def _next_formation(candidates: pd.DatetimeIndex, thresholds: pd.Series, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(thresholds), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(candidates) and valid.any():
        positions = np.flatnonzero(valid)
        indices = candidates.searchsorted(pd.DatetimeIndex(thresholds.iloc[positions]), side="left")
        inside = indices < len(candidates)
        result[positions[inside]] = candidates.to_numpy()[indices[inside]]
    return result


def _date_values(values: pd.Series) -> pd.Series:
    """Parse numeric Tushare YYYYMMDD dates as dates, never epoch nanoseconds."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        try:
            text = values.astype("Int64").astype("string")
            return pd.to_datetime(text, format="%Y%m%d").dt.normalize()
        except (TypeError, ValueError) as exc:
            raise DataContractError("numeric statement dates must be integer YYYYMMDD") from exc
    if pd.api.types.is_object_dtype(values.dtype) and values.map(
        lambda value: isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)
    ).any():
        raise DataContractError("mixed numeric and non-numeric statement dates are ambiguous")
    return pd.to_datetime(values).dt.normalize()


def formation_dates(fundamentals: pd.DataFrame, dates: Any) -> pd.DataFrame:
    """Copy records and append ``annual_available`` and ``monthly_available``.

    Monthly eligibility uses a quarter end plus four calendar months and a
    strictly earlier announcement, selecting the first eligible trading day of
    a month. Q4 annual eligibility starts at the next June's final trading day,
    selecting a later June if that cutoff was missed. Missing or unavailable
    dates remain NaT; records are not dropped, reordered or filled with a
    sentinel. The supplied calendar must be complete between its endpoints.

    Its first month is conservative when the first calendar day is not the
    first of the month. Its last June is conservative unless June is known to
    have ended. Use the same full trading calendar for all truncation runs.
    """
    required = {"instrument", "end_date", "ann_date"}
    if missing := required - set(fundamentals.columns):
        raise DataContractError(f"formation dates need columns: {sorted(missing)}")
    if existing := {"annual_available", "monthly_available"} & set(fundamentals.columns):
        raise DataContractError(f"formation date output columns already exist: {sorted(existing)}")
    date_series = pd.Series(dates)
    calendar = pd.DatetimeIndex(_date_values(date_series) if pd.api.types.is_numeric_dtype(date_series.dtype)
                                else pd.to_datetime(date_series))
    if calendar.tz is not None or calendar.hasnans or not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise DataContractError("formation calendar must be sorted, unique, timezone-naive trading dates")
    if not calendar.equals(calendar.normalize()):
        raise DataContractError("formation calendar must contain whole dates")

    end = _date_values(fundamentals["end_date"])
    ann = _date_values(fundamentals["ann_date"])
    valid = (end.notna() & ann.notna()).to_numpy()
    monthly_days, annual_days = _formation_days(calendar)
    monthly_threshold = pd.concat([end + pd.DateOffset(months=4), ann + pd.Timedelta(days=1)], axis=1).max(axis=1)
    first_june = (end + pd.DateOffset(months=6)).dt.to_period("M").dt.to_timestamp()
    annual_threshold = pd.concat([first_june, ann], axis=1).max(axis=1)
    is_year_end = (end.dt.month.eq(12) & end.dt.day.eq(31)).fillna(False).to_numpy()
    result = fundamentals.copy()
    result["annual_available"] = _next_formation(annual_days, annual_threshold, valid & is_year_end)
    result["monthly_available"] = _next_formation(monthly_days, monthly_threshold, valid)
    return result
