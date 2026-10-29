"""Publish KunQuant statement results at annual or monthly formation anchors.

The statement matrix and quarter calculations remain those of QuarterGrid.
Only the output anchor is replaced; dates are selected from point-in-time
availability of the exact report and the required preceding report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.formation_dates import _date_values, _formation_days
from rquant.factors.statements import QuarterGrid


def _calendar(values: Any, name: str) -> pd.DatetimeIndex:
    series = pd.Series(values)
    index = pd.DatetimeIndex(_date_values(series) if pd.api.types.is_numeric_dtype(series.dtype)
                            else pd.to_datetime(series))
    if index.tz is not None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise DataContractError(f"{name} must be sorted, unique, timezone-naive trading dates")
    if not index.equals(index.normalize()):
        raise DataContractError(f"{name} must contain whole dates")
    return index


class FormationGrid(QuarterGrid):
    """QuarterGrid with frozen formation anchors and no unrelated-report delays.

    ``full_calendar`` is the complete external trading calendar, even when
    ``dates`` is a prefix or subrange. Both must be sorted, unique dates and
    every output date must be in that calendar. Existing statement values and
    quarterly arithmetic are not modified. ``annual_previous`` and
    ``monthly_previous`` require the exact preceding report to be announced;
    missing statement fields do not cause a fallback to an older anchor.
    """

    def __init__(self, fundamentals: pd.DataFrame, dates: Any, instruments: Any, *,
                 full_calendar: Any, mode: str) -> None:
        if mode not in {"annual_current", "annual_previous", "monthly_previous"}:
            raise DataContractError(f"unsupported formation mode: {mode}")
        required = {"instrument", "end_date", "ann_date"}
        if missing := required - set(fundamentals.columns):
            raise DataContractError(f"formation grid needs columns: {sorted(missing)}")
        dates = _calendar(dates, "output calendar")
        full_calendar = _calendar(full_calendar, "full calendar")
        instruments = pd.Index(instruments)
        if instruments.has_duplicates or instruments.hasnans:
            raise DataContractError("formation instruments must be unique and non-null")
        if (full_calendar.get_indexer(dates) < 0).any():
            raise DataContractError("output dates must belong to the full calendar")
        frame = fundamentals[fundamentals["instrument"].isin(instruments)].copy()
        frame["end_date"] = _date_values(frame["end_date"])
        frame["ann_date"] = _date_values(frame["ann_date"])
        frame = frame[frame["end_date"].notna()].copy()
        if not frame["end_date"].dt.is_quarter_end.all():
            raise DataContractError("formation report dates must be calendar quarter ends")
        if frame.duplicated(["instrument", "end_date"]).any():
            raise DataContractError("formation records must contain at most one first report per stock and quarter")
        super().__init__(frame, dates, instruments)

        monthly_days, annual_days = _formation_days(full_calendar)
        events = monthly_days if mode == "monthly_previous" else annual_days
        if len(dates):
            events = events[events <= dates[-1]]
        else:
            events = events[:0]
        selected = self._formation_anchor(frame, instruments, events, mode)
        output = np.full((len(dates), len(instruments)), -1, dtype=np.int64)
        if len(events) and len(dates):
            event_rows = events.searchsorted(dates, side="right") - 1
            valid = event_rows >= 0
            output[valid, :] = selected[event_rows[valid], :]
        # Both aliases intentionally obey the selected mode, even if callers
        # use q_/y_ output naming differently for this private graph.
        output.flags.writeable = False
        self.anchor = output
        self.annual_anchor = output

    def _formation_anchor(self, frame: pd.DataFrame, instruments: pd.Index,
                          events: pd.DatetimeIndex, mode: str) -> np.ndarray:
        result = np.full((len(events), len(instruments)), -1, dtype=np.int64)
        if not len(frame) or not len(events) or not len(instruments):
            return result
        ends = frame["end_date"]
        numbers = (ends.dt.year * 4 + (ends.dt.month - 1) // 3).to_numpy(dtype=np.int64)
        stocks = instruments.get_indexer(frame["instrument"])
        columns = numbers - self.first_quarter
        announcements = np.full((len(instruments), self.quarters), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        announcements[stocks, columns] = frame["ann_date"].to_numpy(dtype="datetime64[ns]")
        event_values = events.to_numpy(dtype="datetime64[ns]")

        if mode != "monthly_previous":
            current = (events.year.to_numpy(dtype=np.int64) - 1) * 4 + 3 - self.first_quarter
            for event_row, column in enumerate(current):
                if not 0 <= column < self.quarters:
                    continue
                eligible = self.reported[:, column] & (announcements[:, column] <= event_values[event_row])
                if mode == "annual_previous":
                    if column < 4:
                        continue
                    eligible &= self.reported[:, column - 4] & (announcements[:, column - 4] <= event_values[event_row])
                result[event_row, eligible] = column
            return result

        # Per stock, place each eligible quarter into its first eligible month,
        # then keep the latest chronological quarter until a newer one qualifies.
        # The preceding report is required for the lagged asset and, when
        # applicable, for differencing cumulative quarterly flows.
        threshold_end = (ends + pd.DateOffset(months=4)).to_numpy(dtype="datetime64[ns]")
        threshold_by_quarter = np.full(self.quarters, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        threshold_by_quarter[columns] = threshold_end
        day = np.timedelta64(1, "D")
        for stock in range(len(instruments)):
            current_columns = np.flatnonzero(self.reported[stock])
            current_columns = current_columns[current_columns > 0]
            if not len(current_columns):
                continue
            previous = current_columns - 1
            current_ann = announcements[stock, current_columns]
            previous_ann = announcements[stock, previous]
            eligible = self.reported[stock, previous] & ~np.isnat(current_ann) & ~np.isnat(previous_ann)
            if not eligible.any():
                continue
            current_columns = current_columns[eligible]
            threshold = np.maximum(threshold_by_quarter[current_columns], current_ann[eligible] + day)
            threshold = np.maximum(threshold, previous_ann[eligible] + day)
            first = events.searchsorted(pd.DatetimeIndex(threshold), side="left")
            inside = first < len(events)
            if inside.any():
                np.maximum.at(result[:, stock], first[inside], current_columns[inside])
                result[:, stock] = np.maximum.accumulate(result[:, stock])
        return result
