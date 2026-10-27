"""First-reported financial statements on a fiscal-quarter grid, re-dated to the trading days they are known.

``QuarterGrid`` aligns a table of first-reported statements (one row per instrument and fiscal quarter, with its
announcement date) to fiscal-quarter columns, and records for every trading day which quarter is the latest one
usable. KunQuant statement graphs (``FactorLibrary.build_statements``) run over these quarters: ``statement_arrays``
lays the statement fields out as quarters x instruments, and ``gather`` publishes a quarterly result on every trading
day on which its quarter is the latest known one.

The NumPy quarter helpers below (``lag``, ``ratio``, ...) serve the pandas libraries that still compute their
statement formulas in Python (jq240, oap).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError

Panel = pd.DataFrame

# Balance-sheet items that Tushare leaves empty when the company reports none.
ZERO_IF_MISSING = (
    "trad_asset",
    "lt_eqt_invest",
    "goodwill",
    "intan_assets",
    "st_borr",
    "non_cur_liab_due_1y",
    "lt_borr",
    "bond_payable",
    "minority_int",
    "oth_eqt_tools_p_shr",
)
# Finance, depreciation and amortisation lines are blank rather than zero for many firms.
ZERO_IF_MISSING_FLOWS = ("sell_exp", "admin_exp", "fin_exp", "depr_fa_coga_dpba", "amort_intang_assets",
                         "lt_amort_deferred_exp")


class QuarterGrid:
    """First-reported fiscal-quarter values per instrument, with the trading day each quarter becomes usable.

    Arrays are ``instruments x quarters``. A quarter's usable row is the first trading date on or after its
    announcement, raised to the latest usable row of every earlier reported quarter, so a characteristic anchored on a
    quarter can read any lagged quarter without looking ahead.
    """

    def __init__(self, fundamentals: Any, dates: pd.DatetimeIndex, instruments: pd.Index) -> None:
        frame = fundamentals[fundamentals["instrument"].isin(set(instruments))].copy()
        self.dates = dates
        self.instruments = instruments
        end = pd.to_datetime(frame["end_date"])
        quarter_number = (end.dt.year * 4 + (end.dt.month - 1) // 3).to_numpy()
        self.first_quarter = int(quarter_number.min()) if len(quarter_number) else 0
        last_quarter = int(quarter_number.max()) if len(quarter_number) else 0
        self.quarters = last_quarter - self.first_quarter + 1
        rows = instruments.get_indexer(frame["instrument"])
        columns = quarter_number - self.first_quarter
        shape = (len(instruments), self.quarters)
        self.reported = np.zeros(shape, dtype=bool)
        self.reported[rows, columns] = True
        usable = np.full(shape, -1, dtype=np.int64)
        usable[rows, columns] = np.searchsorted(dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy())
        self.usable = np.maximum.accumulate(usable, axis=1)
        self._values: dict[str, np.ndarray] = {}
        for column in frame.columns:
            if column in {"instrument", "end_date", "ann_date"}:
                continue
            values = np.full(shape, np.nan)
            values[rows, columns] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if column in ZERO_IF_MISSING or column in ZERO_IF_MISSING_FLOWS:
                values = np.where(self.reported & np.isnan(values), 0.0, values)
            self._values[column] = values
        self.quarter_index = (np.arange(self.quarters) + self.first_quarter) % 4
        self.anchor = self._anchor(self.reported)
        self.annual_anchor = self._anchor(self.reported & (self.quarter_index == 3)[None, :])
        self._singles: dict[str, np.ndarray] = {}

    def _anchor(self, eligible: np.ndarray) -> np.ndarray:
        """Return the latest eligible quarter column usable on each trading date, or -1."""
        anchor = np.full((len(self.dates), len(self.instruments)), -1, dtype=np.int64)
        rows, columns = np.nonzero(eligible & (self.usable < len(self.dates)) & (self.usable >= 0))
        np.maximum.at(anchor, (self.usable[rows, columns], rows), columns)
        return np.maximum.accumulate(anchor, axis=0)

    def value(self, name: str) -> np.ndarray:
        try:
            return self._values[name]
        except KeyError as exc:
            raise DataContractError(f"Fundamental reference lacks {name}") from exc

    def single(self, name: str) -> np.ndarray:
        """Convert a year-to-date flow into single-quarter values."""
        if name not in self._singles:
            ytd = self.value(name)
            previous = lag(ytd, 1)
            self._singles[name] = np.where(self.quarter_index[None, :] == 0, ytd, ytd - previous)
        return self._singles[name]

    def ttm(self, name: str) -> np.ndarray:
        single = self.single(name)
        return single + lag(single, 1) + lag(single, 2) + lag(single, 3)

    def annual(self, name: str) -> np.ndarray:
        """Fiscal-year totals (Q4 year-to-date) at Q4 columns; flows only."""
        return np.where(self.quarter_index[None, :] == 3, self.value(name), np.nan)

    def statement_arrays(self, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
        """The statement fields as contiguous quarters x instruments arrays, plus ``fiscal_quarter`` (0 for Q1)."""
        arrays = {field: np.ascontiguousarray(self.value(field).T) for field in fields}
        quarter = self.quarter_index.astype(float)[:, None]
        shape = (self.quarters, len(self.instruments))
        arrays["fiscal_quarter"] = np.ascontiguousarray(np.broadcast_to(quarter, shape))
        return arrays

    def gather(self, values: np.ndarray, *, annual: bool = False) -> np.ndarray:
        """Publish quarters x instruments ``values`` on every trading day from that day's latest usable quarter.

        With ``annual`` the quarter is the latest usable fiscal year end (Q4).
        """
        anchor = self.annual_anchor if annual else self.anchor
        columns = np.broadcast_to(np.arange(len(self.instruments)), anchor.shape)
        result = values[np.clip(anchor, 0, None), columns] if values.size else np.full(anchor.shape, np.nan)
        return np.where(anchor >= 0, result, np.nan)

    def panel(self, values: np.ndarray, *, annual: bool = False) -> Panel:
        """``gather`` for instruments x quarters ``values``, as a dates x instruments panel."""
        return Panel(self.gather(values.T, annual=annual), index=self.dates, columns=self.instruments)


class StatementResults:
    """Statement-graph results (quarters x instruments) by output name, with the grid that dates each of them."""

    def __init__(self) -> None:
        self._results: dict[str, tuple[QuarterGrid, np.ndarray]] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._results

    def add(self, name: str, grid: QuarterGrid, values: np.ndarray) -> None:
        self._results[name] = (grid, values)

    def daily(self, name: str) -> np.ndarray:
        """The result on every trading day: ``q_`` results from the latest known quarter, ``y_`` from the latest
        fiscal year."""
        grid, values = self._results[name]
        return grid.gather(values, annual=name.startswith("y_"))


def lag(values: np.ndarray, periods: int) -> np.ndarray:
    """Shift quarter columns right so column q holds quarter q - periods."""
    result = np.full(values.shape, np.nan)
    if periods < values.shape[1]:
        result[:, periods:] = values[:, : values.shape[1] - periods]
    return result


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    return np.where(np.isfinite(result), result, np.nan)


def growth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Relative change, defined only for a positive base."""
    return ratio(current - previous, np.where(previous > 0, previous, np.nan))


def nan_mean(stacked: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the finite values along ``axis`` (missing when none) and their count."""
    finite = np.isfinite(stacked)
    count = finite.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, stacked, 0.0).sum(axis=axis) / count
    return np.where(count > 0, mean, np.nan), count


def rolling_quarter_std(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    """Sample standard deviation over the current and ``window - 1`` previous quarters."""
    stacked = np.stack([lag(values, k) for k in range(window)], axis=2)
    mean, count = nan_mean(stacked, 2)
    squares = np.where(np.isfinite(stacked), (stacked - mean[..., None]) ** 2, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.sqrt(squares / (count - 1))
    return np.where(count >= max(minimum, 2), result, np.nan)

