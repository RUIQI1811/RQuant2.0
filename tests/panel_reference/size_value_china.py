"""Pandas reference implementation of the size_value_china factor set.

RQuant computes size_value_china with KunQuant (``rquant.factors.libraries.size_value_china``). This calculator, the
original implementation, is kept for the parity tests that check the KunQuant graph against it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.size_value_china import (
    FITS,
    SIZE_VALUE_CHINA,
    SIZE_VALUE_CHINA_NAMES,
    TURNOVER_DAYS,
    YEAR,
)
from rquant.factors.panel_operators import replace_inf

Panel = pd.DataFrame
LAST_OUTPUT = {
    key: [name for name, _, _, source in SIZE_VALUE_CHINA if source and source[0] == key][-1] for key in FITS
}


def _window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """Sums over every trailing ``window`` rows, one result per window end row from ``window - 1`` on."""
    prefix = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
    return prefix[window:] - prefix[:-window]


def rolling_ols(
    y: np.ndarray,
    x: np.ndarray,
    names: tuple[str, ...],
    window: int,
    minimum: int,
    residual_slices: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """OLS of every stock column of ``y`` on a constant and the columns of ``x`` over trailing windows.

    Rows where the stock or any regressor is missing are dropped inside each window, and statistics are placed on the
    window's last row. Outputs are ``alpha``, ``beta_<name>`` per regressor, ``ivol`` (residual standard deviation),
    ``r2`` and, for every ``residual_slices`` entry ``(first, last)`` of window positions, the mean over standard
    deviation of the full-window fit's residuals in that part of the window.
    """
    dates, stocks = y.shape
    residual_slices = dict(residual_slices or {})
    outputs = ("alpha", *(f"beta_{name}" for name in names), "ivol", "r2", *residual_slices)
    out = {name: np.full((dates, stocks), np.nan) for name in outputs}
    if dates < window:
        return out
    design = np.column_stack([np.ones(dates), x])
    design_ok = np.isfinite(design).all(axis=1)
    design = np.where(design_ok[:, None], design, 0.0)
    k = design.shape[1]
    outer = design[:, :, None] * design[:, None, :]
    identity = np.eye(k)
    rows = slice(window - 1, dates)
    ends = np.arange(window - 1, dates)
    for column in range(stocks):
        series = y[:, column]
        mask = np.isfinite(series) & design_ok
        if mask.sum() < minimum:
            continue
        weight = mask.astype(float)
        value = np.where(mask, series, 0.0)
        count = _window_sums(weight, window)
        valid = count >= minimum
        if not valid.any():
            continue
        gram = _window_sums(outer * weight[:, None, None], window)
        sum_xy = _window_sums(design * value[:, None], window)
        sum_y = _window_sums(value, window)
        sum_yy = _window_sums(value**2, window)
        gram[~valid] = identity
        try:
            beta = np.linalg.solve(gram, sum_xy[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = (np.linalg.pinv(gram) @ sum_xy[..., None])[..., 0]
        squared = sum_yy - 2.0 * np.einsum("nk,nk->n", beta, sum_xy) + np.einsum("nk,nkl,nl->n", beta, gram, beta)
        with np.errstate(invalid="ignore", divide="ignore"):
            total = sum_yy - sum_y**2 / count
            dof = np.where(count > k, count - k, np.nan)
            ivol = np.sqrt(squared / dof)
            r2 = 1.0 - squared / total
        # Prefix-sum rounding can leave a tiny negative or zero residual sum of squares for an exact fit.
        keep = valid & (squared > 1e-14 * np.maximum(sum_yy, 1e-300)) & (total > 0)
        out["alpha"][rows, column] = np.where(valid, beta[:, 0], np.nan)
        for position, name in enumerate(names, 1):
            out[f"beta_{name}"][rows, column] = np.where(valid, beta[:, position], np.nan)
        out["ivol"][rows, column] = np.where(keep, ivol, np.nan)
        out["r2"][rows, column] = np.where(keep, r2, np.nan)
        if not residual_slices:
            continue
        prefix_n = np.concatenate([[0.0], np.cumsum(weight)])
        prefix_y = np.concatenate([[0.0], np.cumsum(value)])
        prefix_yy = np.concatenate([[0.0], np.cumsum(value**2)])
        prefix_x = np.concatenate([np.zeros((1, k)), np.cumsum(design * weight[:, None], axis=0)])
        prefix_xy = np.concatenate([np.zeros((1, k)), np.cumsum(design * value[:, None], axis=0)])
        prefix_xx = np.concatenate([np.zeros((1, k, k)), np.cumsum(outer * weight[:, None, None], axis=0)])
        for name, (first, last) in residual_slices.items():
            start = ends - window + 1 + first
            stop = ends - window + 1 + last
            part_n = prefix_n[stop] - prefix_n[start]
            part_e = prefix_y[stop] - prefix_y[start] - np.einsum("nk,nk->n", beta, prefix_x[stop] - prefix_x[start])
            part_ee = (
                prefix_yy[stop]
                - prefix_yy[start]
                - 2.0 * np.einsum("nk,nk->n", beta, prefix_xy[stop] - prefix_xy[start])
                + np.einsum("nk,nkl,nl->n", beta, prefix_xx[stop] - prefix_xx[start], beta)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = part_e / part_n
                spread = np.sqrt((part_ee - part_n * mean**2) / (part_n - 1))
                score = mean / spread
            enough = valid & (part_n >= (last - first) // 2) & (spread > 0)
            out[name][rows, column] = np.where(enough, score, np.nan)
    return out


class SizeValueChina:
    """Calculator with shared regression fits; ``calculate`` returns one output panel."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna()
        self.returns = close / close.shift(1) - 1.0
        self._fits: dict[str, dict[str, np.ndarray]] = {}

    def calculate(self, name: str) -> Panel:
        _, _, _, source = SIZE_VALUE_CHINA[SIZE_VALUE_CHINA_NAMES.index(name)]
        if source is None:
            method: Callable[[], Panel] = getattr(self, f"_char_{name}")
            result = method()
        else:
            fit, output = source
            result = Panel(self._fit(fit)[output], index=self.dates, columns=self.instruments)
        result = replace_inf(result.astype(float))
        return result.where(self.trading)

    def release(self, name: str) -> None:
        """Drop a regression fit once its last output in catalog order is published."""
        for key in FITS:
            if name == LAST_OUTPUT[key]:
                self._fits.pop(key, None)

    def _panel(self, key: str) -> Panel:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"size_value_china input is missing: {key}")
        return value

    def _fit(self, key: str) -> dict[str, np.ndarray]:
        if key not in self._fits:
            regressors, window, minimum, slices = FITS[key]
            x = np.column_stack(
                [np.asarray(pd.Series(self._panel(f"ch4_{name}")).reindex(self.dates), dtype=float)
                 for name in regressors]
            )
            self._fits[key] = rolling_ols(
                self.returns.to_numpy(dtype=float), x, regressors, window, minimum, slices
            )
        return self._fits[key]

    def _char_size(self) -> Panel:
        value = self._panel("basic_total_mv")
        return np.log(value.where(value > 0))

    def _char_ep(self) -> Panel:
        pe = self._panel("basic_pe_ttm")
        return 1.0 / pe.where(pe > 0)

    def _char_abturn(self) -> Panel:
        turnover = self._panel("basic_turnover_total").where(self.trading)
        recent = turnover.rolling(TURNOVER_DAYS, min_periods=15).mean()
        year = turnover.rolling(YEAR, min_periods=120).mean()
        return recent / year.where(year > 0)
