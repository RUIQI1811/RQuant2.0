"""Shared panel operators with stable semantics across factor families.

Rows are trading dates and columns are symbols. Cross-sectional operators work
across columns; time-series operators work down rows. Family-specific operators
whose published semantics differ remain in their calculator modules.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

Panel = pd.DataFrame


def window(value: float | int) -> int:
    """Floor a lookback to the nearest positive trading-day integer."""

    return max(1, int(np.floor(float(value))))


def replace_inf(value: Panel) -> Panel:
    """Replace positive and negative infinity with missing values."""

    return value.replace([np.inf, -np.inf], np.nan)


def safe_div(numerator: Panel, denominator: Panel | float) -> Panel:
    """Divide panels while treating near-zero denominators as unavailable."""

    if isinstance(denominator, pd.DataFrame):
        denominator = denominator.mask(denominator.abs() < 1e-12)
    elif abs(float(denominator)) < 1e-12:
        denominator = np.nan
    return replace_inf(numerator / denominator)


def rank(value: Panel) -> Panel:
    """Return the percentile rank of every daily stock cross-section."""

    return value.rank(axis=1, method="average", pct=True)


def delay(value: Panel, periods: float | int) -> Panel:
    return value.shift(window(periods))


def correlation(left: Panel, right: Panel, periods: float | int) -> Panel:
    """Return a stable rolling Pearson correlation.

    Pandas can emit ``+/-inf`` for a fully observed window when one side is
    constant because its rolling covariance and variance calculations do not
    cancel at exactly the same floating-point precision.  The formula engines
    treat such a window as zero correlation: it contains no co-movement
    information, but it is not a missing-input or warm-up window.
    """

    lookback = window(periods)
    result = left.rolling(lookback, min_periods=lookback).corr(right)
    left_zero_variance = left.rolling(lookback, min_periods=lookback).std(ddof=0).eq(0.0)
    right_zero_variance = right.rolling(lookback, min_periods=lookback).std(ddof=0).eq(0.0)
    result = result.mask(left_zero_variance | right_zero_variance, 0.0)
    return replace_inf(result).clip(lower=-1.0, upper=1.0)


def covariance(left: Panel, right: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return left.rolling(lookback, min_periods=lookback).cov(right)


def scale(value: Panel, target: float = 1.0) -> Panel:
    denominator = value.abs().sum(axis=1).replace(0.0, np.nan)
    return value.div(denominator, axis=0) * float(target)


def delta(value: Panel, periods: float | int) -> Panel:
    return value.diff(window(periods))


def signed_power(value: Panel, exponent: Panel | float) -> Panel:
    return np.sign(value) * np.power(value.abs(), exponent)


def decay_linear(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    weights = np.arange(1.0, lookback + 1.0)
    weights /= weights.sum()

    def weighted(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(values, weights))

    return value.rolling(lookback, min_periods=lookback).apply(weighted, raw=True)


def ts_min(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).min()


def ts_max(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).max()


def ts_argmin(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(lambda values: float(np.argmin(values) + 1), raw=True)


def ts_argmax(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(lambda values: float(np.argmax(values) + 1), raw=True)


def ts_rank(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)

    def last_rank(values: np.ndarray) -> float:
        return float(pd.Series(values).rank(method="average", pct=True).iloc[-1])

    return value.rolling(lookback, min_periods=lookback).apply(last_rank, raw=True)


def ts_sum(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).sum()


def product(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(np.prod, raw=True)


def stddev(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).std(ddof=1)


def element_min(left: Panel, right: Panel) -> Panel:
    return left.combine(right, np.minimum)


def element_max(left: Panel, right: Panel) -> Panel:
    return left.combine(right, np.maximum)


__all__ = [
    "Panel",
    "correlation",
    "covariance",
    "decay_linear",
    "delay",
    "delta",
    "element_max",
    "element_min",
    "product",
    "rank",
    "replace_inf",
    "safe_div",
    "scale",
    "signed_power",
    "stddev",
    "ts_argmax",
    "ts_argmin",
    "ts_max",
    "ts_min",
    "ts_rank",
    "ts_sum",
    "window",
]


def _window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """Sums over every trailing ``window`` rows, one result per window end row from ``window - 1`` on."""
    prefix = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
    return prefix[window:] - prefix[:-window]


def _range_sums(prefix: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    return prefix[last] - prefix[first]


def rolling_regression(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
    minimum: int,
    *,
    residual_slices: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """OLS of every stock column of ``y`` on a constant and the columns of ``x`` over trailing windows.

    Missing stock or regressor rows are dropped inside each window. Statistics are placed on the window's last row.
    The first ``x`` column is the market for co-skewness. ``residual_slices`` maps output names to positions inside
    the window whose residual mean over residual standard deviation is reported (residual momentum).

    Window statistics come from prefix sums of cross-products up to third order, so the cost does not grow with the
    window length; a window's result depends only on rows up to its last row.
    """
    dates, stocks = y.shape
    residual_slices = dict(residual_slices or {})
    names = ("beta", "slopes", "resid_std", "resid_skew", "coskew", *residual_slices)
    out = {name: np.full((dates, stocks), np.nan) for name in names}
    if dates < window:
        return out
    design = np.column_stack([np.ones(dates), x])
    design_ok = np.isfinite(design).all(axis=1)
    design = np.where(design_ok[:, None], design, 0.0)
    k = design.shape[1]
    outer = design[:, :, None] * design[:, None, :]
    cubic = outer[:, :, :, None] * design[:, None, None, :]
    identity = np.eye(k)
    ends = np.arange(window - 1, dates)
    rows = slice(window - 1, dates)
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
        sum_x = _window_sums(design * weight[:, None], window)
        gram = _window_sums(outer * weight[:, None, None], window)
        sum_y = _window_sums(value, window)
        sum_yy = _window_sums(value**2, window)
        sum_xy = _window_sums(design * value[:, None], window)
        gram[~valid] = identity
        try:
            beta = np.linalg.solve(gram, sum_xy[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = (np.linalg.pinv(gram) @ sum_xy[..., None])[..., 0]
        fitted_xy = np.einsum("nk,nk->n", beta, sum_xy)
        squared = sum_yy - 2.0 * fitted_xy + np.einsum("nk,nkl,nl->n", beta, gram, beta)
        sum_e = sum_y - np.einsum("nk,nk->n", beta, sum_x)
        sum_xyy = _window_sums(design * (value**2)[:, None], window)
        sum_xxy = _window_sums(outer * value[:, None, None], window)
        sum_xxx = _window_sums(cubic * weight[:, None, None, None], window)
        cubed = (
            _window_sums(value**3, window)
            - 3.0 * np.einsum("nk,nk->n", beta, sum_xyy)
            + 3.0 * np.einsum("nk,nkl,nl->n", beta, sum_xxy, beta)
            - np.einsum("ni,nj,nl,nijl->n", beta, beta, beta, sum_xxx)
        )
        # Residual moments against the market column: sum(e m) and sum(e m^2).
        e_m = sum_xy[:, 1] - np.einsum("nk,nk->n", beta, gram[:, :, 1])
        e_mm = sum_xxy[:, 1, 1] - np.einsum("nk,nk->n", beta, sum_xxx[:, :, 1, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            market_mean = sum_x[:, 1] / count
            centered_squares = gram[:, 1, 1] - count * market_mean**2
            co_moment = e_mm - 2.0 * market_mean * e_m + market_mean**2 * sum_e
            variance = squared / count
            dof = np.where(count > k, count - k, np.nan)
            resid_std = np.sqrt(squared / dof)
            resid_skew = (cubed / count) / variance**1.5
            coskew = (co_moment / count) / (np.sqrt(variance) * centered_squares / count)
        # Prefix-sum rounding can leave a tiny negative or zero residual sum of squares for an exact fit.
        keep = valid & (squared > 1e-14 * np.maximum(sum_yy, 1e-300))
        out["beta"][rows, column] = np.where(valid, beta[:, 1], np.nan)
        out["slopes"][rows, column] = np.where(valid, beta[:, 1:].sum(axis=1), np.nan)
        out["resid_std"][rows, column] = np.where(keep, resid_std, np.nan)
        out["resid_skew"][rows, column] = np.where(keep, resid_skew, np.nan)
        out["coskew"][rows, column] = np.where(keep, coskew, np.nan)
        if not residual_slices:
            continue
        prefix = {
            "n": np.concatenate([[0.0], np.cumsum(weight)]),
            "y": np.concatenate([[0.0], np.cumsum(value)]),
            "yy": np.concatenate([[0.0], np.cumsum(value**2)]),
            "x": np.concatenate([np.zeros((1, k)), np.cumsum(design * weight[:, None], axis=0)]),
            "xy": np.concatenate([np.zeros((1, k)), np.cumsum(design * value[:, None], axis=0)]),
            "xx": np.concatenate([np.zeros((1, k, k)), np.cumsum(outer * weight[:, None, None], axis=0)]),
        }
        for name, (first, last) in residual_slices.items():
            start = ends - window + 1 + first
            stop = ends - window + 1 + last
            part_count = _range_sums(prefix["n"], start, stop)
            part_e = _range_sums(prefix["y"], start, stop) - np.einsum(
                "nk,nk->n", beta, _range_sums(prefix["x"], start, stop)
            )
            part_ee = (
                _range_sums(prefix["yy"], start, stop)
                - 2.0 * np.einsum("nk,nk->n", beta, _range_sums(prefix["xy"], start, stop))
                + np.einsum("nk,nkl,nl->n", beta, _range_sums(prefix["xx"], start, stop), beta)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = part_e / part_count
                spread = np.sqrt((part_ee - part_count * mean**2) / (part_count - 1))
                momentum = mean / spread
            enough = valid & (part_count >= (last - first) // 2) & (spread > 0)
            out[name][rows, column] = np.where(enough, momentum, np.nan)
    return out
