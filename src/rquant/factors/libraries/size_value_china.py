"""Stock characteristics and factor loadings of Liu, Stambaugh and Yuan's "Size and Value in China" (JFE 2019).

The paper's CH-3 model explains Chinese stock returns with market, size (SMB) and EP-based value (VMG) factors built
after removing the smallest 30% of stocks, and CH-4 adds a turnover factor (PMO, pessimistic minus optimistic). This
module is an independent implementation from the published definitions with Tushare inputs. It publishes, per stock
and trading day:

* the three sorting characteristics: size (log total market value), EP and abnormal turnover;
* rolling CH-3 and CH-4 loadings, idiosyncratic volatility, fit and alpha from daily regressions of the stock's return
  on the daily factor returns in ``reference/ch4_factors.parquet`` (see ``rquant.data.ch4_factors``);
* CH-4 residual reversal (last month) and residual momentum (months t-12..t-1).

Rows are trading dates and columns are instruments. Every regression window ends on the row it is published on and
uses factor returns of that day and earlier only. Every output is missing on days the stock has no quote.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.panel_operators import replace_inf
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
MONTH = 21
HALF_YEAR = 120
YEAR = 250
TURNOVER_DAYS = 20
CH3 = ("mkt", "smb", "vmg")
CH4 = ("mkt", "smb4", "vmg", "pmo")
# Regression fits: (key, regressors, window, minimum observations, residual slices inside the window).
FITS: dict[str, tuple[tuple[str, ...], int, int, dict[str, tuple[int, int]]]] = {
    "ch3_120": (CH3, HALF_YEAR, 60, {}),
    "ch4_120": (CH4, HALF_YEAR, 60, {"resrev": (HALF_YEAR - MONTH, HALF_YEAR)}),
    "ch4_250": (CH4, YEAR, 120, {"resmom": (0, YEAR - MONTH)}),
}

# (source name, description, maximum lookback in trading days, fit key and output, or None for a characteristic).
SIZE_VALUE_CHINA: tuple[tuple[str, str, int, tuple[str, str] | None], ...] = (
    ("size", "log total market value", 1, None),
    ("ep", "trailing earnings over total market value (1 / pe_ttm); missing for loss makers", 1, None),
    ("abturn", "20-day over 250-day average daily turnover of total shares", YEAR, None),
    ("ch3_beta_mkt", "CH-3 market loading, 120 days", HALF_YEAR, ("ch3_120", "beta_mkt")),
    ("ch3_beta_smb", "CH-3 SMB loading, 120 days", HALF_YEAR, ("ch3_120", "beta_smb")),
    ("ch3_beta_vmg", "CH-3 VMG loading, 120 days", HALF_YEAR, ("ch3_120", "beta_vmg")),
    ("ch3_ivol", "CH-3 residual volatility, 120 days", HALF_YEAR, ("ch3_120", "ivol")),
    ("ch4_beta_mkt", "CH-4 market loading, 120 days", HALF_YEAR, ("ch4_120", "beta_mkt")),
    ("ch4_beta_smb", "CH-4 SMB loading, 120 days", HALF_YEAR, ("ch4_120", "beta_smb4")),
    ("ch4_beta_vmg", "CH-4 VMG loading, 120 days", HALF_YEAR, ("ch4_120", "beta_vmg")),
    ("ch4_beta_pmo", "CH-4 PMO loading, 120 days", HALF_YEAR, ("ch4_120", "beta_pmo")),
    ("ch4_ivol", "CH-4 residual volatility, 120 days", HALF_YEAR, ("ch4_120", "ivol")),
    ("ch4_r2", "CH-4 R-squared, 120 days", HALF_YEAR, ("ch4_120", "r2")),
    ("ch4_resrev", "mean over std of CH-4 residuals in the last 21 days of a 120-day fit", HALF_YEAR,
     ("ch4_120", "resrev")),
    ("ch4_beta_mkt_y", "CH-4 market loading, 250 days", YEAR, ("ch4_250", "beta_mkt")),
    ("ch4_beta_smb_y", "CH-4 SMB loading, 250 days", YEAR, ("ch4_250", "beta_smb4")),
    ("ch4_beta_vmg_y", "CH-4 VMG loading, 250 days", YEAR, ("ch4_250", "beta_vmg")),
    ("ch4_beta_pmo_y", "CH-4 PMO loading, 250 days", YEAR, ("ch4_250", "beta_pmo")),
    ("ch4_ivol_y", "CH-4 residual volatility, 250 days", YEAR, ("ch4_250", "ivol")),
    ("ch4_alpha_y", "CH-4 daily alpha (intercept), 250 days", YEAR, ("ch4_250", "alpha")),
    ("ch4_resmom", "mean over std of CH-4 residuals in days t-249..t-21 of a 250-day fit", YEAR,
     ("ch4_250", "resmom")),
)
SIZE_VALUE_CHINA_NAMES = tuple(name for name, _, _, _ in SIZE_VALUE_CHINA)
LAST_OUTPUT = {
    key: [name for name, _, _, source in SIZE_VALUE_CHINA if source and source[0] == key][-1] for key in FITS
}
DAILY_BASIC_FIELDS = ("total_mv", "pe_ttm", "turnover_total")


class SizeValueChinaLibrary(PanelFactorLibrary):
    """CH-3/CH-4 characteristics and loadings, evaluated as daily point-in-time panels."""

    family = "size_value_china"
    required_inputs = ("close",)

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"svc_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.size_value_china.SizeValueChina.{name}",
                catalog_version=1,
            )
            for ordinal, (name, description, lookback, _) in enumerate(SIZE_VALUE_CHINA, 1)
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.reference import load_reference_manifest

        root = Path(canonical_root) / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "ch4_factors.parquet" not in files:
            raise DataContractError("size_value_china needs CH-4 factor returns; run `rquant data build-reference`")
        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in (*daily_files, "ch4_factors.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        columns = ["instrument", "datetime", *DAILY_BASIC_FIELDS]
        daily = pd.concat([pd.read_parquet(root / name, columns=columns) for name in daily_files], ignore_index=True)
        daily = daily[daily["instrument"].isin(set(instruments))]
        inputs: dict[str, Any] = {}
        for field in DAILY_BASIC_FIELDS:
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments)
        factors = pd.read_parquet(root / "ch4_factors.parquet").set_index("datetime").sort_index()
        for name in sorted(set(CH3) | set(CH4)):
            inputs[f"ch4_{name}"] = factors[name].reindex(dates)
        hashes = {
            "size_value_china_reference": manifest["fingerprint"],
            "size_value_china_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "size_value_china_factors": files["ch4_factors.parquet"],
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = SizeValueChina(inputs)
        return {name: calculator.calculate(name) for name in SIZE_VALUE_CHINA_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = SizeValueChina(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)
            calculator.release(spec.source_name)


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
