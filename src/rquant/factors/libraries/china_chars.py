"""China A-share equity characteristics evaluated as daily point-in-time panels.

The catalog follows the characteristic definitions published by Quantactix/ChinaAShareEquityCharacteristics (monthly,
CSMAR-based, after Hou, Xue and Zhang's anomaly list). This module is an independent re-implementation: no upstream
code is used, inputs come from Tushare, and every characteristic is re-dated to the trading day on which it is known.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days, a quarter 63 and a year 252.
* Market return, SMB and HML come from ``reference/market_factors.parquet`` (value-weighted over all A shares). Daily
  returns are not reduced by a risk-free rate; over the 120-day windows used here that only moves the intercept.
* Financial statements are first-reported values. A fiscal quarter becomes usable on the first trading day on or after
  its announcement, and only once every earlier reported quarter is also out, so lagged quarters are never newer
  than the quarter that anchors them. Income and cash-flow items are converted from year-to-date to single quarters.
* Flow-to-price ratios (ep, cfp, ocfp, em) use trailing-twelve-month flows against the current total market value.
* Every output is missing on days the stock has no quote.

Upstream characteristics that cannot be built from Tushare data at this account level are not in the catalog:
employee counts (hn, lfe), advertising expense (adm, gad), CH-4 factor loadings (mkt_beta, vmg_beta, smb_beta,
pmo_beta), q-factor idiosyncratic moments (idvq, idsq) and the unfinished tail-risk measure (tail).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.panel_operators import replace_inf, safe_div
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
MONTH = 21
QUARTER = 63
HALF_YEAR = 120
YEAR = 252

# (source name, description, maximum lookback in trading days). Order is the stable public column order.
CHINA_CHARS: tuple[tuple[str, str, int], ...] = (
    # Liquidity
    ("size", "log float market value", 1),
    ("size3", "cube of log float market value", 1),
    ("turnm", "log one-month float-share turnover", MONTH),
    ("turnq", "log average monthly turnover over one quarter", QUARTER),
    ("turna", "log average monthly turnover over one year", YEAR),
    ("vturn", "std of daily turnover, 120 days", HALF_YEAR),
    ("cvturn", "coefficient of variation of daily turnover, 120 days", HALF_YEAR),
    ("abturn", "one-month over one-year average daily turnover", 250),
    ("dtvm", "log one-month traded value", MONTH),
    ("dtvq", "log average monthly traded value over one quarter", QUARTER),
    ("dtva", "log average monthly traded value over one year", YEAR),
    ("vdtv", "std of daily traded value, 120 days", HALF_YEAR),
    ("cvd", "coefficient of variation of daily traded value, 120 days", HALF_YEAR),
    ("ami", "Amihud illiquidity |r|/value, one month", MONTH),
    # Risk
    ("idvc", "CAPM residual volatility, 120 days", HALF_YEAR),
    ("idvff", "Fama-French three-factor residual volatility, 120 days", HALF_YEAR),
    ("tv", "total return volatility, 120 days", HALF_YEAR),
    ("idsff", "Fama-French three-factor residual skewness, 120 days", HALF_YEAR),
    ("idsc", "CAPM residual skewness, 120 days", HALF_YEAR),
    ("ts", "total return skewness, 120 days", HALF_YEAR),
    ("cs", "Harvey-Siddique co-skewness, 120 days", HALF_YEAR),
    ("dbeta", "downside market beta, 240 days", 240),
    ("betafp", "Frazzini-Pedersen beta, 120 days", HALF_YEAR + 2),
    ("betadm", "Dimson beta with lead and lag market returns, 120 days", HALF_YEAR + 2),
    ("beta", "CAPM market beta, 120 days", HALF_YEAR),
    # Past returns
    ("m1", "one-month return", MONTH),
    ("m11", "return from month t-12 to t-1", YEAR),
    ("m60", "return from month t-60 to t-12", 5 * YEAR),
    ("m6", "return from month t-6 to t-1", 6 * MONTH),
    ("m3", "return from month t-3 to t-1", 3 * MONTH),
    ("indmom", "value-weighted six-month return of the SW level-1 industry", 6 * MONTH),
    ("m24", "return from month t-36 to t-12", 3 * YEAR),
    ("mchg", "momentum change: months t-6..t-1 minus t-12..t-7", YEAR),
    ("im12", "FF3 residual momentum, months t-12..t-1 of a 720-day fit", 720),
    ("im6", "FF3 residual momentum, months t-6..t-1 of a 720-day fit", 720),
    ("w52", "price over its 240-day high", 240),
    ("mdr", "average of the five largest daily returns, one month", MONTH),
    ("pr", "unadjusted share price", 1),
    ("abr", "abnormal return from day -2 to +1 around the latest earnings announcement", 4),
    ("season", "average same-calendar-month return over the prior five years", 5 * YEAR),
    # Profitability
    ("roe", "quarterly net income to parent over lagged book equity", 1),
    ("droe", "four-quarter change in roe", 1),
    ("roa", "quarterly net income over lagged total assets", 1),
    ("droa", "four-quarter change in roa", 1),
    ("rna", "quarterly operating profit over lagged net operating assets", 1),
    ("pm", "quarterly operating profit over revenue", 1),
    ("ato", "quarterly revenue over lagged net operating assets", 1),
    ("ct", "quarterly revenue over lagged total assets", 1),
    ("gpa", "annual gross profit over total assets", 1),
    ("gpla", "quarterly gross profit over lagged total assets", 1),
    ("ope", "annual operating profit (revenue - costs - SG&A - finance) over book equity", 1),
    ("ople", "quarterly operating profit over lagged book equity", 1),
    ("opa", "annual operating profit over total assets", 1),
    ("opla", "quarterly operating profit over lagged total assets", 1),
    ("tbi", "quarterly pretax income over net income", 1),
    ("bl", "total assets over book equity", 1),
    ("sg", "annual revenue growth", 1),
    ("sgq", "quarterly revenue growth over four quarters", 1),
    ("fscore", "Piotroski F-score from quarterly statements", 1),
    ("oscore", "Ohlson O-score from quarterly and trailing statements", 1),
    # Value
    ("bm", "book equity over total market value", 1),
    ("dm", "total liabilities over total market value", 1),
    ("am", "total assets over total market value", 1),
    ("ep", "trailing net income to parent over total market value", 1),
    ("cfp", "trailing net income plus depreciation and amortisation over total market value", 1),
    ("sr", "weighted five-year annual revenue growth", 1),
    ("em", "enterprise value over trailing EBITDA", 1),
    ("sp", "trailing revenue over total market value", 1),
    ("ocfp", "trailing operating cash flow over total market value", 1),
    ("de", "annual total liabilities over book equity", 1),
    ("ebp", "enterprise book-to-price", 1),
    ("ndp", "net debt over total market value", 1),
    # Investment
    ("ag", "four-quarter total asset growth", 1),
    ("dpia", "change in fixed assets plus inventory over lagged assets", 1),
    ("noa", "net operating assets over lagged total assets", 1),
    ("dnoa", "change in net operating assets over lagged total assets", 1),
    ("ig", "annual capital expenditure growth", 1),
    ("cei", "one-year composite equity issuance", YEAR),
    ("cdi", "five-year log growth of total liabilities", 1),
    ("ivg", "annual inventory growth", 1),
    ("ivchg", "annual inventory change over average total assets", 1),
    ("oacc", "annual net income minus operating cash flow over lagged assets", 1),
    ("tacc", "annual total accruals over lagged assets", 1),
    ("dwc", "change in net non-cash working capital over lagged assets", 1),
    ("dcoa", "change in current operating assets over lagged assets", 1),
    ("dcol", "change in current operating liabilities over lagged assets", 1),
    ("dnco", "change in net non-current operating assets over lagged assets", 1),
    ("dnca", "change in non-current operating assets over lagged assets", 1),
    ("dncl", "change in non-current operating liabilities over lagged assets", 1),
    ("dfin", "change in net financial assets over lagged assets", 1),
    ("dbe", "change in book equity over lagged assets", 1),
    # Other
    ("rdm", "quarterly R&D expense over total market value", 1),
    ("rds", "quarterly R&D expense over revenue", 1),
    ("ol", "annual operating costs over total assets", 1),
    ("age", "months since listing", 1),
    ("dsi", "% change in revenue minus % change in inventory", 1),
    ("dsa", "% change in revenue minus % change in receivables", 1),
    ("dgs", "% change in gross profit minus % change in revenue", 1),
    ("dss", "% change in revenue minus % change in SG&A", 1),
    ("etr", "effective tax rate change times earnings change over market value", 1),
    ("tan", "tangibility: weighted liquid and fixed assets over total assets", 1),
    ("vcf", "volatility of quarterly operating cash flow over revenue, 16 quarters", 1),
    ("cta", "cash over total assets", 1),
    ("esm", "earnings smoothness, 16 quarters", 1),
    ("ala", "asset liquidity over lagged total assets", 1),
    ("alm", "asset liquidity over market value of assets", 1),
    ("sue", "standardized unexpected quarterly earnings", 1),
    ("rs", "standardized unexpected quarterly revenue", 1),
    ("tes", "four-quarter change in tax expense over lagged assets", 1),
)
CHINA_CHARS_NAMES = tuple(name for name, _, _ in CHINA_CHARS)
CHINA_CHARS_DAILY_BASIC = {"turnover_rate": 0.01, "circ_mv": 10_000.0, "close": 1.0}
MARKET_FACTORS = ("mkt", "smb", "hml")
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
YTD_FLOWS = (
    "revenue",
    "oper_cost",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "rd_exp",
    "operate_profit",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "n_cashflow_act",
    "c_pay_acq_const_fiolta",
    "n_incr_cash_cash_equ",
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
)
# Finance, depreciation and amortisation lines are blank rather than zero for many firms.
ZERO_IF_MISSING_FLOWS = ("sell_exp", "admin_exp", "fin_exp", "depr_fa_coga_dpba", "amort_intang_assets",
                         "lt_amort_deferred_exp")


class ChinaCharsLibrary(PanelFactorLibrary):
    """China A-share equity characteristics, re-dated to daily point-in-time panels."""

    family = "china_chars"
    required_inputs = ("close", "amount", "cap", "sector")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"cnc_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.china_chars.ChinaChars.{name}",
                catalog_version=2,
            )
            for ordinal, (name, description, lookback) in enumerate(CHINA_CHARS, 1)
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.canonical import ts_code_to_qlib
        from rquant.data.reference import load_reference_manifest

        root = Path(canonical_root) / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "china_chars needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        for name, expected in files.items():
            if sha256_file(root / name) != expected:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        columns = ["instrument", "datetime", *CHINA_CHARS_DAILY_BASIC]
        daily = pd.concat([pd.read_parquet(root / name, columns=columns) for name in daily_files], ignore_index=True)
        daily = daily[daily["instrument"].isin(set(instruments))]
        inputs: dict[str, Any] = {}
        for field, scale in CHINA_CHARS_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale

        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        for name in MARKET_FACTORS:
            inputs[name] = factors[name].reindex(dates)

        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        basic = pd.read_parquet(root / "stock_basic.parquet", columns=["ts_code", "list_date"])
        stock_basic_hash = sha256_file(root / "stock_basic.parquet")
        basic = basic.assign(instrument=basic["ts_code"].map(ts_code_to_qlib)).drop_duplicates("instrument")
        listed = pd.to_datetime(basic.set_index("instrument")["list_date"], format="%Y%m%d", errors="coerce")
        inputs["list_date"] = listed.reindex(instruments)
        hashes = {
            "china_chars_reference": manifest["fingerprint"],
            "china_chars_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "china_chars_market_factors": files["market_factors.parquet"],
            "china_chars_fundamentals": files["fundamentals.parquet"],
            "china_chars_stock_basic": stock_basic_hash,
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = ChinaChars(inputs)
        return {name: calculator.calculate(name) for name in CHINA_CHARS_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = ChinaChars(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)
            calculator.release(spec.source_name)


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

    def panel(self, values: np.ndarray, *, annual: bool = False) -> Panel:
        anchor = self.annual_anchor if annual else self.anchor
        columns = np.broadcast_to(np.arange(len(self.instruments)), anchor.shape)
        result = values[columns, np.clip(anchor, 0, None)] if values.size else np.full(anchor.shape, np.nan)
        result = np.where(anchor >= 0, result, np.nan)
        return Panel(result, index=self.dates, columns=self.instruments)


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


class ChinaChars:
    """Calculator with shared intermediates; ``calculate`` returns one characteristic panel."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna()
        self.price = close.ffill()
        self.returns = close / close.shift(1) - 1.0
        self._cache: dict[str, Any] = {}
        self._grid: QuarterGrid | None = None

    # ------------------------------------------------------------------ infrastructure
    def calculate(self, name: str) -> Panel:
        method: Callable[[], Panel] = getattr(self, f"_char_{name}")
        result = method()
        if isinstance(result, np.ndarray):
            result = Panel(result, index=self.dates, columns=self.instruments)
        result = replace_inf(result.astype(float))
        return result.where(self.trading)

    def release(self, name: str) -> None:
        """Drop intermediates no later characteristic needs (outputs are computed in catalog order)."""
        position = CHINA_CHARS_NAMES.index(name)
        if position == CHINA_CHARS_NAMES.index("beta"):
            for key in ("capm", "ff3", "dimson"):
                self._cache.pop(key, None)
        if position == CHINA_CHARS_NAMES.index("im6"):
            self._cache.pop("ff3_momentum", None)

    def _panel(self, key: str) -> Panel:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"china_chars input is missing: {key}")
        return value

    def _series(self, key: str) -> np.ndarray:
        return np.asarray(pd.Series(self._panel(key)).reindex(self.dates), dtype=float)

    @property
    def grid(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(self._panel("fundamentals"), self.dates, self.instruments)
        return self._grid

    def _turnover(self) -> Panel:
        return self._panel("basic_turnover_rate")

    def _float_value(self) -> Panel:
        return self._panel("basic_circ_mv")

    def _cap(self) -> Panel:
        return self._panel("cap")

    def _rolling(self, panel: Panel, window: int, minimum: int) -> Any:
        return panel.rolling(window, min_periods=minimum)

    def _regression(self, key: str, regressors: tuple[str, ...], window: int, minimum: int, **kwargs: Any) -> dict:
        if key not in self._cache:
            x = np.column_stack([self._series(name) for name in regressors])
            self._cache[key] = rolling_regression(
                self.returns.to_numpy(dtype=float), x, window, minimum, **kwargs
            )
        return self._cache[key]

    def _capm(self) -> dict:
        return self._regression("capm", ("mkt",), HALF_YEAR, 40)

    def _ff3(self) -> dict:
        return self._regression("ff3", MARKET_FACTORS, HALF_YEAR, 40)

    def _momentum(self, start: int, end: int) -> Panel:
        """Return from ``start`` trading days ago to ``end`` trading days ago (start > end)."""
        return self.price.shift(end) / self.price.shift(start) - 1.0

    def _market_panel(self, values: np.ndarray) -> Panel:
        return Panel(np.repeat(values[:, None], len(self.instruments), axis=1), index=self.dates,
                     columns=self.instruments)

    # ------------------------------------------------------------------ liquidity
    def _char_size(self) -> Panel:
        return np.log(self._float_value().where(self._float_value() > 0))

    def _char_size3(self) -> Panel:
        return self._char_size() ** 3

    def _monthly_turnover(self, window: int, minimum: int) -> Panel:
        mean = self._rolling(self._turnover(), window, minimum).mean()
        return np.log((mean * MONTH).where(mean > 0))

    def _char_turnm(self) -> Panel:
        return self._monthly_turnover(MONTH, 15)

    def _char_turnq(self) -> Panel:
        return self._monthly_turnover(QUARTER, 40)

    def _char_turna(self) -> Panel:
        return self._monthly_turnover(YEAR, 120)

    def _char_vturn(self) -> Panel:
        return self._rolling(self._turnover(), HALF_YEAR, 40).std()

    def _char_cvturn(self) -> Panel:
        rolling = self._rolling(self._turnover(), HALF_YEAR, 40)
        return safe_div(rolling.std(), rolling.mean())

    def _char_abturn(self) -> Panel:
        turnover = self._turnover()
        return safe_div(self._rolling(turnover, MONTH, 7).mean(), self._rolling(turnover, 250, 85).mean())

    def _monthly_value(self, window: int, minimum: int) -> Panel:
        mean = self._rolling(self._panel("amount"), window, minimum).mean()
        return np.log((mean * MONTH).where(mean > 0))

    def _char_dtvm(self) -> Panel:
        return self._monthly_value(MONTH, 15)

    def _char_dtvq(self) -> Panel:
        return self._monthly_value(QUARTER, 40)

    def _char_dtva(self) -> Panel:
        return self._monthly_value(YEAR, 120)

    def _char_vdtv(self) -> Panel:
        return self._rolling(self._panel("amount"), HALF_YEAR, 40).std()

    def _char_cvd(self) -> Panel:
        rolling = self._rolling(self._panel("amount"), HALF_YEAR, 40)
        return safe_div(rolling.std(), rolling.mean())

    def _char_ami(self) -> Panel:
        amount = self._panel("amount")
        illiquidity = safe_div(self.returns.abs(), amount.where(amount > 0))
        return self._rolling(illiquidity, MONTH, 15).mean()

    # ------------------------------------------------------------------ risk
    def _char_idvc(self) -> np.ndarray:
        return self._capm()["resid_std"]

    def _char_idvff(self) -> np.ndarray:
        return self._ff3()["resid_std"]

    def _char_tv(self) -> Panel:
        return self._rolling(self.returns, HALF_YEAR, 40).std()

    def _char_idsff(self) -> np.ndarray:
        return self._ff3()["resid_skew"]

    def _char_idsc(self) -> np.ndarray:
        return self._capm()["resid_skew"]

    def _char_ts(self) -> Panel:
        return self._rolling(self.returns, HALF_YEAR, 40).skew()

    def _char_cs(self) -> np.ndarray:
        return self._capm()["coskew"]

    def _char_dbeta(self) -> np.ndarray:
        window, minimum = 240, 50
        market = self._series("mkt")
        returns = self.returns.to_numpy(dtype=float)
        result = np.full(returns.shape, np.nan)
        if len(market) < window:
            return result
        market_windows = np.lib.stride_tricks.sliding_window_view(market, window)
        with np.errstate(invalid="ignore"):
            down = market_windows < nan_mean(market_windows, 1)[0][:, None]
        market_values = np.where(np.isfinite(market_windows), market_windows, 0.0)
        for column in range(returns.shape[1]):
            stock = returns[:, column]
            observed = np.isfinite(stock)
            mask = np.lib.stride_tricks.sliding_window_view(observed, window) & down
            values = np.lib.stride_tricks.sliding_window_view(np.where(observed, stock, 0.0), window) * mask
            m = market_values * mask
            count = mask.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_r = values.sum(axis=1) / count
                mean_m = m.sum(axis=1) / count
                covariance = (values * m).sum(axis=1) / count - mean_r * mean_m
                variance = (m * m).sum(axis=1) / count - mean_m**2
                beta = covariance / variance
            result[window - 1 :, column] = np.where((count >= minimum) & (variance > 0), beta, np.nan)
        return result

    def _char_betafp(self) -> Panel:
        log_returns = np.log1p(self.returns)
        log_market = np.log1p(self._series("mkt"))
        market = self._market_panel(log_market)
        sigma_stock = self._rolling(log_returns, HALF_YEAR, 40).std()
        sigma_market = self._rolling(market.where(log_returns.notna()), HALF_YEAR, 40).std()
        stock_3d = log_returns.rolling(3, min_periods=3).sum()
        market_3d = self._market_panel(pd.Series(log_market).rolling(3, min_periods=3).sum().to_numpy())
        correlation = stock_3d.rolling(HALF_YEAR, min_periods=40).corr(market_3d)
        return correlation.clip(-1.0, 1.0) * safe_div(sigma_stock, sigma_market)

    def _char_betadm(self) -> np.ndarray:
        if "dimson" not in self._cache:
            market = self._series("mkt")
            lagged = np.concatenate([[np.nan], market[:-1]])
            lead = np.concatenate([market[1:], [np.nan]])
            fit = rolling_regression(
                self.returns.to_numpy(dtype=float), np.column_stack([lagged, market, lead]), HALF_YEAR, 40
            )
            # A window ending on day d uses the market return of d + 1, so its estimate is published on d + 1.
            shifted = np.full(fit["slopes"].shape, np.nan)
            shifted[1:] = fit["slopes"][:-1]
            self._cache["dimson"] = shifted
        return self._cache["dimson"]

    def _char_beta(self) -> np.ndarray:
        return self._capm()["beta"]

    # ------------------------------------------------------------------ past returns
    def _char_m1(self) -> Panel:
        return self._momentum(MONTH, 0)

    def _char_m11(self) -> Panel:
        return self._momentum(YEAR, MONTH)

    def _char_m60(self) -> Panel:
        return self._momentum(5 * YEAR, YEAR)

    def _char_m6(self) -> Panel:
        return self._momentum(6 * MONTH, MONTH)

    def _char_m3(self) -> Panel:
        return self._momentum(3 * MONTH, MONTH)

    def _char_indmom(self) -> Panel:
        sector = self._panel("sector")
        returns = self._momentum(6 * MONTH, 0)
        weights = self._float_value().where(returns.notna() & self._float_value().gt(0))
        result = Panel(np.nan, index=self.dates, columns=self.instruments)
        codes = pd.unique(sector.to_numpy().ravel())
        for code in codes[np.isfinite(codes)]:
            members = sector.eq(code)
            member_weights = weights.where(members)
            industry = (returns * member_weights).sum(axis=1) / member_weights.sum(axis=1)
            result = result.mask(members, np.broadcast_to(industry.to_numpy()[:, None], result.shape))
        return result

    def _char_m24(self) -> Panel:
        return self._momentum(3 * YEAR, YEAR)

    def _char_mchg(self) -> Panel:
        return self._momentum(6 * MONTH, MONTH) - self._momentum(YEAR, 6 * MONTH)

    def _residual_momentum(self) -> dict:
        slices = {"im12": (720 - YEAR, 720 - MONTH), "im6": (720 - 6 * MONTH, 720 - MONTH)}
        return self._regression("ff3_momentum", MARKET_FACTORS, 720, 360, residual_slices=slices)

    def _char_im12(self) -> np.ndarray:
        return self._residual_momentum()["im12"]

    def _char_im6(self) -> np.ndarray:
        return self._residual_momentum()["im6"]

    def _char_w52(self) -> Panel:
        return safe_div(self.price, self.price.rolling(240, min_periods=120).max())

    def _char_mdr(self) -> np.ndarray:
        window, minimum, top = MONTH, 15, 5
        returns = self.returns.to_numpy(dtype=float)
        result = np.full(returns.shape, np.nan)
        if len(returns) < window:
            return result
        for first in range(0, returns.shape[1], 64):
            block = np.lib.stride_tricks.sliding_window_view(returns[:, first : first + 64], window, axis=0)
            count = np.isfinite(block).sum(axis=2)
            largest = -np.sort(np.where(np.isfinite(block), -block, np.inf), axis=2)[:, :, :top]
            mean, _ = nan_mean(largest, 2)
            result[window - 1 :, first : first + 64] = np.where(count >= minimum, mean, np.nan)
        return result

    def _char_pr(self) -> Panel:
        return self._panel("basic_close")

    def _char_abr(self) -> Panel:
        frame = self._panel("fundamentals")
        abnormal = self.returns.sub(self._series("mkt"), axis=0)
        window_sum = abnormal.rolling(4, min_periods=3).sum().to_numpy()
        announcement = np.searchsorted(self.dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy())
        columns = self.instruments.get_indexer(frame["instrument"])
        published = announcement + 1
        keep = (columns >= 0) & (published < len(self.dates))
        events = np.full(window_sum.shape, np.nan)
        events[published[keep], columns[keep]] = window_sum[published[keep], columns[keep]]
        return Panel(events, index=self.dates, columns=self.instruments).ffill()

    def _char_season(self) -> Panel:
        monthly = self._momentum(MONTH, 0)
        stacked = np.stack([monthly.shift(YEAR * k - MONTH).to_numpy() for k in range(1, 6)], axis=2)
        mean, count = nan_mean(stacked, 2)
        return np.where(count >= 3, mean, np.nan)

    # ------------------------------------------------------------------ fundamentals helpers
    def _q(self, values: np.ndarray) -> Panel:
        return self.grid.panel(values)

    def _y(self, values: np.ndarray) -> Panel:
        return self.grid.panel(values, annual=True)

    def _book_equity(self) -> np.ndarray:
        g = self.grid
        return g.value("total_hldr_eqy_exc_min_int") - g.value("oth_eqt_tools_p_shr")

    def _debt(self) -> np.ndarray:
        g = self.grid
        return g.value("st_borr") + g.value("non_cur_liab_due_1y") + g.value("lt_borr") + g.value("bond_payable")

    def _net_operating_assets(self) -> np.ndarray:
        g = self.grid
        return (
            g.value("total_hldr_eqy_inc_min_int")
            + self._debt()
            - g.value("money_cap")
            - g.value("trad_asset")
        )

    def _operating_profit(self, flow: Callable[[str], np.ndarray]) -> np.ndarray:
        return flow("revenue") - flow("oper_cost") - flow("sell_exp") - flow("admin_exp") - flow("fin_exp")

    def _roe(self) -> np.ndarray:
        return ratio(self.grid.single("n_income_attr_p"), lag(self._book_equity(), 1))

    def _roa(self) -> np.ndarray:
        return ratio(self.grid.single("n_income"), lag(self.grid.value("total_assets"), 1))

    def _lagged_assets_change(self, values: np.ndarray) -> np.ndarray:
        assets = self.grid.value("total_assets")
        return ratio(values - lag(values, 4), lag(assets, 4))

    # ------------------------------------------------------------------ profitability
    def _char_roe(self) -> Panel:
        return self._q(self._roe())

    def _char_droe(self) -> Panel:
        roe = self._roe()
        return self._q(roe - lag(roe, 4))

    def _char_roa(self) -> Panel:
        return self._q(self._roa())

    def _char_droa(self) -> Panel:
        roa = self._roa()
        return self._q(roa - lag(roa, 4))

    def _char_rna(self) -> Panel:
        return self._q(ratio(self.grid.single("operate_profit"), lag(self._net_operating_assets(), 1)))

    def _char_pm(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("operate_profit"), g.single("revenue")))

    def _char_ato(self) -> Panel:
        return self._q(ratio(self.grid.single("revenue"), lag(self._net_operating_assets(), 1)))

    def _char_ct(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("revenue"), lag(g.value("total_assets"), 1)))

    def _char_gpa(self) -> Panel:
        g = self.grid
        return self._y(ratio(g.annual("revenue") - g.annual("oper_cost"), g.value("total_assets")))

    def _char_gpla(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("revenue") - g.single("oper_cost"), lag(g.value("total_assets"), 1)))

    def _char_ope(self) -> Panel:
        return self._y(ratio(self._operating_profit(self.grid.annual), self._book_equity()))

    def _char_ople(self) -> Panel:
        return self._q(ratio(self._operating_profit(self.grid.single), lag(self._book_equity(), 1)))

    def _char_opa(self) -> Panel:
        return self._y(ratio(self._operating_profit(self.grid.annual), self.grid.value("total_assets")))

    def _char_opla(self) -> Panel:
        g = self.grid
        return self._q(ratio(self._operating_profit(g.single), lag(g.value("total_assets"), 1)))

    def _char_tbi(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("total_profit"), g.single("n_income")))

    def _char_bl(self) -> Panel:
        return self._q(ratio(self.grid.value("total_assets"), self._book_equity()))

    def _char_sg(self) -> Panel:
        revenue = self.grid.annual("revenue")
        return self._y(growth(revenue, lag(revenue, 4)))

    def _char_sgq(self) -> Panel:
        revenue = self.grid.single("revenue")
        return self._q(growth(revenue, lag(revenue, 4)))

    def _char_fscore(self) -> Panel:
        g = self.grid
        assets = g.value("total_assets")
        income = g.single("n_income")
        cash_flow = g.single("n_cashflow_act")
        revenue = g.single("revenue")
        roa = ratio(income, lag(assets, 1))
        leverage = ratio(g.value("lt_borr") + g.value("bond_payable"), assets)
        current = ratio(g.value("total_cur_assets"), g.value("total_cur_liab"))
        margin = ratio(revenue - g.single("oper_cost"), revenue)
        turnover = ratio(revenue, lag(assets, 1))
        shares = g.value("total_share")
        components = (
            (roa, roa > 0),
            (cash_flow, cash_flow > 0),
            (roa - lag(roa, 4), roa - lag(roa, 4) > 0),
            (cash_flow - income, cash_flow > income),
            (leverage - lag(leverage, 4), leverage < lag(leverage, 4)),
            (current - lag(current, 4), current > lag(current, 4)),
            (shares - lag(shares, 4), shares <= lag(shares, 4)),
            (margin - lag(margin, 4), margin > lag(margin, 4)),
            (turnover - lag(turnover, 4), turnover > lag(turnover, 4)),
        )
        known = np.all([np.isfinite(value) for value, _ in components], axis=0)
        score = np.sum([signal.astype(float) for _, signal in components], axis=0)
        return self._q(np.where(known, score, np.nan))

    def _char_oscore(self) -> Panel:
        g = self.grid
        assets = g.value("total_assets")
        liabilities = g.value("total_liab")
        current_assets = g.value("total_cur_assets")
        current_liabilities = g.value("total_cur_liab")
        income = g.ttm("n_income")
        previous = lag(income, 4)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_assets = np.log(np.where(assets > 0, assets, np.nan))
        score = (
            -1.32
            - 0.407 * log_assets
            + 6.03 * ratio(liabilities, assets)
            - 1.43 * ratio(current_assets - current_liabilities, assets)
            + 0.076 * ratio(current_liabilities, current_assets)
            - 1.72 * np.where(np.isfinite(liabilities - assets), (liabilities > assets).astype(float), np.nan)
            - 2.37 * ratio(income, assets)
            - 1.83 * ratio(g.ttm("n_cashflow_act"), liabilities)
            + 0.285 * np.where(np.isfinite(income + previous), ((income < 0) & (previous < 0)).astype(float), np.nan)
            - 0.521 * ratio(income - previous, np.abs(income) + np.abs(previous))
        )
        return self._q(score)

    # ------------------------------------------------------------------ value
    def _to_price(self, numerator: Panel) -> Panel:
        cap = self._cap()
        return safe_div(numerator, cap.where(cap > 0))

    def _char_bm(self) -> Panel:
        return self._to_price(self._q(self._book_equity()))

    def _char_dm(self) -> Panel:
        return self._to_price(self._q(self.grid.value("total_liab")))

    def _char_am(self) -> Panel:
        return self._to_price(self._q(self.grid.value("total_assets")))

    def _char_ep(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_income_attr_p")))

    def _depreciation_ttm(self) -> np.ndarray:
        g = self.grid
        return g.ttm("depr_fa_coga_dpba") + g.ttm("amort_intang_assets") + g.ttm("lt_amort_deferred_exp")

    def _char_cfp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_income") + self._depreciation_ttm()))

    def _char_sr(self) -> Panel:
        revenue = self.grid.annual("revenue")
        total = np.zeros(revenue.shape)
        for year in range(1, 6):
            total = total + (6 - year) * growth(lag(revenue, 4 * (year - 1)), lag(revenue, 4 * year))
        return self._y(total / 15.0)

    def _char_em(self) -> Panel:
        g = self.grid
        net_debt = self._q(self._debt() + g.value("oth_eqt_tools_p_shr") + g.value("minority_int")
                           - g.value("money_cap") - g.value("trad_asset"))
        ebitda = self._q(g.ttm("total_profit") + g.ttm("fin_exp") + self._depreciation_ttm())
        return safe_div(self._cap() + net_debt, ebitda.where(ebitda > 0))

    def _char_sp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("revenue")))

    def _char_ocfp(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_cashflow_act")))

    def _char_de(self) -> Panel:
        return self._y(ratio(self.grid.value("total_liab"), self._book_equity()))

    def _net_debt(self) -> np.ndarray:
        g = self.grid
        return self._debt() + g.value("oth_eqt_tools_p_shr") - g.value("money_cap") - g.value("trad_asset")

    def _char_ebp(self) -> Panel:
        net_debt = self._q(self._net_debt())
        book = self._q(self._book_equity())
        return safe_div(net_debt + book, net_debt + self._cap())

    def _char_ndp(self) -> Panel:
        return self._to_price(self._q(self._net_debt()))

    # ------------------------------------------------------------------ investment
    def _char_ag(self) -> Panel:
        assets = self.grid.value("total_assets")
        return self._q(growth(assets, lag(assets, 4)))

    def _char_dpia(self) -> Panel:
        g = self.grid
        return self._q(self._lagged_assets_change(g.value("fix_assets") + g.value("inventories")))

    def _char_noa(self) -> Panel:
        return self._q(ratio(self._net_operating_assets(), lag(self.grid.value("total_assets"), 4)))

    def _char_dnoa(self) -> Panel:
        return self._q(self._lagged_assets_change(self._net_operating_assets()))

    def _char_ig(self) -> Panel:
        capex = self.grid.annual("c_pay_acq_const_fiolta")
        return self._y(growth(capex, lag(capex, 4)))

    def _char_cei(self) -> Panel:
        cap = self._cap().ffill()
        with np.errstate(invalid="ignore", divide="ignore"):
            value_growth = np.log(safe_div(cap, cap.shift(YEAR)))
            price_growth = np.log(safe_div(self.price, self.price.shift(YEAR)))
        return value_growth - price_growth

    def _char_cdi(self) -> Panel:
        liabilities = self.grid.value("total_liab")
        base = lag(liabilities, 20)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.log(ratio(liabilities, np.where(base > 0, base, np.nan)))
        return self._y(result)

    def _char_ivg(self) -> Panel:
        inventory = self.grid.value("inventories")
        return self._y(growth(inventory, lag(inventory, 4)))

    def _char_ivchg(self) -> Panel:
        g = self.grid
        inventory = g.value("inventories")
        assets = g.value("total_assets")
        return self._y(ratio(inventory - lag(inventory, 4), (assets + lag(assets, 4)) / 2.0))

    def _char_oacc(self) -> Panel:
        g = self.grid
        return self._y(ratio(g.annual("n_income") - g.annual("n_cashflow_act"), lag(g.value("total_assets"), 4)))

    def _working_capital(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.grid
        operating_assets = g.value("total_cur_assets") - g.value("money_cap") - g.value("trad_asset")
        operating_liabilities = g.value("total_cur_liab") - g.value("st_borr") - g.value("non_cur_liab_due_1y")
        return operating_assets, operating_liabilities

    def _non_current(self) -> tuple[np.ndarray, np.ndarray]:
        g = self.grid
        assets = g.value("total_assets") - g.value("total_cur_assets") - g.value("lt_eqt_invest")
        liabilities = g.value("total_liab") - g.value("total_cur_liab") - g.value("lt_borr") - g.value("bond_payable")
        return assets, liabilities

    def _financial(self) -> np.ndarray:
        g = self.grid
        return g.value("trad_asset") + g.value("lt_eqt_invest") - self._debt() - g.value("oth_eqt_tools_p_shr")

    def _char_tacc(self) -> Panel:
        current_assets, current_liabilities = self._working_capital()
        non_current_assets, non_current_liabilities = self._non_current()
        working = current_assets - current_liabilities
        total = working + (non_current_assets - non_current_liabilities) + self._financial()
        return self._y(self._lagged_assets_change(total))

    def _char_dwc(self) -> Panel:
        current_assets, current_liabilities = self._working_capital()
        return self._y(self._lagged_assets_change(current_assets - current_liabilities))

    def _char_dcoa(self) -> Panel:
        return self._y(self._lagged_assets_change(self._working_capital()[0]))

    def _char_dcol(self) -> Panel:
        return self._y(self._lagged_assets_change(self._working_capital()[1]))

    def _char_dnco(self) -> Panel:
        assets, liabilities = self._non_current()
        return self._y(self._lagged_assets_change(assets - liabilities))

    def _char_dnca(self) -> Panel:
        return self._y(self._lagged_assets_change(self._non_current()[0]))

    def _char_dncl(self) -> Panel:
        return self._y(self._lagged_assets_change(self._non_current()[1]))

    def _char_dfin(self) -> Panel:
        return self._y(self._lagged_assets_change(self._financial()))

    def _char_dbe(self) -> Panel:
        return self._y(self._lagged_assets_change(self._book_equity()))

    # ------------------------------------------------------------------ other
    def _char_rdm(self) -> Panel:
        return self._to_price(self._q(self.grid.single("rd_exp")))

    def _char_rds(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.single("rd_exp"), g.single("revenue")))

    def _char_ol(self) -> Panel:
        g = self.grid
        costs = g.annual("oper_cost") + g.annual("sell_exp") + g.annual("admin_exp")
        return self._y(ratio(costs, g.value("total_assets")))

    def _char_age(self) -> Panel:
        listed = pd.Series(self._panel("list_date")).reindex(self.instruments)
        days = self.dates.to_numpy()[:, None] - pd.to_datetime(listed).to_numpy()[None, :]
        months = days / np.timedelta64(1, "D") / 30.4375
        return np.where(months >= 0, months, np.nan)

    def _annual_change_gap(self, first: np.ndarray, second: np.ndarray) -> Panel:
        return self._y(growth(first, lag(first, 4)) - growth(second, lag(second, 4)))

    def _char_dsi(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.value("inventories"))

    def _char_dsa(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.value("accounts_receiv"))

    def _char_dgs(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue") - g.annual("oper_cost"), g.annual("revenue"))

    def _char_dss(self) -> Panel:
        g = self.grid
        return self._annual_change_gap(g.annual("revenue"), g.annual("sell_exp") + g.annual("admin_exp"))

    def _char_etr(self) -> Panel:
        g = self.grid
        rate = ratio(g.annual("income_tax"), g.annual("total_profit"))
        average, _ = nan_mean(np.stack([lag(rate, 4 * k) for k in (1, 2, 3)], axis=2), 2)
        income = g.annual("n_income_attr_p")
        earnings_change = self._y(income - lag(income, 4))
        return self._y(rate - average) * self._to_price(earnings_change)

    def _char_tan(self) -> Panel:
        g = self.grid
        tangible = (
            g.value("money_cap")
            + 0.715 * g.value("accounts_receiv")
            + 0.547 * g.value("inventories")
            + 0.535 * g.value("fix_assets")
        )
        return self._q(ratio(tangible, g.value("total_assets")))

    def _char_vcf(self) -> Panel:
        g = self.grid
        return self._q(rolling_quarter_std(ratio(g.single("n_cashflow_act"), g.single("revenue")), 16, 8))

    def _char_cta(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.value("money_cap"), g.value("total_assets")))

    def _char_esm(self) -> Panel:
        g = self.grid
        lagged_assets = lag(g.value("total_assets"), 1)
        earnings = rolling_quarter_std(ratio(g.single("n_income"), lagged_assets), 16, 8)
        cash = rolling_quarter_std(ratio(g.single("n_cashflow_act"), lagged_assets), 16, 8)
        return self._q(ratio(earnings, cash))

    def _asset_liquidity(self) -> np.ndarray:
        g = self.grid
        cash = g.value("money_cap")
        current = g.value("total_cur_assets")
        assets = g.value("total_assets")
        return cash + 0.75 * (current - cash) + 0.5 * (assets - current - g.value("goodwill") - g.value("intan_assets"))

    def _char_ala(self) -> Panel:
        return self._q(ratio(self._asset_liquidity(), lag(self.grid.value("total_assets"), 1)))

    def _char_alm(self) -> Panel:
        g = self.grid
        market_assets = self._q(g.value("total_assets") - self._book_equity()) + self._cap()
        return safe_div(self._q(self._asset_liquidity()), market_assets.where(market_assets > 0))

    def _surprise(self, values: np.ndarray) -> np.ndarray:
        change = values - lag(values, 4)
        return ratio(change, rolling_quarter_std(change, 8, 6))

    def _char_sue(self) -> Panel:
        return self._q(self._surprise(self.grid.single("n_income_attr_p")))

    def _char_rs(self) -> Panel:
        return self._q(self._surprise(self.grid.single("revenue")))

    def _char_tes(self) -> Panel:
        g = self.grid
        tax = g.single("income_tax")
        return self._q(ratio(tax - lag(tax, 4), lag(g.value("total_assets"), 4)))
