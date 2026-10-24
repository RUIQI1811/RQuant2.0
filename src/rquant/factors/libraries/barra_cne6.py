"""Barra China Equity Model (CNE6) style descriptors evaluated as daily point-in-time panels.

The catalog follows the descriptor definitions MSCI published for the CNE6 model (size, beta, residual volatility,
liquidity, momentum, long-term reversal, value, earnings yield, leverage, earnings variability, earnings quality,
profitability, investment quality, growth and dividend yield). This module is an independent implementation from
those public definitions: no MSCI code or data is used and inputs come from Tushare. It publishes the descriptors,
not the composite style factors, so a model can weight them itself.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days and a year 252.
* Returns are close-to-close returns of RQuant's backward-adjusted prices; the day after a suspension has none. The
  market return is the value-weighted all-A series of ``reference/market_factors.parquet``. Returns are not reduced
  by a risk-free rate: within the trailing windows used here a constant rate only moves the intercepts.
* Exponentially weighted windows give the newest row weight 1 and halve it every ``half-life`` rows. Lagged
  descriptors (STREV, RSTR, LTRSTR, LTHALPHA) average the raw value over the documented lags.
* Financial statements are the first-reported values of ``reference/fundamentals.parquet`` on the quarter grid of
  ``china_chars``: a quarter is usable once it and every earlier quarter are announced. Flow-to-price and flow-to-assets
  ratios use trailing-twelve-month flows; five-year descriptors use the last five fiscal-year (Q4) values and need at
  least three of them.
* The two cross-sectional descriptors (MIDCAP, INDMOM) use only the stocks in the CSI 300 on that day, so later index
  members do not shape earlier cross-sections.
* Every output is missing on days the stock has no quote.

Left out: the analyst-forecast descriptors EPIBS, EGRLF, ETOPF_STD and DPIBS (Tushare's forecast endpoint allows one
call per minute at this account level).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.libraries.china_chars import QuarterGrid, lag, ratio
from rquant.factors.panel_operators import replace_inf, safe_div
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
MONTH = 21
YEAR = 252
LONG_WINDOW = 1040
LONG_LAG = 273
FISCAL_YEARS = 5
CNE6_DAILY_BASIC = {"turnover_rate": 0.01, "dv_ttm": 0.01}

# (source name, description, maximum lookback in trading days). Order is the stable public column order.
CNE6_DESCRIPTORS: tuple[tuple[str, str, int], ...] = (
    # Size
    ("lncap", "LNCAP: log total market value", 1),
    ("midcap", "MIDCAP: cube of standardised LNCAP, orthogonal to it, winsorised and standardised", 1),
    # Beta and residual volatility
    ("hbeta", "HBETA: market beta, 252 days, half-life 63", YEAR),
    ("hsigma", "HSIGMA: residual volatility of the HBETA regression", YEAR),
    ("dastd", "DASTD: daily return volatility, 252 days, half-life 42", YEAR),
    ("cmra", "CMRA: range of cumulative log returns over the last 1..12 months", YEAR),
    # Liquidity
    ("stom", "STOM: log one-month share turnover", MONTH),
    ("stoq", "STOQ: log average monthly turnover over three months", 3 * MONTH),
    ("stoa", "STOA: log average monthly turnover over twelve months", YEAR),
    ("atvr", "ATVR: turnover summed over 252 days, half-life 63", YEAR),
    # Momentum
    ("strev", "STREV: log return over 21 days, half-life 5, averaged over lags 1..3", MONTH + 3),
    ("season", "SEASON: average next-month return in each of the prior five years", 5 * YEAR),
    ("indmom", "INDMOM: sqrt-cap-weighted industry relative strength minus the stock's, 126 days, half-life 21",
     126),
    ("rstr", "RSTR: log return over 252 days, half-life 126, averaged over lags 11..21", YEAR + 21),
    # Long-term reversal
    ("ltrstr", "LTRSTR: minus the log return over 1040 days, half-life 260, averaged over lags 273..283",
     LONG_WINDOW + LONG_LAG + 10),
    ("lthalpha", "LTHALPHA: minus the market-model alpha over 1040 days, half-life 260, averaged over lags 273..283",
     LONG_WINDOW + LONG_LAG + 10),
    # Value and earnings yield
    ("btop", "BTOP: common equity over total market value", 1),
    ("etop", "ETOP: trailing net income to parent over total market value", 1),
    ("cetop", "CETOP: trailing operating cash flow over total market value", 1),
    ("em", "EM: trailing EBIT over enterprise value", 1),
    # Leverage
    ("mlev", "MLEV: market value plus preferred equity and long-term debt, over market value", 1),
    ("blev", "BLEV: common equity plus preferred equity and long-term debt, over common equity", 1),
    ("dtoa", "DTOA: total liabilities over total assets", 1),
    # Earnings variability
    ("vsal", "VSAL: std over mean of the last five fiscal-year revenues", 1),
    ("vern", "VERN: std over mean absolute value of the last five fiscal-year net incomes to parent", 1),
    ("vflo", "VFLO: std over mean absolute value of the last five fiscal-year net cash changes", 1),
    # Earnings quality
    ("abs", "ABS: minus balance-sheet accruals (change in net operating assets less D&A) over total assets", 1),
    ("acf", "ACF: minus cash-flow accruals (net income less operating and investing cash flow, plus D&A) over total "
     "assets", 1),
    # Profitability
    ("ato", "ATO: trailing revenue over total assets", 1),
    ("gp", "GP: trailing gross profit over total assets", 1),
    ("gpm", "GPM: trailing gross profit over trailing revenue", 1),
    ("roa", "ROA: trailing net income to parent over total assets", 1),
    # Investment quality
    ("agro", "AGRO: minus the five-year slope of fiscal-year total assets over their mean", 1),
    ("igro", "IGRO: minus the five-year slope of fiscal-year total shares over their mean", 1),
    ("cxgro", "CXGRO: minus the five-year slope of fiscal-year capital expenditure over its mean", 1),
    # Growth
    ("egro", "EGRO: five-year slope of fiscal-year earnings per share over its mean absolute value", 1),
    ("sgro", "SGRO: five-year slope of fiscal-year revenue per share over its mean", 1),
    # Dividend yield
    ("dtop", "DTOP: trailing twelve-month dividend yield", 1),
)
CNE6_NAMES = tuple(name for name, _, _ in CNE6_DESCRIPTORS)


class BarraCNE6Library(PanelFactorLibrary):
    """Barra CNE6 style descriptors, re-dated to daily point-in-time panels."""

    family = "barra_cne6"
    required_inputs = ("close", "cap", "sector")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"cne6_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.barra_cne6.CNE6Descriptors.{name}",
                catalog_version=1,
            )
            for ordinal, (name, description, lookback) in enumerate(CNE6_DESCRIPTORS, 1)
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.reference import load_reference_manifest

        canonical_root = Path(canonical_root)
        root = canonical_root / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "barra_cne6 needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in (*daily_files, "market_factors.parquet", "fundamentals.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        wanted = [("instrument", "in", list(instruments))]
        columns = ["instrument", "datetime", *CNE6_DAILY_BASIC]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in CNE6_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale

        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        inputs["mkt"] = factors["mkt"].reindex(dates)
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        bar_files = sorted((canonical_root / "bars").glob("year=*/*.parquet"))
        bar_files = [path for path in bar_files if int(path.parent.name.split("=")[1]) in years]
        if not bar_files:
            raise DataContractError(f"Canonical bars do not exist under {canonical_root / 'bars'}")
        members = pd.concat(
            [pd.read_parquet(path, columns=["datetime", "instrument", "in_csi300"], filters=wanted)
             for path in bar_files],
            ignore_index=True,
        )
        members["datetime"] = pd.to_datetime(members["datetime"])
        membership = members.pivot(index="datetime", columns="instrument", values="in_csi300")
        inputs["universe"] = membership.reindex(index=dates, columns=instruments).fillna(False).astype(bool)
        hashes = {
            "barra_cne6_reference": manifest["fingerprint"],
            "barra_cne6_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "barra_cne6_market_factors": files["market_factors.parquet"],
            "barra_cne6_fundamentals": files["fundamentals.parquet"],
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = CNE6Descriptors(inputs)
        return {name: calculator.calculate(name) for name in CNE6_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = CNE6Descriptors(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)


def decay_rate(half_life: float | None) -> float:
    return 1.0 if half_life is None else 0.5 ** (1.0 / half_life)


def ew_window_sums(values: Mapping[str, np.ndarray], window: int, half_life: float | None) -> dict[str, np.ndarray]:
    """Exponentially weighted sums over every trailing ``window`` rows; the newest row has weight 1.

    Missing values must already be zero. The recursion ``S_t = d S_{t-1} + x_t - d^window x_{t-window}`` keeps the cost
    independent of the window length, and row ``t`` only reads rows ``t - window + 1 .. t``.
    """
    decay = decay_rate(half_life)
    tail = decay**window
    result: dict[str, np.ndarray] = {}
    for key, x in values.items():
        out = np.empty(x.shape)
        running = np.zeros(x.shape[1:])
        for row in range(len(x)):
            running = decay * running + x[row]
            if row >= window:
                running = running - tail * x[row - window]
            out[row] = running
        result[key] = out
    return result


def ew_market_fit(y: np.ndarray, market: np.ndarray, window: int, half_life: float, minimum: int) -> dict:
    """Weighted least squares of every stock column of ``y`` on a constant and the market over trailing windows."""
    x = np.broadcast_to(market[:, None], y.shape)
    mask = np.isfinite(y) & np.isfinite(x)
    y0 = np.where(mask, y, 0.0)
    x0 = np.where(mask, x, 0.0)
    weight = mask.astype(float)
    sums = ew_window_sums(
        {"w": weight, "x": x0, "y": y0, "xx": x0 * x0, "xy": x0 * y0, "yy": y0 * y0}, window, half_life
    )
    count = ew_window_sums({"n": weight}, window, None)["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sums["x"] / sums["w"]
        mean_y = sums["y"] / sums["w"]
        var_x = sums["xx"] / sums["w"] - mean_x**2
        var_y = sums["yy"] / sums["w"] - mean_y**2
        cov = sums["xy"] / sums["w"] - mean_x * mean_y
        beta = cov / var_x
        alpha = mean_y - beta * mean_x
        residual = var_y - beta * cov
    valid = (count > minimum - 0.5) & (var_x > 1e-16)
    return {
        "beta": np.where(valid, beta, np.nan),
        "alpha": np.where(valid, alpha, np.nan),
        # Rounding can leave a tiny negative residual variance for an exact fit.
        "resid_std": np.where(valid & (residual > 0), np.sqrt(np.maximum(residual, 0.0)), np.nan),
    }


def lag_average(values: np.ndarray, first: int, last: int) -> np.ndarray:
    """Mean of ``values`` lagged ``first`` through ``last`` rows; missing unless every lag is known."""
    rows = len(values)
    stacked = np.full((last - first + 1, *values.shape), np.nan)
    for position, periods in enumerate(range(first, last + 1)):
        if periods < rows:
            stacked[position, periods:] = values[: rows - periods]
    return stacked.mean(axis=0)


def five_year_slope(values: np.ndarray, *, absolute_mean: bool = False) -> np.ndarray:
    """OLS slope of the last five fiscal-year values on time over their mean (or mean absolute value).

    ``values`` holds fiscal-year values at Q4 quarter columns; at least three of the five years must be known.
    """
    stacked = np.stack([lag(values, 4 * (FISCAL_YEARS - 1 - k)) for k in range(FISCAL_YEARS)], axis=2)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=2)
    time = np.broadcast_to(np.arange(1.0, FISCAL_YEARS + 1.0), stacked.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = np.where(finite, time, 0.0).sum(axis=2) / count
        mean_v = np.where(finite, stacked, 0.0).sum(axis=2) / count
        centered_t = np.where(finite, time - mean_t[..., None], 0.0)
        slope = (centered_t * np.where(finite, stacked - mean_v[..., None], 0.0)).sum(axis=2) / (
            centered_t**2
        ).sum(axis=2)
        scale = np.where(finite, np.abs(stacked), 0.0).sum(axis=2) / count if absolute_mean else mean_v
    return np.where(count >= 3, ratio(slope, scale), np.nan)


def five_year_variation(values: np.ndarray) -> np.ndarray:
    """Sample std of the last five fiscal-year values over their mean absolute value; at least three years."""
    stacked = np.stack([lag(values, 4 * k) for k in range(FISCAL_YEARS)], axis=2)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, stacked, 0.0).sum(axis=2) / count
        squares = np.where(finite, (stacked - mean[..., None]) ** 2, 0.0).sum(axis=2)
        std = np.sqrt(squares / (count - 1))
        scale = np.where(finite, np.abs(stacked), 0.0).sum(axis=2) / count
    return np.where(count >= 3, ratio(std, scale), np.nan)


class CNE6Descriptors:
    """Calculator with shared intermediates; ``calculate`` returns one descriptor panel."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna()
        self.price = close.ffill()
        self.returns = (close / close.shift(1) - 1.0).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.log_returns = np.log1p(np.where(self.returns > -1.0, self.returns, np.nan))
        self._cache: dict[str, Any] = {}
        self._grid: QuarterGrid | None = None

    # ------------------------------------------------------------------ infrastructure
    def calculate(self, name: str) -> Panel:
        method: Callable[[], Any] = getattr(self, f"_d_{name}")
        result = method()
        if isinstance(result, np.ndarray):
            result = Panel(result, index=self.dates, columns=self.instruments)
        result = replace_inf(result.astype(float))
        return result.where(self.trading)

    def _input(self, key: str) -> Any:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"barra_cne6 input is missing: {key}")
        return value

    def _market(self) -> np.ndarray:
        return np.asarray(pd.Series(self._input("mkt")).reindex(self.dates), dtype=float)

    def _cap(self) -> Panel:
        cap = self._input("cap")
        return cap.where(cap > 0)

    def _universe(self) -> np.ndarray:
        universe = self.inputs.get("universe")
        if universe is None:
            return np.ones((len(self.dates), len(self.instruments)), dtype=bool)
        return np.asarray(universe.reindex(index=self.dates, columns=self.instruments).fillna(False), dtype=bool)

    @property
    def grid(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(self._input("fundamentals"), self.dates, self.instruments)
        return self._grid

    def _ew_log_return(self, window: int, half_life: float, minimum: int) -> np.ndarray:
        observed = np.isfinite(self.log_returns)
        sums = ew_window_sums({"r": np.where(observed, self.log_returns, 0.0)}, window, half_life)["r"]
        count = ew_window_sums({"n": observed.astype(float)}, window, None)["n"]
        return np.where(count > minimum - 0.5, sums, np.nan)

    def _beta_fit(self) -> dict:
        if "beta" not in self._cache:
            self._cache["beta"] = ew_market_fit(self.returns, self._market(), YEAR, 63, 63)
        return self._cache["beta"]

    def _turnover(self) -> Panel:
        return self._input("basic_turnover_rate")

    def _log_turnover(self, window: int, minimum: int, months: int) -> Panel:
        total = self._turnover().rolling(window, min_periods=minimum).sum() / months
        return np.log(total.where(total > 0))

    def _q(self, values: np.ndarray, *, annual: bool = False) -> Panel:
        return self.grid.panel(values, annual=annual)

    def _to_price(self, numerator: Panel) -> Panel:
        return safe_div(numerator, self._cap())

    def _common_equity(self) -> np.ndarray:
        g = self.grid
        return g.value("total_hldr_eqy_exc_min_int") - g.value("oth_eqt_tools_p_shr")

    def _long_term_debt(self) -> np.ndarray:
        g = self.grid
        return g.value("lt_borr") + g.value("bond_payable")

    def _total_debt(self) -> np.ndarray:
        g = self.grid
        return g.value("st_borr") + g.value("non_cur_liab_due_1y") + self._long_term_debt()

    def _fiscal_year(self, values: np.ndarray) -> np.ndarray:
        """Balance-sheet values at fiscal year ends (Q4 columns) only."""
        return np.where(self.grid.quarter_index[None, :] == 3, values, np.nan)

    def _standardise(self, values: np.ndarray, universe: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Cap-weighted mean zero and unit equal-weighted std over each day's universe members."""
        result = np.full(values.shape, np.nan)
        for row in range(len(values)):
            members = universe[row] & np.isfinite(values[row]) & np.isfinite(weights[row])
            if members.sum() < 3:
                continue
            mean = np.average(values[row, members], weights=weights[row, members])
            std = values[row, members].std(ddof=1)
            if std > 0:
                result[row] = (values[row] - mean) / std
        return result

    # ------------------------------------------------------------------ size
    def _d_lncap(self) -> Panel:
        return np.log(self._cap())

    def _d_midcap(self) -> np.ndarray:
        cap = self._cap().to_numpy(dtype=float)
        universe = self._universe()
        size = self._standardise(np.log(cap), universe, cap)
        cube = size**3
        weights = np.sqrt(cap)
        residual = np.full(cube.shape, np.nan)
        for row in range(len(cube)):
            members = universe[row] & np.isfinite(cube[row]) & np.isfinite(weights[row])
            if members.sum() < 3:
                continue
            design = np.column_stack([np.ones(members.sum()), size[row, members]])
            root = np.sqrt(weights[row, members])
            coefficients, *_ = np.linalg.lstsq(design * root[:, None], cube[row, members] * root, rcond=None)
            residual[row] = cube[row] - coefficients[0] - coefficients[1] * size[row]
        # Winsorise at three standard deviations of the members, then standardise again.
        clipped = np.full(residual.shape, np.nan)
        for row in range(len(residual)):
            members = universe[row] & np.isfinite(residual[row])
            if members.sum() < 3:
                continue
            mean = residual[row, members].mean()
            std = residual[row, members].std(ddof=1)
            clipped[row] = np.clip(residual[row], mean - 3.0 * std, mean + 3.0 * std)
        return self._standardise(clipped, universe, cap)

    # ------------------------------------------------------------------ beta and residual volatility
    def _d_hbeta(self) -> np.ndarray:
        return self._beta_fit()["beta"]

    def _d_hsigma(self) -> np.ndarray:
        return self._beta_fit()["resid_std"]

    def _d_dastd(self) -> np.ndarray:
        observed = np.isfinite(self.returns)
        r = np.where(observed, self.returns, 0.0)
        sums = ew_window_sums({"w": observed.astype(float), "r": r, "rr": r * r}, YEAR, 42)
        count = ew_window_sums({"n": observed.astype(float)}, YEAR, None)["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums["r"] / sums["w"]
            variance = sums["rr"] / sums["w"] - mean**2
        return np.where((count > 62.5) & (variance > 0), np.sqrt(np.maximum(variance, 0.0)), np.nan)

    def _d_cmra(self) -> np.ndarray:
        observed = np.isfinite(self.log_returns)
        prefix = np.concatenate([np.zeros((1, observed.shape[1])), np.cumsum(np.where(observed, self.log_returns,
                                                                                         0.0), axis=0)])
        counts = np.concatenate([np.zeros((1, observed.shape[1])), np.cumsum(observed, axis=0)])
        rows = len(observed)
        cumulative = np.full((12, *observed.shape), np.nan)
        for months in range(1, 13):
            days = months * MONTH
            if days > rows:
                continue
            cumulative[months - 1, days - 1 :] = prefix[days:] - prefix[:-days]
        enough = np.full(observed.shape, False)
        if rows >= YEAR:
            enough[YEAR - 1 :] = (counts[YEAR:] - counts[:-YEAR]) >= 200
        with np.errstate(invalid="ignore", divide="ignore"):
            high = np.log1p(np.max(cumulative, axis=0))
            low = np.log1p(np.min(cumulative, axis=0))
        return np.where(enough, high - low, np.nan)

    # ------------------------------------------------------------------ liquidity
    def _d_stom(self) -> Panel:
        return self._log_turnover(MONTH, 15, 1)

    def _d_stoq(self) -> Panel:
        return self._log_turnover(3 * MONTH, 40, 3)

    def _d_stoa(self) -> Panel:
        return self._log_turnover(YEAR, 120, 12)

    def _d_atvr(self) -> np.ndarray:
        turnover = self._turnover().to_numpy(dtype=float)
        observed = np.isfinite(turnover)
        total = ew_window_sums({"v": np.where(observed, turnover, 0.0)}, YEAR, 63)["v"]
        count = ew_window_sums({"n": observed.astype(float)}, YEAR, None)["n"]
        return np.where(count >= 120, total, np.nan)

    # ------------------------------------------------------------------ momentum
    def _d_strev(self) -> np.ndarray:
        return lag_average(self._ew_log_return(MONTH, 5, 15), 1, 3)

    def _d_season(self) -> np.ndarray:
        price = self.price
        # The month that starts k years ago ends 252 k - 21 rows ago, so every term is known on row t.
        stacked = np.stack(
            [(price.shift(YEAR * k - MONTH) / price.shift(YEAR * k) - 1.0).to_numpy(dtype=float) for k in range(1, 6)]
        )
        finite = np.isfinite(stacked)
        count = finite.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(finite, stacked, 0.0).sum(axis=0) / count
        return np.where(count >= 2, mean, np.nan)

    def _d_indmom(self) -> np.ndarray:
        strength = self._ew_log_return(126, 21, 60)
        sector = np.asarray(self._input("sector").reindex(index=self.dates, columns=self.instruments), dtype=float)
        weights = np.sqrt(self._cap().to_numpy(dtype=float))
        universe = self._universe()
        result = np.full(strength.shape, np.nan)
        for row in range(len(strength)):
            codes = sector[row]
            members = universe[row] & np.isfinite(strength[row]) & np.isfinite(weights[row]) & np.isfinite(codes)
            for code in np.unique(codes[members]):
                peers = members & (codes == code)
                industry = np.average(strength[row, peers], weights=weights[row, peers])
                in_industry = codes == code
                result[row, in_industry] = industry - strength[row, in_industry]
        return result

    def _d_rstr(self) -> np.ndarray:
        return lag_average(self._ew_log_return(YEAR, 126, 120), 11, 21)

    # ------------------------------------------------------------------ long-term reversal
    def _d_ltrstr(self) -> np.ndarray:
        return -lag_average(self._ew_log_return(LONG_WINDOW, 260, 500), LONG_LAG, LONG_LAG + 10)

    def _d_lthalpha(self) -> np.ndarray:
        fit = ew_market_fit(self.returns, self._market(), LONG_WINDOW, 260, 500)
        return -lag_average(fit["alpha"], LONG_LAG, LONG_LAG + 10)

    # ------------------------------------------------------------------ value and earnings yield
    def _d_btop(self) -> Panel:
        return self._to_price(self._q(self._common_equity()))

    def _d_etop(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_income_attr_p")))

    def _d_cetop(self) -> Panel:
        return self._to_price(self._q(self.grid.ttm("n_cashflow_act")))

    def _d_em(self) -> Panel:
        g = self.grid
        ebit = self._q(g.ttm("total_profit") + g.ttm("fin_exp"))
        enterprise = self._cap() + self._q(self._total_debt() - g.value("money_cap"))
        return safe_div(ebit, enterprise.where(enterprise > 0))

    # ------------------------------------------------------------------ leverage
    def _d_mlev(self) -> Panel:
        other = self._q(self.grid.value("oth_eqt_tools_p_shr") + self._long_term_debt())
        return safe_div(self._cap() + other, self._cap())

    def _d_blev(self) -> Panel:
        equity = self._common_equity()
        total = equity + self.grid.value("oth_eqt_tools_p_shr") + self._long_term_debt()
        return self._q(ratio(total, np.where(equity > 0, equity, np.nan)))

    def _d_dtoa(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.value("total_liab"), g.value("total_assets")))

    # ------------------------------------------------------------------ earnings variability
    def _d_vsal(self) -> Panel:
        return self._q(five_year_variation(self.grid.annual("revenue")), annual=True)

    def _d_vern(self) -> Panel:
        return self._q(five_year_variation(self.grid.annual("n_income_attr_p")), annual=True)

    def _d_vflo(self) -> Panel:
        return self._q(five_year_variation(self.grid.annual("n_incr_cash_cash_equ")), annual=True)

    # ------------------------------------------------------------------ earnings quality
    def _d_abs(self) -> Panel:
        g = self.grid
        assets = g.value("total_assets")
        operating = (assets - g.value("money_cap")) - (g.value("total_liab") - self._total_debt())
        depreciation = g.ttm("depr_fa_coga_dpba") + g.ttm("amort_intang_assets") + g.ttm("lt_amort_deferred_exp")
        accruals = operating - lag(operating, 4) - depreciation
        return self._q(-ratio(accruals, assets))

    def _d_acf(self) -> Panel:
        g = self.grid
        depreciation = g.ttm("depr_fa_coga_dpba") + g.ttm("amort_intang_assets") + g.ttm("lt_amort_deferred_exp")
        cash_flow = g.ttm("n_cashflow_act") + g.ttm("n_cashflow_inv_act")
        accruals = g.ttm("n_income_attr_p") - cash_flow + depreciation
        return self._q(-ratio(accruals, g.value("total_assets")))

    # ------------------------------------------------------------------ profitability
    def _d_ato(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.ttm("revenue"), g.value("total_assets")))

    def _d_gp(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.ttm("revenue") - g.ttm("oper_cost"), g.value("total_assets")))

    def _d_gpm(self) -> Panel:
        g = self.grid
        revenue = g.ttm("revenue")
        return self._q(ratio(revenue - g.ttm("oper_cost"), revenue))

    def _d_roa(self) -> Panel:
        g = self.grid
        return self._q(ratio(g.ttm("n_income_attr_p"), g.value("total_assets")))

    # ------------------------------------------------------------------ investment quality
    def _d_agro(self) -> Panel:
        return self._q(-five_year_slope(self._fiscal_year(self.grid.value("total_assets"))), annual=True)

    def _d_igro(self) -> Panel:
        return self._q(-five_year_slope(self._fiscal_year(self.grid.value("total_share"))), annual=True)

    def _d_cxgro(self) -> Panel:
        return self._q(-five_year_slope(self.grid.annual("c_pay_acq_const_fiolta")), annual=True)

    # ------------------------------------------------------------------ growth
    def _per_share(self, flow: str) -> np.ndarray:
        g = self.grid
        shares = self._fiscal_year(g.value("total_share"))
        return ratio(g.annual(flow), np.where(shares > 0, shares, np.nan))

    def _d_egro(self) -> Panel:
        return self._q(five_year_slope(self._per_share("n_income_attr_p"), absolute_mean=True), annual=True)

    def _d_sgro(self) -> Panel:
        return self._q(five_year_slope(self._per_share("revenue")), annual=True)

    # ------------------------------------------------------------------ dividend yield
    def _d_dtop(self) -> Panel:
        # Tushare leaves dv_ttm empty for stocks that paid nothing over the last year.
        dividend = self._input("basic_dv_ttm")
        return dividend.fillna(0.0).where(self._turnover().notna() | dividend.notna())
