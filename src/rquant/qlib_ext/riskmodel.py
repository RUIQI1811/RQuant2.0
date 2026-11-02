"""A structured (fundamental) risk model for Qlib's ``EnhancedIndexingOptimizer``: Shenwan industries and CNE6 styles.

Qlib ships only statistical risk models (``StructuredCovEstimator``: PCA or factor analysis of past returns), whose
factors have no meaning, so the optimiser cannot be told to stay industry- or size-neutral. This model gives it named
factors in the CNE6 layout:

- **Exposures** on each day: one dummy per Shenwan level-1 industry (point in time, from the canonical bars; stocks
  without a membership on the day share an ``industry_unknown`` dummy) and the
  CNE6 style factors combined from the ``barra_cne6`` descriptors in ``CNE6_STYLES``. Every descriptor is winsorised
  at its 1% and 99% cross-sectional quantiles and standardised to a float-cap-weighted mean of zero and an equal-
  weighted standard deviation of one; a style is the mean of its available standardised descriptors, standardised
  again. A stock missing a style is put at the mean (zero).
- **Factor returns**: each day's stock returns (in percent, as ``StructuredCovEstimator`` scales them) regressed on
  the previous day's exposures, weighted by the square root of float cap. The industry dummies span the market, so
  there is no separate country factor.
- **Factor covariance** on a signal date: the exponentially weighted covariance of the last ``lookback`` daily factor
  returns (newest weight 1, halving every ``half_life`` days).
- **Specific risk**: the exponentially weighted mean square of each stock's regression residuals over the same
  window; a stock with fewer than ``MIN_SPECIFIC_OBSERVATIONS`` residuals gets the cross-sectional median.

``factor_deviation`` turns the industry, size and other style limits into the optimiser's ``f_dev``: the portfolio's
active exposure to each factor (its weight in an industry minus the benchmark's, or its size exposure minus the
benchmark's, in standard deviations) must stay within the limit.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rquant.errors import DataContractError
from rquant.qlib_ext.strategy import RiskModelData

# CNE6 style factors and the barra_cne6 descriptors each one averages (equal weights).
CNE6_STYLES: dict[str, tuple[str, ...]] = {
    "size": ("lncap",),
    "mid_cap": ("midcap",),
    "beta": ("hbeta",),
    "residual_volatility": ("hsigma", "dastd", "cmra"),
    "liquidity": ("stom", "stoq", "stoa", "atvr"),
    "short_term_reversal": ("strev",),
    "seasonality": ("season",),
    "industry_momentum": ("indmom",),
    "momentum": ("rstr",),
    "long_term_reversal": ("ltrstr", "lthalpha"),
    "book_to_price": ("btop",),
    "earnings_yield": ("etop", "cetop", "em"),
    "leverage": ("mlev", "blev", "dtoa"),
    "earnings_variability": ("vsal", "vern", "vflo"),
    "earnings_quality": ("abs", "acf"),
    "profitability": ("ato", "gp", "gpm", "roa"),
    "investment_quality": ("agro", "igro", "cxgro"),
    "growth": ("egro", "sgro"),
    "dividend_yield": ("dtop",),
}
CNE6_DESCRIPTOR_NAMES = tuple(name for members in CNE6_STYLES.values() for name in members)
WINSOR_QUANTILES = (0.01, 0.99)
MIN_SPECIFIC_OBSERVATIONS = 20
# A limit no active exposure reaches (winsorised styles stay within a few standard deviations): the optimiser's f_dev
# needs a number for every factor, and a huge one makes ECOS report "optimal_inaccurate".
UNBOUNDED = 100.0


def standardise(values: Any, cap: Any) -> Any:
    """Winsorise each column of ``values`` (stocks x columns) at its 1%/99% quantiles and scale it to a cap-weighted
    mean of zero and an equal-weighted standard deviation of one. Missing values stay missing."""
    import numpy as np

    values = np.array(values, dtype=float, copy=True)
    if values.ndim == 1:
        values = values[:, None]
    cap = np.asarray(cap, dtype=float)
    weight = np.where(np.isfinite(cap) & (cap > 0), cap, 0.0)
    for column in range(values.shape[1]):
        x = values[:, column]
        finite = np.isfinite(x)
        if finite.sum() < 3:
            values[:, column] = np.nan
            continue
        low, high = np.quantile(x[finite], WINSOR_QUANTILES)
        x = np.clip(x, low, high)
        weighted = finite & (weight > 0)
        centre = float(np.average(x[weighted], weights=weight[weighted])) if weighted.any() else float(x[finite].mean())
        spread = float(np.std(x[finite] - centre, ddof=1))
        values[:, column] = (x - centre) / spread if spread > 0 else np.where(finite, 0.0, np.nan)
    return values


def style_exposures(descriptors: Any, cap: Any) -> Any:
    """One day's CNE6 style exposures (stocks x ``CNE6_STYLES``) from its descriptors (stocks x descriptor names)."""
    import numpy as np

    columns = list(descriptors.columns)
    standard = standardise(descriptors.to_numpy(dtype=float), cap)
    styles = np.full((len(descriptors), len(CNE6_STYLES)), np.nan)
    for position, members in enumerate(CNE6_STYLES.values()):
        block = standard[:, [columns.index(name) for name in members]]
        available = np.isfinite(block).any(axis=1)
        styles[available, position] = np.nanmean(block[available], axis=1)
    styles = standardise(styles, cap)
    return np.nan_to_num(styles, nan=0.0)


class StructuredRiskModelData(RiskModelData):
    """``RiskModelData`` whose ``risk`` is the industry + CNE6 style model of this module instead of PCA factors."""

    def __init__(
        self,
        *,
        close: Any,
        volume: Any,
        float_cap: Any,
        sector: Any,
        descriptors: Iterable[Any],
        signal_dates: Iterable[Any],
        lookback: int = 240,
        half_life: float = 90.0,
        industry_deviation: float | None = 0.03,
        size_deviation: float | None = 0.2,
        style_deviation: float | None = None,
    ) -> None:
        """``close`` (adjusted), ``volume``, ``float_cap`` and ``sector`` (Shenwan level-1 code) are frames of dates by
        instruments. ``descriptors`` yields frames indexed by ``datetime`` and ``instrument`` with the barra_cne6
        descriptor columns (source names), in date order. Exposures are kept for ``signal_dates`` only."""
        import numpy as np
        import pandas as pd

        super().__init__(close=close, volume=volume, float_cap=float_cap, lookback=lookback, num_factors=1)
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        for name, limit in (
            ("industry_deviation", industry_deviation),
            ("size_deviation", size_deviation),
            ("style_deviation", style_deviation),
        ):
            if limit is not None and limit < 0:
                raise ValueError(f"{name} must not be negative, got {limit}")
        self.half_life = float(half_life)
        self.sector = sector.reindex(index=self.close.index, columns=self.close.columns)
        sectors = self.sector.to_numpy(dtype=float)
        self.industries = [int(code) for code in np.unique(sectors[np.isfinite(sectors)])]
        # Stocks without a Shenwan membership on the day (2-4% of rows before 2021, mostly delisted names) share one
        # industry of their own, so they are neither unexplained by the industries nor exempt from the limits.
        self.factor_names = [f"industry_{code}" for code in self.industries] + ["industry_unknown", *CNE6_STYLES]
        self.limits = {"industry": industry_deviation, "size": size_deviation, "style": style_deviation}

        returns = self.close.pct_change(fill_method=None) * 100.0
        wanted = {pd.Timestamp(day) for day in signal_dates}
        self.exposures: dict[Any, Any] = {}
        factor_returns: dict[Any, Any] = {}
        residuals: dict[Any, Any] = {}
        previous = None
        for frame in descriptors:
            missing = sorted(set(CNE6_DESCRIPTOR_NAMES).difference(frame.columns))
            if missing:
                raise DataContractError(f"barra_cne6 descriptors are missing: {missing}")
            for day, rows in frame.groupby(level="datetime", sort=True):
                day = pd.Timestamp(day)
                if day not in self.close.index:
                    previous = None
                    continue
                exposure = self._exposure(day, rows.droplevel("datetime").loc[:, list(CNE6_DESCRIPTOR_NAMES)])
                if previous is not None:
                    fitted = self._regress(day, previous, returns.loc[day])
                    if fitted is not None:
                        factor_returns[day], residuals[day] = fitted
                if day in wanted:
                    self.exposures[day] = exposure
                previous = exposure
        if not factor_returns:
            raise DataContractError("No day has both barra_cne6 exposures and stock returns for the risk model")
        self.factor_returns = pd.DataFrame(factor_returns).T.sort_index()
        self.residuals = pd.DataFrame(residuals).T.sort_index()

    def _exposure(self, day: Any, descriptors: Any) -> Any:
        """Industry dummies and style exposures of the day's descriptor rows (instruments x factor names)."""
        import numpy as np
        import pandas as pd

        instruments = descriptors.index
        cap = self.float_cap.loc[day].reindex(instruments).to_numpy(dtype=float)
        sector = self.sector.loc[day].reindex(instruments).to_numpy(dtype=float)
        dummies = (sector[:, None] == np.asarray(self.industries, dtype=float)[None, :]).astype(float)
        dummies = np.hstack([dummies, (~np.isfinite(sector)).astype(float)[:, None]])
        styles = style_exposures(descriptors, cap)
        return pd.DataFrame(np.hstack([dummies, styles]), index=instruments, columns=self.factor_names)

    def _regress(self, day: Any, exposure: Any, returns: Any) -> tuple[Any, Any] | None:
        """The day's factor returns and residuals: square-root-cap-weighted least squares of the day's returns on
        the previous day's exposures."""
        import numpy as np
        import pandas as pd

        position = self.close.index.get_loc(day)
        cap = self.float_cap.iloc[position - 1].reindex(exposure.index).to_numpy(dtype=float)
        y = returns.reindex(exposure.index).to_numpy(dtype=float)
        usable = np.isfinite(y) & np.isfinite(cap) & (cap > 0)
        if usable.sum() <= exposure.shape[1]:
            return None
        x = exposure.to_numpy(dtype=float)[usable]
        root = np.sqrt(np.sqrt(cap[usable]))
        coefficients, *_ = np.linalg.lstsq(x * root[:, None], y[usable] * root, rcond=None)
        residual = pd.Series(y[usable] - x @ coefficients, index=exposure.index[usable])
        return pd.Series(coefficients, index=self.factor_names), residual

    def factor_deviation(self) -> Any:
        """The optimiser's ``f_dev``: industry limits for the dummies, the size limit, the style limit elsewhere."""
        import numpy as np

        def limit(value: float | None) -> float:
            return UNBOUNDED if value is None else float(value)

        values = []
        for name in self.factor_names:
            if name.startswith("industry_"):
                values.append(limit(self.limits["industry"]))
            elif name == "size":
                values.append(limit(self.limits["size"]))
            else:
                values.append(limit(self.limits["style"]))
        return np.array(values, dtype=float)

    def risk(self, date: Any, codes: list[str]) -> tuple[Any, Any, Any]:
        """Factor exposures (codes x factors), factor covariance and specific volatility on ``date``, in percent."""
        import numpy as np
        import pandas as pd

        date = pd.Timestamp(date)
        if date not in self.exposures:
            raise DataContractError(f"No barra_cne6 exposures were kept for {date.date()}")
        history = self.factor_returns.loc[:date].tail(self.lookback)
        if len(history) < self.lookback:
            raise DataContractError(f"Fewer than {self.lookback} days of factor returns end on {date.date()}")
        weights = 0.5 ** (np.arange(len(history))[::-1] / self.half_life)
        weights = weights / weights.sum()
        values = history.to_numpy(dtype=float)
        centred = values - weights @ values
        factor_cov = (centred * weights[:, None]).T @ centred

        window = self.residuals.reindex(index=history.index, columns=codes).to_numpy(dtype=float)
        observed = np.isfinite(window)
        squared = np.where(observed, window, 0.0) ** 2
        mass = weights @ observed
        specific_var = np.full(len(codes), np.nan)
        enough = observed.sum(axis=0) >= MIN_SPECIFIC_OBSERVATIONS
        specific_var[enough] = (weights @ squared)[enough] / mass[enough]
        fallback = np.nanmedian(specific_var) if np.isfinite(specific_var).any() else float(np.nanmean(squared))
        specific_var = np.where(np.isfinite(specific_var), specific_var, fallback)

        exposures = self.exposures[date].reindex(codes).fillna(0.0).to_numpy(dtype=float)
        return exposures, factor_cov, np.sqrt(specific_var)


def load_sector(bars_root: Any, start: Any, end: Any, instruments: list[str]) -> Any:
    """Shenwan level-1 sector codes (dates x instruments) from the canonical bars, NaN without a membership."""
    from pathlib import Path

    import pandas as pd

    first, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for year in range(first.year, last.year + 1):
        path = Path(bars_root) / f"year={year}" / "bars.parquet"
        if not path.exists():
            raise DataContractError(f"Canonical bars partition is missing: {path}")
        frame = pd.read_parquet(path, columns=["datetime", "instrument", "sector"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frames.append(frame[frame["instrument"].isin(set(instruments))])
    bars = pd.concat(frames, ignore_index=True)
    bars = bars[(bars["datetime"] >= first) & (bars["datetime"] <= last)]
    return bars.pivot_table(index="datetime", columns="instrument", values="sector", aggfunc="last")


def load_barra_descriptors(factor_root: Any, start: Any, end: Any) -> Iterable[Any]:
    """Yield the barra_cne6 store year by year (rows ``datetime`` x ``instrument``, columns the descriptors' source
    names) from ``start`` through ``end``, so only one year of descriptors is in memory at a time."""
    from pathlib import Path

    import pandas as pd

    from rquant.factors.catalog import get_catalog

    root = Path(factor_root) / "barra_cne6"
    names = {canonical: source for source, canonical in get_catalog().source_to_canonical("barra_cne6").items()}
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    years = [year for year in range(first.year, last.year + 1) if (root / f"year={year}" / "factors.parquet").exists()]
    if not years:
        raise DataContractError(
            f"No barra_cne6 factor partitions under {root}; build them with `rquant factors build` for this universe"
        )
    for year in years:
        frame = pd.read_parquet(root / f"year={year}" / "factors.parquet", columns=["datetime", "instrument", *names])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame[(frame["datetime"] >= first) & (frame["datetime"] <= last)]
        yield frame.rename(columns=names).set_index(["datetime", "instrument"]).sort_index()
