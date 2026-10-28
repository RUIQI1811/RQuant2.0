"""Select factors whose single-factor evaluation is stable across calendar years.

The inputs are the artifacts written by ``rquant factors evaluate``: ``factor_summary.csv`` (full period),
``annual_effectiveness.csv`` (one row per factor and year) and ``factor_daily.parquet`` (one row per factor and date).
A factor's direction is the sign of its full-period Rank IC; every annual statistic is oriented by that direction, so a
factor that reliably predicts lower returns is kept and marked for inverted use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rquant.errors import DataContractError, DependencyError


@dataclass(frozen=True)
class StabilityCriteria:
    # Years with fewer evaluated return days than this (for example a short final year) are not counted.
    min_year_days: int = 120
    # Share of counted years whose oriented annual Rank IC is positive.
    min_positive_rank_ic_year_ratio: float = 0.75
    # Share of counted years whose oriented annual top-minus-bottom quantile return is positive.
    min_positive_long_short_year_ratio: float = 0.75
    # Absolute full-period mean daily Rank IC.
    min_abs_rank_ic: float = 0.01
    # Absolute t-statistic of the full-period mean daily Rank IC (daily Rank ICIR times the square root of its days).
    min_abs_rank_ic_t_stat: float = 0.0
    # Published factor values divided by the largest observation count of any factor, summed over dates.
    min_coverage: float = 0.9
    # Share of counted years in which the oriented long quantile (top for direction +1, bottom for -1) beats the
    # equal-weight universe return. A long-only portfolio can only use this side of the spread. Needs
    # ``universe_returns``; when the share is required, the mean yearly long excess must also be positive.
    min_positive_long_excess_year_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def factor_coverage(daily: Any) -> Any:
    """Return each factor's share of the best-covered factor's observations, pooled over all dates.

    The per-date denominator is the largest ``observations`` count of any factor on that date, which is the number of
    universe members with a finite label when at least one factor is fully populated.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas is required for factor selection") from exc
    required = {"datetime", "factor", "observations"}
    if not required.issubset(daily.columns):
        raise DataContractError(f"Daily evaluation frame is missing columns: {sorted(required - set(daily.columns))}")
    frame = daily.loc[:, ["datetime", "factor", "observations"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    capacity = frame.groupby("datetime")["observations"].transform("max")
    totals = frame.assign(capacity=capacity).groupby("factor")[["observations", "capacity"]].sum()
    coverage = totals["observations"] / totals["capacity"].where(totals["capacity"] > 0)
    return coverage.rename("coverage").fillna(0.0)


def score_factor_stability(
    summary: Any, annual: Any, coverage: Any, criteria: StabilityCriteria, *, universe_returns: Any = None
) -> Any:
    """Return one row per factor with its oriented yearly statistics and the selection decision.

    ``universe_returns`` maps calendar years to the compounded equal-weight return of the evaluated universe over the
    same days and horizon as the annual quantile returns (see ``compound_by_year``).
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas and NumPy are required for factor selection") from exc
    for name, frame, columns in (
        ("summary", summary, {"factor", "rank_ic_mean", "rank_icir", "rank_ic_days", "long_short_return"}),
        ("annual", annual, {"factor", "year", "return_days", "rank_ic_mean", "long_short_return"}),
    ):
        if not columns.issubset(frame.columns):
            raise DataContractError(f"Factor {name} frame is missing columns: {sorted(columns - set(frame.columns))}")
    if summary["factor"].duplicated().any():
        raise DataContractError("Factor summary contains duplicate factors")

    full = summary.set_index("factor")
    direction = np.sign(full["rank_ic_mean"]).replace(0.0, np.nan)
    years = annual[annual["return_days"] >= criteria.min_year_days].copy()
    years["direction"] = years["factor"].map(direction)
    years["oriented_rank_ic"] = years["rank_ic_mean"] * years["direction"]
    years["oriented_long_short"] = years["long_short_return"] * years["direction"]
    grouped = years.groupby("factor")
    counted = grouped["year"].nunique()
    positive_ic = grouped["oriented_rank_ic"].apply(lambda values: int((values > 0.0).sum()))
    positive_ls = grouped["oriented_long_short"].apply(lambda values: int((values > 0.0).sum()))

    result = pd.DataFrame(index=full.index)
    result["direction"] = direction.fillna(0.0).astype(int)
    result["rank_ic_mean"] = full["rank_ic_mean"]
    result["rank_icir"] = full["rank_icir"]
    result["rank_ic_t_stat"] = full["rank_icir"] * np.sqrt(full["rank_ic_days"].astype(float))
    result["oriented_long_short_return"] = full["long_short_return"] * direction
    result["counted_years"] = counted.reindex(result.index).fillna(0).astype(int)
    result["positive_rank_ic_years"] = positive_ic.reindex(result.index).fillna(0).astype(int)
    result["positive_long_short_years"] = positive_ls.reindex(result.index).fillna(0).astype(int)
    denominator = result["counted_years"].where(result["counted_years"] > 0)
    result["positive_rank_ic_year_ratio"] = (result["positive_rank_ic_years"] / denominator).fillna(0.0)
    result["positive_long_short_year_ratio"] = (result["positive_long_short_years"] / denominator).fillna(0.0)
    result["coverage"] = coverage.reindex(result.index).fillna(0.0)
    if universe_returns is not None:
        if not {"top_return", "bottom_return"}.issubset(years.columns):
            raise DataContractError("Annual frame needs top_return and bottom_return to score the long side")
        universe = years["year"].map(universe_returns)
        if universe.isna().any():
            missing = sorted(years.loc[universe.isna(), "year"].unique())
            raise DataContractError(f"Universe returns are missing counted years: {missing}")
        long_leg = years["top_return"].where(years["direction"] > 0, years["bottom_return"])
        years["long_excess"] = long_leg - universe
        long_grouped = years.groupby("factor")["long_excess"]
        positive_long = long_grouped.apply(lambda values: int((values > 0.0).sum()))
        result["positive_long_excess_years"] = positive_long.reindex(result.index).fillna(0).astype(int)
        result["positive_long_excess_year_ratio"] = (result["positive_long_excess_years"] / denominator).fillna(0.0)
        result["mean_long_excess"] = long_grouped.mean().reindex(result.index)
    elif criteria.min_positive_long_excess_year_ratio > 0.0:
        raise DataContractError("min_positive_long_excess_year_ratio needs universe_returns")

    checks = {
        "direction": result["direction"] != 0,
        "rank_ic": result["rank_ic_mean"].abs() >= criteria.min_abs_rank_ic,
        "rank_ic_t_stat": result["rank_ic_t_stat"].abs().fillna(0.0) >= criteria.min_abs_rank_ic_t_stat,
        "rank_ic_years": result["positive_rank_ic_year_ratio"] >= criteria.min_positive_rank_ic_year_ratio,
        "long_short_years": result["positive_long_short_year_ratio"]
        >= criteria.min_positive_long_short_year_ratio,
        "coverage": result["coverage"] >= criteria.min_coverage,
    }
    if criteria.min_positive_long_excess_year_ratio > 0.0:
        checks["long_excess_years"] = (
            result["positive_long_excess_year_ratio"] >= criteria.min_positive_long_excess_year_ratio
        ) & (result["mean_long_excess"].fillna(0.0) > 0.0)
    result["selected"] = np.logical_and.reduce([check.to_numpy() for check in checks.values()])
    result["failed_checks"] = [
        ",".join(name for name, check in checks.items() if not bool(check.loc[factor])) for factor in result.index
    ]
    return result.reset_index()


def compound_by_year(daily_returns: Any) -> Any:
    """Compound a date-indexed series of daily returns within each calendar year."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas is required for factor selection") from exc
    returns = daily_returns.dropna()
    index = pd.DatetimeIndex(returns.index)
    return (1.0 + returns).groupby(index.year).prod() - 1.0


def cross_section_rank_correlation(values: Any, *, min_cross_section: int = 20) -> Any:
    """Return the Spearman correlation matrix of one cross-section (rows are instruments, columns factors).

    Every column is ranked over its own finite values (ties get the average rank), then each pair is correlated with
    Pearson over the rows where both are finite, which is what ``DataFrame.rank().corr(min_periods=...)`` computes.
    Pairs with fewer than ``min_cross_section`` such rows, or without variance on them, are NaN. The pairwise sums are
    matrix products, so a cross-section of a few thousand factors takes a fraction of a second.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas and NumPy are required for factor selection") from exc
    ranked = pd.DataFrame(np.asarray(values, dtype=float)).rank(axis=0).to_numpy()
    valid = np.isfinite(ranked)
    mask = valid.astype(float)
    # Centering each column on its own mean rank keeps the products small; Pearson is invariant to the shift.
    centered = np.where(valid, ranked - np.nanmean(np.where(valid, ranked, np.nan), axis=0, keepdims=True), 0.0)
    count = mask.T @ mask
    sum_x = centered.T @ mask
    sum_xx = (centered * centered).T @ mask
    sum_xy = centered.T @ centered
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = sum_xy - sum_x * sum_x.T / count
        variance_x = sum_xx - sum_x * sum_x / count
        correlation = covariance / np.sqrt(variance_x * variance_x.T)
    tolerance = 1e-9 * np.maximum(sum_xx, 1.0)
    degenerate = (count < min_cross_section) | (variance_x <= tolerance) | (variance_x.T <= tolerance.T)
    correlation[degenerate] = np.nan
    return np.clip(correlation, -1.0, 1.0)


class RankCorrelationAccumulator:
    """Average cross-sectional Spearman correlations over dates that arrive in chunks of rows.

    Feeding the rows of a long sample one chunk at a time (for example one store year) keeps only that chunk in memory;
    each date must arrive in a single chunk.
    """

    def __init__(self, factor_names: list[str] | tuple[str, ...], *, min_cross_section: int = 20) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise DependencyError("NumPy is required for factor selection") from exc
        self.names = list(factor_names)
        self.min_cross_section = min_cross_section
        self._total = np.zeros((len(self.names), len(self.names)))
        self._counts = np.zeros((len(self.names), len(self.names)))

    def add(self, frame: Any) -> None:
        """Add every date of ``frame``, which holds ``datetime`` plus one column per factor."""
        try:
            import numpy as np
        except ImportError as exc:
            raise DependencyError("NumPy is required for factor selection") from exc
        missing = sorted({"datetime", *self.names} - set(frame.columns))
        if missing:
            raise DataContractError(f"Factor frame is missing columns: {missing}")
        for _, group in frame.groupby("datetime", sort=True):
            matrix = cross_section_rank_correlation(
                group[self.names].to_numpy(dtype=float), min_cross_section=self.min_cross_section
            )
            valid = np.isfinite(matrix)
            self._total[valid] += matrix[valid]
            self._counts[valid] += 1

    def mean(self) -> Any:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and NumPy are required for factor selection") from exc
        counts = self._counts
        mean = np.divide(self._total, counts, out=np.full_like(self._total, np.nan), where=counts > 0)
        return pd.DataFrame(mean, index=self.names, columns=self.names)


def mean_rank_correlation(frame: Any, factor_names: list[str] | tuple[str, ...], *, min_cross_section: int = 20) -> Any:
    """Return the mean over dates of the cross-sectional Spearman correlation between every pair of factors.

    ``frame`` holds ``datetime`` plus one column per factor. Each date's values are ranked across instruments and
    correlated pairwise over the instruments where both factors are finite; dates with fewer than
    ``min_cross_section`` such instruments do not count for that pair.
    """
    accumulator = RankCorrelationAccumulator(factor_names, min_cross_section=min_cross_section)
    accumulator.add(frame)
    return accumulator.mean()


def prune_correlated(
    strength: Any, correlation: Any, *, max_abs_correlation: float
) -> tuple[list[str], dict[str, str]]:
    """Greedily keep the strongest factors whose absolute correlation with every kept factor stays below the limit.

    ``strength`` maps factor names to the score used for ordering (for example the absolute Rank IC); ties keep the
    input order. Returns the kept names in strength order and, for every dropped factor, the kept factor it duplicates.
    """
    if not 0.0 < max_abs_correlation < 1.0:
        raise ValueError("max_abs_correlation must lie strictly between 0 and 1")
    ordered = list(strength.sort_values(ascending=False, kind="stable").index)
    missing = [name for name in ordered if name not in correlation.index]
    if missing:
        raise DataContractError(f"Correlation matrix is missing factors: {missing[:5]}")
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for name in ordered:
        overlaps = correlation.loc[name, kept].abs() if kept else None
        if overlaps is not None and (overlaps >= max_abs_correlation).any():
            dropped[name] = str(overlaps.idxmax())
        else:
            kept.append(name)
    return kept, dropped
