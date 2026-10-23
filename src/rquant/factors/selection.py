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
    # Published factor values divided by the largest observation count of any factor, summed over dates.
    min_coverage: float = 0.9

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


def score_factor_stability(summary: Any, annual: Any, coverage: Any, criteria: StabilityCriteria) -> Any:
    """Return one row per factor with its oriented yearly statistics and the selection decision."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas and NumPy are required for factor selection") from exc
    for name, frame, columns in (
        ("summary", summary, {"factor", "rank_ic_mean", "rank_icir", "long_short_return"}),
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
    result["oriented_long_short_return"] = full["long_short_return"] * direction
    result["counted_years"] = counted.reindex(result.index).fillna(0).astype(int)
    result["positive_rank_ic_years"] = positive_ic.reindex(result.index).fillna(0).astype(int)
    result["positive_long_short_years"] = positive_ls.reindex(result.index).fillna(0).astype(int)
    denominator = result["counted_years"].where(result["counted_years"] > 0)
    result["positive_rank_ic_year_ratio"] = (result["positive_rank_ic_years"] / denominator).fillna(0.0)
    result["positive_long_short_year_ratio"] = (result["positive_long_short_years"] / denominator).fillna(0.0)
    result["coverage"] = coverage.reindex(result.index).fillna(0.0)

    checks = {
        "direction": result["direction"] != 0,
        "rank_ic": result["rank_ic_mean"].abs() >= criteria.min_abs_rank_ic,
        "rank_ic_years": result["positive_rank_ic_year_ratio"] >= criteria.min_positive_rank_ic_year_ratio,
        "long_short_years": result["positive_long_short_year_ratio"]
        >= criteria.min_positive_long_short_year_ratio,
        "coverage": result["coverage"] >= criteria.min_coverage,
    }
    result["selected"] = np.logical_and.reduce([check.to_numpy() for check in checks.values()])
    result["failed_checks"] = [
        ",".join(name for name, check in checks.items() if not bool(check.loc[factor])) for factor in result.index
    ]
    return result.reset_index()
