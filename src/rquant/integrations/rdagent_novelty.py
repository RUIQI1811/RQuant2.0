"""Novelty screen for RD-Agent's new factors, installed next to RD-Agent's factor runner by ``rdagent_setup.py``.

RD-Agent 1.0.0 only drops a new factor when its mean daily correlation with an already accepted factor reaches
0.99, and never compares with Alpha158 or any other known library, so its loop keeps rediscovering Alpha158. This
screen rejects a new factor when

* fewer than ``RQUANT_MIN_COVERAGE`` (default 0.7) of the reference rows have a value, or
* its mean daily rank correlation, in absolute value, with a reference factor (Alpha158, WorldQuant 101, GTJA 191
  and TA-Lib values exported by RQuant) or an accepted RD-Agent factor reaches ``RQUANT_MAX_CORRELATION``
  (default 0.7).

Reference values are read from ``RQUANT_REFERENCE_FACTORS`` (default
``git_ignore_folder/rquant_reference_factors.parquet`` under the directory RD-Agent runs from); they hold a sample
of trading days, and correlations are measured on those days. It only needs pandas and NumPy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_MAX_CORRELATION = 0.7
DEFAULT_MIN_COVERAGE = 0.7
DEFAULT_REFERENCE_PATH = Path("git_ignore_folder") / "rquant_reference_factors.parquet"
# Days to sample when there is no reference file to take them from.
FALLBACK_DATES = 150


def screen_new_factors(new: Any, accepted: Any = None, reference: Any = None) -> tuple[Any, list[str]]:
    """Return the new factors that pass, and one line per rejected factor saying why."""
    import pandas as pd

    max_correlation = float(os.environ.get("RQUANT_MAX_CORRELATION", DEFAULT_MAX_CORRELATION))
    min_coverage = float(os.environ.get("RQUANT_MIN_COVERAGE", DEFAULT_MIN_COVERAGE))
    if reference is None:
        reference = load_reference()
    if reference is not None:
        dates = reference.index.get_level_values("datetime").unique()
    else:
        all_dates = new.index.get_level_values("datetime").unique().sort_values()
        dates = all_dates[:: max(1, len(all_dates) // FALLBACK_DATES)]

    sample = _rows_on(new, dates)
    if reference is not None:
        # Coverage counts the rows RQuant trades: point-in-time members on the sampled days.
        sample = sample.reindex(reference.index)
    coverage = sample.notna().mean()

    comparisons = []
    if reference is not None:
        comparisons.append(reference)
    if accepted is not None and not accepted.empty:
        comparisons.append(_rows_on(accepted, dates).add_prefix("accepted."))
    others = pd.concat(comparisons, axis=1) if comparisons else None
    correlation = rank_correlation(sample, others) if others is not None else None

    kept, rejected = [], []
    for column in new.columns:
        if coverage[column] < min_coverage:
            rejected.append(f"{column}: has a value on only {coverage[column]:.0%} of stock-days")
            continue
        if correlation is not None:
            row = correlation.loc[column].abs()
            if not row.dropna().empty and row.max() >= max_correlation:
                rejected.append(
                    f"{column}: rank correlation {correlation.loc[column, row.idxmax()]:+.2f} with existing factor "
                    f"{row.idxmax()}"
                )
                continue
        kept.append(column)
    return new[kept], rejected


def load_reference() -> Any:
    import pandas as pd

    path = Path(os.environ.get("RQUANT_REFERENCE_FACTORS", DEFAULT_REFERENCE_PATH)).expanduser()
    if not path.exists():
        return None
    return pd.read_parquet(path)


def rank_correlation(left: Any, right: Any) -> Any:
    """Mean over days of the cross-sectional Spearman correlation of every left column with every right column.

    Both frames are indexed by <datetime, instrument>. Each day uses the stocks present in both; a stock missing one
    value counts as the day's median rank for it.
    """
    import numpy as np
    import pandas as pd

    left, right = left.align(right, join="inner", axis=0)
    left = _centered_ranks(left)
    right = _centered_ranks(right)
    total = np.zeros((left.shape[1], right.shape[1]))
    days = np.zeros_like(total)
    positions = left.index.get_level_values("datetime")
    left_values = left.to_numpy()
    right_values = right.to_numpy()
    for day in positions.unique():
        mask = positions == day
        x = left_values[mask]
        y = right_values[mask]
        x_norm = np.sqrt((x**2).sum(axis=0))
        y_norm = np.sqrt((y**2).sum(axis=0))
        denominator = np.outer(x_norm, y_norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            daily = (x.T @ y) / denominator
        valid = denominator > 0
        total[valid] += daily[valid]
        days[valid] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(days > 0, total / days, np.nan)
    return pd.DataFrame(mean, index=left.columns, columns=right.columns)


def _rows_on(frame: Any, dates: Any) -> Any:
    return frame[frame.index.get_level_values("datetime").isin(dates)]


def _centered_ranks(frame: Any) -> Any:
    ranks = frame.groupby(level="datetime").rank(pct=True)
    return (ranks - ranks.groupby(level="datetime").transform("mean")).fillna(0.0).astype("float64")
