"""Novelty screen for RD-Agent's new factors, installed next to RD-Agent's factor runner by ``rdagent_setup.py``.

RD-Agent 1.0.0 only drops a new factor when its mean daily correlation with an already accepted factor reaches
0.99, and never compares with Alpha158 or any other known library, so its loop keeps rediscovering Alpha158. This
screen rejects a new factor when

* its code shifts data backwards in time (``shift(-n)``) or uses centred windows, which read future rows,
* it barely differs across stocks on a day (the same market-wide value for every stock, or a ratio that is 1 by
  construction), which a cross-sectional model cannot use and a rank correlation cannot compare,
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
import re
from pathlib import Path
from typing import Any

DEFAULT_MAX_CORRELATION = 0.7
DEFAULT_MIN_COVERAGE = 0.7
DEFAULT_REFERENCE_PATH = Path("git_ignore_folder") / "rquant_reference_factors.parquet"
# Days to sample when there is no reference file to take them from.
FALLBACK_DATES = 150
# shift(-1), shift(periods=-h) and centred rolling windows all read rows after the current day.
LOOK_AHEAD = re.compile(r"shift\(\s*(?:periods\s*=\s*)?-|center\s*=\s*True")
# A factor must take at least this share of distinct values across the stocks of a typical day ...
MIN_DISTINCT_SHARE = 0.05
# ... and spread by more than this relative to its typical size (amount / (volume * vwap) is 1 up to rounding).
MIN_RELATIVE_SPREAD = 1e-4


def screen_new_factors(
    new: Any, accepted: Any = None, reference: Any = None, codes: dict[str, str] | None = None
) -> tuple[Any, list[str]]:
    """Return the new factors that pass, and one line per rejected factor saying why.

    ``codes`` maps factor names to the source of the ``factor.py`` that computed them.
    """
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
        code = (codes or {}).get(column, "")
        match = LOOK_AHEAD.search(code)
        if match:
            rejected.append(
                f"{column}: its code reads future rows ({match.group(0)!r}); use only past and current data"
            )
            continue
        flat = _flat_reason(sample[column])
        if flat:
            rejected.append(f"{column}: {flat}; a factor must rank stocks against each other on each day")
            continue
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


def _flat_reason(values: Any) -> str | None:
    """Why a factor does not separate stocks on a day, or None when it does."""
    days = values.dropna().groupby(level="datetime")
    if days.ngroups == 0:
        return None
    distinct = (days.nunique() / days.size()).median()
    if distinct < MIN_DISTINCT_SHARE:
        return f"on a typical day only {distinct:.0%} of stocks have distinct values"
    spread = (days.std() / days.apply(lambda day: day.abs().median()).replace(0.0, float("nan"))).median()
    if spread == spread and spread < MIN_RELATIVE_SPREAD:
        return "its values are practically identical across stocks"
    return None


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
