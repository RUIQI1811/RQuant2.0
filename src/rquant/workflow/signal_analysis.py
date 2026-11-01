"""Calendar-year summary of the daily IC and Rank IC Qlib's ``SigAnaRecord`` computes for a run's predictions.

``SigAnaRecord`` (run by ``rquant report``) correlates each date's scores with the realised label the model was
trained on (the next-open return over the run's horizon): IC is the Pearson correlation, Rank IC the Spearman one.
ICIR is the mean daily IC over its standard deviation, not annualised, as ``SigAnaRecord`` logs it.
"""

from __future__ import annotations

import math
from typing import Any

from rquant.errors import DataContractError


def summarize_ic(daily: Any) -> dict[str, Any]:
    """Return overall and calendar-year IC, ICIR, Rank IC and Rank ICIR of a frame of ``datetime``, ``ic`` and
    ``rank_ic`` rows."""
    import pandas as pd

    def stats(group: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"days": int(group[["ic", "rank_ic"]].notna().any(axis=1).sum())}
        for name in ("ic", "rank_ic"):
            values = group[name].dropna()
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            result[name] = _finite(mean)
            result[f"{name}_ir"] = _finite(mean / std) if std > 0 else None
            result[f"{name}_positive_ratio"] = _finite(float((values > 0).mean())) if len(values) else None
        return result

    daily = daily.dropna(subset=["ic", "rank_ic"], how="all")
    if daily.empty:
        raise DataContractError("No prediction date has an IC")
    years = pd.to_datetime(daily["datetime"]).dt.year
    return {
        "overall": stats(daily),
        "years": [{"year": int(year), **stats(group)} for year, group in daily.groupby(years, sort=True)],
    }


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None
