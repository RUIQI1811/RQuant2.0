"""Information coefficient of a walk-forward run's predictions, as Qlib's ``SigAnaRecord`` reports it.

Each prediction date is scored on its own cross-section: IC is the Pearson correlation between the score and the
realised label the model was trained on (the next-open return over the run's horizon), Rank IC the Spearman
correlation. ICIR is the mean daily IC over its standard deviation, not annualised, the convention of
``qlib.contrib.eva.alpha.calc_ic`` and ``SigAnaRecord``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json
from rquant.qlib_ext.loader import LABEL_EXPRESSIONS
from rquant.qlib_ext.runtime import qlib_init_kwargs

MIN_CROSS_SECTION = 20


def daily_ic(predictions: Any, labels: Any, *, min_cross_section: int = MIN_CROSS_SECTION) -> Any:
    """Return one row per date with ``ic``, ``rank_ic`` and the number of scored instruments.

    ``predictions`` and ``labels`` are Series on a <datetime, instrument> index. Dates with fewer than
    ``min_cross_section`` pairs where both are finite are left out.
    """
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"score": predictions, "label": labels.reindex(predictions.index)})
    frame = frame[np.isfinite(frame["score"]) & np.isfinite(frame["label"])]
    rows = []
    for day, group in frame.groupby(level="datetime", sort=True):
        if len(group) < min_cross_section:
            continue
        rows.append(
            {
                "datetime": day,
                "ic": group["score"].corr(group["label"]),
                "rank_ic": group["score"].rank().corr(group["label"].rank()),
                "instruments": len(group),
            }
        )
    return pd.DataFrame(rows, columns=["datetime", "ic", "rank_ic", "instruments"])


def summarize_ic(daily: Any) -> dict[str, Any]:
    """Return overall and calendar-year IC, ICIR, Rank IC and Rank ICIR for the rows ``daily_ic`` returns."""
    import pandas as pd

    def stats(group: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"days": len(group)}
        for column, name in (("ic", "ic"), ("rank_ic", "rank_ic")):
            values = group[column].dropna()
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            result[name] = _finite(mean)
            result[f"{name}_ir"] = _finite(mean / std) if std > 0 else None
            result[f"{name}_positive_ratio"] = _finite(float((values > 0).mean())) if len(values) else None
        return result

    if daily.empty:
        raise DataContractError("No prediction date has enough scored instruments for an IC")
    years = pd.to_datetime(daily["datetime"]).dt.year
    return {
        "overall": stats(daily),
        "years": [{"year": int(year), **stats(group)} for year, group in daily.groupby(years, sort=True)],
    }


class SignalAnalysisRunner:
    """Score a walk-forward run's predictions against the label its model was trained on."""

    def __init__(self, *, qlib_root: str | Path, run_directory: str | Path) -> None:
        self.qlib_root = Path(qlib_root)
        self.run_directory = Path(run_directory)

    def run(self) -> dict[str, Any]:
        try:
            import pandas as pd
            import qlib
            from qlib.constant import REG_CN
            from qlib.data import D
        except ImportError as exc:
            raise DependencyError("pyqlib==0.9.7, pandas and pyarrow are required for signal analysis") from exc

        horizon = self._horizon()
        predictions = pd.read_parquet(self.run_directory / "predictions.parquet")
        predictions["datetime"] = pd.to_datetime(predictions["datetime"])
        score = predictions.set_index(["datetime", "instrument"])["score"].sort_index()
        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN, **qlib_init_kwargs())
        dates = score.index.get_level_values("datetime")
        labels = D.features(
            sorted(score.index.get_level_values("instrument").unique()),
            [LABEL_EXPRESSIONS[horizon]],
            start_time=dates.min(),
            end_time=dates.max(),
            freq="day",
        )
        labels = labels.iloc[:, 0].swaplevel().sort_index()
        labels.index = labels.index.set_names(["datetime", "instrument"])

        daily = daily_ic(score, labels)
        daily.to_parquet(self.run_directory / "signal_ic.parquet", index=False)
        summary = {
            "status": "complete",
            "horizon": horizon,
            "label": LABEL_EXPRESSIONS[horizon],
            "min_cross_section": MIN_CROSS_SECTION,
            **summarize_ic(daily),
        }
        atomic_write_json(self.run_directory / "signal_analysis.json", summary)
        return summary

    def _horizon(self) -> str:
        import json

        path = self.run_directory / "walk_forward.json"
        if not path.exists():
            raise DataContractError(f"Walk-forward manifest does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            horizon = json.load(handle).get("horizon")
        if horizon not in LABEL_EXPRESSIONS:
            raise DataContractError(f"Unsupported horizon in {path}: {horizon}")
        return str(horizon)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None
