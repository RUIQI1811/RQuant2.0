from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rquant.errors import DataContractError, DependencyError
from rquant.factors.catalog import FactorSet, get_catalog
from rquant.io import atomic_replace_file, atomic_write_json, sha256_file
from rquant.qlib_ext.loader import HORIZON_STEPS, LABEL_EXPRESSIONS


@dataclass(frozen=True)
class FactorEvaluationConfig:
    factor_set: FactorSet
    horizon: Literal["1d", "5d", "20d"]
    benchmark: str = "SH000300"
    quantile: float = 0.2
    min_cross_section: int = 20
    min_effective_days: int = 20

    def validate(self) -> None:
        if self.horizon not in LABEL_EXPRESSIONS:
            raise ValueError(f"Unsupported horizon: {self.horizon}")
        if not 0.0 < self.quantile <= 0.5:
            raise ValueError("quantile must be greater than 0 and no greater than 0.5")
        if self.min_cross_section < 2:
            raise ValueError("min_cross_section must be at least 2")
        if self.min_effective_days < 1:
            raise ValueError("min_effective_days must be positive")


class FactorEvaluator:
    """Evaluate raw factors as next-open scores with benchmark-adjusted side attribution."""

    def __init__(
        self,
        *,
        qlib_root: str | Path,
        factor_root: str | Path,
        output_directory: str | Path,
        config: FactorEvaluationConfig,
    ) -> None:
        self.qlib_root = Path(qlib_root)
        self.factor_root = Path(factor_root)
        self.output_directory = Path(output_directory)
        self.config = config
        self.config.validate()

    def run(self, *, start: Any, end: Any) -> dict[str, Any]:
        try:
            import pandas as pd
            import qlib
            from qlib.constant import REG_CN
            from qlib.data import D
        except ImportError as exc:
            raise DependencyError(
                "pandas, pyarrow, NumPy and pyqlib==0.9.7 are required for factor evaluation"
            ) from exc

        start_timestamp = pd.Timestamp(start).normalize()
        end_timestamp = pd.Timestamp(end).normalize()
        if end_timestamp < start_timestamp:
            raise ValueError("end must not precede start")
        factor_directory = self.factor_root / self.config.factor_set
        factor_manifest = _load_complete_manifest(factor_directory / "manifest.json", self.config.factor_set)
        qlib_manifest = _load_complete_manifest(self.qlib_root / "manifest.json")
        expected = list(get_catalog().canonical_names(self.config.factor_set))
        partitions = sorted(factor_directory.glob("year=*/factors.parquet"))
        if not partitions:
            raise DataContractError(f"No factor partitions for {self.config.factor_set}: {factor_directory}")

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN)
        calendar = pd.DatetimeIndex(
            D.calendar(start_time=start_timestamp, end_time=end_timestamp, freq="day")
        ).normalize()
        if calendar.empty:
            raise DataContractError("No Qlib trading dates in the requested factor-evaluation range")
        holding_period = HORIZON_STEPS[self.config.horizon]
        daily_parts = []
        for partition in partitions:
            year = int(partition.parent.name.removeprefix("year="))
            if year < start_timestamp.year or year > end_timestamp.year:
                continue
            print(f"factor evaluation: reading year={year}", file=sys.stderr, flush=True)
            frame = pd.read_parquet(partition, columns=["datetime", "instrument", *expected])
            frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
            frame = frame[frame["datetime"].between(start_timestamp, end_timestamp)]
            if frame.empty:
                continue
            partition_start = frame["datetime"].min()
            partition_end = frame["datetime"].max()
            instruments = sorted(frame["instrument"].astype(str).unique())
            labels = D.features(
                instruments,
                [LABEL_EXPRESSIONS[self.config.horizon]],
                start_time=partition_start,
                end_time=partition_end,
                freq="day",
            )
            labels = _feature_series(pd, labels, "label", include_instrument=True)
            benchmark = D.features(
                [self.config.benchmark],
                [LABEL_EXPRESSIONS[self.config.horizon]],
                start_time=partition_start,
                end_time=partition_end,
                freq="day",
            )
            benchmark = _feature_series(pd, benchmark, "benchmark", include_instrument=False)
            evaluated = frame.merge(labels, on=["datetime", "instrument"], how="left", validate="many_to_one")
            evaluated = evaluated.merge(benchmark, on="datetime", how="left", validate="many_to_one")
            daily_parts.append(
                evaluate_factor_frame(
                    evaluated,
                    expected,
                    quantile=self.config.quantile,
                    min_cross_section=self.config.min_cross_section,
                )
            )
            print(f"factor evaluation: completed year={year}", file=sys.stderr, flush=True)

        if not daily_parts:
            raise DataContractError("No factor observations in the requested evaluation range")
        daily = pd.concat(daily_parts, ignore_index=True).sort_values(["factor", "datetime"]).reset_index(drop=True)
        daily = assign_staggered_pockets(daily, calendar, holding_period=holding_period)
        summary, annual = summarize_effectiveness(
            daily,
            expected,
            min_effective_days=self.config.min_effective_days,
            holding_period=holding_period,
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        daily_path = self.output_directory / "factor_daily.parquet"
        summary_path = self.output_directory / "factor_summary.csv"
        annual_path = self.output_directory / "annual_effectiveness.csv"
        _atomic_write_frame(daily, daily_path, format="parquet")
        _atomic_write_frame(summary, summary_path, format="csv")
        _atomic_write_frame(annual, annual_path, format="csv")
        manifest = {
            "status": "complete",
            "factor_set": self.config.factor_set,
            "horizon": self.config.horizon,
            "label_expression": LABEL_EXPRESSIONS[self.config.horizon],
            "benchmark": self.config.benchmark,
            "quantile": self.config.quantile,
            "min_cross_section": self.config.min_cross_section,
            "min_effective_days": self.config.min_effective_days,
            "start": str(daily["datetime"].min().date()),
            "end": str(daily["datetime"].max().date()),
            "anchor_step": 1,
            "holding_period": holding_period,
            "pocket_count": holding_period,
            "evaluation_method": "staggered_pockets_v1",
            "factor_count": len(expected),
            "daily_rows": len(daily),
            "annual_rows": len(annual),
            "factor_fingerprint": factor_manifest.get("fingerprint"),
            "qlib_fingerprint": qlib_manifest.get("fingerprint"),
            "side_definitions": {
                "long_excess_daily": "top_quantile_return - benchmark_return",
                "short_excess_daily": "benchmark_return - bottom_quantile_return",
                "annual_long_excess": "equal_weight_pocket_top_return - equal_weight_pocket_benchmark_return",
                "annual_short_excess": "equal_weight_pocket_benchmark_return - equal_weight_pocket_bottom_return",
            },
            "portfolio_aggregation": (
                "Each trading date is assigned to one of holding_period equal-capital pockets. Each pocket compounds "
                "its non-overlapping holding-period returns; pocket terminal wealth is then combined at equal weight."
            ),
            "effectiveness_rule": (
                "A side is effective when it has at least min_effective_days, every required pocket has a valid "
                "return, and its benchmark-adjusted equal-weight pocket return is positive. IC and Rank IC remain "
                "diagnostics."
            ),
            "artifacts": {
                daily_path.name: sha256_file(daily_path),
                summary_path.name: sha256_file(summary_path),
                annual_path.name: sha256_file(annual_path),
            },
        }
        manifest_path = self.output_directory / "factor_evaluation.json"
        atomic_write_json(manifest_path, manifest)
        return {
            **manifest,
            "manifest": str(manifest_path),
            "daily": str(daily_path),
            "summary": str(summary_path),
            "annual": str(annual_path),
        }


def evaluate_factor_frame(
    frame: Any,
    factor_names: list[str] | tuple[str, ...],
    *,
    quantile: float,
    min_cross_section: int,
) -> Any:
    """Return per-date IC and benchmark-adjusted top/bottom attribution for every factor."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas and NumPy are required for factor evaluation") from exc
    required = {"datetime", "instrument", "label", "benchmark", *factor_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataContractError(f"Factor-evaluation frame is missing columns: {missing}")
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be greater than 0 and no greater than 0.5")
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be at least 2")

    working = frame.loc[:, ["datetime", "instrument", "label", "benchmark", *factor_names]].copy()
    working["datetime"] = pd.to_datetime(working["datetime"]).dt.normalize()
    working = working.sort_values(["datetime", "instrument"])
    output = []
    names = list(factor_names)
    for timestamp, group in working.groupby("datetime", sort=True):
        x = group[names].to_numpy(dtype=float)
        labels = group["label"].to_numpy(dtype=float)
        y = np.broadcast_to(labels[:, None], x.shape)
        valid = np.isfinite(x) & np.isfinite(y)
        counts = valid.sum(axis=0)
        eligible = counts >= min_cross_section
        ic = _columnwise_corr(np, x, y, valid, counts, eligible)

        ranked_x = pd.DataFrame(np.where(valid, x, np.nan)).rank(axis=0, method="average").to_numpy()
        ranked_y = pd.DataFrame(np.where(valid, y, np.nan)).rank(axis=0, method="average").to_numpy()
        rank_ic = _columnwise_corr(np, ranked_x, ranked_y, valid, counts, eligible)

        selection_rank = pd.DataFrame(np.where(valid, x, np.nan)).rank(axis=0, method="first").to_numpy()
        group_size = np.floor(counts * quantile).astype(int)
        selectable = eligible & (group_size >= 1) & np.isfinite(ic)
        top_mask = valid & selectable & (selection_rank > (counts - group_size))
        bottom_mask = valid & selectable & (selection_rank <= group_size)
        top_return = _masked_column_mean(np, y, top_mask, group_size, selectable)
        bottom_return = _masked_column_mean(np, y, bottom_mask, group_size, selectable)
        benchmark_values = group["benchmark"].to_numpy(dtype=float)
        finite_benchmark = benchmark_values[np.isfinite(benchmark_values)]
        benchmark_return = float(finite_benchmark[0]) if len(finite_benchmark) else float("nan")
        if len(finite_benchmark) and not np.allclose(finite_benchmark, benchmark_return, rtol=0.0, atol=1e-12):
            raise DataContractError(f"Benchmark return is not unique on {timestamp.date()}")

        for index, factor in enumerate(names):
            top = float(top_return[index])
            bottom = float(bottom_return[index])
            long_excess = top - benchmark_return
            short_excess = benchmark_return - bottom
            output.append(
                {
                    "datetime": timestamp,
                    "factor": factor,
                    "observations": int(counts[index]),
                    "selected_per_side": int(group_size[index]) if selectable[index] else 0,
                    "ic": _finite_or_nan(math, float(ic[index])),
                    "rank_ic": _finite_or_nan(math, float(rank_ic[index])),
                    "top_return": _finite_or_nan(math, top),
                    "benchmark_return": _finite_or_nan(math, benchmark_return),
                    "bottom_return": _finite_or_nan(math, bottom),
                    "long_excess": _finite_or_nan(math, long_excess),
                    "short_excess": _finite_or_nan(math, short_excess),
                    "long_short": _finite_or_nan(math, top - bottom),
                }
            )
    return pd.DataFrame(output)


def assign_staggered_pockets(daily: Any, calendar: Any, *, holding_period: int) -> Any:
    """Assign every evaluated trading date to one of the equal-capital staggered pockets."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas is required for factor evaluation") from exc
    if holding_period < 1:
        raise ValueError("holding_period must be positive")
    if "datetime" not in daily.columns:
        raise DataContractError("Factor-evaluation frame is missing column: datetime")
    normalized_calendar = pd.DatetimeIndex(calendar).normalize()
    if normalized_calendar.empty:
        raise DataContractError("Cannot assign staggered pockets without trading dates")
    if normalized_calendar.has_duplicates:
        raise DataContractError("Trading calendar contains duplicate dates")
    pockets = {timestamp: index % holding_period + 1 for index, timestamp in enumerate(normalized_calendar)}
    result = daily.copy()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    result["pocket"] = result["datetime"].map(pockets)
    if result["pocket"].isna().any():
        missing = result.loc[result["pocket"].isna(), "datetime"].min()
        raise DataContractError(f"Factor-evaluation date is absent from the Qlib trading calendar: {missing.date()}")
    result["pocket"] = result["pocket"].astype(int)
    return result


def summarize_effectiveness(
    daily: Any,
    factor_names: list[str] | tuple[str, ...],
    *,
    min_effective_days: int,
    holding_period: int,
) -> tuple[Any, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas is required for factor evaluation") from exc
    if min_effective_days < 1:
        raise ValueError("min_effective_days must be positive")
    if holding_period < 1:
        raise ValueError("holding_period must be positive")
    if "pocket" not in daily.columns:
        raise DataContractError("Factor-evaluation frame is missing column: pocket")
    frame = daily.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["year"] = frame["datetime"].dt.year
    summary_rows = []
    annual_rows = []
    for factor in factor_names:
        factor_frame = frame[frame["factor"].eq(factor)]
        summary_rows.append(
            {"factor": factor, **_aggregate_period(factor_frame, min_effective_days, holding_period)}
        )
        for year, group in factor_frame.groupby("year", sort=True):
            annual_rows.append(
                {"factor": factor, "year": int(year), **_aggregate_period(group, min_effective_days, holding_period)}
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows)


def _aggregate_period(frame: Any, min_effective_days: int, holding_period: int) -> dict[str, Any]:
    import numpy as np

    def mean_and_ir(column: str) -> tuple[float, float, int]:
        values = frame[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return float("nan"), float("nan"), 0
        deviation = values.std(ddof=1) if len(values) > 1 else float("nan")
        return float(values.mean()), float(values.mean() / deviation) if deviation > 0 else float("nan"), len(values)

    ic_mean, icir, ic_days = mean_and_ir("ic")
    rank_ic_mean, rank_icir, rank_ic_days = mean_and_ir("rank_ic")
    returns = frame.loc[
        :, ["datetime", "pocket", "top_return", "benchmark_return", "bottom_return", "long_excess", "short_excess"]
    ]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    days = len(returns)
    pocket_count = int(returns["pocket"].nunique()) if days else 0
    if days:
        invalid_pockets = returns.loc[
            ~returns["pocket"].isin(range(1, holding_period + 1)),
            "pocket",
        ]
        if len(invalid_pockets):
            raise DataContractError(f"Pocket is outside 1..{holding_period}: {invalid_pockets.iloc[0]}")
        pocket_returns = []
        for _, pocket in returns.sort_values("datetime").groupby("pocket", sort=True):
            pocket_returns.append(
                (
                    _compound(np, pocket["top_return"].to_numpy(dtype=float)),
                    _compound(np, pocket["benchmark_return"].to_numpy(dtype=float)),
                    _compound(np, pocket["bottom_return"].to_numpy(dtype=float)),
                )
            )
        top_return = float(np.mean([value[0] for value in pocket_returns]))
        benchmark_return = float(np.mean([value[1] for value in pocket_returns]))
        bottom_return = float(np.mean([value[2] for value in pocket_returns]))
        long_excess = top_return - benchmark_return
        short_excess = benchmark_return - bottom_return
        long_positive_ratio = float((returns["long_excess"] > 0.0).mean())
        short_positive_ratio = float((returns["short_excess"] > 0.0).mean())
    else:
        top_return = benchmark_return = bottom_return = long_excess = short_excess = float("nan")
        long_positive_ratio = short_positive_ratio = float("nan")
    sufficient = days >= min_effective_days and pocket_count == holding_period
    long_effective = bool(sufficient and np.isfinite(long_excess) and long_excess > 0.0)
    short_effective = bool(sufficient and np.isfinite(short_excess) and short_excess > 0.0)
    if not sufficient:
        effective_side = "insufficient_data"
        dominant_side = "insufficient_data"
    elif long_effective and short_effective:
        effective_side = "both"
        dominant_side = "long" if long_excess > short_excess else "short" if short_excess > long_excess else "equal"
    elif long_effective:
        effective_side = dominant_side = "long"
    elif short_effective:
        effective_side = dominant_side = "short"
    else:
        effective_side = dominant_side = "neither"
    return {
        "start": str(frame["datetime"].min().date()) if len(frame) else None,
        "end": str(frame["datetime"].max().date()) if len(frame) else None,
        "ic_days": ic_days,
        "ic_mean": ic_mean,
        "icir": icir,
        "rank_ic_days": rank_ic_days,
        "rank_ic_mean": rank_ic_mean,
        "rank_icir": rank_icir,
        "return_days": days,
        "pocket_count": pocket_count,
        "required_pockets": holding_period,
        "top_return": top_return,
        "benchmark_return": benchmark_return,
        "bottom_return": bottom_return,
        "long_excess_return": long_excess,
        "short_excess_return": short_excess,
        "long_short_return": top_return - bottom_return,
        "long_positive_day_ratio": long_positive_ratio,
        "short_positive_day_ratio": short_positive_ratio,
        "long_effective": long_effective,
        "short_effective": short_effective,
        "effective_side": effective_side,
        "dominant_side": dominant_side,
    }


def _columnwise_corr(np: Any, x: Any, y: Any, valid: Any, counts: Any, eligible: Any) -> Any:
    scale_x = np.max(np.where(valid, np.abs(x), 0.0), axis=0)
    scale_y = np.max(np.where(valid, np.abs(y), 0.0), axis=0)
    scalable = eligible & (scale_x > 0.0) & (scale_y > 0.0)
    safe_x = np.where(valid & scalable, x / np.where(scale_x > 0.0, scale_x, 1.0), 0.0)
    safe_y = np.where(valid & scalable, y / np.where(scale_y > 0.0, scale_y, 1.0), 0.0)
    divisor = np.where(counts > 0, counts, 1)
    centered_x = np.where(valid, safe_x - safe_x.sum(axis=0) / divisor, 0.0)
    centered_y = np.where(valid, safe_y - safe_y.sum(axis=0) / divisor, 0.0)
    numerator = (centered_x * centered_y).sum(axis=0)
    denominator = np.sqrt((centered_x**2).sum(axis=0) * (centered_y**2).sum(axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=scalable & (denominator > 0.0),
    )


def _masked_column_mean(np: Any, values: Any, mask: Any, group_size: Any, selectable: Any) -> Any:
    return np.divide(
        np.where(mask, values, 0.0).sum(axis=0),
        np.where(group_size > 0, group_size, 1),
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=selectable,
    )


def _compound(np: Any, values: Any) -> float:
    if (values < -1.0).any():
        raise DataContractError("Return below -100% in factor evaluation")
    return float(np.prod(1.0 + values) - 1.0)


def _feature_series(pd: Any, frame: Any, name: str, *, include_instrument: bool) -> Any:
    if frame.empty:
        columns = ["datetime", name]
        if include_instrument:
            columns.insert(1, "instrument")
        return pd.DataFrame(columns=columns)
    result = frame.iloc[:, 0].rename(name).reset_index()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    if include_instrument:
        return result.loc[:, ["datetime", "instrument", name]]
    return result.loc[:, ["datetime", name]].drop_duplicates("datetime", keep="last")


def _load_complete_manifest(path: Path, factor_set: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise DataContractError(f"Required manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise DataContractError(f"Manifest is not complete: {path}")
    if factor_set is not None and payload.get("factor_set") != factor_set:
        raise DataContractError(f"Factor manifest does not describe {factor_set}: {path}")
    return payload


def _atomic_write_frame(frame: Any, destination: Path, *, format: Literal["csv", "parquet"]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=f".{format}.tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if format == "csv":
            frame.to_csv(temporary, index=False)
        else:
            frame.to_parquet(temporary, index=False)
        atomic_replace_file(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _finite_or_nan(math_module: Any, value: float) -> float:
    return value if math_module.isfinite(value) else float("nan")
