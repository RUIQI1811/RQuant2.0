from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json

UNIVERSE_LABELS = {
    "csi300": "CSI300",
    "main": "Shanghai/Shenzhen main-board (no ST, no first 60 trading days)",
    "sh_main": "Shanghai main-board (no ST, no first 60 trading days)",
    "all": "all A-share",
}

LIMITATIONS = [
    "Long-only point-in-time {universe} portfolio.",
    "No order-book queue model.",
    "No market-impact or participation-capacity model.",
    "Daily bars cannot model intraday execution priority at the opening auction.",
    "Results are research simulations, not live account returns.",
]

YEARLY_PERFORMANCE_DEFINITIONS = {
    "strategy_return": "Compounded daily net return, where net return equals return minus cost.",
    "benchmark_return": "Compounded daily benchmark return.",
    "excess_return": "Strategy return minus benchmark return for the same calendar-year period.",
    "max_drawdown": "Maximum drawdown of the calendar-year net NAV, including the year-opening NAV of 1.0.",
    "average_daily_turnover": "Arithmetic mean of the daily turnover ratio.",
    "turnover_sum": "Sum of the daily turnover ratio over the calendar-year period.",
    "cash_cost": "Increase in cumulative cash trading cost during the calendar-year period.",
    "cost_rate_sum": "Sum of the daily trading-cost ratio over the calendar-year period.",
}


def build_report(run_directory: str | Path) -> dict[str, Any]:
    run_dir = Path(run_directory)
    predictions_path = run_dir / "predictions.parquet"
    if not predictions_path.exists():
        raise DataContractError(f"Predictions do not exist: {predictions_path}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas and pyarrow are required for reporting") from exc

    predictions = pd.read_parquet(predictions_path)
    workflow = None
    walk_forward = run_dir / "walk_forward.json"
    if walk_forward.exists():
        with walk_forward.open("r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    universe = (workflow or {}).get("universe", "csi300")
    summary = {
        "status": "complete",
        "prediction_rows": len(predictions),
        "prediction_start": str(predictions["datetime"].min()),
        "prediction_end": str(predictions["datetime"].max()),
        "instruments": int(predictions["instrument"].nunique()),
        "score_missing": int(predictions["score"].isna().sum()),
        "universe": universe,
        "limitations": [item.format(universe=UNIVERSE_LABELS.get(universe, universe)) for item in LIMITATIONS],
    }
    if workflow is not None:
        summary["workflow"] = workflow
    backtest = run_dir / "backtest.json"
    if backtest.exists():
        with backtest.open("r", encoding="utf-8") as handle:
            summary["backtest"] = json.load(handle)
    portfolio_path = run_dir / "portfolio.parquet"
    if portfolio_path.exists():
        portfolio = pd.read_parquet(portfolio_path)
        summary["yearly_performance"] = {
            "definitions": YEARLY_PERFORMANCE_DEFINITIONS,
            "years": _yearly_performance(pd, portfolio),
        }
    signal_path = run_dir / "signal_analysis.json"
    if signal_path.exists():
        with signal_path.open("r", encoding="utf-8") as handle:
            summary["signal_analysis"] = json.load(handle)
    importance_path = run_dir / "feature_importance.parquet"
    if importance_path.exists():
        summary["feature_importance"] = _feature_importance_summary(pd, pd.read_parquet(importance_path))
    optimizations_path = run_dir / "optimizations.parquet"
    if optimizations_path.exists():
        optimizations = pd.read_parquet(optimizations_path)
        if not optimizations.empty:
            summary["optimizations"] = {
                column: float(optimizations[column].mean())
                for column in ("universe", "holdings", "turnover", "active_share", "tracking_error")
            } | {"rebalances": len(optimizations)}
            if "solved" in optimizations:
                summary["optimizations"]["unsolved"] = int((~optimizations["solved"].astype(bool)).sum())
            if "max_industry_active" in optimizations:
                summary["optimizations"]["max_industry_active"] = float(optimizations["max_industry_active"].max())
                summary["optimizations"]["max_abs_size_active"] = float(optimizations["size_active"].abs().max())
    atomic_write_json(run_dir / "report.json", summary)
    markdown = [
        "# RQuant Research Report",
        "",
        f"- Prediction rows: {summary['prediction_rows']}",
        f"- Prediction period: {summary['prediction_start']} to {summary['prediction_end']}",
        f"- Universe: {universe}",
        f"- Instruments: {summary['instruments']}",
        "",
    ]
    if "backtest" in summary:
        markdown.extend(_backtest_markdown(summary["backtest"]))
    if "optimizations" in summary:
        optimized = summary["optimizations"]
        markdown.extend(
            [
                "## Portfolio optimisation",
                "",
                f"- Rebalances: {optimized['rebalances']}",
                f"- Average universe: {optimized['universe']:.0f} stocks, holdings: {optimized['holdings']:.0f}",
                f"- Average one-way turnover per rebalance: {optimized['turnover'] / 2:.2%}",
                f"- Average active share: {optimized['active_share']:.2%}",
                f"- Average ex-ante tracking error (annualised): {optimized['tracking_error']:.2%}",
            ]
        )
        if "unsolved" in optimized:
            markdown.append(
                f"- Rebalances where the optimiser failed (held the current book, or the benchmark if empty): "
                f"{optimized['unsolved']}"
            )
        if "max_industry_active" in optimized:
            markdown.append(
                f"- Largest active industry weight: {optimized['max_industry_active']:.2%}; largest active size "
                f"exposure: {optimized['max_abs_size_active']:.3f} std"
            )
        markdown.append("")
    if "signal_analysis" in summary:
        markdown.extend(_signal_markdown(summary["signal_analysis"]))
    if "feature_importance" in summary:
        markdown.extend(_importance_markdown(summary["feature_importance"]))
    if "yearly_performance" in summary:
        markdown.extend(
            [
                "## Yearly stability",
                "",
                (
                    "Strategy return compounds daily net returns (`return - cost`). Excess return is the "
                    "strategy return minus the benchmark return for the same period."
                ),
                "",
                (
                    "| Year | Period | Strategy return | Benchmark return | Excess | Max drawdown | "
                    "Avg daily turnover | Turnover sum | Cash cost |"
                ),
                "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["yearly_performance"]["years"]:
            markdown.append(
                "| {year} | {start} to {end} | {strategy} | {benchmark} | {excess} pp | {drawdown} | "
                "{daily_turnover} | {turnover_sum} | {cash_cost} |".format(
                    year=row["year"],
                    start=row["start"],
                    end=row["end"],
                    strategy=_format_percent(row["strategy_return"]),
                    benchmark=_format_percent(row["benchmark_return"]),
                    excess=f"{row['excess_return'] * 100:+.2f}",
                    drawdown=_format_percent(row["max_drawdown"]),
                    daily_turnover=_format_percent(row["average_daily_turnover"]),
                    turnover_sum=f"{row['turnover_sum']:.2f}x",
                    cash_cost=f"{row['cash_cost']:,.2f}",
                )
            )
        markdown.append("")
    markdown.extend(["## Limitations", "", *(f"- {item}" for item in summary["limitations"]), ""])
    (run_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary


METRIC_ROWS = (
    ("Net return (return - cost)", "net_return", "net_return_compounded"),
    ("Benchmark", "benchmark_return", "benchmark_return_compounded"),
    ("Excess return with cost", "excess_return_with_cost", None),
    ("Excess return without cost", "excess_return_without_cost", None),
)


def _backtest_markdown(backtest: dict[str, Any]) -> list[str]:
    """The headline table of Qlib's risk_analysis: compounded annual return and NAV drawdown, the Sharpe (information)
    ratio, and Qlib's default additive annual return and drawdown for comparison with Qlib's benchmarks."""
    metrics = backtest.get("metrics", {})
    lines = ["## Portfolio backtest", ""]
    if not all(isinstance(value, dict) for value in metrics.values()):
        lines.extend([*(f"- {key}: {value}" for key, value in metrics.items()), ""])
        return lines

    def cell(series: str | None, field: str, kind: str = "percent") -> str:
        value = (metrics.get(series) or {}).get(field) if series else None
        if value is None:
            return "n/a"
        return f"{value * 100:+.2f}%" if kind == "percent" else f"{value:.3f}"

    lines.extend(
        [
            "qlib.contrib.evaluate.risk_analysis on daily returns, 252 trading days a year. Compounded columns use "
            "mode `product` (NAV growth and NAV drawdown); the Sharpe ratio is the mean over the standard deviation of "
            "daily returns times sqrt(252) (information ratio for excess returns); the Qlib-sum columns are Qlib's "
            "default mode (mean daily return times 252, drawdown of the summed returns).",
            "",
            "| Series | Annual return | Max drawdown | Sharpe | Annual return (Qlib sum) | Max drawdown (Qlib sum) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, additive, compounded in METRIC_ROWS:
        if additive not in metrics:
            continue
        lines.append(
            f"| {label} | {cell(compounded, 'annualized_return')} | {cell(compounded, 'max_drawdown')} | "
            f"{cell(additive, 'information_ratio', 'ratio')} | {cell(additive, 'annualized_return')} | "
            f"{cell(additive, 'max_drawdown')} |"
        )
    lines.append("")
    return lines


def _yearly_performance(pd: Any, portfolio: Any) -> list[dict[str, Any]]:
    required = {"datetime", "return", "cost", "bench", "turnover", "total_cost"}
    missing = sorted(required.difference(portfolio.columns))
    if missing:
        raise DataContractError(f"Portfolio columns are missing for yearly reporting: {missing}")
    if portfolio.empty:
        raise DataContractError("Portfolio is empty")

    frame = portfolio.loc[:, sorted(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise DataContractError("Portfolio contains invalid datetimes")
    if frame["datetime"].duplicated().any():
        raise DataContractError("Portfolio contains duplicate datetimes")
    frame = frame.sort_values("datetime").reset_index(drop=True)
    numeric_columns = ["return", "cost", "bench", "turnover", "total_cost"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["turnover", "total_cost"]].isna().any().any():
        raise DataContractError("Portfolio turnover or cumulative cash cost contains missing values")
    if (frame["total_cost"].diff().dropna() < -1e-9).any():
        raise DataContractError("Portfolio cumulative cash cost decreases")

    frame["net_return"] = frame["return"].fillna(0.0) - frame["cost"].fillna(0.0)
    frame["benchmark_return"] = frame["bench"].fillna(0.0)
    frame["year"] = frame["datetime"].dt.year

    yearly: list[dict[str, Any]] = []
    previous_total_cost = 0.0
    for year, group in frame.groupby("year", sort=True):
        net_nav = (1.0 + group["net_return"]).cumprod()
        benchmark_nav = (1.0 + group["benchmark_return"]).cumprod()
        running_peak = net_nav.cummax().clip(lower=1.0)
        strategy_return = float(net_nav.iloc[-1] - 1.0)
        benchmark_return = float(benchmark_nav.iloc[-1] - 1.0)
        total_cost = float(group["total_cost"].iloc[-1])
        yearly.append(
            {
                "year": int(year),
                "start": str(group["datetime"].iloc[0].date()),
                "end": str(group["datetime"].iloc[-1].date()),
                "observations": int(len(group)),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
                "max_drawdown": float((net_nav / running_peak - 1.0).min()),
                "average_daily_turnover": float(group["turnover"].mean()),
                "turnover_sum": float(group["turnover"].sum()),
                "cash_cost": total_cost - previous_total_cost,
                "cost_rate_sum": float(group["cost"].fillna(0.0).sum()),
            }
        )
        previous_total_cost = total_cost
    return yearly


IMPORTANCE_TOP = 30


def _feature_importance_summary(pd: Any, frame: Any) -> dict[str, Any]:
    """Average each factor's share of the total gain over the walk-forward windows (one per year, or one per
    ``rolling.retrain_months`` with a ``window`` column), highest first."""
    required = {"prediction_year", "factor", "gain", "split", "gain_share"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"Feature importance columns are missing: {missing}")
    years = int(frame["window" if "window" in frame.columns else "prediction_year"].nunique())
    grouped = frame.groupby("factor", sort=False)
    table = pd.DataFrame(
        {
            "mean_gain_share": grouped["gain_share"].sum() / years,
            "years_used": grouped["split"].apply(lambda values: int((values > 0).sum())),
        }
    ).sort_values("mean_gain_share", ascending=False, kind="mergesort")
    source_names = _source_names()
    return {
        "years": years,
        "factors": int(len(table)),
        "unused_factors": int((table["years_used"] == 0).sum()),
        "top": [
            {
                "factor": str(name),
                "source_name": source_names.get(str(name), str(name)),
                "mean_gain_share": float(row.mean_gain_share),
                "years_used": int(row.years_used),
            }
            for name, row in table.head(IMPORTANCE_TOP).iterrows()
        ],
    }


def _source_names() -> dict[str, str]:
    """Catalog column name (``jq_218``) to the factor's name in its source library (``single_day_VPT``)."""
    from rquant.factors.catalog import get_catalog

    return {spec.canonical_name: spec.source_name for spec in get_catalog().specs}


def _signal_markdown(signal: dict[str, Any]) -> list[str]:
    def value(number: Any, digits: int = 4) -> str:
        return "n/a" if number is None else f"{number:.{digits}f}"

    overall = signal["overall"]
    lines = [
        "## Prediction IC",
        "",
        f"Daily cross-sectional correlation between the score and the label `{signal['label']}`; ICIR is the mean "
        "over the standard deviation of the daily values, not annualised.",
        "",
        "| Period | Days | IC | ICIR | Rank IC | Rank ICIR | Rank IC > 0 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (("All", overall), *((str(item["year"]), item) for item in signal["years"])):
        lines.append(
            f"| {label} | {row['days']} | {value(row['ic'])} | {value(row['ic_ir'], 3)} | {value(row['rank_ic'])} | "
            f"{value(row['rank_ic_ir'], 3)} | {value(row['rank_ic_positive_ratio'], 2)} |"
        )
    lines.append("")
    return lines


def _importance_markdown(importance: dict[str, Any]) -> list[str]:
    lines = [
        "## Feature importance",
        "",
        f"Share of the model's total split gain, averaged over {importance['years']} walk-forward years. "
        f"{importance['unused_factors']} of {importance['factors']} factors were never split on.",
        "",
        "| Rank | Factor | Source name | Mean gain share | Years used |",
        "| ---: | :--- | :--- | ---: | ---: |",
    ]
    for rank, row in enumerate(importance["top"], start=1):
        source = row.get("source_name", row["factor"])
        lines.append(
            f"| {rank} | {row['factor']} | {source} | {row['mean_gain_share']:.2%} | {row['years_used']} |"
        )
    lines.append("")
    return lines


def _format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"
