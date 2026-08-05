from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json

LIMITATIONS = [
    "Long-only point-in-time CSI300 portfolio.",
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
    summary = {
        "status": "complete",
        "prediction_rows": len(predictions),
        "prediction_start": str(predictions["datetime"].min()),
        "prediction_end": str(predictions["datetime"].max()),
        "instruments": int(predictions["instrument"].nunique()),
        "score_missing": int(predictions["score"].isna().sum()),
        "limitations": LIMITATIONS,
    }
    walk_forward = run_dir / "walk_forward.json"
    if walk_forward.exists():
        with walk_forward.open("r", encoding="utf-8") as handle:
            summary["workflow"] = json.load(handle)
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
    atomic_write_json(run_dir / "report.json", summary)
    markdown = [
        "# RQuant Research Report",
        "",
        f"- Prediction rows: {summary['prediction_rows']}",
        f"- Prediction period: {summary['prediction_start']} to {summary['prediction_end']}",
        f"- Instruments: {summary['instruments']}",
        "",
    ]
    if "backtest" in summary:
        markdown.extend(["## Portfolio backtest", ""])
        for key, value in summary["backtest"].get("metrics", {}).items():
            markdown.append(f"- {key}: {value}")
        markdown.append("")
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
    markdown.extend(["## Limitations", "", *(f"- {item}" for item in LIMITATIONS), ""])
    (run_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary


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


def _format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"
