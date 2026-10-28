"""Backtest metrics of finished runs over any date window, read from the saved run outputs (no retraining).

Usage: python scripts/analysis/window_metrics.py <runs_root> <start> [end] label=run_dir [label=run_dir ...]

Each ``run_dir`` is a folder under ``<runs_root>/walk-forward`` that ``rquant report`` has filled. Its
``portfolio.parquet`` holds the daily account path (return, cost, turnover, benchmark), so any window is a slice of it.
Net daily return = return - cost, as the reports use. Sharpe = mean * 252 / (std * sqrt(252)); excess and IR are
against the report benchmark (``bench``, SH000300).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


def window_metrics(run_dir: Path, start: str, end: str) -> dict[str, float]:
    port = pd.read_parquet(run_dir / "portfolio.parquet")
    port["datetime"] = pd.to_datetime(port["datetime"])
    port = port[(port["datetime"] >= start) & (port["datetime"] <= end)].set_index("datetime")
    net = port["return"] - port["cost"]
    excess = net - port["bench"].fillna(0.0)
    nav = (1 + net).cumprod()
    annual = float(nav.iloc[-1] ** (252 / len(net)) - 1)
    drawdown = float((nav / nav.cummax() - 1).min())
    return {
        "annual": annual,
        "sharpe": float(net.mean() * 252 / (net.std(ddof=1) * math.sqrt(252))),
        "max_drawdown": drawdown,
        "calmar": annual / abs(drawdown),
        "cumulative": float(nav.iloc[-1] - 1),
        "excess_annual": float(excess.mean() * 252),
        "information_ratio": float(excess.mean() / excess.std(ddof=1) * math.sqrt(252)),
        "turnover": float(port["turnover"].mean()),
        "cost_annual": float(port["cost"].mean() * 252),
        "days": len(net),
    }


def main() -> None:
    runs_root, start = Path(sys.argv[1]), sys.argv[2]
    rest = sys.argv[3:]
    end = rest.pop(0) if rest and "=" not in rest[0] else "2100-01-01"
    rows = {}
    for item in rest:
        label, run = item.split("=", 1)
        rows[label] = window_metrics(runs_root / "walk-forward" / run, start, end)
    table = pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
    for column in ("annual", "max_drawdown", "cumulative", "excess_annual", "turnover", "cost_annual"):
        table[column] = (table[column] * 100).map("{:+.1f}%".format)
    for column in ("sharpe", "calmar", "information_ratio"):
        table[column] = table[column].map("{:.2f}".format)
    table["days"] = table["days"].astype(int)
    pd.set_option("display.width", 250)
    print(f"window {start} .. {end}")
    print(table.to_string())


if __name__ == "__main__":
    main()
