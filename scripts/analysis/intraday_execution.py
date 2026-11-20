"""Intraday execution of a walk-forward run's daily book on Tushare minute bars (Qlib NestedExecutor).

Run from the project root in the rquant environment. Steps, each resumable:

    # 1. Which stock-days need minute bars, and how many Tushare requests that is (no requests are made).
    python scripts/analysis/intraday_execution.py plan --run runs/walk-forward/<run> --start 2020-01-01

    # 2. Fetch them (needs the Tushare "A股历史分钟" permission; cached, so an interrupted sync resumes).
    python scripts/analysis/intraday_execution.py sync --run runs/walk-forward/<run> --start 2020-01-01

    # 3. Restore and check the bars, write data/minutes/1min/month=YYYYMM/bars.parquet and the minute calendar.
    python scripts/analysis/intraday_execution.py build

    # 4. Backtest one execution variant (repeat with other windows); writes <run>/intraday/<label>/.
    python scripts/analysis/intraday_execution.py backtest --run runs/walk-forward/<run> \
        --sell-window 09:30-09:59 --buy-window 09:30-09:59 --start 2020-01-02

``plan`` uses the run's predictions and, when present, the daily backtest's positions.parquet and trades.parquet
(run ``rquant report`` first with the same top-k). Windows are minute-bar start labels, both ends included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from rquant.data.minutes import (
    PLAN_FILE,
    MinuteBarBuilder,
    monthly_requests,
    plan_minute_requests,
    sync_minutes,
)
from rquant.workflow.intraday import IntradayBacktestRunner, IntradayConfig


def _paths(root: Path) -> dict[str, Path]:
    data = root / "data"
    return {
        "raw": data / "raw",
        "canonical": data / "canonical",
        "qlib": data / "qlib",
        "minutes": data / "minutes" / "1min",
    }


def _calendar(paths: dict[str, Path]) -> pd.DatetimeIndex:
    text = (paths["qlib"] / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    return pd.DatetimeIndex(pd.to_datetime(text))


def _plan(args: argparse.Namespace, paths: dict[str, Path]) -> pd.DataFrame:
    run = Path(args.run)
    predictions = pd.read_parquet(run / "predictions.parquet", columns=["datetime", "instrument", "score"])
    positions = pd.read_parquet(run / "positions.parquet") if (run / "positions.parquet").exists() else None
    trades = pd.read_parquet(run / "trades.parquet") if (run / "trades.parquet").exists() else None
    if positions is None:
        print("warning: no positions.parquet in the run; planning from predictions alone", file=sys.stderr)
    # Books of earlier intraday backtests: their holdings and trades, to fetch the minute bars they fell back on.
    for book in args.book or []:
        book = Path(book)
        extra_positions = pd.read_parquet(book / "positions.parquet", columns=["datetime", "instrument"])
        extra_trades = pd.read_parquet(book / "trades.parquet", columns=["datetime", "instrument"])
        positions = extra_positions if positions is None else pd.concat([positions, extra_positions], ignore_index=True)
        trades = extra_trades if trades is None else pd.concat([trades, extra_trades], ignore_index=True)
    plan = plan_minute_requests(
        predictions,
        _calendar(paths),
        top=args.top,
        positions=positions,
        trades=trades,
        start=args.start,
        end=args.end,
    )
    requests = monthly_requests(plan)
    paths["minutes"].mkdir(parents=True, exist_ok=True)
    plan.to_parquet(paths["minutes"] / PLAN_FILE, index=False)
    per_month = requests.groupby("month")["instrument"].nunique()
    print(
        json.dumps(
            {
                "stock_days": len(plan),
                "instruments": int(plan["instrument"].nunique()),
                "months": int(requests["month"].nunique()),
                "requests": len(requests),
                "instruments_per_month_median": float(per_month.median()) if len(per_month) else 0,
                "estimated_rows_millions": round(len(plan) * 241 / 1e6, 1),
                "hours_at_rate": round(len(requests) / args.max_requests_per_minute / 60, 1),
            },
            indent=2,
        )
    )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="project root holding data/ and runs/")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "sync"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, help="walk-forward run directory")
        command.add_argument("--top", type=int, default=200, help="top-ranked stocks per day to cover")
        command.add_argument("--start", default="2020-01-01")
        command.add_argument("--end", default=None)
        command.add_argument("--max-requests-per-minute", type=int, default=190)
        command.add_argument(
            "--book", action="append", help="an intraday backtest output dir whose holdings and trades to cover too"
        )
        if name == "sync":
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--limit", type=int, default=None, help="fetch only the first N requests (a trial)")
    commands.add_parser("build")
    backtest = commands.add_parser("backtest")
    backtest.add_argument("--run", required=True, help="walk-forward run directory")
    backtest.add_argument("--topk", type=int, default=50)
    backtest.add_argument("--n-drop", type=int, default=5)
    backtest.add_argument("--capital", type=float, default=1_000_000.0)
    backtest.add_argument("--buy-window", default="09:30-09:59")
    backtest.add_argument("--sell-window", default="09:30-09:59")
    backtest.add_argument("--start", default=None)
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--output", default=None, help="default <run>/intraday/<label>")
    backtest.add_argument(
        "--from-empty", action="store_true", help="start with cash only instead of the daily book of the day before"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = _paths(root)
    began = time.time()
    if args.command == "plan":
        _plan(args, paths)
    elif args.command == "sync":
        from rquant.data.collector import TushareCollector

        requests = _plan(args, paths)
        if args.limit is not None:
            requests = requests.head(args.limit)
        collector = TushareCollector(
            paths["raw"], token=os.environ.get("TUSHARE_TOKEN"), max_requests_per_minute=args.max_requests_per_minute
        )
        manifest = sync_minutes(collector, requests, workers=args.workers, show_progress=True)
        summary = {key: manifest[key] for key in ("requests", "cached", "fetched", "empty", "rows")}
        print(json.dumps(summary, indent=2))
    elif args.command == "build":
        manifest = MinuteBarBuilder(paths["raw"], paths["canonical"], paths["minutes"]).build(log=print)
        summary = {key: manifest[key] for key in ("months", "stock_days", "amount_mismatch_share", "volume_units")}
        print(json.dumps(summary, indent=2))
    else:
        config = IntradayConfig(
            initial_capital=args.capital,
            topk=args.topk,
            n_drop=args.n_drop,
            buy_window=args.buy_window,
            sell_window=args.sell_window,
            start=args.start,
            end=args.end,
            start_from_daily=not args.from_empty,
        )
        run = Path(args.run)
        output = Path(args.output) if args.output else run / "intraday" / config.label
        manifest = IntradayBacktestRunner(
            qlib_root=paths["qlib"],
            minute_root=paths["minutes"],
            run_directory=run,
            output_directory=output,
            config=config,
            log=lambda message: print(message, flush=True),
        ).run()
        net = manifest["metrics"]["net_return"]
        print(
            json.dumps(
                {
                    "output": str(output),
                    "annualized_return": net.get("annualized_return"),
                    "information_ratio": net.get("information_ratio"),
                    "max_drawdown": net.get("max_drawdown"),
                    "execution_vs_open_bp": manifest["execution_vs_open_bp"],
                    "start_book": manifest["start_book"],
                    "uncovered_holding_days": manifest["uncovered_holding_days"],
                    "uncovered_share": manifest["uncovered_share"],
                    "unfilled_orders": manifest["unfilled_orders"],
                    "fallback_trades": manifest["fallback_trades"],
                    "fallback_trade_value_share": manifest["fallback_trade_value_share"],
                },
                indent=2,
            )
        )
    print(f"done in {time.time() - began:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
