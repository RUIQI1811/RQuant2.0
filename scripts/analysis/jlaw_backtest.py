"""Backtest the rule-based J Law approximation (``rquant.workflow.jlaw``) on canonical bars, in several variants.

Usage (from the project root, in the rquant environment):

    python scripts/analysis/jlaw_backtest.py --root . --universe sh_main --start 2013-01-04 --end 2026-10-21

It reads ``data/canonical`` only (no factor store, no model) and writes ``runs/jlaw/<UTC time>/``:

* ``summary.csv``: annual return, Sharpe, max drawdown, trade statistics per variant and window
  (``full``: start..end; ``2022+``: the same run from 2022-01-04; ``fresh2022``: a new account started 2022-01-04);
* ``<variant>_daily.csv``, ``<variant>_trades.csv``, ``<variant>_open.csv`` for the full run;
* ``params.json`` and ``rules.md`` (every rule with its source, J Law's own words or an approximation).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rquant.workflow.jlaw import RULES, JLawParams, compute_signals, load_panels, run_backtest, summarize

VARIANTS = {
    "base": {},
    "pullback": {"enable_pullback": True},
    "no_market_filter": {"market_filter": False},
    "no_pyramid": {"add_fractions": ()},
    "no_beta": {"min_beta": None},
    "risk_0.5pct": {"risk_per_trade": 0.005, "risk_reduced": 0.0025},
}
FRESH_START = "2022-01-04"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="project root holding data/ and runs/")
    parser.add_argument("--universe", default="sh_main")
    parser.add_argument("--start", default="2013-01-04")
    parser.add_argument("--end", default="2026-10-21")
    parser.add_argument("--capital", type=float, default=250_000.0)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="repeat; default all")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    warmup_start = pd.Timestamp(args.start) - pd.DateOffset(years=2)
    began = time.time()
    panels, benchmark = load_panels(root / "data" / "canonical", args.universe, warmup_start, args.end)
    print(
        f"loaded {panels['close'].shape[1]} instruments x {panels['close'].shape[0]} days "
        f"in {time.time() - began:.0f}s",
        flush=True,
    )

    out = root / "runs" / "jlaw" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: UP017
    out.mkdir(parents=True, exist_ok=False)
    with (out / "rules.md").open("w", encoding="utf-8") as handle:
        handle.write("| rule | source | whose |\n| --- | --- | --- |\n")
        for rule, source, whose in RULES:
            handle.write(f"| {rule} | {source} | {whose} |\n")

    rows = []
    all_params = {}
    for name in args.variant or list(VARIANTS):
        params = JLawParams(**{**JLawParams().to_dict(), **VARIANTS[name], "initial_capital": args.capital})
        all_params[name] = params.to_dict()
        began = time.time()
        signals = compute_signals(panels, benchmark, params)
        result = run_backtest(panels, benchmark, params, args.start, args.end, signals=signals)
        result.daily.to_csv(out / f"{name}_daily.csv", index=False)
        result.trades.to_csv(out / f"{name}_trades.csv", index=False)
        result.open_positions.to_csv(out / f"{name}_open.csv", index=False)
        windows = [("full", result.daily, result.trades, None)]
        if pd.Timestamp(args.start) < pd.Timestamp(FRESH_START) < pd.Timestamp(args.end):
            fresh = run_backtest(panels, benchmark, params, FRESH_START, args.end, signals=signals)
            windows += [
                ("2022+", result.daily, result.trades, FRESH_START),
                ("fresh2022", fresh.daily, fresh.trades, None),
            ]
        for window, daily, trades, start in windows:
            rows.append({"variant": name, "window": window, **summarize(daily, trades, start=start)})
        print(f"{name}: {time.time() - began:.0f}s", flush=True)

    (out / "params.json").write_text(
        json.dumps({"universe": args.universe, "start": args.start, "end": args.end, "variants": all_params}, indent=2),
        encoding="utf-8",
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    shown = summary.copy()
    for column in ("annual", "max_drawdown", "cumulative", "bench_annual", "excess_annual", "exposure", "win_rate"):
        if column in shown:
            shown[column] = (shown[column] * 100).map("{:+.1f}%".format)
    for column in ("sharpe", "calmar", "avg_win_r", "avg_loss_r", "avg_holding_days"):
        if column in shown:
            shown[column] = shown[column].map("{:.2f}".format)
    pd.set_option("display.width", 250)
    print(shown.drop(columns=["days"]).to_string(index=False))
    print(f"run_directory: {out}")


if __name__ == "__main__":
    main()
