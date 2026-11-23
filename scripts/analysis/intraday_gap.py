"""Where an intraday execution backtest's return differs from the daily backtest of the same run.

Usage (project root, rquant environment):

    python scripts/analysis/intraday_gap.py --root . --daily runs/walk-forward/<run> \
        --intraday runs/walk-forward/<run>/intraday/<label> [--start 2020-01-02]

Both books are re-valued at the adjusted daily close of the Qlib provider and their daily P&L is split into
- hold: stocks held from the previous close to this close (previous holdings minus today's sells);
- buy: today's buys, from the fill price to the close;
- sell: today's sells, from the previous close to the fill price;
- cost: commissions and taxes;
- residual: the book's own account change minus the four parts (valuation or bookkeeping that differs from
  ``cash + holdings x daily close``).
Each part is in basis points of the previous account value, averaged per day; the gap table is intraday - daily.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load(directory: Path, start: str | None) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in ("portfolio", "positions", "trades"):
        frame = pd.read_parquet(directory / f"{name}.parquet")
        frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
        frames[name] = frame
    portfolio = frames["portfolio"].set_index("datetime").sort_index()
    if start is not None:
        # keep the day before the window as the base of the first day's change
        before = portfolio.index[portfolio.index < pd.Timestamp(start)]
        first = before[-1] if len(before) else portfolio.index[0]
        portfolio = portfolio[portfolio.index >= first]
    frames["portfolio"] = portfolio
    return frames


def _attribution(book: dict[str, pd.DataFrame], prices: pd.DataFrame) -> pd.DataFrame:
    portfolio = book["portfolio"]
    days = portfolio.index
    held = book["positions"].pivot_table(
        index="datetime", columns="instrument", values="adjusted_amount", aggfunc="sum"
    )
    held = held.reindex(days).fillna(0.0)
    close = prices["close"].unstack("instrument").reindex(index=days, columns=held.columns)
    trades = book["trades"]
    trades = trades[trades["datetime"].isin(days)]
    sign = trades["direction"].map({"buy": 1.0, "sell": -1.0})
    trade_close = prices["close"].reindex(pd.MultiIndex.from_arrays([trades["instrument"], trades["datetime"]]))
    previous_day = pd.Series(days[:-1], index=days[1:])
    previous_close = prices["close"].reindex(
        pd.MultiIndex.from_arrays([trades["instrument"], trades["datetime"].map(previous_day)])
    )
    trades = trades.assign(
        close=trade_close.to_numpy(),
        previous_close=previous_close.to_numpy(),
        signed=sign.to_numpy() * trades["adjusted_amount"].to_numpy(),
    )
    buys = trades[trades["direction"] == "buy"]
    sells = trades[trades["direction"] == "sell"]
    buy_pnl = (buys["adjusted_amount"] * (buys["close"] - buys["price"])).groupby(buys["datetime"]).sum()
    sell_pnl = (sells["adjusted_amount"] * (sells["price"] - sells["previous_close"])).groupby(sells["datetime"]).sum()
    sold = sells.pivot_table(index="datetime", columns="instrument", values="adjusted_amount", aggfunc="sum")
    sold = sold.reindex(index=days, columns=held.columns).fillna(0.0)
    kept = (held.shift(1).fillna(0.0) - sold).clip(lower=0.0)
    hold_pnl = (kept * (close - close.shift(1))).sum(axis=1, min_count=1).fillna(0.0)
    cost = trades.groupby("datetime")["cost"].sum()
    account = portfolio["account"]
    base = account.shift(1)
    parts = pd.DataFrame(
        {
            "hold": hold_pnl,
            "buy": buy_pnl.reindex(days).fillna(0.0),
            "sell": sell_pnl.reindex(days).fillna(0.0),
            "cost": -cost.reindex(days).fillna(0.0),
        }
    )
    parts["residual"] = account.diff() - parts.sum(axis=1)
    parts = parts.div(base, axis=0) * 10_000.0
    parts["net"] = account.pct_change() * 10_000.0
    parts["cash_share"] = portfolio["cash"] / account * 100.0 if "cash" in portfolio.columns else float("nan")
    parts["valuation_gap_bp"] = (
        (account - portfolio.get("cash", 0.0) - (held * close).sum(axis=1)) / account * 10_000.0
        if "cash" in portfolio.columns
        else float("nan")
    )
    return parts.iloc[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--daily", required=True)
    parser.add_argument("--intraday", required=True)
    parser.add_argument("--start", default="2020-01-02")
    args = parser.parse_args()

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    from rquant.qlib_ext.runtime import qlib_init_kwargs

    root = Path(args.root).resolve()
    qlib.init(provider_uri=str(root / "data" / "qlib"), region=REG_CN, **qlib_init_kwargs())
    daily = _load(Path(args.daily), args.start)
    intraday = _load(Path(args.intraday), args.start)
    common = daily["portfolio"].index.intersection(intraday["portfolio"].index)
    daily["portfolio"] = daily["portfolio"].loc[common]
    intraday["portfolio"] = intraday["portfolio"].loc[common]
    codes = sorted(
        set(daily["positions"]["instrument"])
        | set(intraday["positions"]["instrument"])
        | set(daily["trades"]["instrument"])
        | set(intraday["trades"]["instrument"])
    )
    prices = D.features(codes, ["$close"], start_time=common[0], end_time=common[-1], freq="day")
    prices.columns = ["close"]
    prices = prices.swaplevel().sort_index() if prices.index.names[0] == "datetime" else prices
    prices.index = prices.index.set_names(["instrument", "datetime"])

    tables = {name: _attribution(book, prices) for name, book in (("daily", daily), ("intraday", intraday))}
    columns = ["net", "hold", "buy", "sell", "cost", "residual", "cash_share", "valuation_gap_bp"]
    summary = pd.DataFrame({name: table[columns].mean() for name, table in tables.items()}).T
    summary.loc["gap"] = summary.loc["intraday"] - summary.loc["daily"]
    pd.set_option("display.width", 200)
    print("mean per day, basis points of the previous account (cash_share in %)")
    print(summary.round(3).to_string())
    yearly = {}
    for name, table in tables.items():
        yearly[name] = table[["net", "hold", "buy", "sell", "cost", "residual"]].groupby(table.index.year).mean()
    gap = yearly["intraday"] - yearly["daily"]
    print("\nintraday - daily by year, bp per day")
    print(gap.round(3).to_string())
    worst = (tables["intraday"]["net"] - tables["daily"]["net"]).sort_values().head(10)
    print("\nworst days (intraday - daily net, bp)")
    print(worst.round(1).to_string())
    held = {
        name: book["positions"].pivot_table(
            index="datetime", columns="instrument", values="adjusted_amount", aggfunc="sum"
        )
        for name, book in (("daily", daily), ("intraday", intraday))
    }
    overlap = []
    for day in common[:: max(1, len(common) // 200)]:
        a = held["daily"].loc[day].dropna() if day in held["daily"].index else pd.Series(dtype=float)
        b = held["intraday"].loc[day].dropna() if day in held["intraday"].index else pd.Series(dtype=float)
        a, b = set(a[a > 0].index), set(b[b > 0].index)
        if a or b:
            overlap.append(len(a & b) / max(len(a | b), 1))
    print(json.dumps({"holding_name_overlap_mean": sum(overlap) / len(overlap) if overlap else None}))


if __name__ == "__main__":
    main()
