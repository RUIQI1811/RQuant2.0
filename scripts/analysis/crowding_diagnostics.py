"""Does the post-2022 decline look like crowding? Read-only diagnostics on a finished run and its factors.

Usage (from anywhere; nothing is trained and nothing under data/ is written):

    python scripts/analysis/crowding_diagnostics.py <project_root> <run> [--universe sh_main] [--topk 50]
        [--factors] [--out <dir>]

``<run>`` is a folder under ``<project_root>/runs/walk-forward`` with ``predictions.parquet``. Every signal date t is
traded at the open of t+1 and held to the open of t+2 (the 1d label). Each (t, stock) row gets:

* ``pre``   close t -> open t+1: the overnight move before the buy, which the strategy cannot capture;
* ``day``   open t+1 -> close t+1 and ``night`` close t+1 -> open t+2: the two halves of the captured ``label``;
* ``next``  open t+2 -> open t+3: the day after the holding period (how fast the signal decays);
* ``circ_mv`` float market cap on t (reference daily_basic), for size terciles within each day's scored stocks.

Crowding predicts that more of the signal is priced before we can trade (``pre`` excess of the top-k rises while the
captured ``label`` excess falls), that IC decays fastest where crowded money trades (small caps, price/volume factors),
and that the signal decays faster (``next`` IC shrinks). A regime change (a style crash, a bear market) instead shows up
as a negative small-minus-large spread and a broad IC drop across categories and sizes.

Outputs (CSV, printed too): ``yearly.csv`` (per signal year plus 2013-2021 and 2022+ rows) and, with ``--factors``,
``factor_years.csv`` (each jq240 factor's mean daily Rank IC against the same label on the same stocks, per year, read
from ``data/factors-<universe>/jq240`` a few columns at a time so memory stays small), ``categories.csv`` and
``category_years.csv``: the mean sign-aligned Rank IC per jq240 category, with each factor's sign fixed from its
2013-2021 Rank IC so a factor that flips counts as negative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["datetime", "instrument"]
FIELDS = {
    "pre": "Ref($open, -1) / $close - 1",
    "day": "Ref($close, -1) / Ref($open, -1) - 1",
    "night": "Ref($open, -2) / Ref($close, -1) - 1",
    "label": "Ref($open, -2) / Ref($open, -1) - 1",
    "next": "Ref($open, -3) / Ref($open, -2) - 1",
}
RETURNS = list(FIELDS)
SPLIT_YEAR = 2022
TERCILES = ("small", "mid", "large")


def daily_rank_ic(frame: pd.DataFrame, x: str, y: str, by: list[str] | None = None) -> pd.Series:
    """Spearman correlation of ``x`` and ``y`` on each group of ``by`` (default: each date); NaN rows are dropped."""
    by = by or ["datetime"]
    data = frame.loc[frame[x].notna() & frame[y].notna(), [*by, x, y]]
    groups = data.groupby(by, sort=True, observed=True)
    ranks = pd.DataFrame({x: groups[x].rank(), y: groups[y].rank()})
    keys = [data[key] for key in by]
    centred = ranks - ranks.groupby(keys, observed=True).transform("mean")

    def total(values: pd.Series) -> pd.Series:
        return values.groupby(keys, observed=True).sum()

    covariance = total(centred[x] * centred[y])
    scale = np.sqrt(total(centred[x] ** 2) * total(centred[y] ** 2))
    counts = centred[x].groupby(keys, observed=True).size()
    return (covariance / scale.replace(0.0, np.nan))[counts >= 10]


def add_buckets(frame: pd.DataFrame, topk: int) -> pd.DataFrame:
    """Each day's size tercile (by ``circ_mv`` among the scored stocks) and top-k flag (highest ``score``)."""
    frame = frame.copy()
    days = frame.groupby("datetime")
    size = days["circ_mv"].rank(pct=True)
    codes = np.clip(np.ceil(size.fillna(0.0).to_numpy() * 3), 1, 3).astype(int) - 1
    frame["tercile"] = pd.Categorical.from_codes(np.where(size.isna(), -1, codes), categories=list(TERCILES))
    frame["size_pct"] = size
    frame["top"] = days["score"].rank(ascending=False, method="first") <= topk
    return frame


def _daily_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """Per date: top-k mean minus all-stock mean of every return, small-minus-large tercile label, top-k size pct."""
    days = frame.groupby("datetime")
    everyone = days[RETURNS].mean()
    top = frame[frame["top"]].groupby("datetime")
    spreads = top[RETURNS].mean() - everyone
    spreads.columns = [f"top_{column}" for column in RETURNS]
    by_size = frame.groupby(["datetime", "tercile"], observed=False)["label"].mean().unstack("tercile")
    spreads["small_minus_large"] = by_size["small"] - by_size["large"]
    spreads["top_size_pct"] = top["size_pct"].mean()
    return spreads


def yearly_table(frame: pd.DataFrame, topk: int) -> pd.DataFrame:
    """One row per signal year and per period (2013-2021, 2022+): IC, size-tercile IC and top-k return split."""
    frame = add_buckets(frame, topk)
    daily = pd.DataFrame(
        {
            "rank_ic": daily_rank_ic(frame, "score", "label"),
            "rank_ic_next": daily_rank_ic(frame, "score", "next"),
            "rank_ic_pre": daily_rank_ic(frame, "score", "pre"),
        }
    )
    tercile_ic = daily_rank_ic(frame, "score", "label", by=["datetime", "tercile"]).unstack("tercile")
    for name in TERCILES:
        daily[f"rank_ic_{name}"] = tercile_ic[name] if name in tercile_ic else np.nan
    daily = daily.join(_daily_spreads(frame), how="outer")
    daily.index = pd.DatetimeIndex(daily.index)
    year = daily.index.year
    period = np.where(year < SPLIT_YEAR, f"{year.min()}-{SPLIT_YEAR - 1}", f"{SPLIT_YEAR}+")
    table = pd.concat([daily.groupby(year.astype(str)).mean(), daily.groupby(period).mean()])
    table["days"] = pd.concat([daily.groupby(year.astype(str)).size(), daily.groupby(period).size()])
    for column in [c for c in table.columns if c.startswith("top_") and c != "top_size_pct"] + ["small_minus_large"]:
        table[column] = table[column] * 1e4  # basis points per day
    captured = table["top_label"]
    table["pre_share"] = table["top_pre"] / (table["top_pre"].abs() + captured.abs())
    return table


def category_tables(annual: pd.DataFrame, categories: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean sign-aligned annual Rank IC per category: before/after ``SPLIT_YEAR`` and per year."""
    annual = annual[annual["factor"].isin(categories)].copy()
    annual["category"] = annual["factor"].map(categories)
    before = annual[annual["year"] < SPLIT_YEAR].groupby("factor")["rank_ic_mean"].mean()
    sign = np.sign(before).replace(0.0, 1.0)
    annual["aligned"] = annual["rank_ic_mean"] * annual["factor"].map(sign)
    annual = annual[annual["factor"].isin(sign.index)]
    annual["period"] = np.where(annual["year"] < SPLIT_YEAR, "before", "after")
    per_factor = annual.groupby(["category", "factor", "period"])["aligned"].mean().unstack("period")
    summary = per_factor.groupby("category").agg(
        factors=("before", "size"),
        ic_before=("before", "mean"),
        ic_after=("after", "mean"),
        flipped=("after", lambda values: int((values < 0).sum())),
    )
    summary["retained"] = summary["ic_after"] / summary["ic_before"]
    overall = per_factor.mean()
    summary.loc["ALL"] = [
        len(per_factor), overall["before"], overall["after"], int((per_factor["after"] < 0).sum()),
        overall["after"] / overall["before"],
    ]
    by_year = annual.groupby(["category", "year"])["aligned"].mean().unstack("year")
    return summary.sort_values("retained"), by_year


def factor_years(panel: pd.DataFrame, factor_dir: Path, names: list[str], chunk: int = 16) -> pd.DataFrame:
    """Mean daily Rank IC of each factor against ``panel``'s label on ``panel``'s rows, per partition year.

    Reads ``chunk`` factor columns of one ``year=*/factors.parquet`` at a time (rows of ~1,700 stocks x ~243 days),
    so the peak is a few hundred MB instead of the whole 233-column year.
    """
    labels = panel.loc[panel["label"].notna(), [*KEYS, "label"]]
    rows = []
    for path in sorted(factor_dir.glob("year=*/factors.parquet")):
        year = int(path.parent.name.removeprefix("year="))
        target = labels[labels["datetime"].dt.year == year]
        if target.empty:
            continue
        for start in range(0, len(names), chunk):
            columns = names[start : start + chunk]
            factors = pd.read_parquet(path, columns=[*KEYS, *columns])
            factors["datetime"] = pd.to_datetime(factors["datetime"]).dt.normalize()
            factors["instrument"] = factors["instrument"].astype(str)
            merged = target.merge(factors, on=KEYS, how="inner")
            for name in columns:
                ic = daily_rank_ic(merged, name, "label")
                rows.append({"factor": name, "year": year, "rank_ic_mean": ic.mean(), "days": len(ic)})
        print(f"factors {year}: {len(names)} factors on {len(target)} labelled rows", flush=True)
    return pd.DataFrame(rows)


def jq240_categories() -> dict[str, str]:
    from rquant.factors.libraries.jq240 import JQ240

    return {f"jq_{ordinal:03d}": entry[1] for ordinal, entry in enumerate(JQ240, 1)}


def load_panel(root: Path, run: Path) -> pd.DataFrame:
    """Predictions joined with the return split and float market cap, read one signal year at a time."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    from rquant.qlib_ext.runtime import qlib_init_kwargs

    predictions = pd.read_parquet(run / "predictions.parquet", columns=[*KEYS, "score"])
    predictions["datetime"] = pd.to_datetime(predictions["datetime"]).dt.normalize()
    predictions["instrument"] = predictions["instrument"].astype(str)
    qlib.init(provider_uri=str(root / "data" / "qlib"), region=REG_CN, **qlib_init_kwargs())
    parts = []
    for year, scores in predictions.groupby(predictions["datetime"].dt.year):
        first, last = scores["datetime"].min(), scores["datetime"].max()
        instruments = sorted(scores["instrument"].astype(str).unique())
        features = D.features(instruments, list(FIELDS.values()), start_time=first, end_time=last, freq="day")
        features = features.rename(columns={expression: name for name, expression in FIELDS.items()})[RETURNS]
        features = features.reset_index()
        features["datetime"] = pd.to_datetime(features["datetime"]).dt.normalize()
        features["instrument"] = features["instrument"].astype(str)
        caps = pd.read_parquet(
            root / "data" / "canonical" / "reference" / "daily_basic" / f"year={year}" / "data.parquet",
            columns=[*KEYS, "circ_mv"],
        )
        caps["datetime"] = pd.to_datetime(caps["datetime"]).dt.normalize()
        caps["instrument"] = caps["instrument"].astype(str)
        caps = caps.drop_duplicates(KEYS, keep="last")
        part = scores.merge(features, on=KEYS, how="left").merge(caps, on=KEYS, how="left")
        parts.append(part)
        print(f"{year}: {len(part)} rows, {part['label'].notna().mean():.1%} with label", flush=True)
    return pd.concat(parts, ignore_index=True)


def _print(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.round(4).to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--universe", default="sh_main")
    parser.add_argument("--factors", action="store_true", help="also compute jq240 per-category Rank IC decay")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.root / "runs" / "crowding" / args.run
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    panel = load_panel(args.root, args.root / "runs" / "walk-forward" / args.run)
    yearly = yearly_table(panel, args.topk)
    yearly.to_csv(out / "yearly.csv")
    _print(f"Model signal per signal year (top{args.topk}; top_* and small_minus_large in bp/day)", yearly)

    if args.factors:
        categories = jq240_categories()
        factor_dir = args.root / "data" / ("factors" if args.universe == "csi300" else f"factors-{args.universe}")
        annual = factor_years(panel, factor_dir / "jq240", list(categories))
        annual.to_csv(out / "factor_years.csv", index=False)
        summary, by_year = category_tables(annual, categories)
        summary.to_csv(out / "categories.csv")
        by_year.to_csv(out / "category_years.csv")
        _print(f"jq240 sign-aligned Rank IC by category: before {SPLIT_YEAR} vs {SPLIT_YEAR}+", summary)
        _print("jq240 sign-aligned Rank IC by category and year", by_year)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
