"""Measure the rank correlation of every factor of the twelve formula libraries (input of remove_duplicates).

Usage: python select_remove_duplicates.py <project_root> <out_dir> [max_abs_rank_correlation] [correlation_date_step]

Writes the full mean cross-sectional Spearman matrix (``rank_correlation.parquet``), each factor's coverage, and a
baseline pruning that uses no return information: factors are ordered by coverage (rounded to two decimals,
higher first) and then by registry order (library, then ordinal), and a factor is dropped when its absolute mean daily
cross-sectional Spearman correlation with an already kept factor reaches the limit. The correlation and coverage are
measured on every ``step``-th trading date of the whole store, one store year at a time. ``order_and_prune.py`` prunes
the same matrix in the order of single-factor results instead, which is what remove_duplicates uses.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.libraries.registry import get_library_registry
from rquant.factors.selection import RankCorrelationAccumulator, prune_correlated

SOURCES = (
    "qlib_alpha158", "wq_alpha101", "gtja191", "qlib_alpha360", "china_chars", "size_value_china",
    "talib", "jq240", "oap", "barra_cne6", "tushare_tech", "jkp153",
)
UNIVERSE = os.environ.get("RD_UNIVERSE", "sh_main")
FACTOR_DIR = "factors" if UNIVERSE == "csi300" else f"factors-{UNIVERSE}"


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    max_corr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    out.mkdir(parents=True, exist_ok=True)
    registry = get_library_registry()
    columns = {name: [spec.canonical_name for lib in registry.select(name) for spec in lib.specs] for name in SOURCES}
    names = [factor for name in SOURCES for factor in columns[name]]
    source_of = {factor: name for name in SOURCES for factor in columns[name]}
    print("factors:", len(names), {name: len(columns[name]) for name in SOURCES}, flush=True)

    store = root / "data" / FACTOR_DIR
    years = sorted(int(path.name.split("=")[1]) for path in (store / SOURCES[0]).glob("year=*"))
    # The sampled dates are every step-th date of the first source's store, over all years.
    all_dates = []
    for year in years:
        part = pd.read_parquet(store / SOURCES[0] / f"year={year}" / "factors.parquet", columns=["datetime"])
        all_dates.append(pd.to_datetime(part["datetime"]).dt.normalize().unique())
    dates = pd.DatetimeIndex(np.sort(np.concatenate(all_dates)))[::step]
    print("years:", years[0], "-", years[-1], "sampled dates:", len(dates), flush=True)

    accumulator = RankCorrelationAccumulator(names)
    observations = pd.Series(0.0, index=names)
    capacity = 0.0
    for year in years:
        started = time.time()
        year_dates = dates[dates.year == year]
        if len(year_dates) == 0:
            continue
        frames = []
        for name in SOURCES:
            path = store / name / f"year={year}" / "factors.parquet"
            part = pd.read_parquet(path, columns=["datetime", "instrument", *columns[name]])
            part["datetime"] = pd.to_datetime(part["datetime"]).dt.normalize()
            part = part[part["datetime"].isin(year_dates)].set_index(["datetime", "instrument"])
            frames.append(part.astype("float32"))
        sample = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
        del frames
        # Coverage as in factor_coverage: per date, the best-covered factor's count is the denominator.
        counts = sample.notna().groupby(level="datetime").sum()
        observations += counts.sum(axis=0)
        capacity += float(counts.max(axis=1).sum())
        accumulator.add(sample.reset_index())
        print(f"{year}: {len(year_dates)} dates, {len(sample)} rows, {time.time() - started:.0f}s", flush=True)
        del sample

    coverage = (observations / capacity).rename("coverage")
    correlation = accumulator.mean()
    order = pd.DataFrame({"factor": names, "coverage": coverage.reindex(names).to_numpy()})
    order["position"] = np.arange(len(order))
    order["rounded"] = order["coverage"].round(2)
    order = order.sort_values(["rounded", "position"], ascending=[False, True], kind="stable")
    # prune_correlated sorts by descending strength with stable ties, so a decreasing score reproduces this order.
    strength = pd.Series(np.arange(len(order), 0, -1, dtype=float), index=order["factor"].to_numpy())
    kept, dropped = prune_correlated(strength, correlation, max_abs_correlation=max_corr)
    kept = [factor for factor in names if factor in set(kept)]  # publish in registry order
    print("kept:", len(kept), flush=True)

    table = pd.DataFrame({"factor": names, "factor_set": [source_of[f] for f in names]})
    table["coverage"] = table["factor"].map(coverage)
    table["kept"] = table["factor"].isin(kept)
    table["duplicate_of"] = table["factor"].map(dropped)
    table["duplicate_correlation"] = [
        correlation.loc[f, d] if isinstance(d, str) else np.nan
        for f, d in zip(table["factor"], table["duplicate_of"], strict=True)
    ]
    table.to_csv(out / "remove_duplicates.csv", index=False)
    correlation.to_parquet(out / "rank_correlation.parquet")
    kept_corr = correlation.loc[kept, kept].abs().to_numpy()
    np.fill_diagonal(kept_corr, np.nan)
    meta = {
        "universe": UNIVERSE,
        "max_abs_rank_correlation": max_corr,
        "correlation_date_step": step,
        "correlation_start": str(dates[0].date()),
        "correlation_end": str(dates[-1].date()),
        "sampled_dates": len(dates),
        "factors": len(names),
        "kept_count": len(kept),
        "kept_per_source": (
            table[table["kept"]].groupby("factor_set").size().reindex(SOURCES).fillna(0).astype(int).to_dict()
        ),
        "max_kept_pair_correlation": float(np.nanmax(kept_corr)),
        "never_correlated": sorted(f for f in names if not np.isfinite(correlation.loc[f].drop(f)).any()),
        "kept": kept,
        "dropped": dropped,
        "coverage": {f: round(float(coverage[f]), 4) for f in names},
    }
    (out / "selection.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("kept_count", "kept_per_source", "max_kept_pair_correlation")}), flush=True)
    for limit in (0.7, 0.8, 0.9, 0.95):
        k, _ = prune_correlated(strength, correlation, max_abs_correlation=limit)
        print(f"limit {limit}: kept {len(k)}", flush=True)


if __name__ == "__main__":
    main()
