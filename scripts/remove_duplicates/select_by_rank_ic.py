"""Pick the strongest factors of some libraries by Rank IC, drop near duplicates, and write the locked selection module.

Usage: python select_by_rank_ic.py <project_root> <out_dir> <module.py> <name> <PREFIX> <sources,comma,separated>
       <universe> <start> <end> <summary.csv,...> [top_n] [max_abs_rank_correlation] [min_coverage] [date_step]
       [min_abs_rank_ic]

``top_n`` 0 keeps every factor that survives the filters and the pruning; ``settings.json`` in ``out_dir`` overrides
``top_n`` and ``min_abs_rank_ic``.

Unlike ``select_by_coverage.py`` this selection USES RETURNS. Each ``summary.csv`` is the ``factor_summary.csv`` of a
``rquant factors evaluate --horizon 1d --start <start> --end <end>`` run on ``<universe>`` (one per source, in source
order); its ``rank_ic_mean`` is the mean daily Rank IC of the factor against the walk-forward label (next open to the
open after). Coverage and the mean daily cross-sectional Spearman correlation of every pair are measured here on the
``<universe>`` stores, on every ``date_step``-th trading date from ``start`` to ``end``, one store year at a time.
Factors with coverage below ``min_coverage`` or |Rank IC| below ``min_abs_rank_ic`` are dropped, the rest are
ordered by |Rank IC| (higher first, then registry order), ``prune_correlated`` drops a factor whose correlation with an
already kept one reaches the limit, and every kept factor (or the first ``top_n`` when positive) is a member. When
the selection window overlaps a backtest, that backtest is in sample.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.libraries.registry import get_library_registry
from rquant.factors.selection import RankCorrelationAccumulator, prune_correlated


def main() -> None:
    root, out, module = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    name, prefix = sys.argv[4], sys.argv[5]
    sources = tuple(sys.argv[6].split(","))
    universe, start, end = sys.argv[7], pd.Timestamp(sys.argv[8]), pd.Timestamp(sys.argv[9])
    summaries = [Path(path) for path in sys.argv[10].split(",")]
    top_n = int(sys.argv[11]) if len(sys.argv) > 11 else 100
    limit = float(sys.argv[12]) if len(sys.argv) > 12 else 0.8
    min_coverage = float(sys.argv[13]) if len(sys.argv) > 13 else 0.8
    step = int(sys.argv[14]) if len(sys.argv) > 14 else 5
    min_abs_ic = float(sys.argv[15]) if len(sys.argv) > 15 else 0.0
    # A settings.json in out_dir overrides top_n and min_abs_rank_ic (the 2026-11-19 queue was launched with top_n 100
    # before the owner dropped the cap and asked for |Rank IC| >= 0.01).
    settings_path = out / "settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        top_n = int(settings.get("top_n", top_n))
        min_abs_ic = float(settings.get("min_abs_rank_ic", min_abs_ic))
        print("settings.json:", settings, flush=True)
    if len(summaries) != len(sources):
        raise SystemExit("one factor_summary.csv per source, in source order")
    out.mkdir(parents=True, exist_ok=True)

    registry = get_library_registry()
    columns = {s: [spec.canonical_name for lib in registry.select(s) for spec in lib.specs] for s in sources}
    names = [factor for s in sources for factor in columns[s]]
    source_of = {factor: s for s in sources for factor in columns[s]}
    rank_ic = pd.concat(
        [
            pd.read_csv(path).set_index("factor")["rank_ic_mean"].reindex(columns[s])
            for s, path in zip(sources, summaries, strict=True)
        ]
    )
    print("candidates:", len(names), {s: len(columns[s]) for s in sources}, flush=True)

    store = root / "data" / f"factors-{universe}"
    years = [year for year in range(start.year, end.year + 1) if (store / sources[0] / f"year={year}").exists()]
    all_dates = []
    for year in years:
        part = pd.read_parquet(store / sources[0] / f"year={year}" / "factors.parquet", columns=["datetime"])
        all_dates.append(pd.to_datetime(part["datetime"]).dt.normalize().unique())
    dates = pd.DatetimeIndex(np.sort(np.concatenate(all_dates)))
    dates = dates[(dates >= start) & (dates <= end)][::step]
    print("sampled dates:", len(dates), dates[0].date(), "-", dates[-1].date(), flush=True)

    accumulator = RankCorrelationAccumulator(names)
    observations = pd.Series(0.0, index=names)
    capacity = 0.0
    for year in years:
        began = time.time()
        year_dates = dates[dates.year == year]
        if len(year_dates) == 0:
            continue
        frames = []
        for s in sources:
            part = pd.read_parquet(
                store / s / f"year={year}" / "factors.parquet", columns=["datetime", "instrument", *columns[s]]
            )
            part["datetime"] = pd.to_datetime(part["datetime"]).dt.normalize()
            part = part[part["datetime"].isin(year_dates)].set_index(["datetime", "instrument"])
            frames.append(part.astype("float32"))
        sample = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
        del frames
        counts = sample.notna().groupby(level="datetime").sum()
        observations += counts.sum(axis=0)
        capacity += float(counts.max(axis=1).sum())
        accumulator.add(sample.reset_index())
        print(f"{year}: {len(year_dates)} dates, {len(sample)} rows, {time.time() - began:.0f}s", flush=True)
        del sample
    coverage = (observations / capacity).reindex(names)
    correlation = accumulator.mean()

    covered = [f for f in names if coverage[f] >= min_coverage and np.isfinite(rank_ic[f])]
    eligible = [f for f in covered if abs(rank_ic[f]) >= min_abs_ic]
    strength = rank_ic[eligible].abs()
    kept_all, dropped_all = prune_correlated(strength, correlation.loc[eligible, eligible], max_abs_correlation=limit)
    kept = kept_all[:top_n] if top_n > 0 else kept_all
    for threshold in (0.01, 0.02, 0.03):
        candidates = [f for f in covered if abs(rank_ic[f]) >= threshold]
        survivors, _ = prune_correlated(
            rank_ic[candidates].abs(), correlation.loc[candidates, candidates], max_abs_correlation=limit
        )
        print(f"|Rank IC| >= {threshold}: {len(candidates)} eligible, {len(survivors)} after pruning", flush=True)
    kept_set = set(kept)
    direction = np.sign(rank_ic).replace(0, 1).fillna(1).astype(int)
    per_source = {s: sum(source_of[f] == s for f in kept) for s in sources}
    sub = correlation.loc[kept, kept].abs().to_numpy()
    np.fill_diagonal(sub, np.nan)
    print(f"eligible (coverage >= {min_coverage:.2f}):", len(eligible), "kept after pruning:", len(kept_all))
    print("members:", len(kept), per_source, "max member pair correlation", round(float(np.nanmax(sub)), 4))
    print("top 10:", [(f, round(float(rank_ic[f]), 4)) for f in kept[:10]])

    table = pd.DataFrame({
        "factor": names,
        "factor_set": [source_of[f] for f in names],
        "coverage": coverage.to_numpy(),
        "rank_ic_mean": rank_ic.reindex(names).to_numpy(),
    })
    table["eligible"] = table["factor"].isin(eligible)
    table["member"] = table["factor"].isin(kept_set)
    table["rank"] = table["factor"].map({f: i + 1 for i, f in enumerate(kept)})
    table["duplicate_of"] = table["factor"].map(dropped_all)
    table.to_csv(out / f"{name}.csv", index=False)
    meta = {
        "name": name,
        "sources": sources,
        "universe": universe,
        "selection_start": str(start.date()),
        "selection_end": str(end.date()),
        "correlation_date_step": step,
        "sampled_dates": len(dates),
        "min_coverage": min_coverage,
        "min_abs_rank_ic": min_abs_ic,
        "max_abs_rank_correlation": limit,
        "top_n": top_n,
        "candidates": len(names),
        "eligible": len(eligible),
        "kept_after_pruning": len(kept_all),
        "members": kept,
        "members_per_source": per_source,
        "rank_ic": {f: float(rank_ic[f]) for f in kept},
        "max_member_pair_correlation": float(np.nanmax(sub)),
        "summaries": [str(path) for path in summaries],
    }
    (out / "selection.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    counts = ", ".join(f"{s} {per_source[s]} of {len(columns[s])}" for s in sources)
    paragraphs = [
        f"Locked membership of the {name} factor set. THIS SELECTION USES RETURNS.",
        f"{name} keeps {len(kept)} factors of {', '.join(sources)} ({len(names)} candidates) by mean daily Rank IC, "
        "with near duplicates removed:",
        f"- Rank IC: ``rquant factors evaluate --horizon 1d`` on the ``{universe}`` stores from {start.date()} to "
        f"{end.date()} (the walk-forward label, next open to the open after);",
        f"- factors covering less than {min_coverage} of the best-covered factor's observations, or with |Rank IC| "
        f"below {min_abs_ic}, are left out;",
        f"- the absolute mean daily cross-sectional Spearman correlation of every pair is measured on every {step}th "
        f"trading date of the same window ({len(dates)} dates);",
        "- factors are ordered by |Rank IC| (higher first, then registry order) and ``prune_correlated`` drops a "
        f"factor whose correlation with an already kept factor reaches {limit}; "
        + (f"the first {top_n} kept factors are members." if top_n > 0 else "every kept factor is a member."),
        f"Members per source: {counts}. Built with ``scripts/remove_duplicates/select_by_rank_ic.py``.",
        f"The selection window overlaps every backtest from {start.date()} on, so those backtests are in sample and "
        "overstate what the set can do out of sample. ``direction`` is the sign of the Rank IC and is metadata only.",
    ]
    doc = "\n\n".join(
        textwrap.fill(text, width=118, subsequent_indent="  " if text.startswith("- ") else "") for text in paragraphs
    ).replace("\n\n- ", "\n- ")
    members = "\n".join(f'    SelectedFactor("{f}", {int(direction[f])}),' for f in kept)
    source_lines = "\n".join(f'        "{s}",' for s in sources)
    rank_lines = "\n".join(f'    "{f}": {float(rank_ic[f]):.6f},' for f in kept)
    module.write_text(
        f'''"""{doc}
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

{prefix}_MEMBERS = (
{members}
)

# Mean daily Rank IC of each member over the selection window.
RANK_IC = {{
{rank_lines}
}}

{prefix}_SELECTION = FactorSelection(
    name="{name}",
    source_factor_sets=(
{source_lines}
    ),
    members={prefix}_MEMBERS,
    criteria={{
        "universe": "{universe}",
        "returns_used": True,
        "in_sample": True,
        "score": "mean daily Rank IC, 1d label",
        "selection_start": "{start.date()}",
        "selection_end": "{end.date()}",
        "min_coverage": {min_coverage},
        "min_abs_rank_ic": {min_abs_ic},
        "max_abs_rank_correlation": {limit},
        "top_n": {top_n if top_n > 0 else None},
        "correlation_date_step": {step},
    }},
    provenance={{
        "universe": "{universe}",
        "candidates": {len(names)},
        "eligible": {len(eligible)},
        "rank_ic": RANK_IC,
    }},
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {module}: {len(kept)} members")


if __name__ == "__main__":
    main()
