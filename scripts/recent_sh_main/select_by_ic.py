"""Reselect recent_sh_main by |IC| and |Rank IC| alone and write its locked module.

Usage: python select_by_ic.py <project_root> <factor_stability.csv> <rank_correlation.parquet> <sampled selection.json>
       <output.py> [min_abs_ic] [min_abs_rank_ic] [max_abs_rank_correlation] [stability|plain] [name] [prefix]

Pass ``stability`` to also keep the earlier recent_sh_main checks (|t| >= 3, oriented Rank IC positive in >= 70% of
years, long side beating the universe in >= 70% of years and on average, coverage >= 0.8); ``min_abs_ic`` 0 drops
the IC threshold.

- ``factor_stability.csv`` is the scored sh_main 2020-01-02..2026-10-21 1d evaluation recent_sh_main was selected
  from (direction, Rank IC, coverage, mean yearly long excess per factor);
- the mean daily IC comes from each evaluation's ``factor_summary.csv`` (runs listed in the current module's
  provenance);
- ``rank_correlation.parquet`` and the sampled ``selection.json`` are the outputs of
  ``scripts/remove_duplicates/select_remove_duplicates.py`` (full 1696-factor matrix and its sampled dates).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.libraries.recent_sh_main import RECENT_SH_MAIN_SELECTION
from rquant.factors.selection import prune_correlated

SOURCES = RECENT_SH_MAIN_SELECTION.source_factor_sets


def main() -> None:
    root = Path(sys.argv[1])
    scored = pd.read_csv(sys.argv[2]).set_index("factor")
    correlation = pd.read_parquet(sys.argv[3])
    sampled = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
    output = Path(sys.argv[5])
    min_ic = float(sys.argv[6]) if len(sys.argv) > 6 else 0.02
    min_rank_ic = float(sys.argv[7]) if len(sys.argv) > 7 else 0.02
    max_corr = float(sys.argv[8]) if len(sys.argv) > 8 else 0.8
    stability = len(sys.argv) > 9 and sys.argv[9] == "stability"
    name = sys.argv[10] if len(sys.argv) > 10 else "recent_sh_main"
    prefix = sys.argv[11] if len(sys.argv) > 11 else "RECENT_SH_MAIN"
    provenance = dict(RECENT_SH_MAIN_SELECTION.provenance)

    summaries = []
    for item in provenance["evaluations"]:
        path = root / "runs" / "factors-evaluate" / item["run_id"] / "factor_summary.csv"
        summaries.append(pd.read_csv(path, usecols=["factor", "ic_mean", "rank_ic_mean"]))
    summary = pd.concat(summaries, ignore_index=True).set_index("factor")
    if summary.index.duplicated().any():
        raise SystemExit("duplicate factors across evaluations")
    names = list(correlation.index)
    columns = ["factor_set", "direction", "rank_ic_mean", "rank_ic_t_stat", "positive_rank_ic_year_ratio",
               "positive_long_excess_year_ratio", "mean_long_excess", "coverage"]
    table = scored.reindex(names)[columns].copy()
    table["ic_mean"] = summary["ic_mean"].reindex(names)
    if not np.allclose(table["rank_ic_mean"], summary["rank_ic_mean"].reindex(names), equal_nan=True):
        raise SystemExit("factor_stability.csv and factor_summary.csv disagree on Rank IC")
    checks = (table["ic_mean"].abs() >= min_ic) & (table["rank_ic_mean"].abs() >= min_rank_ic)
    if stability:
        checks &= (
            (table["rank_ic_t_stat"].abs() >= 3.0)
            & (table["positive_rank_ic_year_ratio"] >= 0.7)
            & (table["positive_long_excess_year_ratio"] >= 0.7)
            & (table["mean_long_excess"] > 0.0)
            & (table["coverage"] >= 0.8)
            & (table["direction"] != 0)
        )
    passing = table[checks]
    strength = passing["mean_long_excess"].fillna(passing["mean_long_excess"].min() - 1.0)
    candidates = correlation.loc[passing.index, passing.index]
    kept, dropped = prune_correlated(strength, candidates, max_abs_correlation=max_corr)
    kept_set = set(kept)
    kept = [name for name in names if name in kept_set]
    per_source = passing.loc[kept].groupby("factor_set").size().to_dict()
    counts = ", ".join(f"{source} {per_source[source]}" for source in SOURCES if per_source.get(source))
    passing_counts = passing.groupby("factor_set").size().to_dict()
    print("passing", len(passing), passing_counts)
    print("kept", len(kept), per_source)
    print("kept mean long excess", float(passing.loc[kept, "mean_long_excess"].mean()))

    members = []
    for factor in kept:
        row = passing.loc[factor]
        direction = int(row["direction"]) or 1
        members.append(
            f'    SelectedFactor("{factor}", {direction}),  # IC {row["ic_mean"]:+.4f}, '
            f"RankIC {row['rank_ic_mean']:+.4f}, long excess {row['mean_long_excess'] * 100:+.1f}%, "
            f"cov {row['coverage']:.3f}"
        )
    paragraphs = [
        f"Locked membership of the {name} factor set.",
        f"{name} keeps the factors whose single-factor results on the Shanghai main-board universe "
        "(``sh_main``) are still significant and stable since 2020, then removes duplicates. It was selected from "
        "the 1d evaluations listed in ``provenance`` (``rquant --universe sh_main factors evaluate --horizon 1d "
        "--start 2020-01-01 --end 2026-10-21`` on each of the twelve library factor sets):",
        (
            "- "
            + " and ".join(
                text
                for text, limit in (
                    (f"the absolute 2020+ mean daily IC is at least ``min_abs_ic`` ({min_ic})", min_ic),
                    (
                        f"the absolute 2020+ mean daily Rank IC is at least ``min_abs_rank_ic`` ({min_rank_ic})",
                        min_rank_ic,
                    ),
                )
                if limit > 0
            )
            + (
                "; its t-statistic (daily Rank ICIR times the square root of the number of days) is at least 3 in "
                "absolute value, the oriented annual Rank IC is positive in at least 70% of the counted years "
                "(5 of 7), the oriented long quantile beats the equal-weight universe in at least 70% of them and on "
                "average, and coverage is at least 0.8"
                if stability
                else "; no t-statistic, yearly, long-side or coverage check applies"
            )
            + f" ({len(passing)} of {len(names)} factors pass);"
        ),
        "- direction is the sign of the 2020+ mean daily Rank IC;",
        "- duplicates are pruned with ``prune_correlated``: passing factors are ordered by their mean yearly long "
        "excess (the oriented top or bottom 20% against the equal-weight sh_main universe, years in "
        "``provenance['universe_annual_returns']``), and a factor is dropped when its absolute mean daily "
        "cross-sectional Spearman correlation with an already kept factor reaches ``max_abs_rank_correlation`` "
        f"({max_corr}). The correlation is the remove_duplicates matrix: every {sampled['correlation_date_step']}th "
        f"trading date from {sampled['correlation_start']} to {sampled['correlation_end']} on the sh_main stores. "
        "``DEDUPLICATED`` maps each dropped factor to the kept one it duplicates.",
        f"Kept per source: {counts}.",
        "The selection uses 2020-2026 sh_main returns, so walk-forward test years from 2020 on are in-sample for "
        "membership, and the membership is only meant for the ``sh_main`` universe.",
        "Members keep their source canonical names and values. ``direction == -1`` marks a factor used inverted; the "
        "stored column is not negated. Changing membership changes the catalog and engine fingerprints, so a stale "
        "store is refused rather than reused.",
    ]
    doc = "\n\n".join(
        textwrap.fill(text, width=118, subsequent_indent="  " if text.startswith("- ") else "") for text in paragraphs
    ).replace("\n\n- ", "\n- ")
    dropped_lines = "\n".join(f'    "{key}": "{value}",' for key, value in sorted(dropped.items()))
    sources = "\n".join(f'        "{source}",' for source in SOURCES)
    annual = "\n".join(
        f"            {year}: {value!r}," for year, value in provenance["universe_annual_returns"].items()
    )
    evaluations = "\n".join(
        "            {\n"
        f'                "factor_set": "{item["factor_set"]}",\n'
        f'                "run_id": "{item["run_id"]}",\n'
        f'                "factor_fingerprint": "{item["factor_fingerprint"]}",\n'
        "            },"
        for item in provenance["evaluations"]
    )
    stability_criteria = (
        '\n        "min_abs_rank_ic_t_stat": 3.0,'
        '\n        "min_positive_rank_ic_year_ratio": 0.7,'
        '\n        "min_positive_long_excess_year_ratio": 0.7,'
        '\n        "min_coverage": 0.8,'
        if stability
        else ""
    )
    text = f'''"""{doc}
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

{prefix}_MEMBERS = (
{chr(10).join(members)}
)

DEDUPLICATED = {{
{dropped_lines}
}}

{prefix}_SELECTION = FactorSelection(
    name="{name}",
    source_factor_sets=(
{sources}
    ),
    members={prefix}_MEMBERS,
    criteria={{
        "universe": "sh_main",
        "min_abs_ic": {min_ic},
        "min_abs_rank_ic": {min_rank_ic},{stability_criteria}
        "deduplication_order": "mean_long_excess",
        "max_abs_rank_correlation": {max_corr},
        "correlation_date_step": {sampled["correlation_date_step"]},
        "correlation_start": "{sampled["correlation_start"]}",
        "correlation_end": "{sampled["correlation_end"]}",
    }},
    provenance={{
        "universe": "sh_main",
        "horizon": "1d",
        "evaluation_start": "{provenance["evaluation_start"]}",
        "evaluation_end": "{provenance["evaluation_end"]}",
        "qlib_fingerprint": "{provenance["qlib_fingerprint"]}",
        "universe_annual_returns": {{
{annual}
        }},
        "deduplicated": DEDUPLICATED,
        "evaluations": (
{evaluations}
        ),
    }},
)
'''
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output}: {len(kept)} members, {len(dropped)} duplicates")


if __name__ == "__main__":
    main()
