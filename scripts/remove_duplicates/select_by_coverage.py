"""Deduplicate the union of some libraries with no return information and write its locked selection module.

Usage: python select_by_coverage.py <project_root> <out_dir> <module.py> <name> <PREFIX> <sources,comma,separated>
       [max_abs_rank_correlation]

Candidates are every factor of the given libraries. They are ordered by coverage (rounded to two decimals, higher
first) and then by registry order (library in the given order, then ordinal), as ``select_remove_duplicates.py`` does,
and ``prune_correlated`` drops a factor when its absolute mean daily cross-sectional Spearman correlation with an
already kept factor reaches the limit. Correlation and coverage come from the full sh_main matrix
``runs/remove_duplicates-selection/out`` (``select_remove_duplicates.py``). Directions are the sign of the sh_main
2020+ Rank IC in ``runs/recent_sh_main-selection/factor_stability.csv``; they are metadata only and do not affect which
factor is kept.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.libraries.registry import get_library_registry
from rquant.factors.selection import prune_correlated


def main() -> None:
    root, out, module = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    name, prefix = sys.argv[4], sys.argv[5]
    sources = tuple(sys.argv[6].split(","))
    limit = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8
    out.mkdir(parents=True, exist_ok=True)
    registry = get_library_registry()
    source_of = {spec.canonical_name: s for s in sources for lib in registry.select(s) for spec in lib.specs}
    names = list(source_of)
    base = root / "runs" / "remove_duplicates-selection" / "out"
    sampled = json.loads((base / "selection.json").read_text(encoding="utf-8"))
    correlation = pd.read_parquet(base / "rank_correlation.parquet").loc[names, names]
    coverage = pd.Series(sampled["coverage"]).reindex(names)
    order = pd.DataFrame({"factor": names, "rounded": coverage.round(2).to_numpy(), "position": np.arange(len(names))})
    order = order.sort_values(["rounded", "position"], ascending=[False, True], kind="stable")
    # prune_correlated keeps the strongest first with stable ties, so a decreasing score reproduces this order.
    strength = pd.Series(np.arange(len(order), 0, -1, dtype=float), index=order["factor"].to_numpy())
    kept, dropped = prune_correlated(strength, correlation, max_abs_correlation=limit)
    kept_set = set(kept)
    kept = [factor for factor in names if factor in kept_set]
    scored = pd.read_csv(root / "runs" / "recent_sh_main-selection" / "factor_stability.csv").set_index("factor")
    direction = scored["direction"].reindex(names).fillna(0).astype(int).replace(0, 1)
    sub = correlation.loc[kept, kept].abs().to_numpy()
    np.fill_diagonal(sub, np.nan)
    candidates = {s: sum(source_of[f] == s for f in names) for s in sources}
    per_source = {s: sum(source_of[f] == s for f in kept) for s in sources}
    at = {lim: len(prune_correlated(strength, correlation, max_abs_correlation=lim)[0]) for lim in (0.7, 0.9)}
    print("candidates:", len(names), candidates)
    print(f"kept at {limit}:", len(kept), per_source, "max kept pair correlation", round(float(np.nanmax(sub)), 4))
    print("kept at 0.7 / 0.9:", at[0.7], "/", at[0.9])

    table = pd.DataFrame(
        {"factor": names, "factor_set": [source_of[f] for f in names], "coverage": coverage.to_numpy()}
    )
    table["kept"] = table["factor"].isin(kept_set)
    table["duplicate_of"] = table["factor"].map(dropped)
    table.to_csv(out / f"{name}.csv", index=False)
    (out / "selection.json").write_text(
        json.dumps(
            {
                "name": name,
                "sources": sources,
                "max_abs_rank_correlation": limit,
                "deduplication_order": "coverage, then registry order",
                "candidates": len(names),
                "kept_count": len(kept),
                "kept_per_source": per_source,
                "kept_count_at_0.7": at[0.7],
                "kept_count_at_0.9": at[0.9],
                "max_kept_pair_correlation": float(np.nanmax(sub)),
                "kept": kept,
                "dropped": dropped,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    counts = ", ".join(f"{s} {per_source[s]} of {candidates[s]}" for s in sources)
    paragraphs = [
        f"Locked membership of the {name} factor set.",
        f"{name} is every factor of {', '.join(sources)} ({len(names)} candidates) with near duplicates removed. "
        "No return, Rank IC or long-side information is used to pick members or to decide which duplicate survives:",
        f"- the absolute mean daily cross-sectional Spearman correlation of every pair was measured on the ``sh_main`` "
        f"stores on every {sampled['correlation_date_step']}th trading date from {sampled['correlation_start']} to "
        f"{sampled['correlation_end']} ({sampled['sampled_dates']} dates; the full matrix of "
        "``scripts/remove_duplicates/select_remove_duplicates.py``);",
        "- factors are ordered by coverage (rounded to two decimals, higher first) and then by registry order "
        f"(library in the order {', '.join(sources)}, then ordinal);",
        "- ``prune_correlated`` drops a factor when its correlation with an already kept factor reaches "
        f"``max_abs_rank_correlation`` ({limit}). ``DEDUPLICATED`` maps each dropped factor to the kept one it "
        "duplicates.",
        f"Kept per source: {counts}. Built with ``scripts/remove_duplicates/select_by_coverage.py``.",
        "Members keep their source canonical names and values. ``direction`` is the sign of the sh_main 2020+ Rank IC "
        "and is metadata only: it plays no part in membership and the stored column is not negated. Membership is "
        "meant for the ``sh_main`` universe, where the correlations were measured.",
    ]
    doc = "\n\n".join(
        textwrap.fill(text, width=118, subsequent_indent="  " if text.startswith("- ") else "") for text in paragraphs
    ).replace("\n\n- ", "\n- ")
    members = "\n".join(f'    SelectedFactor("{f}", {int(direction[f])}),' for f in kept)
    duplicates = "\n".join(f'    "{key}": "{value}",' for key, value in sorted(dropped.items()))
    source_lines = "\n".join(f'        "{s}",' for s in sources)
    module.write_text(
        f'''"""{doc}
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

{prefix}_MEMBERS = (
{members}
)

DEDUPLICATED = {{
{duplicates}
}}

{prefix}_SELECTION = FactorSelection(
    name="{name}",
    source_factor_sets=(
{source_lines}
    ),
    members={prefix}_MEMBERS,
    criteria={{
        "universe": "sh_main",
        "thresholds": None,
        "deduplication_order": "coverage, then registry order",
        "returns_used": False,
        "max_abs_rank_correlation": {limit},
        "correlation_date_step": {sampled["correlation_date_step"]},
        "correlation_start": "{sampled["correlation_start"]}",
        "correlation_end": "{sampled["correlation_end"]}",
    }},
    provenance={{
        "universe": "sh_main",
        "candidates": {len(names)},
        "deduplicated": DEDUPLICATED,
    }},
)
''',
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {module}: {len(kept)} members, {len(dropped)} duplicates")


if __name__ == "__main__":
    main()
