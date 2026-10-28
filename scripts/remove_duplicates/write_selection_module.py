"""Write the locked remove_duplicates selection module from an ``order_and_prune.py`` result.

Usage: python write_selection_module.py <selection_<ordering>.json> <kept_<ordering>.csv> <correlation selection.json>
       <output.py> [factor_set_name] [variable_prefix]

The correlation ``selection.json`` is the one ``select_remove_duplicates.py`` wrote; it supplies the sampled dates.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

from rquant.factors.libraries.recent_sh_main import RECENT_SH_MAIN_SELECTION

SOURCES = (
    "qlib_alpha158", "wq_alpha101", "gtja191", "qlib_alpha360", "china_chars", "size_value_china",
    "talib", "jq240", "oap", "barra_cne6", "tushare_tech", "jkp153",
)
ORDERINGS = {
    "long_excess": ("mean_long_excess", "mean yearly long excess"),
    "abs_rank_ic": ("abs_rank_ic", "absolute Rank IC"),
}


def main() -> None:
    selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    kept = pd.read_csv(sys.argv[2]).set_index("factor")
    sampled = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    output = Path(sys.argv[4])
    name = sys.argv[5] if len(sys.argv) > 5 else "remove_duplicates"
    prefix = sys.argv[6] if len(sys.argv) > 6 else "REMOVE_DUPLICATES"
    ordering, label = ORDERINGS[selection["ordering"]]
    limit = selection["max_abs_rank_correlation"]
    per_source = selection["kept_per_source"]
    counts = ", ".join(f"{source} {per_source[source]}" for source in SOURCES if per_source.get(source))
    provenance = RECENT_SH_MAIN_SELECTION.provenance
    members = []
    for member in selection["kept"]:
        factor = member["factor"]
        row = kept.loc[factor]
        excess = "n/a" if pd.isna(row["mean_long_excess"]) else f"{row['mean_long_excess'] * 100:+.1f}%"
        ic = "n/a" if pd.isna(row["rank_ic_mean"]) else f"{row['rank_ic_mean']:+.4f}"
        members.append(
            f'    SelectedFactor("{factor}", {member["direction"]}),  # long excess {excess}, RankIC {ic}, '
            f"cov {row['coverage']:.3f}"
        )
    dropped = "\n".join(f'    "{key}": "{value}",' for key, value in sorted(selection["dropped"].items()))
    evaluations = "\n".join(
        "            {\n"
        f'                "factor_set": "{item["factor_set"]}",\n'
        f'                "run_id": "{item["run_id"]}",\n'
        f'                "factor_fingerprint": "{item["factor_fingerprint"]}",\n'
        "            },"
        for item in provenance["evaluations"]
    )
    source_lines = "\n".join(f'        "{source}",' for source in SOURCES)
    paragraphs = [
        f"Locked membership of the {name} factor set.",
        f"{name} keeps every factor of the twelve formula libraries except duplicates: no Rank IC, t-statistic, "
        "yearly or long-side threshold removes a factor. Only the choice of which member of a group of near "
        "duplicates survives uses returns:",
        f"- the absolute mean daily cross-sectional Spearman correlation of every pair of the {sampled['factors']} "
        f"library factors was measured on the ``sh_main`` stores on every {sampled['correlation_date_step']}th "
        f"trading date from {sampled['correlation_start']} to {sampled['correlation_end']} "
        f"({sampled['sampled_dates']} dates; ``rquant.factors.selection.RankCorrelationAccumulator``, scripts in "
        "``scripts/remove_duplicates``);",
        f"- factors are ordered by their {label} in the sh_main 1d evaluation of 2020-01-02 to 2026-10-21 listed in "
        "``provenance`` (the evaluation recent_sh_main was selected from; the long side is the top or bottom 20% "
        "by the sign of the 2020+ Rank IC, against the equal-weight sh_main universe), factors without a score last;",
        "- ``prune_correlated`` then drops a factor when its correlation with an already kept factor reaches "
        f"``max_abs_rank_correlation`` ({limit}). ``DEDUPLICATED`` maps each dropped factor to the kept one it "
        "duplicates.",
        f"Kept per source: {counts}.",
        "The ordering uses 2020-2026 sh_main returns, so for walk-forward test years from 2020 on the choice among "
        "duplicates is in-sample; membership is only meant for the ``sh_main`` universe. Members keep their source "
        "canonical names and values; ``direction == -1`` marks a factor whose 2020+ Rank IC is negative, the stored "
        "column is not negated.",
    ]
    doc = "\n\n".join(
        textwrap.fill(text, width=118, subsequent_indent="  " if text.startswith("- ") else "") for text in paragraphs
    ).replace("\n\n- ", "\n- ")
    text = f'''"""{doc}
"""

from __future__ import annotations

from rquant.factors.libraries.base import FactorSelection, SelectedFactor

{prefix}_MEMBERS = (
{chr(10).join(members)}
)

DEDUPLICATED = {{
{dropped}
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
        "deduplication_order": "{ordering}",
        "max_abs_rank_correlation": {limit},
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
        "deduplicated": DEDUPLICATED,
        "evaluations": (
{evaluations}
        ),
    }},
)
'''
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output}: {len(selection['kept'])} members, {len(selection['dropped'])} duplicates")


if __name__ == "__main__":
    main()
