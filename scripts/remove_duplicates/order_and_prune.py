"""Prune the full rank-correlation matrix with an ordering taken from the sh_main 2020+ 1d evaluation.

Usage: python order_and_prune.py <rank_correlation.parquet> <factor_stability.csv> <out_dir> [max_abs_rank_correlation]

Writes one selection per ordering: ``long_excess`` (mean yearly excess of the oriented top/bottom 20% over the
equal-weight universe) and ``abs_rank_ic``. Factors without the score are ordered last, in registry order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.selection import prune_correlated


def main() -> None:
    correlation = pd.read_parquet(sys.argv[1])
    scored = pd.read_csv(sys.argv[2]).set_index("factor")
    out = Path(sys.argv[3])
    max_corr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    out.mkdir(parents=True, exist_ok=True)
    names = list(correlation.index)
    scores = {
        "long_excess": scored["mean_long_excess"],
        "abs_rank_ic": scored["rank_ic_mean"].abs(),
    }
    for key, raw in scores.items():
        raw = raw.reindex(names)
        floor = raw.min() - 1.0
        strength = raw.fillna(floor)
        kept, dropped = prune_correlated(strength, correlation, max_abs_correlation=max_corr)
        kept_set = set(kept)
        kept = [name for name in names if name in kept_set]
        sub = correlation.loc[kept, kept].abs().to_numpy()
        np.fill_diagonal(sub, np.nan)
        direction = scored["direction"].reindex(names).fillna(0).astype(int)
        table = scored.loc[kept, ["factor_set", "direction", "rank_ic_mean", "rank_ic_t_stat", "mean_long_excess",
                                  "coverage"]].reset_index()
        table.to_csv(out / f"kept_{key}.csv", index=False)
        meta = {
            "ordering": key,
            "max_abs_rank_correlation": max_corr,
            "kept_count": len(kept),
            "kept_per_source": table.groupby("factor_set").size().to_dict(),
            "max_kept_pair_correlation": float(np.nanmax(sub)),
            "unscored": sorted(raw.index[raw.isna()]),
            "kept": [{"factor": f, "direction": int(direction[f]) or 1} for f in kept],
            "dropped": dropped,
            "kept_mean_long_excess": float(table["mean_long_excess"].mean()),
            "kept_median_abs_rank_ic": float(table["rank_ic_mean"].abs().median()),
        }
        for limit in (0.7, 0.9):
            meta[f"kept_count_at_{limit}"] = len(prune_correlated(strength, correlation, max_abs_correlation=limit)[0])
        (out / f"selection_{key}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(key, {k: meta[k] for k in ("kept_count", "kept_count_at_0.7", "kept_count_at_0.9", "kept_per_source",
                                         "max_kept_pair_correlation", "kept_mean_long_excess",
                                         "kept_median_abs_rank_ic")}, flush=True)


if __name__ == "__main__":
    main()
