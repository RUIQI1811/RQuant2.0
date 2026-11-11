"""Average the out-of-sample predictions of finished walk-forward runs into a new run that ``rquant report`` can read.

Usage: python scripts/analysis/blend_runs.py <runs_root> <name> run_dir [run_dir ...]

Every ``run_dir`` is a folder under ``<runs_root>/walk-forward`` that ``rquant walk-forward`` wrote, such as the same
workflow trained with different ``--seed`` values. Each run's scores are first z-scored across the stocks of each day,
so runs whose scores have different scales weigh the same, then averaged over the runs that score the stock that day.
Only the (day, stock) rows every run scores are kept, so a blend never mixes in a stock some runs could not see.

The blend is written to ``<runs_root>/walk-forward/<name>``: ``predictions.parquet`` and a ``walk_forward.json``
copied from the first run with a ``blend`` entry naming its members. Then, from the project root:

    rquant --universe sh_main report <name> --topk 50 --n-drop 5 --rebalance-days 1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

KEYS = ["datetime", "instrument"]


def blend(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean daily z-score of ``frames`` (``datetime``, ``instrument``, ``score``) on the rows all of them score."""
    if len(frames) < 2:
        raise ValueError("A blend needs at least two runs")
    columns = []
    for position, frame in enumerate(frames):
        frame = frame.dropna(subset=["score"]).copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Run {position} has duplicate datetime/instrument rows")
        columns.append(frame.set_index(KEYS)["score"].rename(f"run{position}"))
    joined = pd.concat(columns, axis=1, join="inner").sort_index()
    if joined.empty:
        raise ValueError("The runs share no scored rows")
    # Each day's scores minus that day's mean, over that day's standard deviation (zero when every score is equal).
    days = joined.groupby(level="datetime")
    standardised = ((joined - days.transform("mean")) / days.transform("std").replace(0.0, float("nan"))).fillna(0.0)
    return standardised.mean(axis=1).rename("score").reset_index()


def main(argv: list[str]) -> None:
    if len(argv) < 4:
        raise SystemExit(__doc__)
    walk_forward = Path(argv[0]) / "walk-forward"
    name, members = argv[1], argv[2:]
    output = walk_forward / name
    if output.exists():
        raise SystemExit(f"{output} already exists")
    manifests = []
    frames = []
    for member in members:
        directory = walk_forward / member
        frames.append(pd.read_parquet(directory / "predictions.parquet"))
        with (directory / "walk_forward.json").open("r", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    for key in ("horizon", "factor_set", "universe"):
        values = {manifest.get(key) for manifest in manifests}
        if len(values) != 1:
            raise SystemExit(f"The runs differ in {key}: {sorted(map(str, values))}")
    blended = blend(frames)
    output.mkdir(parents=True)
    blended.to_parquet(output / "predictions.parquet", index=False)
    manifest = {
        **{key: value for key, value in manifests[0].items() if key not in ("windows", "mlflow_tracking_uri")},
        "model": f"blend of {len(members)} runs",
        "prediction_rows": len(blended),
        "blend": {
            "method": "mean of daily z-scored scores on the rows every run scores",
            "runs": members,
            "seeds": [manifest.get("seed") for manifest in manifests],
            "rows_per_run": [len(frame) for frame in frames],
        },
    }
    with (output / "walk_forward.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"{output}: {len(blended)} rows from {len(members)} runs")


if __name__ == "__main__":
    main(sys.argv[1:])
