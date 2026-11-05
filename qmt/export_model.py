"""Export one walk-forward LightGBM window so the QMT strategy can score with numpy (QMT has no lightgbm).

The export holds what the window's inference needs and nothing else:

* the factor names, in the order the booster's columns follow (the factor store's column order);
* the ZScoreNorm statistics of the window's fit period, recomputed from the factor store exactly as training computed
  them (float32 panel, ProcessInf, then ``ZScoreNorm.fit``); the run keeps the model but not the processors;
* the booster's trees as flat arrays (``qmt/runtime/lgb_numpy.py`` walks them).

It then checks itself: the exported model scores the factor store's rows of one day with ``lgb_numpy`` and the result
is compared with the booster's own prediction and with the run's saved predictions for that day.

Usage (Windows, from the repository root)::

    python qmt/export_model.py --run runs/walk-forward/<run id> --year 2026 \
        --store data/factors-sh_main/jq240 --out qmt_model_2026.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LGB_NUMPY = runpy.run_path(str(ROOT / "qmt" / "runtime" / "lgb_numpy.py"))
FORMAT = 1


def window_manifest(run: Path, year: int) -> dict[str, Any]:
    return json.loads((run / f"year={year}" / "window.json").read_text(encoding="utf-8"))


def fit_period(window: dict[str, Any]) -> tuple[str, str]:
    """The (start, end) dates whose rows the window's ZScoreNorm was fitted on: the refit training segment."""
    if "segments" in window:
        start, end = window["segments"]["train"]
        return start, end
    return window["train_start"], window["train_end_after_purge"]


def load_booster(run: Path, year: int, window: dict[str, Any]) -> Any:
    """The window's refit booster: ``year=Y/model.pkl`` (sequential runner) or the recorder's ``params.pkl``."""
    candidates = [run / f"year={year}" / "model.pkl"]
    if window.get("recorder_id"):
        candidates.append(run / "mlruns" / window["recorder_id"] / "artifacts" / "params.pkl")
    for path in candidates:
        if path.exists():
            with path.open("rb") as handle:
                model = pickle.load(handle)
            return getattr(model, "model", model)
    raise FileNotFoundError(f"No model for {year} under {run}: tried {[str(path) for path in candidates]}")


def store_columns(store: Path) -> list[str]:
    import pyarrow.parquet as pq

    first = sorted(store.glob("year=*/factors.parquet"))[0]
    return [name for name in pq.ParquetFile(first).schema_arrow.names if name not in {"datetime", "instrument"}]


def load_panel(store: Path, start: str, end: str, names: list[str]) -> pd.DataFrame:
    """Rows of the store dated in [start, end] as one float32 column-major block, sorted by (datetime, instrument),
    as KunQuantDataLoader builds the training panel."""
    import pyarrow.parquet as pq

    first, last = pd.Timestamp(start), pd.Timestamp(end)
    pieces = []
    for path in sorted(store.glob("year=*/factors.parquet")):
        year = int(path.parent.name.removeprefix("year="))
        if year < first.year or year > last.year:
            continue
        keys = pd.read_parquet(path, columns=["datetime", "instrument"])
        keys["datetime"] = pd.to_datetime(keys["datetime"])
        keep = ((keys["datetime"] >= first) & (keys["datetime"] <= last)).to_numpy()
        if keep.any():
            pieces.append((path, keep, keys[keep]))
    if not pieces:
        raise ValueError(f"No store rows between {start} and {end} in {store}")
    keys = pd.concat([frame for _, _, frame in pieces], ignore_index=True)
    order = np.lexsort((keys["instrument"].to_numpy(), keys["datetime"].to_numpy()))
    block = np.empty((len(keys), len(names)), dtype=np.float32, order="F")
    offset = 0
    for path, keep, frame in pieces:
        parquet = pq.ParquetFile(path)
        for begin in range(0, len(names), 32):
            chunk = names[begin : begin + 32]
            table = parquet.read(columns=chunk)
            for position, name in enumerate(chunk, begin):
                block[offset : offset + len(frame), position] = table.column(name).to_numpy()[keep]
        offset += len(frame)
    block = np.asfortranarray(block[order])
    index = pd.MultiIndex.from_arrays(
        [keys["datetime"].to_numpy()[order], keys["instrument"].to_numpy()[order]], names=["datetime", "instrument"]
    )
    return pd.DataFrame(block, index=index, columns=names, copy=False)


def zscore_statistics(panel: pd.DataFrame, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    from rquant.qlib_ext.processors import ProcessInf, ZScoreNorm

    ProcessInf()(panel)
    norm = ZScoreNorm(fit_start_time=start, fit_end_time=end)
    norm.fit(panel)
    return np.asarray(norm.mean_train, dtype=np.float32), np.asarray(norm.std_train, dtype=np.float32)


def flat_trees(booster: Any) -> list[dict[str, list]]:
    """Each tree as arrays indexed by split index; a child c < 0 is leaf ~c."""
    dumped = booster.dump_model()
    if dumped.get("average_output") or dumped.get("num_tree_per_iteration", 1) != 1:
        raise ValueError("Only single-output, summed boosters are supported")
    trees = []
    for info in dumped["tree_info"]:
        root = info["tree_structure"]
        if "split_index" not in root:
            trees.append({"feature": [], "threshold": [], "left": [], "right": [], "leaf": [root["leaf_value"]]})
            continue
        splits = info["num_leaves"] - 1
        tree = {"feature": [0] * splits, "threshold": [0.0] * splits, "left": [0] * splits, "right": [0] * splits,
                "leaf": [0.0] * info["num_leaves"]}
        stack = [root]
        while stack:
            node = stack.pop()
            if node["decision_type"] != "<=" or node["missing_type"] not in {"None", "NaN"}:
                raise ValueError(f"Unsupported split {node['decision_type']} / missing {node['missing_type']}")
            at = node["split_index"]
            tree["feature"][at] = node["split_feature"]
            tree["threshold"][at] = node["threshold"]
            for side in ("left", "right"):
                child = node[f"{side}_child"]
                if "split_index" in child:
                    tree[side][at] = child["split_index"]
                    stack.append(child)
                else:
                    tree[side][at] = ~child["leaf_index"]
                    tree["leaf"][child["leaf_index"]] = child["leaf_value"]
        trees.append(tree)
    return trees


def verify(model: dict[str, Any], booster: Any, store: Path, run: Path, year: int, day: str | None) -> dict[str, Any]:
    """Score one day of store rows three ways and report how far apart they are."""
    saved = pd.read_parquet(run / f"year={year}" / "predictions.parquet")
    saved["datetime"] = pd.to_datetime(saved["datetime"])
    when = pd.Timestamp(day) if day else saved["datetime"].max()
    rows = load_panel(store, when.strftime("%Y-%m-%d"), when.strftime("%Y-%m-%d"), model["features"])
    prepared = LGB_NUMPY["prepare_model"](model)
    matrix = LGB_NUMPY["normalise"](rows.to_numpy(), prepared)
    mine = LGB_NUMPY["predict"](prepared, matrix)
    theirs = booster.predict(matrix)
    expected = saved[saved["datetime"] == when].set_index("instrument")["score"]
    instruments = rows.index.get_level_values("instrument")
    expected = expected.reindex(instruments).to_numpy()
    return {
        "date": when.strftime("%Y-%m-%d"),
        "rows": len(rows),
        "max_abs_vs_booster": float(np.max(np.abs(mine - theirs))),
        "max_abs_vs_saved": float(np.nanmax(np.abs(mine - expected))),
        "saved_missing": int(np.isnan(expected).sum()),
    }


def export(run: Path, year: int, store: Path, day: str | None = None) -> dict[str, Any]:
    window = window_manifest(run, year)
    booster = load_booster(run, year, window)
    names = store_columns(store)
    if booster.num_feature() != len(names):
        raise ValueError(f"Booster has {booster.num_feature()} features, store {store} has {len(names)}")
    start, end = fit_period(window)
    panel = load_panel(store, start, end, names)
    mean, std = zscore_statistics(panel, start, end)
    fitted_rows = len(panel)
    del panel
    model = {
        "format": FORMAT,
        "run": run.name,
        "prediction_year": year,
        "fit_period": [start, end],
        "fit_rows": fitted_rows,
        "iterations": booster.num_trees(),
        "features": names,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "trees": flat_trees(booster),
    }
    model["verify"] = verify(model, booster, store, run, year, day)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="walk-forward run directory")
    parser.add_argument("--year", type=int, required=True, help="prediction year of the window to export")
    parser.add_argument("--store", type=Path, required=True, help="factor store, e.g. data/factors-sh_main/jq240")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--verify-date", help="day to verify on (default: the window's last predicted day)")
    args = parser.parse_args(argv)
    model = export(args.run, args.year, args.store, args.verify_date)
    args.out.write_text(json.dumps(model), encoding="utf-8")
    print(json.dumps({key: model[key] for key in ("run", "prediction_year", "fit_period", "fit_rows", "iterations",
                                                     "verify")}, indent=2))
    check = model["verify"]
    ok = check["max_abs_vs_booster"] < 1e-9 and check["max_abs_vs_saved"] < 1e-6
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
