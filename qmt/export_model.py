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


class StoreRows:
    """The store's rows dated in [start, end], in (datetime, instrument) order as KunQuantDataLoader sorts them."""

    def __init__(self, store: Path, start: str, end: str) -> None:
        import pyarrow.parquet as pq

        first, last = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        self.pieces = []
        dates, instruments = [], []
        for path in sorted(store.glob("year=*/factors.parquet")):
            year = int(path.parent.name.removeprefix("year="))
            if year < pd.Timestamp(start).year or year > pd.Timestamp(end).year:
                continue
            # Instruments as dictionary codes: millions of Python strings would cost far more than the factor blocks.
            table = pq.read_table(path, columns=["datetime", "instrument"], read_dictionary=["instrument"])
            day = table.column("datetime").to_numpy().astype("datetime64[ns]")
            keep = (day >= first) & (day <= last)
            if keep.any():
                self.pieces.append((path, keep, int(keep.sum())))
                dates.append(day[keep])
                code = table.column("instrument").combine_chunks()
                names = np.asarray(code.dictionary.to_pylist(), dtype=object)
                instruments.append(names[code.indices.to_numpy()[keep]])
            del table
        if not self.pieces:
            raise ValueError(f"No store rows between {start} and {end} in {store}")
        day = np.concatenate(dates)
        instrument = pd.Categorical(np.concatenate(instruments))
        del dates, instruments
        # Category codes follow the sorted instrument names, so sorting by code sorts by name.
        self.order = np.lexsort((instrument.codes, day))
        self.index = pd.MultiIndex.from_arrays(
            [day[self.order], instrument[self.order]], names=["datetime", "instrument"]
        )

    def __len__(self) -> int:
        return len(self.index)

    def frame(self, names: list[str]) -> pd.DataFrame:
        """``names`` as one float32 column-major block (the store is float64; training rounds to float32).

        Read one column at a time through the system allocator: Arrow's default pool keeps decoded pages after they
        are freed, which on a 16 GB machine is the difference between fitting and not.
        """
        import pyarrow
        import pyarrow.parquet as pq

        block = np.empty((len(self.order), len(names)), dtype=np.float32, order="F")
        column = np.empty(len(self.order), dtype=np.float32)
        previous = pyarrow.default_memory_pool()
        pyarrow.set_memory_pool(pyarrow.system_memory_pool())
        try:
            files = [(pq.ParquetFile(path), keep, count) for path, keep, count in self.pieces]
            for position, name in enumerate(names):
                offset = 0
                for parquet, keep, count in files:
                    column[offset : offset + count] = parquet.read(columns=[name]).column(0).to_numpy()[keep]
                    offset += count
                block[:, position] = column[self.order]
        finally:
            pyarrow.set_memory_pool(previous)
        return pd.DataFrame(block, index=self.index, columns=names, copy=False)


def load_panel(store: Path, start: str, end: str, names: list[str]) -> pd.DataFrame:
    """Rows of the store dated in [start, end] as one float32 column-major block, sorted by (datetime, instrument),
    as KunQuantDataLoader builds the training panel."""
    return StoreRows(store, start, end).frame(names)


def zscore_statistics(rows: StoreRows, start: str, end: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """ProcessInf then ``ZScoreNorm.fit`` over the fit-period rows, 16 columns at a time.

    ZScoreNorm already fits 16 columns at a time and ProcessInf works column by column, so loading the panel in the
    same 16-column blocks gives the same statistics while holding about 1/15 of the panel in memory.
    """
    from rquant.qlib_ext.processors import ProcessInf, ZScoreNorm

    means, stds = [], []
    for begin in range(0, len(names), 16):
        panel = rows.frame(names[begin : begin + 16])
        ProcessInf()(panel)
        norm = ZScoreNorm(fit_start_time=start, fit_end_time=end)
        norm.fit(panel)
        means.append(np.asarray(norm.mean_train, dtype=np.float32))
        stds.append(np.asarray(norm.std_train, dtype=np.float32))
        del panel
    return np.concatenate(means), np.concatenate(stds)


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
    rows = StoreRows(store, start, end)
    mean, std = zscore_statistics(rows, start, end, names)
    fitted_rows = len(rows)
    del rows
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
