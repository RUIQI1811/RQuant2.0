from __future__ import annotations

import ast
import json
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))

import export_model  # noqa: E402

lgb = pytest.importorskip("lightgbm")
LGB_NUMPY = runpy.run_path(str(ROOT / "qmt" / "runtime" / "lgb_numpy.py"))
NAMES = [f"jq_{number:03d}" for number in range(1, 7)]


def fake_store(root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", "2024-03-29")
    instruments = [f"SH600{number:03d}" for number in range(40)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    values = rng.normal(size=(len(index), len(NAMES))) * [1, 5, 0.1, 100, 1, 1] + [0, 3, 0, 50, 0, 0]
    values[rng.random(values.shape) < 0.05] = np.nan
    values[rng.random(values.shape) < 0.002] = np.inf
    frame = pd.DataFrame(values, index=index, columns=NAMES).reset_index()
    # Store partitions are not guaranteed to be sorted; the export sorts like the training loader.
    frame = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    for year, part in frame.groupby(frame["datetime"].dt.year):
        (root / f"year={year}").mkdir(parents=True)
        part.to_parquet(root / f"year={year}" / "factors.parquet", index=False)
    return frame


def train_window(store: Path, run: Path) -> None:
    """Mimic one refit window: ProcessInf, ZScoreNorm over 2023, Fillna 0, LightGBM, predictions for 2024."""
    from rquant.qlib_ext.processors import Fillna, ProcessInf, ZScoreNorm

    panel = export_model.load_panel(store, "2023-01-01", "2024-12-31", NAMES)
    for processor in (ProcessInf(), ZScoreNorm("2023-01-02", "2023-12-29"), Fillna()):
        processor.fit(panel)
        processor(panel)
    dates = panel.index.get_level_values("datetime")
    train = panel[dates <= "2023-12-29"]
    target = np.tanh(train["jq_001"].to_numpy()) + 0.3 * train["jq_004"].to_numpy() ** 2
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 15, "learning_rate": 0.1, "verbose": -1, "seed": 1},
        lgb.Dataset(np.asfortranarray(train.to_numpy(dtype=np.float32)), label=target),
        num_boost_round=12,
    )
    test = panel[dates >= "2024-01-01"]
    window = run / "year=2024"
    window.mkdir(parents=True)
    pd.DataFrame({
        "datetime": test.index.get_level_values("datetime"),
        "instrument": test.index.get_level_values("instrument"),
        "score": booster.predict(test.to_numpy(dtype=np.float32)),
    }).to_parquet(window / "predictions.parquet", index=False)
    with (window / "model.pkl").open("wb") as handle:
        pickle.dump(booster, handle)
    manifest = {"prediction_year": 2024, "train_start": "2023-01-02", "train_end_after_purge": "2023-12-29"}
    (window / "window.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_export_reproduces_the_window_predictions(tmp_path: Path) -> None:
    store, run = tmp_path / "jq240", tmp_path / "run"
    fake_store(store)
    train_window(store, run)
    model = export_model.export(run, 2024, store)
    assert model["features"] == NAMES
    assert model["iterations"] == 12
    check = model["verify"]
    assert check["date"] == "2024-03-29" and check["rows"] == 40 and check["saved_missing"] == 0
    assert check["max_abs_vs_booster"] < 1e-12
    assert check["max_abs_vs_saved"] < 1e-12
    # The JSON round trip keeps every number exactly.
    again = json.loads(json.dumps(model))
    assert again["mean"] == model["mean"] and again["trees"] == model["trees"]


def test_main_writes_the_export(tmp_path: Path) -> None:
    store, run = tmp_path / "jq240", tmp_path / "run"
    fake_store(store)
    train_window(store, run)
    out = tmp_path / "model.json"
    assert export_model.main(["--run", str(run), "--year", "2024", "--store", str(store), "--out", str(out),
                              "--verify-date", "2024-02-01"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verify"]["date"] == "2024-02-01"


def test_numpy_trees_match_lightgbm_on_unseen_values() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 5)).astype(np.float32)
    y = x[:, 0] * 2 + np.sin(x[:, 1] * 3) + (x[:, 2] > 0.5)
    booster = lgb.train({"objective": "regression", "num_leaves": 31, "verbose": -1, "seed": 2},
                        lgb.Dataset(x, label=y), num_boost_round=20)
    model = {"features": list("abcde"), "mean": [0.0] * 5, "std": [1.0] * 5,
             "trees": export_model.flat_trees(booster)}
    prepared = LGB_NUMPY["prepare_model"](model)
    probe = rng.normal(scale=3, size=(500, 5)).astype(np.float32)
    np.testing.assert_allclose(LGB_NUMPY["predict"](prepared, probe), booster.predict(probe), rtol=0, atol=1e-12)


def test_normalise_matches_the_training_processors() -> None:
    from rquant.qlib_ext.processors import Fillna, ProcessInf

    values = np.array([[1.0, np.inf, np.nan], [3.0, 2.0, 1.0], [np.nan, -np.inf, 4.0]])
    prepared = {"mean": np.array([1, 0.5, 2], dtype=np.float32), "std": np.array([2, 4, 1], dtype=np.float32)}
    index = pd.MultiIndex.from_product([[pd.Timestamp("2024-01-02")], ["a", "b", "c"]],
                                       names=["datetime", "instrument"])
    frame = pd.DataFrame(values.astype(np.float32), index=index)
    ProcessInf()(frame)
    expected = (frame.to_numpy() - prepared["mean"]) / prepared["std"]
    expected = Fillna()(pd.DataFrame(expected, index=index)).to_numpy()
    np.testing.assert_array_equal(LGB_NUMPY["normalise"](values, prepared), expected)


def test_lgb_numpy_is_python36_ascii() -> None:
    source = (ROOT / "qmt" / "runtime" / "lgb_numpy.py").read_bytes().decode("ascii")
    tree = ast.parse(source, feature_version=(3, 6))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported == {"numpy"}


def test_chunked_statistics_equal_the_whole_panel_fit(tmp_path: Path) -> None:
    from rquant.qlib_ext.processors import ProcessInf, ZScoreNorm

    store = tmp_path / "jq240"
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2023-01-02", "2023-06-30")
    index = pd.MultiIndex.from_product([dates, [f"SH600{n:03d}" for n in range(30)]], names=["datetime", "instrument"])
    names = [f"jq_{n:03d}" for n in range(1, 41)]
    values = rng.normal(size=(len(index), len(names))) * rng.uniform(0.01, 1000, size=len(names))
    values[rng.random(values.shape) < 0.05] = np.nan
    values[rng.random(values.shape) < 0.003] = -np.inf
    frame = pd.DataFrame(values, index=index, columns=names).reset_index()
    (store / "year=2023").mkdir(parents=True)
    frame.sample(frac=1.0, random_state=2).to_parquet(store / "year=2023" / "factors.parquet", index=False)

    panel = export_model.load_panel(store, "2023-01-01", "2023-06-30", names)
    ProcessInf()(panel)
    whole = ZScoreNorm("2023-01-01", "2023-06-30")
    whole.fit(panel)
    rows = export_model.StoreRows(store, "2023-01-01", "2023-06-30")
    mean, std = export_model.zscore_statistics(rows, "2023-01-01", "2023-06-30", names)
    np.testing.assert_array_equal(mean, whole.mean_train)
    np.testing.assert_array_equal(std, whole.std_train)
