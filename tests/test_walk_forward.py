from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("qlib", "lightgbm", "pyarrow"))


def write_store(root: Path, *, stocks: int = 30, seed: int = 0) -> tuple[Path, Path, list]:  # type: ignore[type-arg]
    """A Qlib day provider and a wq_alpha101 factor store whose first factor predicts the next-open return."""
    import numpy as np
    import pandas as pd

    from rquant.factors.catalog import get_catalog

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-10-01", "2021-12-31")
    qlib_root = root / "qlib"
    (qlib_root / "calendars").mkdir(parents=True)
    (qlib_root / "calendars" / "day.txt").write_text("".join(f"{day:%Y-%m-%d}\n" for day in dates), encoding="utf-8")
    instruments = [f"SH6{value:05d}" for value in range(stocks)]
    (qlib_root / "instruments").mkdir()
    (qlib_root / "instruments" / "all.txt").write_text(
        "".join(f"{code}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n" for code in instruments), encoding="utf-8"
    )
    signal = rng.normal(size=(len(dates), stocks))
    for position, code in enumerate(instruments):
        returns = rng.normal(0.0, 0.02, len(dates))
        returns[2:] += 0.01 * signal[:-2, position]
        price = 10.0 * np.cumprod(1.0 + returns)
        directory = qlib_root / "features" / code.lower()
        directory.mkdir(parents=True)
        for field, values in (("open", price), ("close", price), ("factor", np.ones(len(dates)))):
            np.concatenate(([0.0], values)).astype(np.float32).tofile(directory / f"{field}.day.bin")
    names = list(get_catalog().canonical_names("wq_alpha101"))
    store = root / "factors" / "wq_alpha101"
    for year in sorted(set(dates.year)):
        days = dates[dates.year == year]
        index = pd.MultiIndex.from_product([days, instruments], names=["datetime", "instrument"])
        values = rng.normal(size=(len(index), len(names)))
        values[:, 0] = signal[np.searchsorted(dates, days)].reshape(-1) + rng.normal(size=len(index))
        frame = pd.DataFrame(values, columns=names)
        frame.insert(0, "instrument", index.get_level_values("instrument"))
        frame.insert(0, "datetime", index.get_level_values("datetime"))
        (store / f"year={year}").mkdir(parents=True)
        frame.to_parquet(store / f"year={year}" / "factors.parquet", index=False)
    return qlib_root, root / "factors", [day.date() for day in dates]


@unittest.skipUnless(HAS_STACK, "pyqlib, LightGBM and pyarrow are required")
class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.temporary.name)
        cls.qlib_root, cls.factor_root, cls.calendar = write_store(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    # Break caught: rolled segments that leave the calendar-year windows and trading-day purges of every earlier run
    # (a training label reading into the test year, or a validation segment cut short).
    def test_calendar_year_rolling_keeps_the_yearly_windows_and_purges(self) -> None:
        import qlib
        from qlib.constant import REG_CN
        from qlib.workflow.task.gen import task_generator

        from rquant.workflow.rolling import generate_rolling_windows, previous_trading_day
        from rquant.workflow.tasks import (
            BUILT_IN_WORKFLOWS,
            CalendarYearRollingGen,
            load_workflow,
            rolling_template,
        )
        from rquant.workflow.trainer import inner_split_task, lean_window

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN)
        for horizon in ("1d", "5d"):
            with self.subTest(horizon=horizon):
                template = rolling_template(
                    load_workflow(BUILT_IN_WORKFLOWS["lgb"])["task"], factor_root=self.factor_root,
                    factor_set="wq_alpha101", horizon=horizon, universe="all",
                )
                options = {"first_year": 2020, "last_year": 2021, "through": date(2021, 12, 20),
                           "train_years": 1, "validation_months": 6, "formal_start_year": 2019}
                tasks = task_generator(template, CalendarYearRollingGen(horizon=horizon, **options))
                windows = generate_rolling_windows(
                    first_prediction_year=2020, last_prediction_year=2021, horizon=horizon, through=date(2021, 12, 20),
                    train_years=1, validation_months=6, formal_start_year=2019,
                )
                self.assertEqual(2, len(tasks))
                for task, window in zip(tasks, windows, strict=True):
                    segments = task["dataset"]["kwargs"]["segments"]
                    refit_end = previous_trading_day(self.calendar, window.test_start, window.purge_trading_days)
                    inner_end = previous_trading_day(self.calendar, window.validation_start, window.purge_trading_days)
                    day = {name: (start.date(), end.date()) for name, (start, end) in segments.items()}
                    self.assertEqual(refit_end, day["train"][1])
                    self.assertEqual(refit_end, day["valid"][1])
                    self.assertEqual(window.validation_start, _next_trading_day(self.calendar, day["valid"][0]))
                    self.assertEqual(min(window.test_end, date(2021, 12, 20)), day["test"][1])
                    lean = lean_window(task)
                    self.assertEqual((day["train"][0], day["test"][1]),
                                     (lean["start_time"].date(), lean["end_time"].date()))
                    self.assertEqual(day["train"], (lean["fit_start_time"].date(), lean["fit_end_time"].date()))
                    inner = inner_split_task(task)
                    inner_segments = inner["dataset"]["kwargs"]["segments"]
                    self.assertNotIn("test", inner_segments)
                    self.assertEqual(inner_end, inner_segments["train"][1].date())
                    last_before_test = previous_trading_day(self.calendar, window.test_start, 0)
                    self.assertEqual(last_before_test, inner_segments["valid"][1].date())
                    self.assertEqual(inner_end, lean_window(inner)["fit_end_time"].date())
                    self.assertEqual(last_before_test, lean_window(inner)["end_time"].date())

    # Break caught: TrainerR recorders, the RollingEnsemble merge or the per-year outputs going missing or overlapping.
    def test_walk_forward_trains_each_year_in_its_own_recorder_and_merges_them(self) -> None:
        import pandas as pd
        from qlib.workflow import R

        from rquant.workflow.runner import ENSEMBLE_RECORDER, RollingWorkflowRunner

        run_directory = self.root / "run-5d"
        run_directory.mkdir()
        runner = RollingWorkflowRunner(
            qlib_root=self.qlib_root, factor_root=self.factor_root, run_directory=run_directory, workflow="lgb",
            horizon="5d", factor_set="wq_alpha101", seed=7, train_years=1, validation_months=6,
            formal_start_year=2019, universe="all",
        )
        runner.template["model"]["kwargs"].update(num_boost_round=30, num_threads=1)
        manifest = runner.run(first_year=2020, last_year=2021, through=date(2021, 12, 20))

        predictions = pd.read_parquet(run_directory / "predictions.parquet")
        yearly = pd.concat(
            [pd.read_parquet(run_directory / f"year={year}" / "predictions.parquet") for year in (2020, 2021)]
        ).sort_values(["datetime", "instrument"], kind="mergesort")
        pd.testing.assert_frame_equal(yearly.reset_index(drop=True), predictions)
        self.assertEqual(["datetime", "instrument", "score"], list(predictions.columns))
        positions = {day: position for position, day in enumerate(day for day in self.calendar if day.year >= 2009)}
        self.assertTrue(all(positions[day.date()] % 5 == 0 for day in predictions["datetime"].unique()))
        self.assertEqual([2020, 2021], [window["prediction_year"] for window in manifest["windows"]])
        self.assertTrue(all(1 <= window["chosen_iterations"] <= 30 for window in manifest["windows"]))
        with (run_directory / "year=2021" / "window.json").open("r", encoding="utf-8") as handle:
            self.assertEqual("2021-12-20", json.load(handle)["segments"]["test"][1])
        importance = pd.read_parquet(run_directory / "feature_importance.parquet")
        self.assertEqual({2020, 2021}, set(importance["prediction_year"]))

        recorders = R.list_recorders(experiment_name=runner.experiment_name)
        names = sorted(recorder.info["name"] for recorder in recorders.values())
        self.assertEqual([ENSEMBLE_RECORDER, "year=2020", "year=2021"], names)
        ensemble = next(rec for rec in recorders.values() if rec.info["name"] == ENSEMBLE_RECORDER)
        pd.testing.assert_frame_equal(
            predictions.set_index(["datetime", "instrument"]), ensemble.load_object("pred.pkl")
        )
        self.assertTrue((run_directory / "mlruns").is_dir())


def _next_trading_day(calendar: list, value: date) -> date:  # type: ignore[type-arg]
    return min(day for day in calendar if day >= value)


if __name__ == "__main__":
    unittest.main()
