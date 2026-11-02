from __future__ import annotations

import contextlib
import importlib.util
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from test_walk_forward import write_store

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("qlib", "torch", "pyarrow"))


def write_industries(root: Path, instruments: list[str], dates: list) -> Path:  # type: ignore[type-arg]
    """Canonical bars holding only the Shenwan level-2 industry; the first two stocks are never classified."""
    import pandas as pd

    bars = root / "canonical" / "bars"
    frame = pd.MultiIndex.from_product([pd.to_datetime(dates), instruments], names=["datetime", "instrument"])
    frame = frame.to_frame(index=False)
    codes = {code: (None if position < 2 else float(1 + position % 4)) for position, code in enumerate(instruments)}
    frame["industry"] = frame["instrument"].map(codes).astype(float)
    for year, rows in frame.groupby(frame["datetime"].dt.year):
        (bars / f"year={year}").mkdir(parents=True)
        rows.to_parquet(bars / f"year={year}" / "bars.parquet", index=False)
    return root / "canonical"


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow is required")
class IndustryConceptTests(unittest.TestCase):
    # Break caught: HIST concepts reading industries after the last prediction date, a stock that changed industry
    # losing its earlier membership, or an unclassified stock left with an empty concept row.
    def test_concepts_are_point_in_time_industry_memberships(self) -> None:
        import numpy as np
        import pandas as pd

        from rquant.workflow.concepts import write_industry_concepts

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-01-04", "2021-06-01", "2022-01-04"]),
                    "instrument": ["SH600000", "SH600000", "SH600001", "SH600000", "SH600002"],
                    "industry": [1.0, 2.0, np.nan, 3.0, 4.0],
                }
            )
            for year, frame in rows.groupby(rows["datetime"].dt.year):
                (root / "bars" / f"year={year}").mkdir(parents=True)
                frame.to_parquet(root / "bars" / f"year={year}" / "bars.parquet", index=False)

            concept_path, index_path = write_industry_concepts(root / "bars", root / "hist", date(2021, 3, 1))
            matrix = np.load(concept_path)
            index = np.load(index_path, allow_pickle=True).item()

        self.assertEqual({"SH600000": 0, "SH600001": 1}, index)
        np.testing.assert_array_equal([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], matrix)


@unittest.skipUnless(HAS_STACK, "pyqlib, PyTorch and pyarrow are required")
class NeuralWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.temporary.name)
        cls.qlib_root, cls.factor_root, cls.calendar = write_store(
            cls.root, stocks=24, factor_set="qlib_alpha360", start="2019-06-03"
        )
        write_store(cls.root / "a101", stocks=24, start="2019-06-03")
        cls.instruments = [f"SH6{value:05d}" for value in range(24)]
        cls.canonical_root = write_industries(cls.root, cls.instruments, cls.calendar)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    # Break caught: a network early-stopping on validation months it also trained on, a time-series window that
    # starts at the train start (first samples padded with zeros), or MTSDatasetH guessing the label horizon.
    def test_neural_tasks_hold_out_validation_and_pad_time_series_windows(self) -> None:
        import qlib
        from qlib.constant import REG_CN
        from qlib.workflow.task.gen import task_generator

        from rquant.workflow.rolling import generate_rolling_windows, previous_trading_day
        from rquant.workflow.tasks import BUILT_IN_WORKFLOWS, CalendarYearRollingGen, load_workflow, rolling_template

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN)
        window = generate_rolling_windows(
            first_prediction_year=2021, last_prediction_year=2021, horizon="1d", through=date(2021, 12, 20),
            train_years=1, validation_months=6, formal_start_year=2020,
        )[0]
        for name, padding in (("gru", 0), ("tra", 60), ("gru-ts", 20)):
            with self.subTest(model=name):
                workflow = load_workflow(BUILT_IN_WORKFLOWS[name])
                self.assertTrue(workflow["rolling"]["holdout_valid"])
                template = rolling_template(
                    workflow["task"], factor_root=self.factor_root, factor_set=None, horizon="1d", universe="all"
                )
                (task,) = task_generator(
                    template,
                    CalendarYearRollingGen(
                        first_year=2021, last_year=2021, through=date(2021, 12, 20), horizon="1d", train_years=1,
                        validation_months=6, formal_start_year=2020, holdout_valid=True,
                    ),
                )
                segments = {key: (start.date(), end.date()) for key, (start, end) in
                            task["dataset"]["kwargs"]["segments"].items()}
                self.assertEqual(previous_trading_day(self.calendar, window.validation_start, 1),
                                 segments["train"][1])
                self.assertEqual(previous_trading_day(self.calendar, window.test_start, 1), segments["valid"][1])
                self.assertLess(segments["train"][1], segments["valid"][0])
                handler = task["dataset"]["kwargs"]["handler"]["kwargs"]
                start = handler["start_time"].date()
                trading_before = [day for day in self.calendar if start <= day < segments["train"][0]]
                self.assertEqual(padding, len(trading_before))
                zscore = handler["infer_processors"][1]["kwargs"]
                self.assertEqual(segments["train"], (zscore["fit_start_time"].date(), zscore["fit_end_time"].date()))
                if name == "tra":
                    self.assertEqual(2, task["dataset"]["kwargs"]["horizon"])

    # Break caught: a Qlib PyTorch model failing inside RQuant's rolling training (dataset, concepts, d_feat,
    # prediction shape) or leaving test days unpredicted.
    def test_each_network_trains_and_predicts_a_rolling_year(self) -> None:
        import numpy as np
        import pandas as pd
        import torch

        from rquant.workflow.runner import RollingWorkflowRunner

        cases = {
            "gru": {"n_epochs": 2, "hidden_size": 8, "batch_size": 256},
            "alstm": {"n_epochs": 2, "hidden_size": 8, "batch_size": 256},
            # HIST copies a default-sized (64) backbone into itself, so it keeps hidden_size 64.
            "hist": {"n_epochs": 2},
            "tra": {"n_epochs": 2, "model_config": {"input_size": 6, "hidden_size": 8, "num_layers": 1,
                                                     "rnn_arch": "LSTM", "use_attn": True, "dropout": 0.0}},
            "gru-ts": {"n_epochs": 2, "hidden_size": 8, "batch_size": 256, "n_jobs": 0},
        }
        for name, overrides in cases.items():
            with self.subTest(model=name):
                run_directory = self.root / f"run-{name}"
                run_directory.mkdir()
                runner = RollingWorkflowRunner(
                    qlib_root=self.qlib_root,
                    factor_root=self.root / "a101" / "factors" if name == "gru-ts" else self.factor_root,
                    run_directory=run_directory, workflow=name, horizon="1d", seed=3, train_years=1,
                    validation_months=6, formal_start_year=2020, universe="all", canonical_root=self.canonical_root,
                    factor_set="wq_alpha101" if name == "gru-ts" else None,
                )
                # TRA picks its device itself; the others take GPU (-1 = CPU).
                runner.template["model"]["kwargs"].update(**({} if name == "tra" else {"GPU": -1}), **overrides)
                if name == "tra":
                    runner.template["dataset"]["kwargs"]["batch_size"] = 128
                with contextlib.ExitStack() as stack:
                    if name == "hist":
                        # The test runs HIST on the CPU (GPU -1) even on a CUDA machine; Qlib's HISTModel.forward asks
                        # torch.get_device for its device, which is -1 on the CPU.
                        stack.enter_context(mock.patch.object(torch, "get_device", lambda tensor: "cpu"))
                    manifest = runner.run(first_year=2021, last_year=2021, through=date(2021, 12, 20))

                predictions = pd.read_parquet(run_directory / "predictions.parquet")
                self.assertTrue(np.isfinite(predictions["score"]).all())
                days = predictions["datetime"].dt.date.unique()
                test_days = [day for day in self.calendar if date(2021, 1, 1) <= day <= date(2021, 12, 20)]
                # The provider runs past `through`, so even TRA (which predicts from the learn view and so needs a
                # label) covers every test day here.
                self.assertEqual(set(test_days), set(days))
                self.assertEqual(1, len(manifest["windows"]))
                if name == "gru-ts":
                    self.assertEqual(101, manifest["model_config"]["kwargs"]["d_feat"])
                if name == "hist":
                    concepts = np.load(run_directory / "hist" / "stock2concept.npy")
                    self.assertEqual((24, 5), concepts.shape)
                    self.assertTrue((concepts.sum(axis=1) >= 1).all())


if __name__ == "__main__":
    unittest.main()
