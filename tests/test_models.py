from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is required")
class ModelConstructionTests(unittest.TestCase):
    def test_both_locked_model_adapters_construct(self) -> None:
        from rquant.workflow.runner import RollingWorkflowRunner

        for name, expected in (("lgb", "LGBModel"), ("double-ensemble", "DEnsembleModel")):
            with self.subTest(model=name):
                runner = RollingWorkflowRunner(
                    qlib_root="unused",
                    factor_root="unused",
                    run_directory="unused",
                    model=name,
                    horizon="1d",
                    factor_set="Hquality",
                )
                self.assertEqual(expected, type(runner._model(iterations=7, refit=True)).__name__)

    def test_fixed_panel_runs_two_models_two_horizons_three_factor_sets(self) -> None:
        import numpy as np
        import pandas as pd
        import qlib
        from mlflow.tracking import MlflowClient
        from qlib.constant import REG_CN
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.data.dataset.loader import StaticDataLoader
        from qlib.workflow import R

        from rquant.factors.catalog import get_catalog
        from rquant.workflow.runner import RollingWorkflowRunner

        rng = np.random.default_rng(42)
        # MLflow keeps its SQLite file open for the process lifetime, which blocks deletion on Windows.
        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        provider = root / "provider"
        provider.mkdir()
        tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        MlflowClient(tracking_uri=tracking_uri).create_experiment(
            "RQuantTest", artifact_location=(root / "artifacts").as_uri()
        )
        qlib.init(
            provider_uri=str(provider),
            region=REG_CN,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": tracking_uri, "default_exp_name": "RQuantTest"},
            },
        )
        dates = pd.bdate_range("2022-01-03", periods=56)
        instruments = [f"S{value:03d}" for value in range(12)]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        maximum_features = max(len(get_catalog().canonical_names(name)) for name in ("qlib_alpha158", "Hquality"))
        matrix = rng.normal(size=(len(index), maximum_features))
        segments = {
            "train": (dates[0], dates[34]),
            "valid": (dates[35], dates[44]),
            "test": (dates[45], dates[-1]),
        }
        with R.start(experiment_name="RQuantTest", recorder_name="acceptance-grid"):
            for factor_set in ("qlib_alpha158", "wq_alpha101", "Hquality"):
                names = get_catalog().canonical_names(factor_set)
                features = pd.DataFrame(matrix[:, : len(names)], index=index, columns=names)
                for horizon, multiplier in (("1d", 1.0), ("5d", 5.0)):
                    label = multiplier * (0.4 * matrix[:, 0] - 0.2 * matrix[:, 1])
                    handler = DataHandlerLP(
                        instruments=None,
                        start_time=dates[0],
                        end_time=dates[-1],
                        data_loader=StaticDataLoader(
                            {"feature": features, "label": pd.DataFrame({"LABEL0": label}, index=index)}
                        ),
                        infer_processors=[],
                        learn_processors=[],
                    )
                    dataset = DatasetH(handler=handler, segments=segments)
                    for model_name in ("lgb", "double-ensemble"):
                        with self.subTest(factor_set=factor_set, horizon=horizon, model=model_name):
                            runner = RollingWorkflowRunner(
                                qlib_root="unused",
                                factor_root="unused",
                                run_directory="unused",
                                model=model_name,
                                horizon=horizon,
                                factor_set=factor_set,
                                seed=42,
                            )
                            model = runner._model(iterations=2, refit=True)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FutureWarning)
                                model.fit(dataset)
                            prediction = model.predict(dataset, segment="test")
                            self.assertEqual(len(dates[45:]) * len(instruments), len(prediction))
                            self.assertTrue(np.isfinite(prediction).all())



@unittest.skipUnless(importlib.util.find_spec("qlib") is not None, "pyqlib is required")
class LeanLGBWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        import qlib
        from mlflow.tracking import MlflowClient
        from qlib.constant import REG_CN

        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "provider").mkdir()
        tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        MlflowClient(tracking_uri=tracking_uri).create_experiment(
            "RQuantLean", artifact_location=(root / "artifacts").as_uri()
        )
        qlib.init(
            provider_uri=str(root / "provider"),
            region=REG_CN,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": tracking_uri, "default_exp_name": "RQuantLean"},
            },
        )

    def _panel(self, rng):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        from rquant.factors.catalog import get_catalog

        dates = pd.bdate_range("2022-01-03", periods=90)
        instruments = [f"SH6{value:05d}" for value in range(40)]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        names = get_catalog().canonical_names("wq_alpha101")
        values = rng.normal(size=(len(index), len(names) + 1)).astype(np.float32)
        values[:, -1] = 0.3 * values[:, 0] - 0.2 * values[:, 1] + rng.normal(scale=0.5, size=len(index))
        values[rng.random(values.shape) < 0.05] = np.nan
        values[rng.random(values.shape) < 0.002] = np.inf
        values[:40, 3] = -np.inf  # a whole date infinite in one column
        values[len(index) - 40 :, -1] = np.nan  # labels not yet known at the end
        columns = pd.MultiIndex.from_tuples([("feature", name) for name in names] + [("label", "LABEL0")])
        return pd.DataFrame(np.asfortranarray(values), index=index, columns=columns, copy=False), dates

    # Break caught: a lean LightGBM window whose processing, segments or binning drift from the Qlib dataset path.
    def test_lean_window_predicts_exactly_what_the_qlib_dataset_path_predicts(self) -> None:
        from unittest import mock

        import numpy as np
        import pandas as pd
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R

        from rquant.workflow import runner as runner_module
        from rquant.workflow.runner import RollingWorkflowRunner

        panel, dates = self._panel(np.random.default_rng(5))

        from qlib.data.dataset.loader import DataLoader

        class FixedLoader(DataLoader):
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            def load(self, instruments=None, start_time=None, end_time=None):  # type: ignore[no-untyped-def]
                rows = panel.index.get_level_values("datetime")
                keep = (rows >= pd.Timestamp(start_time)) & (rows <= pd.Timestamp(end_time))
                return pd.DataFrame(np.asfortranarray(panel.to_numpy()[keep]), index=panel.index[keep],
                                    columns=panel.columns, copy=False)

        runner = RollingWorkflowRunner(
            qlib_root="unused", factor_root="unused", run_directory="unused", model="lgb", horizon="1d",
            factor_set="wq_alpha101", seed=42,
        )
        day = [value.date() for value in dates]
        cases = {
            "tune": (runner._model(), day[60], day[70], {
                "train": (day[0].isoformat(), day[60].isoformat()),
                "valid": (day[62].isoformat(), day[70].isoformat()),
            }),
            "refit": (runner._model(iterations=7, refit=True), day[75], day[-1], {
                "train": (day[1].isoformat(), day[75].isoformat()),
                "valid": (day[62].isoformat(), day[75].isoformat()),
                "test": (day[78].isoformat(), day[-1].isoformat()),
            }),
        }
        for name, (model, fit_end, data_end, segments) in cases.items():
            with self.subTest(case=name), mock.patch.object(runner_module, "KunQuantDataLoader", FixedLoader):
                qlib_model = runner._model(**({} if name == "tune" else {"iterations": 7, "refit": True}))
                if name == "tune":
                    # Early stopping stays on, but the run ends while the validation loss still falls: further out it
                    # flattens on this small panel and multithreaded metric sums pick among equal values, so the stop
                    # would vary from run to run within either path.
                    qlib_model.num_boost_round = model.num_boost_round = 60
                with R.start(experiment_name="RQuantLean", recorder_name=f"qlib-{name}"):
                    handler = runner._handler(DataHandlerLP, fit_start=day[0], fit_end=fit_end, data_end=data_end)
                    dataset = DatasetH(handler=handler, segments=segments)
                    qlib_model.fit(dataset)
                    expected = qlib_model.predict(dataset, segment="test") if "test" in segments else None
                with R.start(experiment_name="RQuantLean", recorder_name=f"lean-{name}"):
                    actual = runner._fit(
                        model, DataHandlerLP, DatasetH, R, fit_start=day[0], fit_end=fit_end, data_end=data_end,
                        segments=segments,
                    )
                self.assertEqual(qlib_model.model.best_iteration, model.model.best_iteration)
                self.assertEqual(qlib_model.model.num_trees(), model.model.num_trees())
                self.assertEqual(
                    qlib_model.model.model_to_string(num_iteration=-1), model.model.model_to_string(num_iteration=-1)
                )
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    pd.testing.assert_series_equal(expected, actual, check_exact=True)

    def test_moving_rows_to_front_leaves_a_column_major_copy_of_them(self) -> None:
        import numpy as np

        from rquant.workflow.lgb_window import move_rows_to_front

        values = np.asfortranarray(np.arange(60, dtype=np.float32).reshape(12, 5))
        expected = values[2:9, :4].copy()
        moved = move_rows_to_front(values, slice(2, 9), 4)
        self.assertTrue(moved.flags["F_CONTIGUOUS"])
        self.assertTrue(np.shares_memory(moved, values))
        np.testing.assert_array_equal(expected, moved)


if __name__ == "__main__":
    unittest.main()
