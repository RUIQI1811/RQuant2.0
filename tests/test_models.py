from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path


def built_in_model(name: str, **overrides):  # type: ignore[no-untyped-def]
    """The model of a built-in workflow, with ``overrides`` on its kwargs."""
    import copy

    from qlib.utils import init_instance_by_config

    from rquant.workflow.tasks import BUILT_IN_WORKFLOWS, load_workflow

    config = copy.deepcopy(load_workflow(BUILT_IN_WORKFLOWS[name])["task"]["model"])
    config["kwargs"].update(overrides)
    return init_instance_by_config(config)


def refit_overrides(name: str, iterations: int) -> dict:  # type: ignore[type-arg]
    if name == "lgb":
        return {"num_boost_round": iterations, "early_stopping_rounds": 0}
    return {"epochs": iterations}


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is required")
class ModelConstructionTests(unittest.TestCase):
    def test_both_built_in_workflows_construct_their_models(self) -> None:
        for name, expected in (("lgb", "LGBModel"), ("double-ensemble", "DEnsembleModel")):
            with self.subTest(model=name):
                self.assertEqual(expected, type(built_in_model(name, **refit_overrides(name, 7))).__name__)

    # Break caught: built-in workflows whose model parameters drift from the ones every past run was trained with.
    def test_built_in_models_keep_the_trained_parameters(self) -> None:
        lgb_model = built_in_model("lgb")
        self.assertEqual((50, 1000), (lgb_model.early_stopping_rounds, lgb_model.num_boost_round))
        self.assertEqual(
            {
                "objective": "mse", "verbosity": -1, "learning_rate": 0.0421, "colsample_bytree": 0.8879,
                "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210,
                "num_threads": 8, "seed": 42,
            },
            lgb_model.params,
        )
        ensemble = built_in_model("double-ensemble")
        self.assertEqual((3, 28, None), (ensemble.num_models, ensemble.epochs, ensemble.early_stopping_rounds))
        self.assertEqual([0.8, 0.7, 0.6, 0.5, 0.4], ensemble.sample_ratios)
        self.assertEqual(0.2, ensemble.params["learning_rate"])

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
        maximum_features = max(len(get_catalog().canonical_names(name)) for name in ("qlib_alpha158", "oap_jq_cnc"))
        matrix = rng.normal(size=(len(index), maximum_features))
        segments = {
            "train": (dates[0], dates[34]),
            "valid": (dates[35], dates[44]),
            "test": (dates[45], dates[-1]),
        }
        with R.start(experiment_name="RQuantTest", recorder_name="acceptance-grid"):
            for factor_set in ("qlib_alpha158", "wq_alpha101", "oap_jq_cnc"):
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
                            model = built_in_model(model_name, **refit_overrides(model_name, 2))
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

    def _task(self, panel, dates_kept, *, fit_end, data_end, segments, model_name):  # type: ignore[no-untyped-def]
        """A rolling task over ``panel`` whose data loader is a KunQuantDataLoader serving it."""
        import numpy as np
        import pandas as pd

        from rquant.qlib_ext.loader import KunQuantDataLoader
        from rquant.workflow.tasks import handler_config

        class FixedLoader(KunQuantDataLoader):
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            def load(self, instruments=None, start_time=None, end_time=None):  # type: ignore[no-untyped-def]
                rows = panel.index.get_level_values("datetime")
                keep = (rows >= pd.Timestamp(start_time)) & (rows <= pd.Timestamp(end_time))
                return pd.DataFrame(np.asfortranarray(panel.to_numpy()[keep]), index=panel.index[keep],
                                    columns=panel.columns, copy=False)

        handler = handler_config(
            factor_root="unused", factor_set="wq_alpha101", horizon="1d", universe="all",
            start_time=dates_kept[0], end_time=data_end, fit_start_time=dates_kept[0], fit_end_time=fit_end,
        )
        handler["kwargs"]["data_loader"] = {"class": FixedLoader, "kwargs": {}}
        return {
            "model": {"class": "unused"},
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {"handler": handler, "segments": segments},
            },
        }

    # Break caught: a lean LightGBM window whose processing, segments or binning drift from the Qlib dataset path.
    def test_lean_window_predicts_exactly_what_the_qlib_dataset_path_predicts(self) -> None:
        import numpy as np
        import pandas as pd
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R

        from rquant.workflow.trainer import fit_task, lean_window

        panel, dates = self._panel(np.random.default_rng(5))
        day = [value.date().isoformat() for value in dates]
        cases = {
            "tune": (built_in_model("lgb"), day[60], day[70], {
                "train": (day[0], day[60]),
                "valid": (day[62], day[70]),
            }),
            "refit": (built_in_model("lgb", **refit_overrides("lgb", 7)), day[75], day[-1], {
                "train": (day[1], day[75]),
                "valid": (day[62], day[75]),
                "test": (day[78], day[-1]),
            }),
        }
        for name, (model, fit_end, data_end, segments) in cases.items():
            with self.subTest(case=name):
                task = self._task(panel, day, fit_end=fit_end, data_end=data_end, segments=segments, model_name="lgb")
                self.assertIsNotNone(lean_window(task))
                qlib_model = built_in_model("lgb", **({} if name == "tune" else refit_overrides("lgb", 7)))
                if name == "tune":
                    # Early stopping stays on, but the run ends while the validation loss still falls: further out it
                    # flattens on this small panel and multithreaded metric sums pick among equal values, so the stop
                    # would vary from run to run within either path.
                    qlib_model.num_boost_round = model.num_boost_round = 60
                with R.start(experiment_name="RQuantLean", recorder_name=f"qlib-{name}"):
                    dataset = init_instance_by_config(task["dataset"])
                    qlib_model.fit(dataset)
                    expected = qlib_model.predict(dataset, segment="test") if "test" in segments else None
                with R.start(experiment_name="RQuantLean", recorder_name=f"lean-{name}"):
                    actual = fit_task(model, task, log_metrics=R.log_metrics)
                self.assertEqual(qlib_model.model.best_iteration, model.model.best_iteration)
                self.assertEqual(qlib_model.model.num_trees(), model.model.num_trees())
                self.assertEqual(
                    qlib_model.model.model_to_string(num_iteration=-1), model.model.model_to_string(num_iteration=-1)
                )
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    pd.testing.assert_series_equal(expected, actual, check_exact=True)

    # Break caught: a lean DoubleEnsemble window whose segments, labels or feature names drift from the Qlib dataset
    # path, which would change its sample re-weighting, feature selection or predictions.
    def test_lean_double_ensemble_predicts_exactly_what_the_qlib_dataset_path_predicts(self) -> None:
        import numpy as np
        import pandas as pd
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R

        from rquant.workflow.trainer import fit_task

        panel, dates = self._panel(np.random.default_rng(11))
        day = [value.date().isoformat() for value in dates]
        segments = {
            "train": (day[1], day[75]),
            "valid": (day[62], day[75]),
            "test": (day[78], day[-1]),
        }
        task = self._task(panel, day, fit_end=day[75], data_end=day[-1], segments=segments, model_name="de")
        qlib_model = built_in_model("double-ensemble", epochs=6)
        # One thread: LightGBM's multithreaded histogram sums may round differently from run to run.
        qlib_model.params["num_threads"] = 1
        np.random.seed(3)
        dataset = init_instance_by_config(task["dataset"])
        qlib_model.fit(dataset)
        expected = qlib_model.predict(dataset, segment="test")
        model = built_in_model("double-ensemble", epochs=6)
        model.params["num_threads"] = 1
        np.random.seed(3)
        with R.start(experiment_name="RQuantLean", recorder_name="lean-double-ensemble"):
            actual = fit_task(model, task)
        self.assertEqual(3, len(model.ensemble))
        for left, right in zip(qlib_model.sub_features, model.sub_features, strict=True):
            self.assertEqual(sorted(left), sorted(right))
        for left, right in zip(qlib_model.ensemble, model.ensemble, strict=True):
            self.assertEqual(left.model_to_string(num_iteration=-1), right.model_to_string(num_iteration=-1))
        pd.testing.assert_series_equal(expected, actual, check_exact=True)

        from rquant.factors.catalog import get_catalog
        from rquant.workflow.lgb_window import feature_importance

        names = get_catalog().canonical_names("wq_alpha101")
        importance = feature_importance(model, names)
        self.assertEqual(list(names), importance["factor"].tolist())
        expected_gain = qlib_model.get_feature_importance(importance_type="gain").reindex(names).fillna(0.0)
        np.testing.assert_array_equal(expected_gain.to_numpy(), importance["gain"].to_numpy())

    # Break caught: a task whose handler differs from RQuant's processors silently taking the lean path, which would
    # train on differently processed data than the task describes.
    def test_only_rquant_processors_take_the_lean_path(self) -> None:
        import copy

        import numpy as np

        from rquant.workflow.trainer import lean_window

        panel, dates = self._panel(np.random.default_rng(1))
        day = [value.date().isoformat() for value in dates]
        task = self._task(panel, day, fit_end=day[60], data_end=day[-1], segments={}, model_name="lgb")
        self.assertEqual(day[60], lean_window(task)["fit_end_time"])
        changed = copy.deepcopy(task)
        changed["dataset"]["kwargs"]["handler"]["kwargs"]["infer_processors"].pop()
        self.assertIsNone(lean_window(changed))
        changed = copy.deepcopy(task)
        changed["dataset"]["kwargs"]["handler"]["kwargs"]["learn_processors"][1]["kwargs"]["fields_group"] = "feature"
        self.assertIsNone(lean_window(changed))
        changed = copy.deepcopy(task)
        changed["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"] = {"class": "StaticDataLoader"}
        self.assertIsNone(lean_window(changed))

    def test_lgb_feature_importance_names_every_factor_in_store_order(self) -> None:
        import numpy as np
        import pandas as pd

        from rquant.workflow.lgb_window import feature_importance

        rng = np.random.default_rng(2)
        names = [f"f{position}" for position in range(6)]
        x = pd.DataFrame(rng.normal(size=(400, 6)).astype(np.float32), columns=names)
        y = (0.5 * x["f0"] - 0.3 * x["f3"] + rng.normal(scale=0.1, size=400)).to_numpy(np.float32)
        lgb_model = built_in_model("lgb", **refit_overrides("lgb", 5))
        import lightgbm as lgb

        params = dict(lgb_model.params, lambda_l1=0.0, lambda_l2=0.0, min_data_in_leaf=5, verbosity=-1)
        lgb_model.model = lgb.train(params, lgb.Dataset(x.to_numpy(), label=y), num_boost_round=5)
        importance = feature_importance(lgb_model, names)
        self.assertEqual(names, importance["factor"].tolist())
        self.assertAlmostEqual(1.0, importance["gain_share"].sum())
        self.assertEqual("f0", importance.sort_values("gain", ascending=False)["factor"].iloc[0])

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
