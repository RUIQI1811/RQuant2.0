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
                    factor_set="combined",
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
        temporary = tempfile.TemporaryDirectory()
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
        maximum_features = len(get_catalog().canonical_names("combined"))
        matrix = rng.normal(size=(len(index), maximum_features))
        segments = {
            "train": (dates[0], dates[34]),
            "valid": (dates[35], dates[44]),
            "test": (dates[45], dates[-1]),
        }
        with R.start(experiment_name="RQuantTest", recorder_name="acceptance-grid"):
            for factor_set in ("qlib_alpha158", "wq_alpha101", "combined"):
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


if __name__ == "__main__":
    unittest.main()
