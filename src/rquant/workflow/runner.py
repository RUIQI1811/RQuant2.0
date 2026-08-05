from __future__ import annotations

import pickle
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

from rquant.errors import DataContractError, DependencyError
from rquant.factors.catalog import FactorSet, get_catalog
from rquant.io import atomic_write_json, sha256_file
from rquant.qlib_ext.loader import KunQuantDataLoader
from rquant.workflow.rolling import generate_rolling_windows, previous_trading_day

ModelName = Literal["lgb", "double-ensemble"]


class RollingWorkflowRunner:
    def __init__(
        self,
        *,
        qlib_root: str | Path,
        factor_root: str | Path,
        run_directory: str | Path,
        model: ModelName,
        horizon: Literal["1d", "5d"],
        factor_set: FactorSet,
        seed: int = 42,
    ) -> None:
        self.qlib_root = Path(qlib_root)
        self.factor_root = Path(factor_root)
        self.run_directory = Path(run_directory)
        self.model_name = model
        self.horizon = horizon
        self.factor_set = factor_set
        self.seed = seed
        get_catalog().select(factor_set)

    def run(self, *, first_year: int, last_year: int, through: date) -> dict[str, Any]:
        qlib, pd, np, R, DataHandlerLP, DatasetH = _qlib_dependencies()
        from qlib.constant import REG_CN

        tracking_uri = f"sqlite:///{self.run_directory / 'mlflow.db'}"
        qlib.init(
            provider_uri=str(self.qlib_root),
            region=REG_CN,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": tracking_uri, "default_exp_name": "RQuant"},
            },
        )
        from qlib.data import D

        calendar = [
            value.date() for value in D.calendar(start_time="2009-01-01", end_time=through.isoformat(), freq="day")
        ]
        calendar_positions = {value: position for position, value in enumerate(calendar)}
        windows = generate_rolling_windows(
            first_prediction_year=first_year,
            last_prediction_year=last_year,
            horizon=self.horizon,
            through=through,
        )
        if not windows:
            raise DataContractError("No rolling windows are available for the requested range")

        all_predictions = []
        window_manifests = []
        for window in windows:
            np.random.seed(self.seed + window.prediction_year)
            window_dir = self.run_directory / f"year={window.prediction_year}"
            window_dir.mkdir(parents=True, exist_ok=False)
            usable_inner_end = previous_trading_day(calendar, window.validation_start, window.purge_trading_days)
            usable_refit_end = previous_trading_day(calendar, window.test_start, window.purge_trading_days)

            tune_handler = self._handler(
                DataHandlerLP,
                fit_start=window.train_start,
                fit_end=usable_inner_end,
                data_end=window.validation_end,
            )
            tune_dataset = DatasetH(
                handler=tune_handler,
                segments={
                    "train": (window.train_start.isoformat(), usable_inner_end.isoformat()),
                    "valid": (window.validation_start.isoformat(), window.validation_end.isoformat()),
                },
            )
            tuning_model = self._model()
            with R.start(
                experiment_name=f"rquant-{self.factor_set}-{self.horizon}-{self.model_name}",
                recorder_name=f"tune-{window.prediction_year}",
            ):
                tuning_model.fit(tune_dataset)
            chosen_iterations = _best_iterations(tuning_model)

            final_handler = self._handler(
                DataHandlerLP,
                fit_start=window.train_start,
                fit_end=usable_refit_end,
                data_end=window.test_end,
            )
            final_dataset = DatasetH(
                handler=final_handler,
                segments={
                    "train": (window.train_start.isoformat(), usable_refit_end.isoformat()),
                    # DoubleEnsemble requires a validation segment. It remains inside the three-year training window.
                    "valid": (window.validation_start.isoformat(), usable_refit_end.isoformat()),
                    "test": (window.test_start.isoformat(), window.test_end.isoformat()),
                },
            )
            final_model = self._model(iterations=chosen_iterations, refit=True)
            with R.start(
                experiment_name=f"rquant-{self.factor_set}-{self.horizon}-{self.model_name}",
                recorder_name=f"refit-{window.prediction_year}",
            ):
                final_model.fit(final_dataset)
                prediction = final_model.predict(final_dataset, segment="test").rename("score")
            prediction_frame = prediction.reset_index()
            if self.horizon == "5d":
                global_positions = pd.to_datetime(prediction_frame["datetime"]).dt.date.map(calendar_positions)
                prediction_frame = prediction_frame[global_positions.mod(5).eq(0)].reset_index(drop=True)
            prediction_path = window_dir / "predictions.parquet"
            prediction_frame.to_parquet(prediction_path, index=False)
            with (window_dir / "model.pkl").open("wb") as handle:
                pickle.dump(final_model, handle)

            window_manifest = {
                **asdict(window),
                "train_end_after_purge": usable_refit_end.isoformat(),
                "inner_train_end_after_purge": usable_inner_end.isoformat(),
                "chosen_iterations": chosen_iterations,
                "prediction_rows": len(prediction_frame),
                "prediction_fingerprint": sha256_file(prediction_path),
            }
            for key, value in tuple(window_manifest.items()):
                if isinstance(value, date):
                    window_manifest[key] = value.isoformat()
            atomic_write_json(window_dir / "window.json", window_manifest)
            window_manifests.append(window_manifest)
            all_predictions.append(prediction_frame)

        combined = pd.concat(all_predictions, ignore_index=True).sort_values(["datetime", "instrument"])
        if combined.duplicated(["datetime", "instrument"]).any():
            raise DataContractError("Rolling predictions overlap across test years")
        combined.to_parquet(self.run_directory / "predictions.parquet", index=False)
        manifest = {
            "status": "complete",
            "model": self.model_name,
            "horizon": self.horizon,
            "factor_set": self.factor_set,
            "seed": self.seed,
            "windows": window_manifests,
            "prediction_rows": len(combined),
            "catalog_fingerprint": get_catalog().fingerprint,
            "mlflow_tracking_uri": tracking_uri,
        }
        atomic_write_json(self.run_directory / "walk_forward.json", manifest)
        return manifest

    def _handler(self, DataHandlerLP: Any, *, fit_start: date, fit_end: date, data_end: date) -> Any:
        loader = KunQuantDataLoader(str(self.factor_root), self.factor_set, self.horizon)
        return DataHandlerLP(
            instruments="csi300",
            start_time=fit_start.isoformat(),
            end_time=data_end.isoformat(),
            data_loader=loader,
            infer_processors=[
                {"class": "ProcessInf"},
                {
                    "class": "ZScoreNorm",
                    "kwargs": {"fit_start_time": fit_start.isoformat(), "fit_end_time": fit_end.isoformat()},
                },
                {"class": "Fillna"},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ],
            process_type=DataHandlerLP.PTYPE_A,
        )

    def _model(self, *, iterations: int | None = None, refit: bool = False) -> Any:
        try:
            from qlib.contrib.model.double_ensemble import DEnsembleModel
            from qlib.contrib.model.gbdt import LGBModel
        except ImportError as exc:
            raise DependencyError("pyqlib==0.9.7 and lightgbm are required for rolling training") from exc
        common = {
            "loss": "mse",
            "learning_rate": 0.0421,
            "colsample_bytree": 0.8879,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 8,
            "seed": self.seed,
        }
        if self.model_name == "lgb":
            return LGBModel(
                early_stopping_rounds=0 if refit else 50,
                num_boost_round=iterations or 1000,
                **common,
            )
        return DEnsembleModel(
            num_models=6,
            enable_sr=True,
            enable_fs=True,
            alpha1=0.23,
            alpha2=0.05,
            bins_sr=10,
            bins_fs=5,
            decay=0.5,
            sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
            sub_weights=[1, 0.2, 0.2, 0.2, 0.2, 0.2],
            epochs=iterations or 100,
            early_stopping_rounds=None if refit else 20,
            **common,
        )


def _best_iterations(model: Any) -> int:
    booster = getattr(model, "model", None)
    if booster is not None:
        value = int(getattr(booster, "best_iteration", 0) or getattr(booster, "num_trees", lambda: 0)())
        return max(1, value)
    ensemble = getattr(model, "ensemble", None)
    if ensemble:
        values = sorted(int(item.best_iteration or item.num_trees()) for item in ensemble)
        return max(1, values[len(values) // 2])
    raise DataContractError("Unable to determine validated boosting iterations")


def _qlib_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.workflow import R
    except ImportError as exc:
        raise DependencyError("pyqlib==0.9.7, pandas, NumPy and LightGBM are required for rolling training") from exc
    return qlib, pd, np, R, DataHandlerLP, DatasetH
