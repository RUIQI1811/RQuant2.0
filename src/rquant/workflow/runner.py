"""Walk-forward training with Qlib's rolling tools.

The workflow file's task is rolled by ``CalendarYearRollingGen`` (a ``RollingGen``), every yearly task is trained by
``TrainerR`` with ``rquant.workflow.trainer.train_rolling_task`` as its ``train_func``, and the yearly predictions are
merged by ``RecorderCollector`` with ``RollingEnsemble``, as ``qlib.contrib.rolling`` does. Every recorder lives in
the run directory: ``mlflow.db`` holds the runs and ``mlruns/`` their artifacts.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.factors.catalog import get_catalog
from rquant.io import atomic_write_json
from rquant.qlib_ext.runtime import qlib_init_kwargs
from rquant.workflow.tasks import (
    BUILT_IN_WORKFLOWS,
    CalendarYearRollingGen,
    load_workflow,
    rolling_template,
    task_factor_set,
    task_horizon,
)
from rquant.workflow.trainer import train_rolling_task

ENSEMBLE_RECORDER = "rolling-ensemble"


class RollingWorkflowRunner:
    def __init__(
        self,
        *,
        qlib_root: str | Path,
        factor_root: str | Path,
        run_directory: str | Path,
        workflow: str | Path,
        horizon: str | None = None,
        factor_set: str | None = None,
        seed: int = 42,
        train_years: int = 3,
        validation_months: int = 6,
        formal_start_year: int = 2010,
        universe: str = "csi300",
    ) -> None:
        """``workflow`` is a qrun workflow file or the name of a built-in one (``lgb``, ``double-ensemble``);
        ``horizon`` and ``factor_set`` override its data loader's."""
        self.qlib_root = Path(qlib_root)
        self.factor_root = Path(factor_root)
        self.run_directory = Path(run_directory)
        self.model_name = str(workflow) if str(workflow) in BUILT_IN_WORKFLOWS else Path(workflow).stem
        self.workflow_path = BUILT_IN_WORKFLOWS.get(str(workflow), Path(workflow))
        self.workflow = load_workflow(self.workflow_path)
        self.seed = seed
        self.train_years = train_years
        self.validation_months = validation_months
        self.formal_start_year = formal_start_year
        self.universe = universe
        self.template = rolling_template(
            self.workflow["task"],
            factor_root=self.factor_root,
            factor_set=factor_set,
            horizon=horizon,
            universe=universe,
        )
        model_kwargs = self.template["model"].setdefault("kwargs", {})
        if "seed" in model_kwargs:
            model_kwargs["seed"] = seed
        self.horizon = task_horizon(self.template)
        self.factor_set = task_factor_set(self.template)
        get_catalog().select(self.factor_set)

    @property
    def experiment_name(self) -> str:
        return f"rquant-{self.factor_set}-{self.horizon}-{self.model_name}"

    def run(self, *, first_year: int, last_year: int, through: date) -> dict[str, Any]:
        try:
            import pandas as pd
            import qlib
            from qlib.constant import REG_CN
            from qlib.model.ens.ensemble import RollingEnsemble
            from qlib.model.trainer import TrainerR
            from qlib.workflow import R
            from qlib.workflow.task.collect import RecorderCollector
            from qlib.workflow.task.gen import task_generator
        except ImportError as exc:
            raise DependencyError("pyqlib==0.9.7, pandas and LightGBM are required for rolling training") from exc

        tracking_uri = init_run_tracking(
            self.run_directory, self.experiment_name, provider_uri=self.qlib_root, region=REG_CN, qlib_module=qlib
        )
        from qlib.data import D

        calendar = D.calendar(start_time="2009-01-01", end_time=through.isoformat(), freq="day")
        anchor_positions = (
            {value.date(): position for position, value in enumerate(calendar)} if self.horizon == "5d" else None
        )
        tasks = task_generator(
            self.template,
            CalendarYearRollingGen(
                first_year=first_year,
                last_year=last_year,
                through=through,
                horizon=self.horizon,
                train_years=self.train_years,
                validation_months=self.validation_months,
                formal_start_year=self.formal_start_year,
            ),
        )
        if not tasks:
            raise DataContractError("No rolling windows are available for the requested range")

        trainer = TrainerR(experiment_name=self.experiment_name, train_func=train_rolling_task)
        recorders = trainer.train(
            tasks,
            seed=self.seed,
            refit=bool(self.workflow["rolling"]["refit"]),
            anchor_positions=anchor_positions,
            run_directory=self.run_directory,
            factor_names=list(get_catalog().canonical_names(self.factor_set)),
        )
        trainer.end_train(recorders)
        trained = {recorder.id for recorder in recorders}
        collected = RecorderCollector(
            experiment=self.experiment_name,
            process_list=[RollingEnsemble()],
            rec_filter_func=lambda recorder: recorder.id in trained,
            artifacts_key=["pred"],
        )()
        combined = collected["pred"]
        with R.start(experiment_name=self.experiment_name, recorder_name=ENSEMBLE_RECORDER):
            R.log_params(rolling_experiment=self.experiment_name, windows=len(tasks))
            R.save_objects(**{"pred.pkl": combined})
        combined.reset_index().to_parquet(self.run_directory / "predictions.parquet", index=False)

        windows = []
        importance = []
        for directory in sorted(self.run_directory.glob("year=*")):
            with (directory / "window.json").open("r", encoding="utf-8") as handle:
                windows.append(json.load(handle))
            path = directory / "feature_importance.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                frame.insert(0, "prediction_year", windows[-1]["prediction_year"])
                importance.append(frame)
        if importance:
            pd.concat(importance, ignore_index=True).to_parquet(
                self.run_directory / "feature_importance.parquet", index=False
            )
        manifest = {
            "status": "complete",
            "engine": "qlib RollingGen (calendar years) + TrainerR + RollingEnsemble",
            "model": self.model_name,
            "workflow": str(self.workflow_path),
            "model_config": self.template["model"],
            "rolling": self.workflow["rolling"],
            "horizon": self.horizon,
            "factor_set": self.factor_set,
            "seed": self.seed,
            "train_years": self.train_years,
            "validation_months": self.validation_months,
            "universe": self.universe,
            "windows": windows,
            "prediction_rows": len(combined),
            "catalog_fingerprint": get_catalog().fingerprint_for(self.factor_set),
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment": self.experiment_name,
        }
        atomic_write_json(self.run_directory / "walk_forward.json", manifest)
        release_run_tracking(self.run_directory)
        return manifest


def init_run_tracking(
    run_directory: Path, experiment_name: str, *, provider_uri: Path, region: str, qlib_module: Any
) -> str:
    """Initialise Qlib with an MLflow store inside ``run_directory`` and create ``experiment_name`` there.

    MLflow's file store refuses to start unless MLFLOW_ALLOW_FILE_STORE is set, and a SQLite store puts artifacts under
    the working directory unless the experiment names a location, so the experiment is created here with its
    artifacts in ``run_directory/mlruns``.
    """
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{run_directory / 'mlflow.db'}"
    qlib_module.init(
        provider_uri=str(provider_uri),
        region=region,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": tracking_uri, "default_exp_name": experiment_name},
        },
        **qlib_init_kwargs(),
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=str(run_directory / "mlruns"))
    return tracking_uri


def release_run_tracking(run_directory: Path) -> None:
    """Close MLflow's SQLite connections to ``run_directory/mlflow.db``.

    MLflow keeps a pooled engine per database for the life of the process, which holds the file open; Windows then
    refuses to delete or replace the run's store until the process exits.
    """
    import gc

    try:
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    except ImportError:  # pragma: no cover - MLflow comes with Qlib
        return
    database = (run_directory / "mlflow.db").resolve()

    def ours(engine: Any) -> bool:
        name = getattr(getattr(engine, "url", None), "database", None)
        return bool(name) and Path(name).resolve() == database

    engines = getattr(SqlAlchemyStore, "_engine_map", {})
    for uri, engine in list(engines.items()):
        if ours(engine):
            engines.pop(uri, None)
            engine.dispose()
    for item in gc.get_objects():
        if isinstance(item, SqlAlchemyStore) and ours(getattr(item, "engine", None)):
            item.engine.dispose()
