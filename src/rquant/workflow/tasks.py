"""Qlib task configurations for RQuant walk-forward runs, and their calendar-year rolling.

A walk-forward run is described by a qrun workflow file (``qlib_init`` plus a ``task`` of ``model``, ``dataset`` and
``record``), the format ``qrun`` itself reads. ``rquant walk-forward`` takes the ``task`` as the template of Qlib's
``RollingGen``: ``CalendarYearRollingGen`` turns it into one task per prediction year, and ``TrainerR`` trains them
(``rquant.workflow.trainer``). The built-in workflows ``lgb`` and ``double-ensemble`` live next to this module.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError
from rquant.workflow.rolling import generate_rolling_windows

try:
    from qlib.workflow.task.gen import RollingGen as _RollingGen
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _RollingGen:  # type: ignore[no-redef]
        pass

WORKFLOW_DIRECTORY = Path(__file__).resolve().parent / "workflows"
BUILT_IN_WORKFLOWS = {
    "lgb": WORKFLOW_DIRECTORY / "lgb.yaml",
    "double-ensemble": WORKFLOW_DIRECTORY / "double_ensemble.yaml",
}

# The label of each horizon is evaluated by KunQuantDataLoader; it needs this many trading days after the prediction
# date, so a training segment must end this many days before the test segment starts (Qlib's ``trunc_days``).
TRUNC_DAYS = {"1d": 2, "5d": 6}

ROLLING_DEFAULTS = {
    # Pick the boosting rounds by early stopping on an inner train/valid split, then train that many rounds on the
    # whole training window (rquant.workflow.trainer). The window lengths are run settings (workflow.train_years and
    # workflow.validation_months in config/default.yaml).
    "refit": False,
}


def infer_processors(fit_start: Any, fit_end: Any) -> list[dict[str, Any]]:
    """The infer processors of every RQuant handler; the lean training path computes exactly these."""
    return [
        {"class": "ProcessInf", "module_path": "rquant.qlib_ext.processors"},
        {
            "class": "ZScoreNorm",
            "module_path": "rquant.qlib_ext.processors",
            "kwargs": {"fit_start_time": fit_start, "fit_end_time": fit_end},
        },
        {"class": "Fillna", "module_path": "rquant.qlib_ext.processors"},
    ]


LEARN_PROCESSORS = [
    {"class": "DropnaLabel", "module_path": "rquant.qlib_ext.processors"},
    {"class": "CSZScoreNorm", "module_path": "qlib.data.dataset.processor", "kwargs": {"fields_group": "label"}},
]


def handler_config(
    *,
    factor_root: str | Path,
    factor_set: str,
    horizon: str,
    universe: str,
    start_time: Any,
    end_time: Any,
    fit_start_time: Any,
    fit_end_time: Any,
) -> dict[str, Any]:
    """A ``DataHandlerLP`` over a KunQuant factor store with RQuant's processors."""
    return {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": universe,
            "start_time": start_time,
            "end_time": end_time,
            "data_loader": {
                "class": "KunQuantDataLoader",
                "module_path": "rquant.qlib_ext.loader",
                "kwargs": {"factor_root": str(factor_root), "factor_set": factor_set, "horizon": horizon},
            },
            "infer_processors": infer_processors(fit_start_time, fit_end_time),
            "learn_processors": copy.deepcopy(LEARN_PROCESSORS),
            "process_type": "append",
            # Training reads the learn and infer views only; keeping the raw panel too would hold a third copy.
            "drop_raw": True,
        },
    }


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Read a qrun workflow file; ``rolling`` (RQuant's rolling settings) is filled with its defaults."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise DataContractError(f"Workflow file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("task"), dict):
        raise DataContractError(f"{path} is not a qrun workflow: it has no task section")
    task = workflow["task"]
    for key in ("model", "dataset"):
        if key not in task:
            raise DataContractError(f"{path}: task.{key} is missing")
    rolling = {**ROLLING_DEFAULTS, **(workflow.get("rolling") or {})}
    unknown = sorted(set(rolling).difference(ROLLING_DEFAULTS))
    if unknown:
        raise DataContractError(f"{path}: unknown rolling keys {unknown}")
    workflow["rolling"] = rolling
    return workflow


def rolling_template(
    task: dict[str, Any],
    *,
    factor_root: str | Path,
    factor_set: str | None,
    horizon: str | None,
    universe: str,
) -> dict[str, Any]:
    """The workflow's task with the run's factor store, factor set, horizon and universe, and SignalRecord only.

    Each rolling task saves its predictions; signal and portfolio analysis run once on the combined predictions
    (``rquant report``), as Qlib's own rolling module postpones them.
    """
    task = copy.deepcopy(task)
    handler = _handler(task)
    kwargs = handler.setdefault("kwargs", {})
    kwargs["instruments"] = universe
    loader = kwargs.get("data_loader")
    if isinstance(loader, dict) and str(loader.get("class", "")).endswith("KunQuantDataLoader"):
        loader_kwargs = loader.setdefault("kwargs", {})
        loader_kwargs["factor_root"] = str(factor_root)
        if factor_set is not None:
            loader_kwargs["factor_set"] = factor_set
        if horizon is not None:
            loader_kwargs["horizon"] = horizon
    task["record"] = [{"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"}]
    return task


def task_horizon(task: dict[str, Any]) -> str:
    loader = _handler(task).get("kwargs", {}).get("data_loader")
    horizon = loader.get("kwargs", {}).get("horizon") if isinstance(loader, dict) else None
    if horizon not in TRUNC_DAYS:
        raise DataContractError(f"The task's data loader must set a horizon of {sorted(TRUNC_DAYS)}, got {horizon!r}")
    return str(horizon)


def task_factor_set(task: dict[str, Any]) -> str:
    loader = _handler(task).get("kwargs", {}).get("data_loader")
    factor_set = loader.get("kwargs", {}).get("factor_set") if isinstance(loader, dict) else None
    if not factor_set:
        raise DataContractError("The task's data loader must name a factor_set")
    return str(factor_set)


def _handler(task: dict[str, Any]) -> dict[str, Any]:
    handler = task.get("dataset", {}).get("kwargs", {}).get("handler")
    if not isinstance(handler, dict):
        raise DataContractError("task.dataset.kwargs.handler must be a handler config")
    return handler


def window_handler_mod(task: dict[str, Any], rolling_gen: Any) -> None:
    """``RollingGen``'s ``ds_extra_mod_func``: load the handler from the train start through the test end, and fit
    every processor that takes a fit period (``ZScoreNorm``) on the train segment only."""
    segments = task["dataset"]["kwargs"]["segments"]
    train, test = segments[rolling_gen.train_key], segments[rolling_gen.test_key]
    kwargs = _handler(task).setdefault("kwargs", {})
    kwargs["start_time"] = train[0]
    kwargs["end_time"] = test[1]
    for processor in kwargs.get("infer_processors", []) + kwargs.get("learn_processors", []):
        if isinstance(processor, dict) and "fit_start_time" in processor.get("kwargs", {}):
            processor["kwargs"]["fit_start_time"] = train[0]
            processor["kwargs"]["fit_end_time"] = train[1]


class CalendarYearRollingGen(_RollingGen):  # type: ignore[misc,valid-type]
    """Qlib's ``RollingGen`` with one test segment per calendar year instead of a fixed number of trading days.

    For prediction year Y the segments are train = Jan 1 of Y - train_years through Dec 31 of Y - 1, valid = the last
    ``validation_months`` of it, and test = Y (clipped to ``through``), aligned to trading days. ``trunc_segments``
    then ends train and valid ``trunc_days`` trading days before the test starts, so no training label looks into the
    test year, and ``window_handler_mod`` moves the handler's data and fit period with them.
    """

    def __init__(
        self,
        *,
        first_year: int,
        last_year: int,
        through: date,
        horizon: str,
        train_years: int = 3,
        validation_months: int = 6,
        formal_start_year: int = 2010,
    ) -> None:
        if horizon not in TRUNC_DAYS:
            raise DataContractError(f"Unsupported horizon: {horizon}")
        super().__init__(step=1, ds_extra_mod_func=window_handler_mod, trunc_days=TRUNC_DAYS[horizon])
        self.windows = generate_rolling_windows(
            first_prediction_year=first_year,
            last_prediction_year=last_year,
            horizon=horizon,
            through=through,
            train_years=train_years,
            validation_months=validation_months,
            formal_start_year=formal_start_year,
        )

    def generate(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        from qlib.workflow.task.gen import trunc_segments

        tasks = []
        for window in self.windows:
            segments = self.ta.align_seg(
                {
                    self.train_key: (window.train_start, window.refit_end),
                    "valid": (window.validation_start, window.validation_end),
                    self.test_key: (window.test_start, window.test_end),
                }
            )
            if segments[self.test_key][0] > segments[self.test_key][1]:
                continue  # no trading day of the year falls on or before `through`
            trunc_segments(self.ta, segments, self.trunc_days, self.test_key)
            rolled = self.task_copy_func(task)
            self._update_task_segs(rolled, segments)
            tasks.append(rolled)
        return tasks
