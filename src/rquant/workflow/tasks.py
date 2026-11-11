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
from rquant.workflow.rolling import RETRAIN_MONTHS, generate_rolling_windows

try:
    from qlib.workflow.task.gen import RollingGen as _RollingGen
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _RollingGen:  # type: ignore[no-redef]
        pass

WORKFLOW_DIRECTORY = Path(__file__).resolve().parent / "workflows"
BUILT_IN_WORKFLOWS = {
    "lgb": WORKFLOW_DIRECTORY / "lgb.yaml",
    "double-ensemble": WORKFLOW_DIRECTORY / "double_ensemble.yaml",
    # PyTorch models (pip install -e ".[nn]"). The Alpha360 ones read qlib_alpha360 as 60 days x 6 fields.
    "gru": WORKFLOW_DIRECTORY / "gru.yaml",
    "alstm": WORKFLOW_DIRECTORY / "alstm.yaml",
    "hist": WORKFLOW_DIRECTORY / "hist.yaml",
    "tra": WORKFLOW_DIRECTORY / "tra.yaml",
    # TSDatasetH / MTSDatasetH windows of every column of any factor set.
    "gru-ts": WORKFLOW_DIRECTORY / "gru_ts.yaml",
    "alstm-ts": WORKFLOW_DIRECTORY / "alstm_ts.yaml",
    "tra-ts": WORKFLOW_DIRECTORY / "tra_ts.yaml",
    # Qlib's Alpha158 benchmarks: TSDatasetH / MTSDatasetH windows of 20 qlib_alpha158 factors.
    "gru-alpha158": WORKFLOW_DIRECTORY / "gru_alpha158.yaml",
    "alstm-alpha158": WORKFLOW_DIRECTORY / "alstm_alpha158.yaml",
    "tra-alpha158": WORKFLOW_DIRECTORY / "tra_alpha158.yaml",
}

# Datasets that feed a model a window of past rows per sample, and the kwarg naming its length in trading days.
TIME_SERIES_DATASETS = {"TSDatasetH": "step_len", "MTSDatasetH": "seq_len"}

# The label of each horizon is evaluated by KunQuantDataLoader; it needs this many trading days after the prediction
# date, so a training segment must end this many days before the test segment starts (Qlib's ``trunc_days``).
TRUNC_DAYS = {"1d": 2, "5d": 6}
# The task key CalendarYearRollingGen stores each window's label under (``2013``, or ``2013-04`` with sub-year
# retraining); the trainer names the window's recorder and run sub-directory after it.
WINDOW_KEY = "rquant_window"

ROLLING_DEFAULTS = {
    # Pick the boosting rounds by early stopping on an inner train/valid split, then train that many rounds on the
    # whole training window (rquant.workflow.trainer). The window lengths are run settings (workflow.train_years and
    # workflow.validation_months in config/default.yaml).
    "refit": False,
    # Train only up to `trunc_days` trading days before the validation months instead of through them, so a model
    # that early-stops on the validation set (the neural networks) never trains on it.
    "holdout_valid": False,
    # What an LGBModel's early stopping watches on the validation months: `loss` (the model's own objective, l2 for
    # mse) or `ic` (the mean daily Pearson correlation of the score with the processed label, higher is better).
    # `ic` stops where the cross-sectional ranking is best rather than where the squared error is; it needs the lean
    # LightGBM path (rquant.workflow.lgb_window).
    "early_stopping_metric": "loss",
    # Months each model predicts before the next one is trained: 12 trains once per calendar year, 3 every quarter.
    # Each window trains on the `train_years` before its first day.
    "retrain_months": 12,
    # Weight each training row by 0.5 ** (its age / half-life), its age counted back from the window's last training
    # day in years, so recent rows count more; null weighs every row alike. Needs the lean LightGBM path.
    "sample_half_life_years": None,
}

EARLY_STOPPING_METRICS = ("loss", "ic")


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


# How the label is normalised across each day's stocks before training: Qlib's CSZScoreNorm (the default) or
# CSRankNorm (the daily percentile rank, centred and scaled to unit variance), which a few limit-up or limit-down
# moves cannot dominate.
LABEL_NORMS = ("CSZScoreNorm", "CSRankNorm")


def learn_processors(label_norm: str = "CSZScoreNorm") -> list[dict[str, Any]]:
    """The learn processors of an RQuant handler: drop rows without a label, then normalise it per day."""
    if label_norm not in LABEL_NORMS:
        raise DataContractError(f"Unsupported label normalisation {label_norm!r}; use one of {LABEL_NORMS}")
    return [
        {"class": "DropnaLabel", "module_path": "rquant.qlib_ext.processors"},
        {"class": label_norm, "module_path": "qlib.data.dataset.processor", "kwargs": {"fields_group": "label"}},
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
    label_norm: str = "CSZScoreNorm",
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
            "learn_processors": learn_processors(label_norm),
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
    if rolling["retrain_months"] not in RETRAIN_MONTHS:
        raise DataContractError(f"{path}: rolling.retrain_months must be one of {RETRAIN_MONTHS}")
    half_life = rolling["sample_half_life_years"]
    if half_life is not None and (
        isinstance(half_life, bool) or not isinstance(half_life, (int, float)) or half_life <= 0
    ):
        raise DataContractError(f"{path}: rolling.sample_half_life_years must be null or a positive number of years")
    if rolling["early_stopping_metric"] not in EARLY_STOPPING_METRICS:
        raise DataContractError(
            f"{path}: rolling.early_stopping_metric must be one of {EARLY_STOPPING_METRICS}, "
            f"got {rolling['early_stopping_metric']!r}"
        )
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
    dataset = task.get("dataset", {})
    if _class_name(dataset) == "MTSDatasetH":
        # MTSDatasetH can only guess the label horizon from Qlib expression fields, which KunQuantDataLoader has not.
        dataset.setdefault("kwargs", {}).setdefault("horizon", TRUNC_DAYS[task_horizon(task)])
    # Only SignalRecord's label reads the raw panel, and rolling training saves its predictions itself: dropping it
    # saves one copy of the panel on Qlib's own dataset path.
    kwargs["drop_raw"] = True
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


def time_series_window(task: dict[str, Any]) -> int:
    """Trading days of history a sample of the task's dataset spans (0 for a cross-sectional DatasetH)."""
    dataset = task.get("dataset", {})
    key = TIME_SERIES_DATASETS.get(_class_name(dataset))
    if key is None:
        return 0
    default = 30 if key == "step_len" else 60  # TSDatasetH's and MTSDatasetH's defaults
    return int(dataset.get("kwargs", {}).get(key) or default)


def selected_columns(task: dict[str, Any]) -> list[str] | None:
    """The feature columns a ``FilterCol`` processor of the task's handler keeps, or None to keep every factor."""
    kwargs = _handler(task).get("kwargs", {})
    for processor in kwargs.get("infer_processors", []) + kwargs.get("learn_processors", []):
        if isinstance(processor, dict) and _class_name(processor) == "FilterCol":
            return list(processor.get("kwargs", {}).get("col_list", []))
    return None


def _class_name(config: dict[str, Any]) -> str:
    value = config.get("class", "")
    return value if isinstance(value, str) else getattr(value, "__name__", "")


def _handler(task: dict[str, Any]) -> dict[str, Any]:
    handler = task.get("dataset", {}).get("kwargs", {}).get("handler")
    if not isinstance(handler, dict):
        raise DataContractError("task.dataset.kwargs.handler must be a handler config")
    return handler


def window_handler_mod(task: dict[str, Any], rolling_gen: Any) -> None:
    """``RollingGen``'s ``ds_extra_mod_func``: load the handler from the train start through the test end, and fit
    every processor that takes a fit period (``ZScoreNorm``) on the train segment only. A time-series dataset's
    handler also starts its window length earlier, so the first training samples have full windows."""
    segments = task["dataset"]["kwargs"]["segments"]
    train, test = segments[rolling_gen.train_key], segments[rolling_gen.test_key]
    kwargs = _handler(task).setdefault("kwargs", {})
    kwargs["start_time"] = train[0]
    window = time_series_window(task)
    if window:
        from qlib.workflow.task.utils import TimeAdjuster

        adjuster = getattr(rolling_gen, "ta", None) or TimeAdjuster()
        kwargs["start_time"] = adjuster.get(max(0, adjuster.align_idx(train[0]) - window))
    kwargs["end_time"] = test[1]
    for processor in kwargs.get("infer_processors", []) + kwargs.get("learn_processors", []):
        if isinstance(processor, dict) and "fit_start_time" in processor.get("kwargs", {}):
            processor["kwargs"]["fit_start_time"] = train[0]
            processor["kwargs"]["fit_end_time"] = train[1]


class CalendarYearRollingGen(_RollingGen):  # type: ignore[misc,valid-type]
    """Qlib's ``RollingGen`` with one test segment per calendar year (or per ``retrain_months`` of it) instead of a
    fixed number of trading days.

    For prediction year Y the segments are train = Jan 1 of Y - train_years through Dec 31 of Y - 1, valid = the last
    ``validation_months`` of it, and test = Y (clipped to ``through``), aligned to trading days. With
    ``retrain_months`` 3 the same holds per quarter: the window starting April 1 of Y trains from April 1 of
    Y - train_years through March 31 of Y. With
    ``holdout_valid`` train stops before the validation months instead. ``trunc_segments``
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
        holdout_valid: bool = False,
        retrain_months: int = 12,
    ) -> None:
        self.holdout_valid = holdout_valid
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
            retrain_months=retrain_months,
        )

    def generate(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        from qlib.workflow.task.gen import trunc_segments

        tasks = []
        for window in self.windows:
            train_end = window.inner_train_end if self.holdout_valid else window.refit_end
            segments = self.ta.align_seg(
                {
                    self.train_key: (window.train_start, train_end),
                    "valid": (window.validation_start, window.validation_end),
                    self.test_key: (window.test_start, window.test_end),
                }
            )
            if segments[self.test_key][0] > segments[self.test_key][1]:
                continue  # no trading day of the year falls on or before `through`
            trunc_segments(self.ta, segments, self.trunc_days, self.test_key)
            if self.holdout_valid:
                # Train labels must not read into the validation months either.
                segments[self.train_key] = self.ta.truncate(
                    segments[self.train_key], segments["valid"][0], self.trunc_days
                )
            rolled = self.task_copy_func(task)
            self._update_task_segs(rolled, segments)
            rolled[WINDOW_KEY] = window.label
            tasks.append(rolled)
        return tasks
