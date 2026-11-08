"""Train one rolling task: the ``train_func`` RQuant gives Qlib's ``TrainerR``.

It does what ``qlib.model.trainer.task_train`` does (start a recorder, log the task, build the model, fit it, save
``params.pkl`` and the ``SignalRecord`` artifact ``pred.pkl``), with two differences:

- A task whose dataset is RQuant's handler (``KunQuantDataLoader`` with RQuant's processors, see
  ``rquant.workflow.tasks.handler_config``) and whose model is ``LGBModel`` or ``DEnsembleModel`` trains through the
  memory-lean window functions in ``rquant.workflow.lgb_window``. They predict exactly what ``DatasetH`` would give
  the model (``tests/test_models.py`` compares them) while holding one float32 copy of the panel instead of up to
  four, which a 16 GB machine needs for multi-year main-board windows. Any other task goes through ``DatasetH``.
- With ``refit``, an ``LGBModel`` first picks its boosting rounds by early stopping on an inner split (train ends
  ``trunc_days`` before valid, valid ends the day before the test), then trains that many rounds on the task's own
  train segment without early stopping. ``early_stopping_metric`` ``ic`` (the workflow's
  ``rolling.early_stopping_metric``) makes that early stopping keep the round with the best mean daily IC on the
  validation months instead of the lowest loss; it needs the lean path.
"""

from __future__ import annotations

import copy
import gc
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rquant.errors import DataContractError
from rquant.io import atomic_write_json, sha256_file
from rquant.workflow.lgb_window import Segment, feature_importance, fit_double_ensemble_window, fit_lgb_window
from rquant.workflow.tasks import (
    LABEL_NORMS,
    TRUNC_DAYS,
    infer_processors,
    learn_processors,
    task_horizon,
    window_handler_mod,
)

QLIB_PROCESSOR_MODULE = "qlib.data.dataset.processor"
FIT_PERIOD_KEYS = {"fit_start_time", "fit_end_time"}


def train_rolling_task(
    task_config: dict[str, Any],
    experiment_name: str,
    recorder_name: str | None = None,
    *,
    seed: int | None = None,
    refit: bool = False,
    anchor_positions: dict[Any, int] | None = None,
    run_directory: str | Path | None = None,
    factor_names: list[str] | None = None,
    early_stopping_metric: str = "loss",
) -> Any:
    """Train ``task_config`` in a new recorder of ``experiment_name`` and return the recorder.

    ``seed`` seeds NumPy with ``seed + prediction year`` before anything is fitted. ``anchor_positions`` (trading day
    to its position in the calendar) keeps only the predictions of every fifth trading day, the 5-day anchors.
    With ``run_directory``, the window's predictions, feature importance and segments are also written to
    ``<run_directory>/year=<prediction year>/``.
    """
    import numpy as np
    import pandas as pd
    from qlib.model.base import Model
    from qlib.model.trainer import _log_task_info
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    segments = task_config["dataset"]["kwargs"]["segments"]
    year = pd.Timestamp(segments["test"][0]).year
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name or f"year={year}"):
        _log_task_info(task_config)
        if seed is not None:
            np.random.seed(seed + year)
        model_config = task_config["model"]
        chosen_iterations = None
        inner = None
        if refit:
            inner = inner_split_task(task_config)
            tuning = init_instance_by_config(model_config, accept_types=Model)
            if not _is_lgb(tuning):
                raise DataContractError("rolling.refit picks boosting rounds by early stopping and needs an LGBModel")
            fit_task(
                tuning,
                inner,
                log_metrics=_prefixed(R.log_metrics, "tune."),
                early_stopping_metric=early_stopping_metric,
            )
            chosen_iterations = best_iterations(tuning)
            # Release the tuning model before the refit panel is loaded.
            del tuning
            gc.collect()
            model_config = copy.deepcopy(model_config)
            model_config.setdefault("kwargs", {}).update(num_boost_round=chosen_iterations, early_stopping_rounds=0)
            R.log_params(chosen_iterations=chosen_iterations)
        model = init_instance_by_config(model_config, accept_types=Model)
        prediction = fit_task(
            model, task_config, log_metrics=R.log_metrics, early_stopping_metric=early_stopping_metric
        )
        if prediction is None:
            raise DataContractError(f"The {year} task has no test segment to predict")
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.iloc[:, 0]
        prediction = prediction.rename("score")
        if anchor_positions is not None:
            days = prediction.index.get_level_values("datetime")
            positions = pd.Series(days.date).map(anchor_positions).to_numpy()
            prediction = prediction[np.asarray(positions % 5 == 0)]
        R.save_objects(**{"params.pkl": model})
        R.save_objects(**{"pred.pkl": prediction.to_frame("score")})
        recorder = R.get_recorder()
        if run_directory is not None:
            _write_window(
                Path(run_directory) / f"year={year}",
                task_config,
                inner,
                prediction,
                model,
                factor_names,
                chosen_iterations=chosen_iterations,
                recorder_id=recorder.id,
            )
        return recorder


def fit_task(
    model: Any, task: dict[str, Any], *, log_metrics: Any = None, early_stopping_metric: str = "loss"
) -> Any:
    """Fit ``model`` on ``task``'s dataset and return its test predictions (None without a test segment)."""
    from qlib.data.dataset import Dataset
    from qlib.utils import init_instance_by_config

    segments = task["dataset"]["kwargs"]["segments"]
    lean = lean_window(task)
    if early_stopping_metric != "loss" and (lean is None or not _is_lgb(model)):
        raise DataContractError(
            f"early_stopping_metric {early_stopping_metric!r} needs an LGBModel on RQuant's lean handler"
        )
    if lean is not None and (_is_lgb(model) or _is_double_ensemble(model)):
        loader = init_instance_by_config(lean["data_loader"])
        instruments, start, end = lean["instruments"], _day(lean["start_time"]), _day(lean["end_time"])

        def load_panel() -> Any:
            return loader.load(instruments, start, end)

        window = {
            "fit_start": _day(lean["fit_start_time"]),
            "fit_end": _day(lean["fit_end_time"]),
            "train": _segment(segments["train"]),
            "valid": _segment(segments["valid"]),
            "test": _segment(segments["test"]) if "test" in segments else None,
            "label_norm": lean["label_norm"],
        }
        if _is_lgb(model):
            return fit_lgb_window(
                model,
                load_panel,
                log_metrics=log_metrics or _ignore,
                early_stopping_metric=early_stopping_metric,
                **window,
            )
        return fit_double_ensemble_window(model, load_panel, **window)
    dataset = init_instance_by_config(task["dataset"], accept_types=Dataset)
    model.fit(dataset)
    if "test" not in segments:
        return None
    # Qlib's time-series models (pytorch_*_ts) always predict the test segment and take no segment argument.
    if "segment" in inspect.signature(model.predict).parameters:
        return model.predict(dataset, segment="test")
    return model.predict(dataset)


def lean_window(task: dict[str, Any]) -> dict[str, Any] | None:
    """The loader, instruments, data range, fit period and label normalisation of ``task`` when its dataset is a
    ``DatasetH`` over a ``DataHandlerLP`` with a ``KunQuantDataLoader`` and exactly RQuant's processors (the label
    normalised by one of ``LABEL_NORMS``); None otherwise."""
    from rquant.qlib_ext.loader import KunQuantDataLoader

    dataset = task.get("dataset", {})
    if _class_name(dataset) != "DatasetH":
        return None
    handler = dataset.get("kwargs", {}).get("handler")
    if not isinstance(handler, dict) or _class_name(handler) != "DataHandlerLP":
        return None
    kwargs = handler.get("kwargs", {})
    loader = kwargs.get("data_loader")
    if not isinstance(loader, dict):
        return None
    loader_class = loader.get("class")
    if isinstance(loader_class, str):
        if loader_class != "KunQuantDataLoader" or loader.get("module_path") != "rquant.qlib_ext.loader":
            return None
    elif not (isinstance(loader_class, type) and issubclass(loader_class, KunQuantDataLoader)):
        return None
    if kwargs.get("process_type", "append") != "append":
        return None
    infer = kwargs.get("infer_processors", [])
    fit = [item.get("kwargs", {}) for item in infer if isinstance(item, dict) and _class_name(item) == "ZScoreNorm"]
    if len(fit) != 1 or "fit_start_time" not in fit[0] or "fit_end_time" not in fit[0]:
        return None
    fit_start, fit_end = fit[0]["fit_start_time"], fit[0]["fit_end_time"]
    if _normalised(infer) != _normalised(infer_processors(fit_start, fit_end)):
        return None
    learn = _normalised(kwargs.get("learn_processors", []))
    label_norm = next((name for name in LABEL_NORMS if learn == _normalised(learn_processors(name))), None)
    if label_norm is None:
        return None
    return {
        "label_norm": label_norm,
        "data_loader": loader,
        "instruments": kwargs.get("instruments"),
        "start_time": kwargs.get("start_time"),
        "end_time": kwargs.get("end_time"),
        "fit_start_time": fit_start,
        "fit_end_time": fit_end,
    }


def inner_split_task(task: dict[str, Any]) -> dict[str, Any]:
    """The task the boosting rounds are picked on: valid runs from its start to the last trading day before the test,
    train ends ``trunc_days`` trading days before valid starts, and there is no test segment."""
    from qlib.workflow.task.gen import trunc_segments
    from qlib.workflow.task.utils import TimeAdjuster

    adjuster = TimeAdjuster(future=True)
    segments = task["dataset"]["kwargs"]["segments"]
    valid_end = adjuster.get(adjuster.align_idx(segments["test"][0]) - 1)
    inner = {"train": (segments["train"][0], valid_end), "valid": (segments["valid"][0], valid_end)}
    trunc_segments(adjuster, inner, TRUNC_DAYS[task_horizon(task)], test_key="valid")
    inner_task = copy.deepcopy(task)
    inner_task["dataset"]["kwargs"]["segments"] = inner
    window_handler_mod(inner_task, SimpleNamespace(train_key="train", test_key="valid"))
    return inner_task


def best_iterations(model: Any) -> int:
    booster = getattr(model, "model", None)
    if booster is not None:
        value = int(getattr(booster, "best_iteration", 0) or getattr(booster, "num_trees", lambda: 0)())
        return max(1, value)
    raise DataContractError("Unable to determine validated boosting iterations")


def _write_window(
    directory: Path,
    task: dict[str, Any],
    inner: dict[str, Any] | None,
    prediction: Any,
    model: Any,
    factor_names: list[str] | None,
    *,
    chosen_iterations: int | None,
    recorder_id: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    prediction_path = directory / "predictions.parquet"
    prediction.reset_index().to_parquet(prediction_path, index=False)
    if factor_names is not None and (_is_lgb(model) or _is_double_ensemble(model)):
        feature_importance(model, factor_names).to_parquet(directory / "feature_importance.parquet", index=False)
    segments = task["dataset"]["kwargs"]["segments"]
    manifest = {
        "prediction_year": int(_timestamp(segments["test"][0]).year),
        "segments": {name: [_day(start), _day(end)] for name, (start, end) in segments.items()},
        "trunc_days": TRUNC_DAYS[task_horizon(task)],
        "chosen_iterations": chosen_iterations,
        "prediction_rows": len(prediction),
        "prediction_fingerprint": sha256_file(prediction_path),
        "recorder_id": recorder_id,
    }
    if inner is not None:
        manifest["inner_segments"] = {
            name: [_day(start), _day(end)] for name, (start, end) in inner["dataset"]["kwargs"]["segments"].items()
        }
    atomic_write_json(directory / "window.json", manifest)


def _normalised(processors: list[Any]) -> list[tuple[str, str, Any]]:
    result = []
    for item in processors:
        if not isinstance(item, dict):
            return [("<instance>", "", id(item))]
        kwargs = {key: value for key, value in item.get("kwargs", {}).items() if key not in FIT_PERIOD_KEYS}
        result.append((_class_name(item), item.get("module_path", QLIB_PROCESSOR_MODULE), sorted(kwargs.items())))
    return result


def _class_name(config: dict[str, Any]) -> str:
    value = config.get("class", "")
    return value if isinstance(value, str) else getattr(value, "__name__", "")


def _is_lgb(model: Any) -> bool:
    from qlib.contrib.model.gbdt import LGBModel

    return isinstance(model, LGBModel)


def _is_double_ensemble(model: Any) -> bool:
    from qlib.contrib.model.double_ensemble import DEnsembleModel

    return isinstance(model, DEnsembleModel)


def _prefixed(log_metrics: Any, prefix: str) -> Any:
    def log(step: int | None = None, **metrics: Any) -> None:
        log_metrics(step=step, **{f"{prefix}{name}": value for name, value in metrics.items()})

    return log


def _ignore(**_: Any) -> None:
    return None


def _timestamp(value: Any) -> Any:
    import pandas as pd

    return pd.Timestamp(value)


def _day(value: Any) -> str:
    return _timestamp(value).date().isoformat()


def _segment(segment: Any) -> Segment:
    return Segment(_day(segment[0]), _day(segment[1]))
