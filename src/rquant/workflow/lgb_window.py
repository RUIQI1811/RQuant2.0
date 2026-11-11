"""Train and predict one walk-forward window without DataHandlerLP, DatasetH or LGBModel copies.

The steps and their results are those of RollingWorkflowRunner's Qlib path for ``lgb``: the infer processors run on
the loaded panel with fitting (ProcessInf, ZScoreNorm over the fit period, Fillna), the learn processors (DropnaLabel,
then a cross-sectional z-score of the label) run on the label only, and ``LGBModel.fit`` and ``predict`` are
reproduced with the same LightGBM call. What changes is where the panel lives:

- the processors write into the loaded block, so there is no raw, infer and learn copy of it;
- the validation and test rows are copied out, then the training rows are moved to the front of the same buffer in
  column-major order, which LightGBM reads without a copy;
- the training Dataset is binned before training starts and the panel buffer is then released, so boosting holds
  only the binned data and the two small copies.

On a 672-factor sh_main window (about 1.57 million rows) one copy of the panel is 4.2 GB; the Qlib path held up to
four of them at once.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rquant.errors import DataContractError
from rquant.qlib_ext.processors import DropnaLabel, Fillna, ProcessInf, ZScoreNorm
from rquant.workflow.tasks import EARLY_STOPPING_METRICS, LABEL_NORMS

LOGGER = logging.getLogger(__name__)


def _log_memory(stage: str) -> None:
    """Log the process memory at a step of the window, so a run's log shows where the peak builds up."""
    try:
        import psutil
    except ImportError:
        return
    info = psutil.Process().memory_info()
    private = getattr(info, "private", None)
    LOGGER.info(
        "lgb window %s: rss %.2f GB%s",
        stage,
        info.rss / 1e9,
        "" if private is None else f", private {private / 1e9:.2f} GB",
    )


@dataclass(frozen=True)
class Segment:
    start: str
    end: str


def segment_rows(dates: Any, segment: Segment) -> slice:
    """Rows of a date-sorted panel whose date lies in ``[start, end]``, as DatasetH's ``loc`` slice selects them."""
    import numpy as np
    import pandas as pd

    first = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(segment.start)), side="left"))
    last = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(segment.end)), side="right"))
    return slice(first, last)


def process_panel(
    panel: Any, *, fit_start: str, fit_end: str, label_norm: str = "CSZScoreNorm"
) -> tuple[Any, Any]:
    """Run the infer processors on ``panel`` in place, then return the learn labels and the rows that keep one.

    ``panel`` is the single-block frame KunQuantDataLoader returns (features, then the label). The returned label is
    the label after the infer and learn processors (DropnaLabel, then ``label_norm``: Qlib's CSZScoreNorm or
    CSRankNorm); the keep mask is None when DropnaLabel keeps every row.
    """
    import numpy as np
    import pandas as pd
    from qlib.data.dataset import processor as qlib_processors

    if label_norm not in LABEL_NORMS:
        raise DataContractError(f"Unsupported label normalisation {label_norm!r}; use one of {LABEL_NORMS}")
    for processor in (ProcessInf(), ZScoreNorm(fit_start_time=fit_start, fit_end_time=fit_end), Fillna()):
        processor.fit(panel)
        processor(panel)
    label = panel.iloc[:, [-1]].copy()
    learn = DropnaLabel()(label)
    keep = None if learn is label else label.iloc[:, 0].notna().to_numpy()
    learn = getattr(qlib_processors, label_norm)(fields_group="label")(learn)
    values = learn.iloc[:, 0].to_numpy(dtype=np.float32)
    if keep is not None:
        full = np.full(len(panel), np.nan, dtype=np.float32)
        full[keep] = values
        values = full
    return pd.Series(values, index=panel.index), keep


def take_features(block: Any, rows: slice, width: int) -> Any:
    """Copy the feature columns of ``rows`` into a new column-major array."""
    import numpy as np

    return np.array(block[rows, :width], dtype=np.float32, order="F")


def move_rows_to_front(block: Any, rows: slice, width: int) -> Any:
    """Overwrite ``block`` so its first ``width`` columns of ``rows`` form a column-major array, and return it.

    The leading ``rows.stop - rows.start`` rows of every column are moved down to their place in a column-major array
    of that height; each destination lies at or before its source and after every column already moved, so nothing
    still to be read is overwritten. The rest of the buffer is left as garbage.
    """
    if not block.flags["F_CONTIGUOUS"]:
        raise DataContractError("Walk-forward panel must be column-major")
    total = block.shape[0]
    height = rows.stop - rows.start
    flat = block.reshape(-1, order="F")
    for column in range(width):
        source = column * total + rows.start
        target = column * height
        if source != target:
            flat[target : target + height] = flat[source : source + height]
    return flat[: width * height].reshape((height, width), order="F")


def fit_lgb_window(
    model: Any,
    load_panel: Callable[[], Any],
    *,
    fit_start: str,
    fit_end: str,
    train: Segment,
    valid: Segment,
    test: Segment | None,
    log_metrics: Any,
    label_norm: str = "CSZScoreNorm",
    early_stopping_metric: str = "loss",
    sample_half_life_years: float | None = None,
) -> Any:
    """Fit ``model`` (a Qlib ``LGBModel``) on the panel ``load_panel`` returns and return the test predictions (None
    without a test segment).

    The panel is loaded here so that no caller keeps a reference to it: its buffer is reused for the training rows and
    released before boosting starts. ``label_norm`` names the label's daily normalisation (``process_panel``).
    With ``early_stopping_metric`` ``ic`` the validation rows are scored by ``daily_ic`` instead of the objective, and
    early stopping keeps the round with the best mean daily IC. ``sample_half_life_years`` weights the training rows
    by ``recency_weights``; the validation rows stay unweighted.
    """
    if early_stopping_metric not in EARLY_STOPPING_METRICS:
        raise DataContractError(
            f"Unsupported early stopping metric {early_stopping_metric!r}; use one of {EARLY_STOPPING_METRICS}"
        )
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    panel = load_panel()
    _log_memory("loaded")
    label, keep = process_panel(panel, fit_start=fit_start, fit_end=fit_end, label_norm=label_norm)
    index = panel.index
    dates = index.get_level_values("datetime").to_numpy()
    width = panel.shape[1] - 1
    if len(panel._mgr.blocks) != 1 or panel._mgr.blocks[0].dtype != np.float32:
        raise DataContractError("Walk-forward panel must be a single float32 block")
    block = panel.to_numpy()
    del panel
    _log_memory("processed")

    test_rows = segment_rows(dates, test) if test is not None else None
    x_test = take_features(block, test_rows, width) if test_rows is not None else None
    valid_rows = segment_rows(dates, valid)
    train_rows = segment_rows(dates, train)
    y = label.to_numpy()
    x_valid, y_valid = _learn_rows(block, valid_rows, width, y, keep)
    if keep is None or keep[train_rows].all():
        x_train, y_train = move_rows_to_front(block, train_rows, width), y[train_rows]
    else:
        x_train, y_train = _learn_rows(block, train_rows, width, y, keep)
    for name, x in (("train", x_train), ("valid", x_valid)):
        if len(x) == 0:
            raise ValueError(f"Empty {name} segment, please check the walk-forward windows.")
    _log_memory("segments copied")

    # Bin the training rows with the model's parameters, as lgb.train would, then drop the panel buffer.
    weight = None
    if sample_half_life_years is not None:
        weight = recency_weights(_learn_dates(dates, train_rows, keep), sample_half_life_years)
    train_set = lgb.Dataset(x_train, label=y_train, weight=weight, params=model.params)
    train_set.construct()
    _log_memory("training rows binned")
    del x_train, block
    gc.collect()
    _log_memory("panel released")
    valid_set = lgb.Dataset(x_valid, label=y_valid)

    evals_result: dict[str, Any] = {}
    params = model.params
    valid_sets = [train_set, valid_set]
    names: tuple[str, ...] = ("train", "valid")
    feval = None
    if early_stopping_metric == "ic":
        # Only the validation rows are scored, by IC alone: LightGBM's early stopping would otherwise also stop on
        # the objective's own metric.
        valid_dates = _learn_dates(dates, valid_rows, keep)
        feval = daily_ic(valid_dates, y_valid)
        params = {**model.params, "metric": "None"}
        valid_sets, names = [valid_set], ("valid",)
    model.model = lgb.train(
        params,
        train_set,
        num_boost_round=model.num_boost_round,
        valid_sets=valid_sets,
        valid_names=list(names),
        feval=feval,
        callbacks=[
            lgb.early_stopping(model.early_stopping_rounds),
            lgb.log_evaluation(period=20),
            lgb.record_evaluation(evals_result),
        ],
    )
    _log_memory("trained")
    for name in names:
        for key, values in evals_result[name].items():
            metric = f"{key}.{name}".replace("@", "_")
            for epoch, value in enumerate(values):
                log_metrics(**{metric: value}, step=epoch)
    del train_set, valid_set, x_valid
    gc.collect()
    if x_test is None:
        return None
    return pd.Series(model.model.predict(x_test), index=index[test_rows])


def recency_weights(dates: Any, half_life_years: float) -> Any:
    """``0.5 ** (age / half_life_years)`` for every row, its age in years counted back from the latest of ``dates``,
    scaled to a mean of one so LightGBM's hessian-sum limits keep their meaning."""
    import numpy as np

    if half_life_years <= 0:
        raise ValueError(f"half_life_years must be positive, got {half_life_years}")
    days = np.asarray(dates, dtype="datetime64[D]")
    age_years = (days.max() - days).astype(np.float64) / 365.25
    weight = np.power(0.5, age_years / half_life_years)
    return (weight / weight.mean()).astype(np.float32)


def daily_ic(dates: Any, label: Any) -> Callable[[Any, Any], tuple[str, float, bool]]:
    """A LightGBM ``feval`` scoring predictions by their mean daily Pearson correlation with ``label``.

    ``dates`` and ``label`` are the validation rows' trading days (sorted, as the panel is) and processed labels, in
    the order LightGBM predicts them. Days with fewer than two stocks or a constant score or label count as zero, so
    a first tree that scores every stock alike cannot look good.
    """
    import numpy as np

    dates = np.asarray(dates)
    label = np.asarray(label, dtype=np.float64)
    if len(dates) != len(label):
        raise DataContractError("daily_ic needs one date per label")
    if len(dates) == 0:
        raise DataContractError("daily_ic needs validation rows")
    if (dates[1:] < dates[:-1]).any():
        raise DataContractError("daily_ic needs date-sorted rows")
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    counts = np.diff(np.r_[starts, len(dates)])
    label_dev = label - np.repeat(np.add.reduceat(label, starts) / counts, counts)
    label_var = np.add.reduceat(label_dev * label_dev, starts)

    def evaluate(predictions: Any, _data: Any) -> tuple[str, float, bool]:
        scores = np.asarray(predictions, dtype=np.float64)
        dev = scores - np.repeat(np.add.reduceat(scores, starts) / counts, counts)
        covariance = np.add.reduceat(dev * label_dev, starts)
        variance = np.add.reduceat(dev * dev, starts)
        denominator = np.sqrt(variance * label_var)
        usable = (counts > 1) & (denominator > 1e-12)
        ic = np.zeros(len(starts))
        ic[usable] = covariance[usable] / denominator[usable]
        return "ic", float(ic.mean()), True

    return evaluate


def _learn_dates(dates: Any, rows: slice, keep: Any) -> Any:
    """The trading days of the learn rows ``_learn_rows`` takes from ``rows``."""
    import numpy as np

    if keep is None or keep[rows].all():
        return dates[rows]
    return dates[np.arange(rows.start, rows.stop)[keep[rows]]]


def _learn_rows(block: Any, rows: slice, width: int, label: Any, keep: Any) -> tuple[Any, Any]:
    import numpy as np

    if keep is None or keep[rows].all():
        return take_features(block, rows, width), label[rows]
    positions = np.arange(rows.start, rows.stop)[keep[rows]]
    return np.array(block[positions, :width], dtype=np.float32, order="F"), label[positions]


class _SegmentFrame:
    """The parts of a ``DatasetH.prepare(..., col_set=["feature", "label"])`` frame that ``DEnsembleModel`` reads.

    ``frame["feature"]`` and ``frame["label"]`` return the two frames as they are, so the features stay the single
    float32 block they were built as instead of being concatenated with the label into a new MultiIndex frame.
    """

    def __init__(self, features: Any, label: Any) -> None:
        self._parts = {"feature": features, "label": label}

    @property
    def empty(self) -> bool:
        return bool(self._parts["feature"].empty)

    def __getitem__(self, key: str) -> Any:
        return self._parts[key]


class _PreparedSegments:
    """A stand-in for ``DatasetH`` that hands ``DEnsembleModel.fit`` and ``predict`` already prepared segments."""

    def __init__(self, learn: dict[str, _SegmentFrame], infer: dict[str, Any]) -> None:
        self.learn = learn
        self.infer = infer

    def prepare(self, segments: Any, col_set: Any = None, data_key: Any = None) -> Any:
        from qlib.data.dataset.handler import DataHandlerLP

        if data_key == DataHandlerLP.DK_L:
            if col_set != ["feature", "label"]:
                raise DataContractError(f"Unsupported learn columns: {col_set}")
            return [self.learn[name] for name in segments] if isinstance(segments, list) else self.learn[segments]
        if col_set != "feature" or isinstance(segments, list):
            raise DataContractError(f"Unsupported infer request: {segments}, {col_set}")
        return self.infer[segments]


def fit_double_ensemble_window(
    model: Any,
    load_panel: Callable[[], Any],
    *,
    fit_start: str,
    fit_end: str,
    train: Segment,
    valid: Segment,
    test: Segment | None,
    label_norm: str = "CSZScoreNorm",
) -> Any:
    """Fit ``model`` (a Qlib ``DEnsembleModel``) on the panel ``load_panel`` returns and return the test predictions
    (None without a test segment).

    The panel goes through the same processors as ``fit_lgb_window``; ``DEnsembleModel`` then receives the learn rows
    of the train and valid segments, and predicts the test rows, exactly as ``DatasetH`` would hand them over. Only
    the panel buffer and one copy of each segment exist before the model starts its own copies.
    """
    import numpy as np
    import pandas as pd

    panel = load_panel()
    _log_memory("loaded")
    label, keep = process_panel(panel, fit_start=fit_start, fit_end=fit_end, label_norm=label_norm)
    index = panel.index
    dates = index.get_level_values("datetime").to_numpy()
    width = panel.shape[1] - 1
    names = pd.Index(panel.columns.get_level_values(-1)[:width])
    if len(panel._mgr.blocks) != 1 or panel._mgr.blocks[0].dtype != np.float32:
        raise DataContractError("Walk-forward panel must be a single float32 block")
    block = panel.to_numpy()
    del panel
    _log_memory("processed")

    y = label.to_numpy()
    infer: dict[str, Any] = {}
    if test is not None:
        rows = segment_rows(dates, test)
        infer["test"] = pd.DataFrame(take_features(block, rows, width), index=index[rows], columns=names, copy=False)
    learn: dict[str, _SegmentFrame] = {}
    for name, segment in (("valid", valid), ("train", train)):
        rows = segment_rows(dates, segment)
        if keep is None or keep[rows].all():
            positions: Any = rows
            x = take_features(block, rows, width) if name == "valid" else move_rows_to_front(block, rows, width)
        else:
            positions = np.arange(rows.start, rows.stop)[keep[rows]]
            x = np.array(block[positions, :width], dtype=np.float32, order="F")
        if len(x) == 0:
            raise ValueError(f"Empty {name} segment, please check the walk-forward windows.")
        frame_index = index[positions]
        learn[name] = _SegmentFrame(
            pd.DataFrame(x, index=frame_index, columns=names, copy=False),
            pd.DataFrame({"LABEL0": y[positions]}, index=frame_index),
        )
    del block
    gc.collect()
    _log_memory("segments copied")

    segments = _PreparedSegments(learn, infer)
    model.fit(segments)
    _log_memory("trained")
    del learn, segments.learn
    gc.collect()
    if test is None:
        return None
    return model.predict(segments, segment="test")


def feature_importance(model: Any, names: Any) -> Any:
    """Gain and split importance of a fitted ``LGBModel`` or ``DEnsembleModel``, one row per factor in ``names``.

    DoubleEnsemble sums its sub-models weighted by ``sub_weights`` (Qlib's ``get_feature_importance``); a factor no
    sub-model used scores zero. ``gain_share`` is the factor's share of the total gain.
    """
    import pandas as pd

    names = pd.Index(list(names))
    columns = {}
    for kind in ("gain", "split"):
        booster = getattr(model, "model", None)
        if booster is not None:
            values = pd.Series(booster.feature_importance(importance_type=kind), index=names, dtype=float)
        else:
            values = model.get_feature_importance(importance_type=kind).astype(float)
        columns[kind] = values.reindex(names).fillna(0.0)
    frame = pd.DataFrame(columns)
    total = float(frame["gain"].sum())
    frame["gain_share"] = frame["gain"] / total if total > 0 else 0.0
    return frame.rename_axis("factor").reset_index()
