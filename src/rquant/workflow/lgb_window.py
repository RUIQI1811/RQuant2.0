"""Train and predict one LightGBM walk-forward window without DataHandlerLP, DatasetH or LGBModel copies.

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
from dataclasses import dataclass
from typing import Any

from rquant.errors import DataContractError
from rquant.qlib_ext.processors import DropnaLabel, Fillna, ProcessInf, ZScoreNorm


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


def process_panel(panel: Any, *, fit_start: str, fit_end: str) -> tuple[Any, Any]:
    """Run the infer processors on ``panel`` in place, then return the learn labels and the rows that keep one.

    ``panel`` is the single-block frame KunQuantDataLoader returns (features, then the label). The returned label is
    the label after the infer and learn processors; the keep mask is None when DropnaLabel keeps every row.
    """
    import numpy as np
    import pandas as pd
    from qlib.data.dataset.processor import CSZScoreNorm

    for processor in (ProcessInf(), ZScoreNorm(fit_start_time=fit_start, fit_end_time=fit_end), Fillna()):
        processor.fit(panel)
        processor(panel)
    label = panel.iloc[:, [-1]].copy()
    learn = DropnaLabel()(label)
    keep = None if learn is label else label.iloc[:, 0].notna().to_numpy()
    learn = CSZScoreNorm(fields_group="label")(learn)
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
    panel: Any,
    *,
    fit_start: str,
    fit_end: str,
    train: Segment,
    valid: Segment,
    test: Segment | None,
    log_metrics: Any,
) -> Any:
    """Fit ``model`` (a Qlib ``LGBModel``) on ``panel`` and return the test predictions (None without a test segment).

    ``panel`` is consumed: its buffer is reused for the training rows and released before boosting starts.
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    label, keep = process_panel(panel, fit_start=fit_start, fit_end=fit_end)
    index = panel.index
    dates = index.get_level_values("datetime").to_numpy()
    width = panel.shape[1] - 1
    if len(panel._mgr.blocks) != 1 or panel._mgr.blocks[0].dtype != np.float32:
        raise DataContractError("Walk-forward panel must be a single float32 block")
    block = panel.to_numpy()
    del panel

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

    # Bin the training rows with the model's parameters, as lgb.train would, then drop the panel buffer.
    train_set = lgb.Dataset(x_train, label=y_train, params=model.params)
    train_set.construct()
    del x_train, block
    gc.collect()
    valid_set = lgb.Dataset(x_valid, label=y_valid)

    evals_result: dict[str, Any] = {}
    names = ("train", "valid")
    model.model = lgb.train(
        model.params,
        train_set,
        num_boost_round=model.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=list(names),
        callbacks=[
            lgb.early_stopping(model.early_stopping_rounds),
            lgb.log_evaluation(period=20),
            lgb.record_evaluation(evals_result),
        ],
    )
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


def _learn_rows(block: Any, rows: slice, width: int, label: Any, keep: Any) -> tuple[Any, Any]:
    import numpy as np

    if keep is None or keep[rows].all():
        return take_features(block, rows, width), label[rows]
    positions = np.arange(rows.start, rows.stop)[keep[rows]]
    return np.array(block[positions, :width], dtype=np.float32, order="F"), label[positions]
