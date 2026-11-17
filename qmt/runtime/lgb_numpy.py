# Score a walk-forward LightGBM window inside QMT, where lightgbm is not installed: the window's inference steps
# (ProcessInf, ZScoreNorm with the fit-period statistics, Fillna 0) and the booster's trees, in numpy 1.19.
# The model dict comes from qmt/export_model.py; it holds no code.
import numpy as np


def prepare_model(model):
    """Turn the exported lists into numpy arrays once."""
    trees = []
    for tree in model["trees"]:
        trees.append({
            "feature": np.asarray(tree["feature"], dtype=np.int64),
            "threshold": np.asarray(tree["threshold"], dtype=np.float64),
            "left": np.asarray(tree["left"], dtype=np.int64),
            "right": np.asarray(tree["right"], dtype=np.int64),
            "leaf": np.asarray(tree["leaf"], dtype=np.float64),
        })
    return {
        "features": list(model["features"]),
        "mean": np.asarray(model["mean"], dtype=np.float32),
        "std": np.asarray(model["std"], dtype=np.float32),
        "trees": trees,
    }


def normalise(values, prepared):
    """Rows x features factor values of one day -> the float32 matrix the booster was trained on.

    +/-inf becomes the mean of the column's finite values that day (ProcessInf), then the fit-period z-score is applied
    in single precision, as the training panel was, and missing values become 0 (Fillna).
    """
    block = np.array(values, dtype=np.float32)
    infinite = np.isinf(block)
    if infinite.any():
        finite = np.where(infinite, np.nan, block)
        for column in np.nonzero(infinite.any(axis=0))[0]:
            kept = finite[:, column]
            kept = kept[~np.isnan(kept)]
            fill = kept.mean(dtype=np.float32) if len(kept) else np.float32(np.nan)
            block[infinite[:, column], column] = fill
    block = (block - prepared["mean"]) / prepared["std"]
    block[np.isnan(block)] = 0
    return block


def predict(prepared, matrix):
    """Sum of the trees' leaf values for each row of ``matrix`` (numerical splits, ``value <= threshold`` goes left).

    Child indices >= 0 are internal nodes; a negative child c is leaf ~c. LightGBM compares in double precision, and
    with missing type None it reads NaN as 0, which ``normalise`` has already done.
    """
    values = np.asarray(matrix, dtype=np.float64)
    rows = np.arange(len(values))
    total = np.zeros(len(values))
    for tree in prepared["trees"]:
        if len(tree["feature"]) == 0:
            total += tree["leaf"][0]
            continue
        node = np.zeros(len(values), dtype=np.int64)
        active = np.ones(len(values), dtype=bool)
        while active.any():
            current = node[active]
            goes_left = values[rows[active], tree["feature"][current]] <= tree["threshold"][current]
            node[active] = np.where(goes_left, tree["left"][current], tree["right"][current])
            active = node >= 0
        total += tree["leaf"][~node]
    return total
