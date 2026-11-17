# Back-fills for the numpy/pandas calls the jq240 reference makes that QMT's numpy 1.19 / pandas 0.22 lack.
# Python 3.6 syntax only; every patch is a no-op on newer versions.
import numpy as np
import pandas as pd


def _to_numpy(self, dtype=None, copy=False):
    values = np.asarray(self.values, dtype=dtype)
    return values.copy() if copy else values


for _cls in (pd.DataFrame, pd.Series, pd.Index):
    if not hasattr(_cls, "to_numpy"):
        _cls.to_numpy = _to_numpy


def _sliding_window_view(x, window_shape, axis=None):
    # Only the form the reference uses: one integer window along one axis, appended as the last dimension.
    x = np.asarray(x)
    axis = axis % x.ndim
    count = x.shape[axis] - window_shape + 1
    shape = x.shape[:axis] + (count,) + x.shape[axis + 1:] + (window_shape,)
    strides = x.strides + (x.strides[axis],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)


if not hasattr(np.lib.stride_tricks, "sliding_window_view"):
    np.lib.stride_tricks.sliding_window_view = _sliding_window_view
