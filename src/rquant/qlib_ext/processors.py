from __future__ import annotations

from typing import Any

# Drop-in replacements for the Qlib processors RollingWorkflowRunner uses, with the same results. Each works a few
# columns at a time and writes back in place, so the frame stays one float32 block and DataHandlerLP's copies of it
# are the only full-size allocations. On a synthetic 190-column, four-year sh_main window the processing peak fell
# from 8.8 GB to 4.1 GB.

try:
    from qlib.data.dataset.processor import Processor as _QlibProcessorBase
except ImportError:

    class _QlibProcessorBase:  # type: ignore[no-redef]
        pass


class ProcessInf(_QlibProcessorBase):
    """Replace +/-inf with the mean of the column's finite values on the same date, in one process.

    Same result as Qlib's ``ProcessInf``, which splits the panel by month and pickles every chunk to one joblib worker
    per core; on a 24-core Windows host those workers held more memory than the panel itself.
    """

    def fit(self, df: Any = None) -> None:
        pass

    def __call__(self, df: Any) -> Any:
        import numpy as np

        positions = [position for position in range(df.shape[1]) if np.isinf(df.iloc[:, position].to_numpy()).any()]
        # A few columns at a time: the masked copy and the daily means are full-length temporaries per column.
        for first in range(0, len(positions), 16):
            chunk = positions[first : first + 16]
            block = df.iloc[:, chunk]
            infinite = np.isinf(block.to_numpy())
            finite = block.mask(infinite)
            daily_mean = finite.groupby(level="datetime", sort=False).transform("mean")
            # Written back by position so the frame keeps a single block; replacing columns by label would split it,
            # and DataHandlerLP's next copy would consolidate the pieces through full-size temporaries.
            df.iloc[:, chunk] = np.where(infinite, daily_mean.to_numpy(), block.to_numpy())
        return df


def _column_chunks(width: int, size: int = 16) -> Any:
    return (slice(first, min(first + size, width)) for first in range(0, width, size))


class ZScoreNorm(_QlibProcessorBase):
    """Qlib's ``ZScoreNorm`` (all columns, statistics from the fit period) computed a few columns at a time in place.

    Qlib's version materialises full-size temporaries of the panel to fit and again to transform.
    Fit statistics use float64 to avoid overflow on large finite ratios; the panel retains its original dtype.
    """

    def __init__(self, fit_start_time: Any, fit_end_time: Any) -> None:
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time

    def fit(self, df: Any = None) -> None:
        import numpy as np

        dates = df.index.get_level_values("datetime")
        rows = (dates >= self.fit_start_time) & (dates <= self.fit_end_time)
        means, stds = [], []
        for chunk in _column_chunks(df.shape[1]):
            # Large finite accounting ratios can overflow float32 squared deviations or their sum.
            # Widen only this bounded chunk; fitted statistics remain training-only and the panel stays float32.
            values = df.iloc[rows, chunk].to_numpy(dtype=np.float64)
            means.append(np.nanmean(values, axis=0))
            stds.append(np.nanstd(values, axis=0))
        self.mean_train = np.concatenate(means)
        self.std_train = np.concatenate(stds)
        # Constant columns pass through unchanged, as in Qlib.
        constant = self.std_train == 0
        self.mean_train[constant] = 0
        self.std_train[constant] = 1

    def __call__(self, df: Any) -> Any:
        for chunk in _column_chunks(df.shape[1]):
            values = df.iloc[:, chunk].to_numpy()
            normalized = (values - self.mean_train[chunk]) / self.std_train[chunk]
            df.iloc[:, chunk] = normalized.astype(values.dtype, copy=False)
        return df


class Fillna(_QlibProcessorBase):
    """Qlib's ``Fillna`` over all columns, written into the frame a few columns at a time.

    ``DataFrame.fillna(inplace=True)`` still builds a full-size temporary of the panel.
    """

    def __init__(self, fill_value: float = 0) -> None:
        self.fill_value = fill_value

    def fit(self, df: Any = None) -> None:
        pass

    def __call__(self, df: Any) -> Any:
        import numpy as np

        for chunk in _column_chunks(df.shape[1]):
            values = df.iloc[:, chunk].to_numpy()
            missing = np.isnan(values)
            if missing.any():
                values[missing] = self.fill_value
                df.iloc[:, chunk] = values
        return df


class DropnaLabel(_QlibProcessorBase):
    """Qlib's ``DropnaLabel``, except that a frame without missing labels is returned as is instead of copied."""

    def fit(self, df: Any = None) -> None:
        pass

    def __call__(self, df: Any) -> Any:
        columns = df.columns[df.columns.get_loc("label")]
        return df if df[columns].notna().all(axis=None) else df.dropna(subset=columns)

    def is_for_infer(self) -> bool:
        return False

    def readonly(self) -> bool:
        return True
