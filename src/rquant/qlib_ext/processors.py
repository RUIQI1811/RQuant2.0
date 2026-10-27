from __future__ import annotations

from typing import Any

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

        columns = [column for column in df.columns if np.isinf(df[column].to_numpy()).any()]
        if not columns:
            return df
        block = df[columns]
        finite = block.mask(np.isinf(block.to_numpy()))
        daily_mean = finite.groupby(level="datetime", sort=False).transform("mean")
        df[columns] = finite.where(finite.notna() | block.isna(), daily_mean)
        return df
