from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.factors.catalog import FactorSet, get_catalog

try:
    from qlib.data.dataset.loader import DataLoader as _QlibDataLoaderBase
except ImportError:

    class _QlibDataLoaderBase:  # type: ignore[no-redef]
        pass


LABEL_EXPRESSIONS = {
    "1d": "Ref($open, -2) / Ref($open, -1) - 1",
    "5d": "Ref($open, -6) / Ref($open, -1) - 1",
    "20d": "Ref($open, -21) / Ref($open, -1) - 1",
}

HORIZON_STEPS = {"1d": 1, "5d": 5, "20d": 20}


class KunQuantDataLoader(_QlibDataLoaderBase):
    """Load a registered RQuant factor store and a Qlib-evaluated next-open label."""

    def __init__(self, factor_root: str, factor_set: FactorSet = "Hquality", horizon: str = "1d") -> None:
        if horizon not in LABEL_EXPRESSIONS:
            raise DataContractError(f"Unsupported horizon: {horizon}")
        self.factor_root = Path(factor_root)
        self.factor_set = factor_set
        self.horizon = horizon
        self.catalog = get_catalog()
        self.catalog.select(factor_set)

    def load(
        self,
        instruments: Any = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> Any:
        try:
            import numpy as np
            import pandas as pd
            from qlib.data import D
        except ImportError as exc:
            raise DependencyError("pandas, pyarrow and pyqlib==0.9.7 are required by KunQuantDataLoader") from exc

        paths = _partitions_in_range(
            sorted((self.factor_root / self.factor_set).glob("year=*/factors.parquet")), pd, start_time, end_time
        )
        if not paths:
            raise DataContractError(f"No factor partitions for {self.factor_set}: {self.factor_root}")
        instrument_list = _resolve_instruments(D, instruments, start_time, end_time)
        expected = list(self.catalog.canonical_names(self.factor_set))

        def load_labels(index: Any) -> Any:
            labels = D.features(
                sorted(index.get_level_values("instrument").unique()),
                [LABEL_EXPRESSIONS[self.horizon]],
                start_time=start_time,
                end_time=end_time,
                freq="day",
            )
            labels.columns = ["LABEL0"]
            labels = labels.swaplevel().sort_index()
            labels.index = labels.index.set_names(["datetime", "instrument"])
            return labels["LABEL0"].reindex(index).to_numpy(dtype=np.float32)

        # One float32 block for features and label, filled column by column straight from the partitions: no
        # concatenated copy of the store and no float64 frame of a whole year exist next to it. DataHandlerLP copies
        # the panel for its processors, and copying a frame of several blocks consolidates them through extra
        # full-size temporaries.
        with _system_arrow_allocator():
            index, panel = _load_feature_block(
                pd, paths, expected, start_time, end_time, instrument_list, labels=load_labels
            )
        columns = pd.MultiIndex.from_tuples([("feature", column) for column in expected] + [("label", "LABEL0")])
        return pd.DataFrame(panel, index=index, columns=columns, copy=False)


def _load_feature_block(
    pd: Any,
    paths: list[Path],
    expected: list[str],
    start_time: Any,
    end_time: Any,
    instruments: list[str] | None,
    *,
    labels: Callable[[Any], Any] | None = None,
    columns_per_read: int = 32,
) -> tuple[Any, Any]:
    """Return the sorted <datetime, instrument> index and a Fortran-ordered float32 block of the factor columns plus
    one trailing column for the label: ``labels(index)`` when given, else left uninitialised.

    ``labels`` runs before the block is allocated. Qlib evaluates the label in a pool of worker processes, fresh
    interpreters that took about 6.3 GB together on Windows; with the block already committed next to them, a
    672-factor sh_main window left 0.2 GB of a 16 GB machine free (1.3 GB with the label first).

    Rows are filtered by date range and instruments. Factor stores are written in double precision; models train on
    single precision, which halves the memory of every copy DataHandlerLP and LightGBM make of a multi-year panel.
    Each partition is read a few columns at a time, and every column is rounded as it is written into the block.
    """
    import numpy as np
    import pyarrow.parquet as pq

    keys = []
    for path in paths:
        names = [name for name in pq.ParquetFile(path).schema_arrow.names if name not in {"datetime", "instrument"}]
        if names != expected:
            raise DataContractError("Factor store columns do not match the locked catalog")
        frame = pd.read_parquet(path, columns=["datetime", "instrument"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        keep = _row_filter(pd, frame, start_time, end_time, instruments)
        keys.append((path, keep, frame[keep]))
    rows = pd.concat([frame for _, _, frame in keys], ignore_index=True)
    index = pd.MultiIndex.from_arrays([rows.pop("datetime"), rows.pop("instrument")], names=["datetime", "instrument"])
    # Destination row of every source row; the identity when the partitions are already in index order.
    destination = None
    if not index.is_monotonic_increasing:
        order = index.argsort()
        index = index[order]
        destination = np.empty_like(order)
        destination[order] = np.arange(len(order))
    label = labels(index) if labels is not None else None
    panel = np.empty((len(index), len(expected) + 1), dtype=np.float32, order="F")
    if label is not None:
        panel[:, -1] = label
        del label
    offset = 0
    for path, keep, frame in keys:
        mask = keep.to_numpy()
        target = slice(offset, offset + len(frame))
        if destination is not None:
            target = destination[target]
        parquet = pq.ParquetFile(path)
        for first in range(0, len(expected), columns_per_read):
            names = expected[first : first + columns_per_read]
            table = parquet.read(columns=names)
            for position, name in enumerate(names, first):
                values = table.column(name).to_numpy()
                panel[target, position] = values[mask] if not mask.all() else values
            del table
        offset += len(frame)
    return index, panel


def _row_filter(pd: Any, frame: Any, start_time: Any, end_time: Any, instruments: list[str] | None) -> Any:
    keep = pd.Series(True, index=frame.index)
    if start_time is not None:
        keep &= frame["datetime"] >= pd.Timestamp(start_time)
    if end_time is not None:
        keep &= frame["datetime"] <= pd.Timestamp(end_time)
    if instruments:
        keep &= frame["instrument"].isin(instruments)
    return keep


@contextmanager
def _system_arrow_allocator() -> Iterator[None]:
    """Read through the C allocator: Arrow's default mimalloc pool keeps the decoded float64 pages after they are freed
    (about 1.5 GB per 100 sh_main factor columns), and ``release_unused`` does not return them."""
    import pyarrow

    previous = pyarrow.default_memory_pool()
    pyarrow.set_memory_pool(pyarrow.system_memory_pool())
    try:
        yield
    finally:
        pyarrow.set_memory_pool(previous)


def _partitions_in_range(paths: list[Path], pd: Any, start_time: Any, end_time: Any) -> list[Path]:
    """Keep only yearly partitions that can contain rows between start_time and end_time."""
    first_year = pd.Timestamp(start_time).year if start_time is not None else None
    last_year = pd.Timestamp(end_time).year if end_time is not None else None
    selected = []
    for path in paths:
        year = int(path.parent.name.removeprefix("year="))
        if first_year is not None and year < first_year:
            continue
        if last_year is not None and year > last_year:
            continue
        selected.append(path)
    return selected


def _resolve_instruments(D: Any, instruments: Any, start_time: Any, end_time: Any) -> list[str] | None:
    if instruments is None:
        return None
    if isinstance(instruments, (list, tuple, set)):
        return list(instruments)
    if isinstance(instruments, str):
        config = D.instruments(market=instruments)
        return D.list_instruments(config, start_time=start_time, end_time=end_time, freq="day", as_list=True)
    return D.list_instruments(instruments, start_time=start_time, end_time=end_time, freq="day", as_list=True)
