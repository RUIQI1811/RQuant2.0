"""Daily ``dates x instruments`` reference panels read one Parquet file at a time."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError


def daily_basic_panels(
    paths: Iterable[str | Path],
    fields: Mapping[str, float] | Iterable[str],
    dates: Any,
    instruments: Any,
) -> dict[str, pd.DataFrame]:
    """``basic_<field>`` panels of daily_basic ``fields`` for ``instruments`` on ``dates``.

    The same values as pivoting the concatenated files and reindexing to ``dates`` and ``instruments`` (times the
    field's scale when ``fields`` maps fields to scales), without holding every daily row of the market at once: each
    file is read for ``instruments`` only and written into the panels before the next one is read.
    """
    scales = dict(fields) if isinstance(fields, Mapping) else dict.fromkeys(fields)
    dates = pd.DatetimeIndex(dates)
    instruments = pd.Index(instruments)
    shape = (len(dates), len(instruments))
    values = {field: np.full(shape, np.nan) for field in scales}
    seen = np.zeros(shape, dtype=bool)
    wanted = [("instrument", "in", list(instruments))]
    for path in paths:
        frame = pd.read_parquet(path, columns=["instrument", "datetime", *scales], filters=wanted)
        frame = frame[frame["instrument"].isin(set(instruments))]
        stamps = pd.to_datetime(frame["datetime"])
        if pd.MultiIndex.from_arrays([stamps, frame["instrument"]]).has_duplicates:
            raise DataContractError(f"daily_basic repeats a (datetime, instrument) pair in {path}")
        rows = dates.get_indexer(stamps)
        columns = instruments.get_indexer(frame["instrument"])
        keep = (rows >= 0) & (columns >= 0)
        if seen[rows[keep], columns[keep]].any():
            raise DataContractError(f"daily_basic repeats a (datetime, instrument) pair across files: {path}")
        seen[rows[keep], columns[keep]] = True
        for field in scales:
            column = frame[field].to_numpy(dtype=np.float64, na_value=np.nan)
            values[field][rows[keep], columns[keep]] = column[keep]
    panels = {}
    for field, scale in scales.items():
        panel = pd.DataFrame(values.pop(field), index=dates, columns=instruments)
        panels[f"basic_{field}"] = panel if scale is None else panel * scale
    return panels
