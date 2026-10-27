"""Reference calculation of the talib factor set through the TA-Lib Python wrapper.

RQuant computes the talib set in KunQuant with TA-Lib's C code vendored into the graph
(``rquant.factors.libraries.talib_indicators``). This calculator, the original pandas implementation, calls the
installed TA-Lib Python package per stock instead and is kept for the parity tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DependencyError
from rquant.factors.libraries.talib_indicators import (
    ADOSC_VOLUME_DAYS,
    FIELDS,
    FLOW_DAYS,
    TALIB_INDICATORS,
    TALIB_NAMES,
)

Panel = pd.DataFrame
INSTRUMENT_BLOCK = 64
Transform = Callable[[np.ndarray, tuple[np.ndarray, ...], Mapping[str, np.ndarray]], np.ndarray]
TRANSFORMS: dict[str, tuple[Transform, int]] = {
    "level": (lambda x, out, a: x, 0),
    "price": (lambda x, out, a: x / a["close"] - 1.0, 0),
    "spread": (lambda x, out, a: x / a["close"], 0),
    "flow": (lambda x, out, a: _flow(x, a["volume"]), FLOW_DAYS),
    "per_volume": (
        lambda x, out, a: x / _trailing_mean(a["volume"], ADOSC_VOLUME_DAYS),
        ADOSC_VOLUME_DAYS - 1,
    ),
    "bb_pctb": (lambda x, out, a: (a["close"] - out[2]) / (out[0] - out[2]), 0),
    "bb_width": (lambda x, out, a: (out[0] - out[2]) / out[1], 0),
}


def _trailing_mean(values: np.ndarray, days: int) -> np.ndarray:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    out = np.full(len(values), np.nan)
    if len(values) >= days:
        out[days - 1 :] = (prefix[days:] - prefix[:-days]) / days
    return out


def _flow(line: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Change of a cumulative volume line over ``FLOW_DAYS`` quotes, per unit of volume traded in those quotes."""
    out = np.full(len(line), np.nan)
    if len(line) > FLOW_DAYS:
        prefix = np.concatenate([[0.0], np.cumsum(volume)])
        traded = prefix[FLOW_DAYS + 1 :] - prefix[1:-FLOW_DAYS]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[FLOW_DAYS:] = (line[FLOW_DAYS:] - line[:-FLOW_DAYS]) / traded
    return out


def _talib() -> Any:
    try:
        import talib
    except ImportError as exc:
        raise DependencyError("The talib factor set needs the TA-Lib package: pip install TA-Lib==0.8.2") from exc
    return talib


class TALibReference:
    """The talib factor set computed per stock by the TA-Lib Python package."""

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        close = inputs["close"]
        buffers = {name: np.empty(close.shape) for name in TALIB_NAMES}
        self._calculate_columns(inputs, lambda name, columns, values: buffers[name].__setitem__(
            (slice(None), columns), values))
        return {name: Panel(buffers[name], index=close.index, columns=close.columns) for name in TALIB_NAMES}

    def _calculate_columns(
        self, inputs: Mapping[str, Any], store: Callable[[str, slice, np.ndarray], None]
    ) -> None:
        """Compute blocks of instruments at a time, so writes to a date-major store stay mostly contiguous."""
        talib = _talib()
        panels = {field: np.asarray(inputs[field], dtype=float) for field in FIELDS}
        dates, stocks = panels["close"].shape
        for first in range(0, stocks, INSTRUMENT_BLOCK):
            columns = slice(first, min(first + INSTRUMENT_BLOCK, stocks))
            width = columns.stop - columns.start
            block = {name: np.full((dates, width), np.nan) for name in TALIB_NAMES}
            for offset in range(width):
                column = first + offset
                quoted = np.isfinite(np.column_stack([panels[field][:, column] for field in FIELDS])).all(axis=1)
                rows = np.flatnonzero(quoted)
                if not len(rows):
                    continue
                arrays = {field: np.ascontiguousarray(panels[field][rows, column]) for field in FIELDS}
                for name, values in _stock_indicators(talib, arrays).items():
                    block[name][rows, offset] = values
            for name in TALIB_NAMES:
                store(name, columns, block[name])


def _stock_indicators(talib: Any, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every published indicator for one stock's consecutive quotes."""
    count = len(arrays["close"])
    calls: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    out: dict[str, np.ndarray] = {}
    for name, function, inputs, parameters, position, transform, lookback in TALIB_INDICATORS:
        key = (function, repr(sorted(parameters.items())))
        if key not in calls:
            result = getattr(talib, function)(*(arrays[field] for field in inputs), **parameters)
            result = result if isinstance(result, tuple) else (result,)
            calls[key] = tuple(np.asarray(value, dtype=float) for value in result)
        outputs = calls[key]
        formula, extra = TRANSFORMS[transform]
        with np.errstate(invalid="ignore", divide="ignore"):
            values = formula(outputs[position], outputs, arrays)
        values = np.where(np.isfinite(values), values, np.nan)
        values[: min(count, lookback + extra)] = np.nan
        out[name] = values
    return out
