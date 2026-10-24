"""Classic TA-Lib technical indicators and candlestick patterns as a daily panel factor set.

Every factor calls one function of the TA-Lib C library through the ``TA-Lib`` Python wrapper on a single stock's
own trading days: rows where the stock has no complete open, high, low, close and volume quote (suspensions, days
before listing) are removed before the call and stay missing afterwards, so recursive indicators (EMA, Wilder
smoothing, Hilbert transform) run over consecutive quotes instead of being reset by a gap. Prices are RQuant's
backward-adjusted canonical prices, whose level is arbitrary per stock, so indicators measured in price units are
published scale-free (see ``TRANSFORMS``). The first ``lookback`` quotes of every stock, for which TA-Lib has no
value or returns a placeholder 0 for candlestick patterns, are missing.

The set is the TA-Lib function list of the classic 0.4 C library (TA-Lib 0.6+ adds indicators that are not used
here) without the math operators and transforms, and without outputs that are an exact affine function of another
published output (ROCP/ROCR/ROCR100 of ROC, CMO = 2 RSI - 100, MA of SMA, TYPPRICE/WCLPRICE of MEDPRICE, VAR of
STDDEV, NATR of ATR) or a near copy of one (MACDFIX, MACDEXT, SAREXT, MAVP, MOM and APO beside ROC and PPO).
LINEARREG_ANGLE and BETA are left out: the first depends on the price scale and TA-Lib's BETA has no market input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DependencyError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.io import stable_hash

Panel = pd.DataFrame
FLOW_DAYS = 20
ADOSC_VOLUME_DAYS = 10
INSTRUMENT_BLOCK = 64
FIELDS = ("open", "high", "low", "close", "volume")
OHLC = ("open", "high", "low", "close")
HLC = ("high", "low", "close")
HL = ("high", "low")
HLCV = ("high", "low", "close", "volume")
C = ("close",)
CV = ("close", "volume")

# Published value from the selected TA-Lib output ``x``, all outputs of the call ``out`` and the stock's inputs ``a``.
Transform = Callable[[np.ndarray, tuple[np.ndarray, ...], Mapping[str, np.ndarray]], np.ndarray]
TRANSFORMS: dict[str, tuple[str, Transform, int]] = {
    # name: (description, transform, extra lookback beyond TA-Lib's)
    "level": ("TA-Lib output", lambda x, out, a: x, 0),
    "price": ("output / close - 1", lambda x, out, a: x / a["close"] - 1.0, 0),
    "spread": ("output / close", lambda x, out, a: x / a["close"], 0),
    "flow": (
        f"{FLOW_DAYS}-day change of the cumulative line over the volume of those days",
        lambda x, out, a: _flow(x, a["volume"]),
        FLOW_DAYS,
    ),
    "per_volume": (
        f"output over the {ADOSC_VOLUME_DAYS}-day mean volume",
        lambda x, out, a: x / _trailing_mean(a["volume"], ADOSC_VOLUME_DAYS),
        ADOSC_VOLUME_DAYS - 1,
    ),
    "bb_pctb": (
        "(close - lower band) / (upper - lower band)",
        lambda x, out, a: (a["close"] - out[2]) / (out[0] - out[2]),
        0,
    ),
    "bb_width": ("(upper - lower band) / middle band", lambda x, out, a: (out[0] - out[2]) / out[1], 0),
}

# (source name, TA-Lib function, inputs, parameters, output position, transform, TA-Lib lookback).
# The lookback is TA-Lib's own count of leading quotes without output; tests check it against the installed library.
TALIB_FACTORS: tuple[tuple[str, str, tuple[str, ...], dict[str, Any], int, str, int], ...] = (
    # Momentum
    ("adx", "ADX", HLC, {"timeperiod": 14}, 0, "level", 27),
    ("adxr", "ADXR", HLC, {"timeperiod": 14}, 0, "level", 40),
    ("aroon_down", "AROON", HL, {"timeperiod": 14}, 0, "level", 14),
    ("aroon_up", "AROON", HL, {"timeperiod": 14}, 1, "level", 14),
    ("aroonosc", "AROONOSC", HL, {"timeperiod": 14}, 0, "level", 14),
    ("bop", "BOP", OHLC, {}, 0, "level", 0),
    ("cci", "CCI", HLC, {"timeperiod": 14}, 0, "level", 13),
    ("dx", "DX", HLC, {"timeperiod": 14}, 0, "level", 14),
    ("macd", "MACD", C, {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, 0, "spread", 33),
    ("macd_signal", "MACD", C, {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, 1, "spread", 33),
    ("macd_hist", "MACD", C, {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, 2, "spread", 33),
    ("mfi", "MFI", HLCV, {"timeperiod": 14}, 0, "level", 14),
    ("minus_di", "MINUS_DI", HLC, {"timeperiod": 14}, 0, "level", 14),
    ("minus_dm", "MINUS_DM", HL, {"timeperiod": 14}, 0, "spread", 13),
    ("plus_di", "PLUS_DI", HLC, {"timeperiod": 14}, 0, "level", 14),
    ("plus_dm", "PLUS_DM", HL, {"timeperiod": 14}, 0, "spread", 13),
    ("ppo", "PPO", C, {"fastperiod": 12, "slowperiod": 26, "matype": 0}, 0, "level", 25),
    ("roc", "ROC", C, {"timeperiod": 10}, 0, "level", 10),
    ("rsi", "RSI", C, {"timeperiod": 14}, 0, "level", 14),
    ("stoch_k", "STOCH", HLC, {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3}, 0, "level", 8),
    ("stoch_d", "STOCH", HLC, {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3}, 1, "level", 8),
    ("stochf_k", "STOCHF", HLC, {"fastk_period": 5, "fastd_period": 3}, 0, "level", 6),
    ("stochf_d", "STOCHF", HLC, {"fastk_period": 5, "fastd_period": 3}, 1, "level", 6),
    ("stochrsi_k", "STOCHRSI", C, {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3}, 0, "level", 20),
    ("stochrsi_d", "STOCHRSI", C, {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3}, 1, "level", 20),
    ("trix", "TRIX", C, {"timeperiod": 30}, 0, "level", 88),
    ("ultosc", "ULTOSC", HLC, {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28}, 0, "level", 28),
    ("willr", "WILLR", HLC, {"timeperiod": 14}, 0, "level", 13),
    # Overlap studies, relative to the close
    ("bb_pctb", "BBANDS", C, {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0, "matype": 0}, 0, "bb_pctb", 19),
    ("bb_width", "BBANDS", C, {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0, "matype": 0}, 0, "bb_width", 19),
    ("dema", "DEMA", C, {"timeperiod": 30}, 0, "price", 58),
    ("ema", "EMA", C, {"timeperiod": 30}, 0, "price", 29),
    ("ht_trendline", "HT_TRENDLINE", C, {}, 0, "price", 63),
    ("kama", "KAMA", C, {"timeperiod": 30}, 0, "price", 30),
    ("mama", "MAMA", C, {"fastlimit": 0.5, "slowlimit": 0.05}, 0, "price", 32),
    ("fama", "MAMA", C, {"fastlimit": 0.5, "slowlimit": 0.05}, 1, "price", 32),
    ("midpoint", "MIDPOINT", C, {"timeperiod": 14}, 0, "price", 13),
    ("midprice", "MIDPRICE", HL, {"timeperiod": 14}, 0, "price", 13),
    ("sar", "SAR", HL, {"acceleration": 0.02, "maximum": 0.2}, 0, "price", 1),
    ("sma", "SMA", C, {"timeperiod": 30}, 0, "price", 29),
    ("t3", "T3", C, {"timeperiod": 5, "vfactor": 0.7}, 0, "price", 24),
    ("tema", "TEMA", C, {"timeperiod": 30}, 0, "price", 87),
    ("trima", "TRIMA", C, {"timeperiod": 30}, 0, "price", 29),
    ("wma", "WMA", C, {"timeperiod": 30}, 0, "price", 29),
    # Volume
    ("ad", "AD", HLCV, {}, 0, "flow", 0),
    ("adosc", "ADOSC", HLCV, {"fastperiod": 3, "slowperiod": 10}, 0, "per_volume", 9),
    ("obv", "OBV", CV, {}, 0, "flow", 0),
    # Volatility
    ("natr", "NATR", HLC, {"timeperiod": 14}, 0, "level", 14),
    ("trange", "TRANGE", HLC, {}, 0, "spread", 1),
    # Price transform
    ("avgprice", "AVGPRICE", OHLC, {}, 0, "price", 0),
    ("medprice", "MEDPRICE", HL, {}, 0, "price", 0),
    # Cycle
    ("ht_dcperiod", "HT_DCPERIOD", C, {}, 0, "level", 32),
    ("ht_dcphase", "HT_DCPHASE", C, {}, 0, "level", 63),
    ("ht_inphase", "HT_PHASOR", C, {}, 0, "spread", 32),
    ("ht_quadrature", "HT_PHASOR", C, {}, 1, "spread", 32),
    ("ht_sine", "HT_SINE", C, {}, 0, "level", 63),
    ("ht_leadsine", "HT_SINE", C, {}, 1, "level", 63),
    ("ht_trendmode", "HT_TRENDMODE", C, {}, 0, "level", 63),
    # Statistic
    ("correl_cv", "CORREL", CV, {"timeperiod": 30}, 0, "level", 29),
    ("linearreg", "LINEARREG", C, {"timeperiod": 14}, 0, "price", 13),
    ("linearreg_intercept", "LINEARREG_INTERCEPT", C, {"timeperiod": 14}, 0, "price", 13),
    ("linearreg_slope", "LINEARREG_SLOPE", C, {"timeperiod": 14}, 0, "spread", 13),
    ("stddev", "STDDEV", C, {"timeperiod": 20, "nbdev": 1.0}, 0, "spread", 19),
    ("tsf", "TSF", C, {"timeperiod": 14}, 0, "price", 13),
)

# Candlestick patterns: +100/-100 (some +-200) on the day a bullish/bearish pattern completes, else 0.
CANDLESTICK_PATTERNS: tuple[tuple[str, int], ...] = (
    ("CDL2CROWS", 12), ("CDL3BLACKCROWS", 13), ("CDL3INSIDE", 12), ("CDL3LINESTRIKE", 8), ("CDL3OUTSIDE", 3),
    ("CDL3STARSINSOUTH", 12), ("CDL3WHITESOLDIERS", 12), ("CDLABANDONEDBABY", 12), ("CDLADVANCEBLOCK", 12),
    ("CDLBELTHOLD", 10), ("CDLBREAKAWAY", 14), ("CDLCLOSINGMARUBOZU", 10), ("CDLCONCEALBABYSWALL", 13),
    ("CDLCOUNTERATTACK", 11), ("CDLDARKCLOUDCOVER", 11), ("CDLDOJI", 10), ("CDLDOJISTAR", 11),
    ("CDLDRAGONFLYDOJI", 10), ("CDLENGULFING", 2), ("CDLEVENINGDOJISTAR", 12), ("CDLEVENINGSTAR", 12),
    ("CDLGAPSIDESIDEWHITE", 7), ("CDLGRAVESTONEDOJI", 10), ("CDLHAMMER", 11), ("CDLHANGINGMAN", 11),
    ("CDLHARAMI", 11), ("CDLHARAMICROSS", 11), ("CDLHIGHWAVE", 10), ("CDLHIKKAKE", 5), ("CDLHIKKAKEMOD", 10),
    ("CDLHOMINGPIGEON", 11), ("CDLIDENTICAL3CROWS", 12), ("CDLINNECK", 11), ("CDLINVERTEDHAMMER", 11),
    ("CDLKICKING", 11), ("CDLKICKINGBYLENGTH", 11), ("CDLLADDERBOTTOM", 14), ("CDLLONGLEGGEDDOJI", 10),
    ("CDLLONGLINE", 10), ("CDLMARUBOZU", 10), ("CDLMATCHINGLOW", 6), ("CDLMATHOLD", 14), ("CDLMORNINGDOJISTAR", 12),
    ("CDLMORNINGSTAR", 12), ("CDLONNECK", 11), ("CDLPIERCING", 11), ("CDLRICKSHAWMAN", 10),
    ("CDLRISEFALL3METHODS", 14), ("CDLSEPARATINGLINES", 11), ("CDLSHOOTINGSTAR", 11), ("CDLSHORTLINE", 10),
    ("CDLSPINNINGTOP", 10), ("CDLSTALLEDPATTERN", 12), ("CDLSTICKSANDWICH", 7), ("CDLTAKURI", 10),
    ("CDLTASUKIGAP", 7), ("CDLTHRUSTING", 11), ("CDLTRISTAR", 12), ("CDLUNIQUE3RIVER", 12),
    ("CDLUPSIDEGAP2CROWS", 12), ("CDLXSIDEGAP3METHODS", 2),
)
PATTERN_PARAMETERS: dict[str, dict[str, Any]] = {
    name: {"penetration": 0.3}
    for name in ("CDLABANDONEDBABY", "CDLEVENINGDOJISTAR", "CDLEVENINGSTAR", "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR")
} | {"CDLDARKCLOUDCOVER": {"penetration": 0.5}, "CDLMATHOLD": {"penetration": 0.5}}

TALIB_INDICATORS: tuple[tuple[str, str, tuple[str, ...], dict[str, Any], int, str, int], ...] = TALIB_FACTORS + tuple(
    (name[3:].lower(), name, OHLC, PATTERN_PARAMETERS.get(name, {}), 0, "level", lookback)
    for name, lookback in CANDLESTICK_PATTERNS
)
TALIB_NAMES = tuple(entry[0] for entry in TALIB_INDICATORS)


def _call(function: str, inputs: tuple[str, ...], parameters: Mapping[str, Any]) -> str:
    arguments = [*inputs, *(f"{key}={value}" for key, value in parameters.items())]
    return f"{function}({', '.join(arguments)})"


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


class TALibLibrary(PanelFactorLibrary):
    """TA-Lib indicators and candlestick patterns, evaluated per stock over its own trading days."""

    family = "talib"
    required_inputs = FIELDS

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"ta_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=f"{_call(function, inputs, parameters)}"
                + ("" if transform == "level" else f"; {TRANSFORMS[transform][0]}"),
                max_lookback=lookback + TRANSFORMS[transform][2] + 1,
                implementation=f"talib.{function}",
                catalog_version=1,
            )
            for ordinal, (name, function, inputs, parameters, _, transform, lookback) in enumerate(
                TALIB_INDICATORS, 1
            )
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Pin the TA-Lib wrapper and C library versions as well as this module's source."""
        from importlib.metadata import version

        talib = _talib()
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "ta_lib_python": version("TA-Lib"),
                "ta_lib_c": str(talib.__ta_version__.decode() if isinstance(talib.__ta_version__, bytes)
                                else talib.__ta_version__).split()[0],
            }
        )

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        close = inputs["close"]
        buffers = {name: np.empty(close.shape) for name in TALIB_NAMES}
        self._calculate_columns(inputs, lambda name, columns, values: buffers[name].__setitem__(
            (slice(None), columns), values))
        return {name: Panel(buffers[name], index=close.index, columns=close.columns) for name in TALIB_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        canonical = {spec.source_name: spec.canonical_name for spec in self.specs}

        def store(name: str, columns: slice, values: np.ndarray) -> None:
            output_buffers[canonical[name]][:, columns] = values

        self._calculate_columns(inputs, store)

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
        _, formula, extra = TRANSFORMS[transform]
        with np.errstate(invalid="ignore", divide="ignore"):
            values = formula(outputs[position], outputs, arrays)
        values = np.where(np.isfinite(values), values, np.nan)
        values[: min(count, lookback + extra)] = np.nan
        out[name] = values
    return out
