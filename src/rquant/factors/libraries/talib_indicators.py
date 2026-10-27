"""Classic TA-Lib technical indicators and candlestick patterns as a daily KunQuant factor set.

Every factor runs one function of the TA-Lib C library on a single stock's own trading days. The functions are
TA-Lib 0.8.1's own C code, vendored into ``rquant/factors/extensions/ta_lib/ta_lib.hpp`` by
``scripts/vendor_ta_lib.py`` and compiled into the KunQuant graph; each graph node reads a stock's whole series, calls
the function once and publishes its output. The engine first moves every stock's quotes (rows with a complete open,
high, low, close and volume) to the top of the time axis and puts the outputs back on their dates afterwards, so
suspensions and days before listing stay missing and recursive indicators (EMA, Wilder smoothing, Hilbert transform)
run over consecutive quotes instead of being reset by a gap. Prices are RQuant's backward-adjusted canonical prices,
whose level is arbitrary per stock, so indicators measured in price units are published scale-free (see
``TRANSFORMS``). The first ``lookback`` quotes of every stock, for which TA-Lib has no value or returns a placeholder 0
for candlestick patterns, are missing.

The set is the TA-Lib function list of the classic 0.4 C library (TA-Lib 0.6+ adds indicators that are not used
here) without the math operators and transforms, and without outputs that are an exact affine function of another
published output (ROCP/ROCR/ROCR100 of ROC, CMO = 2 RSI - 100, MA of SMA, TYPPRICE/WCLPRICE of MEDPRICE, VAR of
STDDEV, NATR of ATR) or a near copy of one (MACDFIX, MACDEXT, SAREXT, MAVP, MOM and APO beside ROC and PPO).
LINEARREG_ANGLE and BETA are left out: the first depends on the price scale and TA-Lib's BETA has no market input.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.io import stable_hash

TA_LIB_HEADER = Path(__file__).resolve().parents[1] / "extensions" / "ta_lib" / "ta_lib.hpp"
FLOW_DAYS = 20
ADOSC_VOLUME_DAYS = 10
FIELDS = ("open", "high", "low", "close", "volume")
OHLC = ("open", "high", "low", "close")
HLC = ("high", "low", "close")
HL = ("high", "low")
HLCV = ("high", "low", "close", "volume")
C = ("close",)
CV = ("close", "volume")

# name: (description, C++ expression of the published value at quote ``t``, extra lookback beyond TA-Lib's). The
# expression sees the selected TA-Lib output ``x``, all outputs ``o0, o1, ...``, the stock's quotes ``in_<field>`` and
# ``volume_sum(t, days)``, the volume of the ``days`` quotes ending at ``t``.
TRANSFORMS: dict[str, tuple[str, str, int]] = {
    "level": ("TA-Lib output", "x[t]", 0),
    "price": ("output / close - 1", "x[t] / in_close[t] - 1.0", 0),
    "spread": ("output / close", "x[t] / in_close[t]", 0),
    "flow": (
        f"{FLOW_DAYS}-day change of the cumulative line over the volume of those days",
        f"t < {FLOW_DAYS} ? nan : (x[t] - x[t - {FLOW_DAYS}]) / volume_sum(t, {FLOW_DAYS})",
        FLOW_DAYS,
    ),
    "per_volume": (
        f"output over the {ADOSC_VOLUME_DAYS}-day mean volume",
        f"t < {ADOSC_VOLUME_DAYS - 1} ? nan : x[t] / (volume_sum(t, {ADOSC_VOLUME_DAYS}) / {ADOSC_VOLUME_DAYS}.0)",
        ADOSC_VOLUME_DAYS - 1,
    ),
    "bb_pctb": ("(close - lower band) / (upper - lower band)", "(in_close[t] - o2[t]) / (o0[t] - o2[t])", 0),
    "bb_width": ("(upper - lower band) / middle band", "(o0[t] - o2[t]) / o1[t]", 0),
}
TRANSFORM_FIELDS = {"price": ("close",), "spread": ("close",), "flow": ("volume",), "per_volume": ("volume",),
                    "bb_pctb": ("close",)}

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


@lru_cache(maxsize=1)
def _header() -> str:
    return TA_LIB_HEADER.read_text(encoding="utf-8")


@cache
def _signature(function: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The optional parameters and outputs of a vendored TA-Lib batch function, as (C type, name) in call order."""
    match = re.search(rf"static inline TA_RetCode TA_{function}\((.*?)\);", _header(), flags=re.S)
    if match is None:
        raise KeyError(f"TA-Lib function {function} is not vendored")
    parameters: list[tuple[str, str]] = []
    outputs: list[tuple[str, str]] = []
    for argument in (part.strip() for part in match.group(1).split(",")):
        kind, name = argument.rsplit(None, 1)
        if name.startswith("optIn"):
            parameters.append((kind, name))
        elif name.startswith("out") and name.endswith("[]"):
            outputs.append((kind, name[:-2]))
    return parameters, outputs


def _normalized(name: str) -> str:
    return name.removeprefix("optIn").replace("_", "").lower()


def _kernel(
    function: str, inputs: tuple[str, ...], parameters: Mapping[str, Any], position: int, transform: str, cut: int
) -> tuple[tuple[str, ...], str]:
    """The kernel fields and C++ body publishing one indicator for a stock's quotes."""
    declared, outputs = _signature(function)
    given = {_normalized(key): value for key, value in parameters.items()}
    arguments = []
    for kind, name in declared:
        value = given.pop(_normalized(name), None)
        if value is None:
            arguments.append("INT_MIN" if kind in {"int", "TA_MAType"} else "-4e+37")
        else:
            arguments.append(repr(float(value)) if kind == "double" else str(int(value)))
    if given:
        raise KeyError(f"{function} has no parameters {sorted(given)}")
    fields = tuple(dict.fromkeys((*inputs, *TRANSFORM_FIELDS.get(transform, ()))))
    pointers = " ".join(f"const double *in_{field} = x{index};" for index, field in enumerate(fields))
    quoted = " && ".join(f"std::isfinite(in_{field}[m])" for field in fields)
    buffers = " ".join(
        f"std::vector<{'int' if kind == 'int' else 'double'}> r{index}(m);" for index, (kind, _) in enumerate(outputs)
    )
    aligned = " ".join(f"std::vector<double> o{index}(m, nan);" for index in range(len(outputs)))
    scatter = " ".join(f"o{index}[beg + j] = (double)r{index}[j];" for index in range(len(outputs)))
    call = ", ".join(
        [
            "0, m - 1",
            *(f"in_{field}" for field in inputs),
            *arguments,
            "&beg, &nb",
            *(f"r{index}.data()" for index in range(len(outputs))),
        ]
    )
    expression = TRANSFORMS[transform][1]
    volume_sum = ""
    if "volume_sum" in expression:
        volume_sum = """
            std::vector<double> volume_prefix(m + 1, 0.0);
            for (int i = 0; i < m; ++i) volume_prefix[i + 1] = volume_prefix[i] + in_volume[i];
            auto volume_sum = [&](int t, int days) { return volume_prefix[t + 1] - volume_prefix[t + 1 - days]; };"""
    body = f"""
        {pointers}
        const double nan = std::numeric_limits<double>::quiet_NaN();
        int m = 0;
        while (m < n && {quoted}) ++m;
        if (m > 0) {{
            {buffers} {aligned}
            int beg = 0, nb = 0;
            if (rq_ta_lib::TA_{function}({call}) == rq_ta_lib::TA_SUCCESS) {{
                for (int j = 0; j < nb; ++j) {{ {scatter} }}
            }}
            const std::vector<double> &x = o{position};
            (void)x;{volume_sum}
            for (int t = 0; t < m; ++t) {{
                const double value = {expression};
                out[t] = t < {cut} || !std::isfinite(value) ? nan : value;
            }}
        }}"""
    return fields, body


class TALibLibrary(FactorLibrary):
    """TA-Lib indicators and candlestick patterns, evaluated in KunQuant over every stock's own trading days."""

    family = "talib"
    required_inputs = FIELDS
    quote_fields = FIELDS
    strict_float = True

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
                implementation=f"rquant.factors.extensions.ta_lib.TA_{function}",
                catalog_version=2,
            )
            for ordinal, (name, function, inputs, parameters, _, transform, lookback) in enumerate(
                TALIB_INDICATORS, 1
            )
        )

    @property
    def cpp_prelude(self) -> str:
        return _header()

    @property
    def implementation_fingerprint(self) -> str:
        """Pin the vendored TA-Lib C source as well as this module's source."""
        return stable_hash({"module": super().implementation_fingerprint, "ta_lib": stable_hash(_header())})

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = {}
        for name, function, fields, parameters, position, transform, lookback in TALIB_INDICATORS:
            kernel_fields, body = _kernel(
                function, fields, parameters, position, transform, lookback + TRANSFORMS[transform][2]
            )
            outputs[name] = k.series_kernel([inputs[field] for field in kernel_fields], body)
        return outputs
