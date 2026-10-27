# ruff: noqa: E501
"""Guotai Junan Alpha191 factors executed by KunQuant.

The formulas follow the Guotai Junan report as audited in RQuant's original pandas implementation (kept as the
parity reference in ``tests/panel_reference/gtja191.py``) and use its panel semantics through
``rquant.factors.extensions.kunquant_panel``: complete rolling windows, missing comparisons are false, near-zero
denominators and infinite results are missing. Benchmark open and close are the CSI300 index series, the same value
for every stock of a day.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.libraries.base import FactorLibrary, FactorSpec

GTJA191_FORMULA_NAMES = tuple(f"gtja_{number:03d}" for number in range(1, 192))
# Alpha030 regresses returns on daily mkt/smb/hml factor returns and stays out of the build catalog until an audited
# style-factor source is wired in; its pandas formula is kept in the parity reference.
GTJA191_BUILD_EXCLUSIONS = frozenset({"gtja_030"})
GTJA191_NAMES = tuple(name for name in GTJA191_FORMULA_NAMES if name not in GTJA191_BUILD_EXCLUSIONS)
GTJA191_AMOUNT_FACTORS = frozenset(f"gtja_{number:03d}" for number in (70, 95, 132, 144))
GTJA191_VWAP_FACTORS = frozenset(
    f"gtja_{number:03d}"
    for number in (
        7,
        8,
        12,
        13,
        16,
        17,
        26,
        36,
        39,
        41,
        44,
        45,
        61,
        64,
        73,
        74,
        77,
        87,
        90,
        92,
        101,
        108,
        114,
        119,
        120,
        121,
        124,
        125,
        130,
        131,
        138,
        154,
        156,
        163,
        170,
        179,
    )
)
GTJA191_MARKET_RELATED_FACTORS = frozenset(f"gtja_{number:03d}" for number in (30, 75, 149, 181, 182))
GTJA191_LIQUIDITY_FACTORS = frozenset(
    f"gtja_{number:03d}" for number in (70, 80, 81, 95, 97, 100, 102, 132, 145, 155, 168)
)
GTJA191_TRADITIONAL_TECHNICAL_FACTORS = frozenset(
    f"gtja_{number:03d}"
    for number in (
        23,
        24,
        27,
        28,
        34,
        46,
        47,
        63,
        65,
        67,
        69,
        72,
        78,
        79,
        82,
        89,
        96,
        137,
        159,
        161,
        162,
        172,
        173,
        175,
        177,
        183,
        186,
        187,
        188,
        189,
    )
)
GTJA191_PRICE_VOLUME_FACTORS = frozenset(
    f"gtja_{number:03d}"
    for number in (
        1,
        4,
        5,
        7,
        9,
        11,
        16,
        25,
        29,
        32,
        33,
        35,
        36,
        39,
        40,
        42,
        44,
        45,
        48,
        56,
        60,
        61,
        62,
        64,
        68,
        73,
        74,
        76,
        77,
        83,
        84,
        85,
        90,
        91,
        92,
        94,
        99,
        101,
        104,
        105,
        108,
        111,
        113,
        114,
        115,
        117,
        119,
        121,
        123,
        125,
        128,
        130,
        131,
        134,
        136,
        138,
        139,
        140,
        141,
        142,
        144,
        148,
        150,
        154,
        163,
        170,
        176,
        178,
        179,
        180,
        191,
    )
)
GTJA191_FORMULA_NOTES: dict[str, str] = {
    "gtja_028": "Use TSMAX(HIGH,9)-TSMIN(LOW,9) in both stochastic terms.",
    "gtja_030": "Map MKT/SMB/HML to explicit external daily factor returns.",
    "gtja_035": "The published OPEN*0.65+OPEN*0.35 term simplifies to OPEN.",
    "gtja_054": "Use the original report's explicit 10-day STD window.",
    "gtja_075": "Benchmark OPEN and CLOSE map to their semantic index fields.",
    "gtja_078": "MA is interpreted as the report-defined rolling MEAN.",
    "gtja_131": "Correct the published DELAT spelling to DELTA.",
    "gtja_159": "Preserve the report's literal weighted 6/12/24-day expression and correct HGIH to HIGH.",
    "gtja_165": "Resolve the published numerator parentheses using the original appendix.",
    "gtja_166": "Resolve the published denominator parentheses using the original appendix.",
    "gtja_181": "Use a 20-day denominator window; the report omits SUM's required window argument.",
    "gtja_183": "Resolve the published SUMAC parentheses using the original appendix.",
}


class GTJA191Error(ValueError):
    """Base error for GTJA191 calculation failures."""


class GTJA191DataError(GTJA191Error):
    """Raised when a factor's required point-in-time input is unavailable."""


def normalize_gtja_name(name: str | int) -> str:
    """Normalize supported aliases to the non-conflicting ``gtja_NNN`` form."""

    if isinstance(name, int):
        number = name
    else:
        raw = str(name).strip().lower().replace("-", "_")
        if raw.startswith("gtja_"):
            raw = raw.removeprefix("gtja_")
        elif raw.startswith("gtja"):
            raw = raw.removeprefix("gtja")
        else:
            raise KeyError(f"invalid GTJA191 factor name: {name}")
        try:
            number = int(raw)
        except ValueError as exc:
            raise KeyError(f"invalid GTJA191 factor name: {name}") from exc
    if not 1 <= number <= 191:
        raise KeyError(f"GTJA191 factor number must be in [1, 191], got {number}")
    return f"gtja_{number:03d}"


def gtja_factor_category(name: str | int) -> str:
    """Classify every GTJA191 formula by its dominant economic input family.

    Lifecycle status is intentionally independent: a disabled formula still
    receives a research category, but classification does not make it active.
    Explicit configuration may override this deterministic fallback.
    """

    normalized = normalize_gtja_name(name)
    if normalized in GTJA191_MARKET_RELATED_FACTORS:
        return "market_related"
    if normalized in GTJA191_TRADITIONAL_TECHNICAL_FACTORS:
        return "traditional_technical"
    if normalized in GTJA191_LIQUIDITY_FACTORS:
        return "liquidity"
    if normalized in GTJA191_PRICE_VOLUME_FACTORS:
        return "price_volume"
    return "price_behavior"


# ---------------------------------------------------------------------------------------------------- operators
FILTERED_REGRESSION_DAYS = 252


def wma(value: Any, periods: int) -> Any:
    """Report WMA with weights proportional to ``0.9**distance`` from the newest row."""
    weights = np.power(0.9, np.arange(periods, dtype=float))
    return k.weighted_sum(value, list(weights / weights.sum()))


def highday(value: Any, periods: int) -> Any:
    """Distance from today to the most recent window maximum."""
    return k.days_since_extreme(value, periods, maximum=True)


def lowday(value: Any, periods: int) -> Any:
    """Distance from today to the most recent window minimum."""
    return k.days_since_extreme(value, periods, maximum=False)


def sequence_regbeta(value: Any, periods: int) -> Any:
    """Rolling OLS slope (with intercept) of the window's values on the sequence 1..periods."""
    sequence = np.arange(1.0, periods + 1.0)
    centered = sequence - sequence.mean()
    weights = centered / (centered**2).sum()
    # weights[i] applies to the i-th oldest row; weighted_sum counts lags from the newest row.
    return k.weighted_sum(value, list(weights[::-1]))


def filtered_regbeta(dependent: Any, independent: Any, condition: Any, periods: int) -> Any:
    """OLS slope over the latest ``periods`` rows that satisfy FILTER's condition, held until the next such row.

    FILTER shortens the sequence, so the window spans as many calendar rows as it takes to collect ``periods``
    qualifying observations; a fixed calendar window would require every row in it to qualify.
    """
    step = f"""
        if (x2 != 0 && std::isfinite(x0) && std::isfinite(x1)) {{
            s.ys[s.head] = x0; s.xs[s.head] = x1; s.head = (s.head + 1) % {periods};
            if (s.filled < {periods}) s.filled++;
            if (s.filled == {periods}) {{
                {{T}} sx = 0, sy = 0, sxx = 0, sxy = 0;
                for (int i = 0; i < {periods}; ++i) {{
                    sx += s.xs[i]; sy += s.ys[i]; sxx += s.xs[i] * s.xs[i]; sxy += s.xs[i] * s.ys[i];
                }}
                const {{T}} denominator = {periods} * sxx - sx * sx;
                s.beta = std::abs(denominator) <= 1e-8 ? std::numeric_limits<{{T}}>::quiet_NaN()
                    : ({periods} * sxy - sx * sy) / denominator;
            }}
        }}
        y = s.beta;
    """
    return k.cpp_state(
        [dependent, independent, k.indicator(condition)],
        step,
        members=f"{{T}} xs[{periods}]; {{T}} ys[{periods}]; int head; int filled; {{T}} beta;",
        init="s.head = 0; s.filled = 0; s.beta = std::numeric_limits<{T}>::quiet_NaN();",
    )


def _neg(value: Any) -> Any:
    return value * -1.0


class _Data:
    """The formula inputs as KunQuant nodes, with shared intermediates built once."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.o = inputs["open"]
        self.h = inputs["high"]
        self.l = inputs["low"]  # noqa: E741 - the conventional name of the low price
        self.c = inputs["close"]
        self.v = inputs["volume"]
        self.amt = inputs["amount"]
        self.vwap = inputs["vwap"]
        self.bo = inputs["benchmark_open"]
        self.bc = inputs["benchmark_close"]
        self.prev_c = k.delay(self.c, 1)
        self.ret = k.safe_div(self.c, self.prev_c) - 1.0
        self._cache: dict[Any, Any] = {}

    def cached(self, key: Any, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def mean(self, value: str, window: int) -> Any:
        return self.cached(("mean", value, window), lambda: k.mean(getattr(self, value), window))

    def change(self) -> Any:
        return self.cached("change", lambda: self.c - self.prev_c)

    def signed_volume(self) -> Any:
        def build() -> Any:
            c, prev, v = self.c, self.prev_c, self.v
            value = k.select(k.eq(c, prev), 0.0, k.select(k.gt(c, prev), v, _neg(v)))
            return k.mask(value, k.valid(c, prev, v))

        return self.cached("signed_volume", build)

    def dtm(self) -> Any:
        def build() -> Any:
            previous_open = k.delay(self.o, 1)
            value = k.emax(self.h - self.o, self.o - previous_open)
            usable = k.valid(self.o, previous_open, self.h)
            return k.mask(k.select(k.gt(self.o, previous_open), value, 0.0), usable)

        return self.cached("dtm", build)

    def dbm(self) -> Any:
        def build() -> Any:
            previous_open = k.delay(self.o, 1)
            value = k.emax(self.o - self.l, self.o - previous_open)
            usable = k.valid(self.o, previous_open, self.l)
            return k.mask(k.select(k.lt(self.o, previous_open), value, 0.0), usable)

        return self.cached("dbm", build)

    def directional_range_parts(self) -> tuple[Any, Any]:
        def build() -> tuple[Any, Any]:
            previous_high = k.delay(self.h, 1)
            previous_low = k.delay(self.l, 1)
            movement = k.emax(k.absolute(self.h - previous_high), k.absolute(self.l - previous_low))
            current_sum = self.h + self.l
            previous_sum = previous_high + previous_low
            usable = k.valid(movement, current_sum, previous_sum)
            down = k.mask(k.select(k.lt(current_sum, previous_sum), movement, 0.0), usable)
            up = k.mask(k.select(k.gt(current_sum, previous_sum), movement, 0.0), usable)
            return up, down

        return self.cached("directional", build)

    def stochastic(self, window: int, percent: bool = True) -> Any:
        def build() -> Any:
            lowest = k.ts_min(self.l, window)
            value = k.safe_div(self.c - lowest, k.ts_max(self.h, window) - lowest)
            return value * 100.0 if percent else value

        return self.cached(("stochastic", window, percent), build)

    def williams(self, window: int) -> Any:
        def build() -> Any:
            highest = k.ts_max(self.h, window)
            return k.safe_div(highest - self.c, highest - k.ts_min(self.l, window)) * 100.0

        return self.cached(("williams", window), build)

    def rsi(self, value: Any, periods: int) -> Any:
        return k.safe_div(k.sma_cn(k.clip(value, lower=0.0), periods, 1), k.sma_cn(k.absolute(value), periods, 1)) * 100.0

    def true_range(self) -> Any:
        def build() -> Any:
            previous = self.prev_c
            return k.emax(k.emax(self.h - self.l, k.absolute(self.h - previous)), k.absolute(self.l - previous))

        return self.cached("true_range", build)

    def adx(self) -> Any:
        def build() -> Any:
            hd = self.h - k.delay(self.h, 1)
            ld = k.delay(self.l, 1) - self.l
            tr_sum = k.ts_sum(self.true_range(), 14)
            usable = k.valid(hd, ld)
            plus = k.safe_div(
                k.ts_sum(k.mask(k.select(k.all_of(k.gt(ld, 0.0), k.gt(ld, hd)), ld, 0.0), usable), 14) * 100.0,
                tr_sum,
            )
            minus = k.safe_div(
                k.ts_sum(k.mask(k.select(k.all_of(k.gt(hd, 0.0), k.gt(hd, ld)), hd, 0.0), usable), 14) * 100.0,
                tr_sum,
            )
            return k.mean(k.safe_div(k.absolute(plus - minus), plus + minus) * 100.0, 6)

        return self.cached("adx", build)

    def accumulated_deviation(self, window: int) -> Any:
        accumulated = k.ts_sum(self.c - k.mean(self.c, window), window)
        return k.safe_div(k.ts_max(accumulated, window) - k.ts_min(accumulated, window), k.stddev(self.c, window))

    def gap_parts(self) -> tuple[Any, Any, Any, Any, Any]:
        """|H - C[-1]|, |L - C[-1]|, |H - L[-1]|, |C[-1] - O[-1]| and the shared numerator of #055 and #137."""

        def build() -> tuple[Any, ...]:
            previous_close = self.prev_c
            previous_open = k.delay(self.o, 1)
            previous_low = k.delay(self.l, 1)
            high_gap = k.absolute(self.h - previous_close)
            low_gap = k.absolute(self.l - previous_close)
            cross_gap = k.absolute(self.h - previous_low)
            open_gap = k.absolute(previous_close - previous_open)
            numerator = (self.c - previous_close + (self.c - self.o) / 2.0 + previous_close - previous_open) * 16.0
            return high_gap, low_gap, cross_gap, open_gap, numerator

        return self.cached("gaps", build)


def _conditional(condition: Any, when_true: float, when_false: float, *operands: Any) -> Any:
    return k.mask(k.select(condition, when_true, when_false), k.valid(*operands))


def _up_down(d: _Data, value: Any, up_condition: Any, down_condition: Any, usable: Any, window: int) -> tuple:
    up = k.ts_sum(k.mask(k.select(up_condition, value, 0.0), usable), window)
    down = k.ts_sum(k.mask(k.select(down_condition, value, 0.0), usable), window)
    return up, down


def _close_location_sum(d: _Data, window: int) -> Any:
    c, prev = d.c, d.prev_c
    up = c - k.emin(d.l, prev)
    down = c - k.emax(d.h, prev)
    value = k.select(k.eq(c, prev), 0.0, k.select(k.gt(c, prev), up, down))
    return k.ts_sum(k.mask(value, k.valid(c, prev, d.l, d.h)), window)


def _volatility_sma(d: _Data, rising: bool) -> Any:
    volatility = k.stddev(d.c, 20)
    usable = k.valid(volatility, d.c, d.prev_c)
    condition = k.gt(d.c, d.prev_c) if rising else k.le(d.c, d.prev_c)
    return k.sma_cn(k.mask(k.select(condition, volatility, 0.0), usable), 20, 1)


# ---------------------------------------------------------------------------------------------------- formulas
def gtja_001(d: _Data) -> Any:
    log_volume = k.log(k.mask(d.v, k.gt(d.v, 0.0)))
    return _neg(k.correlation(k.rank(k.delta(log_volume, 1)), k.rank(k.safe_div(d.c - d.o, d.o)), 6))


def gtja_002(d: _Data) -> Any:
    position = k.safe_div((d.c - d.l) - (d.h - d.c), d.h - d.l)
    return _neg(k.delta(position, 1))


def gtja_003(d: _Data) -> Any:
    return _close_location_sum(d, 6)


def gtja_004(d: _Data) -> Any:
    mean8 = k.mean(d.c, 8)
    mean2 = k.mean(d.c, 2)
    std8 = k.stddev(d.c, 8)
    volume_ratio = k.safe_div(d.v, d.mean("v", 20))
    result = k.mask(k.select(k.ge(volume_ratio, 1.0), 1.0, -1.0), k.notna(volume_ratio))
    result = k.select(k.lt(mean2, mean8 - std8), 1.0, result)
    result = k.select(k.lt(mean8 + std8, mean2), -1.0, result)
    return k.mask(result, k.valid(mean8, mean2, std8))


def gtja_005(d: _Data) -> Any:
    return _neg(k.ts_max(k.correlation(k.ts_rank(d.v, 5), k.ts_rank(d.h, 5), 5), 3))


def gtja_006(d: _Data) -> Any:
    return _neg(k.rank(k.sign(k.delta(d.o * 0.85 + d.h * 0.15, 4))))


def gtja_007(d: _Data) -> Any:
    spread = d.vwap - d.c
    return (k.rank(k.ts_max(spread, 3)) + k.rank(k.ts_min(spread, 3))) * k.rank(k.delta(d.v, 3))


def gtja_008(d: _Data) -> Any:
    value = (d.h + d.l) / 2.0 * 0.2 + d.vwap * 0.8
    return _neg(k.rank(k.delta(value, 4)))


def gtja_009(d: _Data) -> Any:
    midpoint = (d.h + d.l) / 2.0
    value = k.safe_div((midpoint - k.delay(midpoint, 1)) * (d.h - d.l), d.v)
    return k.sma_cn(value, 7, 2)


def gtja_010(d: _Data) -> Any:
    value = k.mask(k.select(k.ge(d.ret, 0.0), d.c, k.stddev(d.ret, 20)), k.valid(d.c, d.ret))
    return k.rank(k.ts_max(value * value, 5))


def gtja_011(d: _Data) -> Any:
    return k.ts_sum(k.safe_div((d.c - d.l) - (d.h - d.c), d.h - d.l) * d.v, 6)


def gtja_012(d: _Data) -> Any:
    return k.rank(d.o - k.mean(d.vwap, 10)) * _neg(k.rank(k.absolute(d.c - d.vwap)))


def gtja_013(d: _Data) -> Any:
    return k.sqrt(d.h * d.l) - d.vwap


def gtja_014(d: _Data) -> Any:
    return d.c - k.delay(d.c, 5)


def gtja_015(d: _Data) -> Any:
    return k.safe_div(d.o, d.prev_c) - 1.0


def gtja_016(d: _Data) -> Any:
    return _neg(k.ts_max(k.rank(k.correlation(k.rank(d.v), k.rank(d.vwap), 5)), 5))


def gtja_017(d: _Data) -> Any:
    base = k.rank(d.vwap - k.ts_max(d.vwap, 15))
    return k.power(base, k.delta(d.c, 5))


def gtja_018(d: _Data) -> Any:
    return k.safe_div(d.c, k.delay(d.c, 5))


def gtja_019(d: _Data) -> Any:
    previous = k.delay(d.c, 5)
    lower = k.safe_div(d.c - previous, previous)
    higher = k.safe_div(d.c - previous, d.c)
    value = k.select(k.eq(d.c, previous), 0.0, k.select(k.lt(d.c, previous), lower, higher))
    return k.mask(value, k.valid(d.c, previous))


def gtja_020(d: _Data) -> Any:
    previous = k.delay(d.c, 6)
    return k.safe_div(d.c - previous, previous) * 100.0


def gtja_021(d: _Data) -> Any:
    return sequence_regbeta(d.mean("c", 6), 6)


def gtja_022(d: _Data) -> Any:
    ratio = k.safe_div(d.c - d.mean("c", 6), d.mean("c", 6))
    return k.sma_cn(ratio - k.delay(ratio, 3), 12, 1)


def gtja_023(d: _Data) -> Any:
    up = _volatility_sma(d, rising=True)
    down = _volatility_sma(d, rising=False)
    return k.safe_div(up, up + down) * 100.0


def gtja_024(d: _Data) -> Any:
    return k.sma_cn(d.c - k.delay(d.c, 5), 5, 1)


def gtja_025(d: _Data) -> Any:
    liquidity = k.safe_div(d.v, d.mean("v", 20))
    first = _neg(k.rank(k.delta(d.c, 7) * (1.0 - k.rank(k.decay_linear(liquidity, 9)))))
    return first * (1.0 + k.rank(k.ts_sum(d.ret, 250)))


def gtja_026(d: _Data) -> Any:
    return d.mean("c", 7) - d.c + k.correlation(d.vwap, k.delay(d.c, 5), 230)


def gtja_027(d: _Data) -> Any:
    roc3 = k.safe_div(d.c - k.delay(d.c, 3), k.delay(d.c, 3))
    roc6 = k.safe_div(d.c - k.delay(d.c, 6), k.delay(d.c, 6))
    return wma((roc3 + roc6) * 100.0, 12)


def gtja_028(d: _Data) -> Any:
    first = k.sma_cn(d.stochastic(9), 3, 1)
    return first * 3.0 - k.sma_cn(first, 3, 1) * 2.0


def gtja_029(d: _Data) -> Any:
    previous = k.delay(d.c, 6)
    return k.safe_div(d.c - previous, previous) * d.v


def gtja_031(d: _Data) -> Any:
    average = d.mean("c", 12)
    return k.safe_div(d.c - average, average) * 100.0


def gtja_032(d: _Data) -> Any:
    return _neg(k.ts_sum(k.rank(k.correlation(k.rank(d.h), k.rank(d.v), 3)), 3))


def gtja_033(d: _Data) -> Any:
    low5 = k.ts_min(d.l, 5)
    return (
        (k.delay(low5, 5) - low5)
        * k.rank((k.ts_sum(d.ret, 240) - k.ts_sum(d.ret, 20)) / 220.0)
        * k.ts_rank(d.v, 5)
    )


def gtja_034(d: _Data) -> Any:
    return k.safe_div(d.mean("c", 12), d.c)


def gtja_035(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.o, 1), 15))
    right = k.rank(k.decay_linear(k.correlation(d.v, d.o, 17), 7))
    return _neg(k.emin(left, right))


def gtja_036(d: _Data) -> Any:
    return k.rank(k.ts_sum(k.correlation(k.rank(d.v), k.rank(d.vwap), 6), 2))


def gtja_037(d: _Data) -> Any:
    value = k.ts_sum(d.o, 5) * k.ts_sum(d.ret, 5)
    return _neg(k.rank(value - k.delay(value, 10)))


def gtja_038(d: _Data) -> Any:
    average = d.mean("h", 20)
    change = _neg(k.delta(d.h, 2))
    return k.mask(k.select(k.lt(average, d.h), change, 0.0), k.valid(average, change, d.h))


def gtja_039(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.c, 2), 8))
    mixed = d.vwap * 0.3 + d.o * 0.7
    right = k.rank(k.decay_linear(k.correlation(mixed, k.ts_sum(d.mean("v", 180), 37), 14), 12))
    return _neg(left - right)


def gtja_040(d: _Data) -> Any:
    usable = k.valid(d.c, d.prev_c, d.v)
    up, down = _up_down(d, d.v, k.gt(d.c, d.prev_c), k.le(d.c, d.prev_c), usable, 26)
    return k.safe_div(up, down) * 100.0


def gtja_041(d: _Data) -> Any:
    return _neg(k.rank(k.ts_max(k.delta(d.vwap, 3), 5)))


def gtja_042(d: _Data) -> Any:
    return _neg(k.rank(k.stddev(d.h, 10))) * k.correlation(d.h, d.v, 10)


def gtja_043(d: _Data) -> Any:
    return k.ts_sum(d.signed_volume(), 6)


def gtja_044(d: _Data) -> Any:
    first = k.ts_rank(k.decay_linear(k.correlation(d.l, d.mean("v", 10), 7), 6), 4)
    second = k.ts_rank(k.decay_linear(k.delta(d.vwap, 3), 10), 15)
    return first + second


def gtja_045(d: _Data) -> Any:
    return k.rank(k.delta(d.c * 0.6 + d.o * 0.4, 1)) * k.rank(k.correlation(d.vwap, d.mean("v", 150), 15))


def gtja_046(d: _Data) -> Any:
    return (d.mean("c", 3) + d.mean("c", 6) + d.mean("c", 12) + d.mean("c", 24)) / (d.c * 4.0)


def gtja_047(d: _Data) -> Any:
    return k.sma_cn(d.williams(6), 9, 1)


def gtja_048(d: _Data) -> Any:
    signs = (
        k.sign(d.c - k.delay(d.c, 1))
        + k.sign(k.delay(d.c, 1) - k.delay(d.c, 2))
        + k.sign(k.delay(d.c, 2) - k.delay(d.c, 3))
    )
    return _neg(k.rank(signs)) * k.safe_div(k.ts_sum(d.v, 5), k.ts_sum(d.v, 20))


def gtja_049(d: _Data) -> Any:
    up, down = d.directional_range_parts()
    down_sum = k.ts_sum(down, 12)
    up_sum = k.ts_sum(up, 12)
    return k.safe_div(down_sum, down_sum + up_sum)


def gtja_050(d: _Data) -> Any:
    up, down = d.directional_range_parts()
    up_sum = k.ts_sum(up, 12)
    down_sum = k.ts_sum(down, 12)
    total = up_sum + down_sum
    return k.safe_div(up_sum, total) - k.safe_div(down_sum, total)


def gtja_051(d: _Data) -> Any:
    up, down = d.directional_range_parts()
    up_sum = k.ts_sum(up, 12)
    return k.safe_div(up_sum, up_sum + k.ts_sum(down, 12))


def gtja_052(d: _Data) -> Any:
    typical = (d.h + d.l + d.c) / 3.0
    previous = k.delay(typical, 1)
    numerator = k.ts_sum(k.emax(d.h - previous, d.h * 0.0), 26)
    denominator = k.ts_sum(k.emax(previous - d.l, d.l * 0.0), 26)
    return k.safe_div(numerator, denominator) * 100.0


def gtja_053(d: _Data) -> Any:
    return k.count(k.gt(d.c, d.prev_c), 12, valid_rows=k.valid(d.c, d.prev_c)) / 12.0 * 100.0


def gtja_054(d: _Data) -> Any:
    value = k.stddev(k.absolute(d.c - d.o), 10) + (d.c - d.o)
    return _neg(k.rank(value + k.correlation(d.c, d.o, 10)))


def gtja_055(d: _Data) -> Any:
    high_gap, low_gap, cross_gap, open_gap, numerator = d.gap_parts()
    first = high_gap + low_gap / 2.0 + open_gap / 4.0
    second = low_gap + high_gap / 2.0 + open_gap / 4.0
    third = cross_gap + open_gap / 4.0
    denominator = k.select(k.all_of(k.gt(low_gap, cross_gap), k.gt(low_gap, high_gap)), second, third)
    denominator = k.select(k.all_of(k.gt(high_gap, low_gap), k.gt(high_gap, cross_gap)), first, denominator)
    value = k.safe_div(numerator, denominator) * k.emax(high_gap, low_gap)
    return k.ts_sum(value, 20)


def gtja_056(d: _Data) -> Any:
    left = k.rank(d.o - k.ts_min(d.o, 12))
    corr = k.correlation(k.ts_sum((d.h + d.l) / 2.0, 19), k.ts_sum(d.mean("v", 40), 19), 13)
    right = k.rank(k.power(k.rank(corr), 5.0))
    return _conditional(k.lt(left, right), 1.0, 0.0, left, right)


def gtja_057(d: _Data) -> Any:
    return k.sma_cn(d.stochastic(9), 3, 1)


def gtja_058(d: _Data) -> Any:
    return k.count(k.gt(d.c, d.prev_c), 20, valid_rows=k.valid(d.c, d.prev_c)) / 20.0 * 100.0


def gtja_059(d: _Data) -> Any:
    return _close_location_sum(d, 20)


def gtja_060(d: _Data) -> Any:
    return k.ts_sum(k.safe_div((d.c - d.l) - (d.h - d.c), d.h - d.l) * d.v, 20)


def gtja_061(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.vwap, 1), 12))
    right = k.rank(k.decay_linear(k.rank(k.correlation(d.l, d.mean("v", 80), 8)), 17))
    return _neg(k.emax(left, right))


def gtja_062(d: _Data) -> Any:
    return _neg(k.correlation(d.h, k.rank(d.v), 5))


def gtja_063(d: _Data) -> Any:
    return d.rsi(d.change(), 6)


def gtja_064(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.correlation(k.rank(d.vwap), k.rank(d.v), 4), 4))
    right = k.rank(
        k.decay_linear(k.ts_max(k.correlation(k.rank(d.c), k.rank(d.mean("v", 60)), 4), 13), 14)
    )
    return _neg(k.emax(left, right))


def gtja_065(d: _Data) -> Any:
    return k.safe_div(d.mean("c", 6), d.c)


def gtja_066(d: _Data) -> Any:
    average = d.mean("c", 6)
    return k.safe_div(d.c - average, average) * 100.0


def gtja_067(d: _Data) -> Any:
    return d.rsi(d.change(), 24)


def gtja_068(d: _Data) -> Any:
    midpoint = (d.h + d.l) / 2.0
    value = k.safe_div((midpoint - k.delay(midpoint, 1)) * (d.h - d.l), d.v)
    return k.sma_cn(value, 15, 2)


def gtja_069(d: _Data) -> Any:
    dtm = k.ts_sum(d.dtm(), 20)
    dbm = k.ts_sum(d.dbm(), 20)
    difference = dtm - dbm
    result = k.select(k.le(dtm, dbm), k.safe_div(difference, dbm), k.safe_div(difference, dtm))
    result = k.select(k.eq(dtm, dbm), 0.0, result)
    return k.mask(result, k.valid(dtm, dbm))


def gtja_070(d: _Data) -> Any:
    return k.stddev(d.amt, 6)


def gtja_071(d: _Data) -> Any:
    average = d.mean("c", 24)
    return k.safe_div(d.c - average, average) * 100.0


def gtja_072(d: _Data) -> Any:
    return k.sma_cn(d.williams(6), 15, 1)


def gtja_073(d: _Data) -> Any:
    first = k.ts_rank(k.decay_linear(k.decay_linear(k.correlation(d.c, d.v, 10), 16), 4), 5)
    second = k.rank(k.decay_linear(k.correlation(d.vwap, d.mean("v", 30), 4), 3))
    return _neg(first - second)


def gtja_074(d: _Data) -> Any:
    first = k.rank(k.correlation(k.ts_sum(d.l * 0.35 + d.vwap * 0.65, 20), k.ts_sum(d.mean("v", 40), 20), 7))
    second = k.rank(k.correlation(k.rank(d.vwap), k.rank(d.v), 6))
    return first + second


def gtja_075(d: _Data) -> Any:
    benchmark_down = k.lt(d.bc, d.bo)
    usable = k.valid(d.c, d.o)
    numerator = k.count(k.all_of(k.gt(d.c, d.o), benchmark_down), 50, valid_rows=usable)
    denominator = k.count(benchmark_down, 50, valid_rows=usable)
    return k.safe_div(numerator, denominator)


def gtja_076(d: _Data) -> Any:
    value = k.safe_div(k.absolute(d.ret), d.v)
    return k.safe_div(k.stddev(value, 20), k.mean(value, 20))


def gtja_077(d: _Data) -> Any:
    midpoint = (d.h + d.l) / 2.0
    left = k.rank(k.decay_linear(midpoint - d.vwap, 20))
    right = k.rank(k.decay_linear(k.correlation(midpoint, d.mean("v", 40), 3), 6))
    return k.emin(left, right)


def gtja_078(d: _Data) -> Any:
    typical = (d.h + d.l + d.c) / 3.0
    typical_mean = k.mean(typical, 12)
    denominator = k.mean(k.absolute(d.c - typical_mean), 12) * 0.015
    return k.safe_div(typical - typical_mean, denominator)


def gtja_079(d: _Data) -> Any:
    return d.rsi(d.change(), 12)


def gtja_080(d: _Data) -> Any:
    previous = k.delay(d.v, 5)
    return k.safe_div(d.v - previous, previous) * 100.0


def gtja_081(d: _Data) -> Any:
    return k.sma_cn(d.v, 21, 2)


def gtja_082(d: _Data) -> Any:
    return k.sma_cn(d.williams(6), 20, 1)


def gtja_083(d: _Data) -> Any:
    return _neg(k.rank(k.covariance(k.rank(d.h), k.rank(d.v), 5)))


def gtja_084(d: _Data) -> Any:
    return k.ts_sum(d.signed_volume(), 20)


def gtja_085(d: _Data) -> Any:
    return k.ts_rank(k.safe_div(d.v, d.mean("v", 20)), 20) * k.ts_rank(_neg(k.delta(d.c, 7)), 8)


def gtja_086(d: _Data) -> Any:
    slope = (k.delay(d.c, 20) - k.delay(d.c, 10)) / 10.0 - (k.delay(d.c, 10) - d.c) / 10.0
    middle = _neg(d.c - k.delay(d.c, 1))
    result = k.select(k.ge(slope, 0.0), middle, 1.0)
    result = k.select(k.le(slope, 0.25), result, -1.0)
    return k.mask(result, k.notna(slope))


def gtja_087(d: _Data) -> Any:
    first = k.rank(k.decay_linear(k.delta(d.vwap, 4), 7))
    second_input = k.safe_div(d.l - d.vwap, d.o - (d.h + d.l) / 2.0)
    return _neg(first + k.ts_rank(k.decay_linear(second_input, 11), 7))


def gtja_088(d: _Data) -> Any:
    previous = k.delay(d.c, 20)
    return k.safe_div(d.c - previous, previous) * 100.0


def gtja_089(d: _Data) -> Any:
    difference = k.sma_cn(d.c, 13, 2) - k.sma_cn(d.c, 27, 2)
    return (difference - k.sma_cn(difference, 10, 2)) * 2.0


def gtja_090(d: _Data) -> Any:
    return _neg(k.rank(k.correlation(k.rank(d.vwap), k.rank(d.v), 5)))


def gtja_091(d: _Data) -> Any:
    return _neg(k.rank(d.c - k.ts_max(d.c, 5))) * k.rank(k.correlation(d.mean("v", 40), d.l, 5))


def gtja_092(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.c * 0.35 + d.vwap * 0.65, 2), 3))
    right = k.ts_rank(k.decay_linear(k.absolute(k.correlation(d.mean("v", 180), d.c, 13)), 5), 15)
    return _neg(k.emax(left, right))


def gtja_093(d: _Data) -> Any:
    return k.ts_sum(d.dbm(), 20)


def gtja_094(d: _Data) -> Any:
    return k.ts_sum(d.signed_volume(), 30)


def gtja_095(d: _Data) -> Any:
    return k.stddev(d.amt, 20)


def gtja_096(d: _Data) -> Any:
    return k.sma_cn(k.sma_cn(d.stochastic(9), 3, 1), 3, 1)


def gtja_097(d: _Data) -> Any:
    return k.stddev(d.v, 10)


def gtja_098(d: _Data) -> Any:
    average = d.mean("c", 100)
    slope = k.safe_div(k.delta(average, 100), k.delay(d.c, 100))
    low_branch = _neg(d.c - k.ts_min(d.c, 100))
    high_branch = _neg(k.delta(d.c, 3))
    value = k.select(k.le(slope, 0.05), low_branch, high_branch)
    return k.mask(value, k.valid(slope, low_branch, high_branch))


def gtja_099(d: _Data) -> Any:
    return _neg(k.rank(k.covariance(k.rank(d.c), k.rank(d.v), 5)))


def gtja_100(d: _Data) -> Any:
    return k.stddev(d.v, 20)


def gtja_101(d: _Data) -> Any:
    left = k.rank(k.correlation(d.c, k.ts_sum(d.mean("v", 30), 37), 15))
    right = k.rank(k.correlation(k.rank(d.h * 0.1 + d.vwap * 0.9), k.rank(d.v), 11))
    return _conditional(k.lt(left, right), -1.0, 0.0, left, right)


def gtja_102(d: _Data) -> Any:
    return d.rsi(d.v - k.delay(d.v, 1), 6)


def gtja_103(d: _Data) -> Any:
    return (20.0 - lowday(d.l, 20)) / 20.0 * 100.0


def gtja_104(d: _Data) -> Any:
    return _neg(k.delta(k.correlation(d.h, d.v, 5), 5)) * k.rank(k.stddev(d.c, 20))


def gtja_105(d: _Data) -> Any:
    return _neg(k.correlation(k.rank(d.o), k.rank(d.v), 10))


def gtja_106(d: _Data) -> Any:
    return d.c - k.delay(d.c, 20)


def gtja_107(d: _Data) -> Any:
    return (
        _neg(k.rank(d.o - k.delay(d.h, 1)))
        * k.rank(d.o - k.delay(d.c, 1))
        * k.rank(d.o - k.delay(d.l, 1))
    )


def gtja_108(d: _Data) -> Any:
    base = k.rank(d.h - k.ts_min(d.h, 2))
    exponent = k.rank(k.correlation(d.vwap, d.mean("v", 120), 6))
    return _neg(k.power(base, exponent))


def gtja_109(d: _Data) -> Any:
    first = k.sma_cn(d.h - d.l, 10, 2)
    return k.safe_div(first, k.sma_cn(first, 10, 2))


def gtja_110(d: _Data) -> Any:
    numerator = k.ts_sum(k.clip(d.h - d.prev_c, lower=0.0), 20)
    denominator = k.ts_sum(k.clip(d.prev_c - d.l, lower=0.0), 20)
    return k.safe_div(numerator, denominator) * 100.0


def gtja_111(d: _Data) -> Any:
    raw = k.safe_div(((d.c - d.l) - (d.h - d.c)) * d.v, d.h - d.l)
    return k.sma_cn(raw, 11, 2) - k.sma_cn(raw, 4, 2)


def gtja_112(d: _Data) -> Any:
    change = d.change()
    positive = k.ts_sum(k.clip(change, lower=0.0), 12)
    negative = k.ts_sum(k.clip(_neg(change), lower=0.0), 12)
    return k.safe_div(positive - negative, positive + negative) * 100.0


def gtja_113(d: _Data) -> Any:
    return (
        _neg(k.rank(k.mean(k.delay(d.c, 5), 20)))
        * k.correlation(d.c, d.v, 2)
        * k.rank(k.correlation(k.ts_sum(d.c, 5), k.ts_sum(d.c, 20), 2))
    )


def gtja_114(d: _Data) -> Any:
    ratio = k.safe_div(d.h - d.l, d.mean("c", 5))
    return k.safe_div(
        k.rank(k.delay(ratio, 2)) * k.rank(k.rank(d.v)), k.safe_div(ratio, d.vwap - d.c)
    )


def gtja_115(d: _Data) -> Any:
    base = k.rank(k.correlation(d.h * 0.9 + d.c * 0.1, d.mean("v", 30), 10))
    exponent = k.rank(k.correlation(k.ts_rank((d.h + d.l) / 2.0, 4), k.ts_rank(d.v, 10), 7))
    return k.power(base, exponent)


def gtja_116(d: _Data) -> Any:
    return sequence_regbeta(d.c, 20)


def gtja_117(d: _Data) -> Any:
    return (
        k.ts_rank(d.v, 32)
        * (1.0 - k.ts_rank(d.c + d.h - d.l, 16))
        * (1.0 - k.ts_rank(d.ret, 32))
    )


def gtja_118(d: _Data) -> Any:
    return k.safe_div(k.ts_sum(d.h - d.o, 20), k.ts_sum(d.o - d.l, 20)) * 100.0


def gtja_119(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.correlation(d.vwap, k.ts_sum(d.mean("v", 5), 26), 5), 7))
    inner = k.ts_min(k.correlation(k.rank(d.o), k.rank(d.mean("v", 15)), 21), 9)
    right = k.rank(k.decay_linear(k.ts_rank(inner, 7), 8))
    return left - right


def gtja_120(d: _Data) -> Any:
    return k.safe_div(k.rank(d.vwap - d.c), k.rank(d.vwap + d.c))


def gtja_121(d: _Data) -> Any:
    base = k.rank(d.vwap - k.ts_min(d.vwap, 12))
    exponent = k.ts_rank(k.correlation(k.ts_rank(d.vwap, 20), k.ts_rank(d.mean("v", 60), 2), 18), 3)
    return _neg(k.power(base, exponent))


def gtja_122(d: _Data) -> Any:
    smoothed = k.sma_cn(k.sma_cn(k.sma_cn(k.log(d.c), 13, 2), 13, 2), 13, 2)
    return k.safe_div(smoothed - k.delay(smoothed, 1), k.delay(smoothed, 1))


def gtja_123(d: _Data) -> Any:
    left = k.rank(k.correlation(k.ts_sum((d.h + d.l) / 2.0, 20), k.ts_sum(d.mean("v", 60), 20), 9))
    right = k.rank(k.correlation(d.l, d.v, 6))
    return _conditional(k.lt(left, right), -1.0, 0.0, left, right)


def gtja_124(d: _Data) -> Any:
    return k.safe_div(d.c - d.vwap, k.decay_linear(k.rank(k.ts_max(d.c, 30)), 2))


def gtja_125(d: _Data) -> Any:
    numerator = k.rank(k.decay_linear(k.correlation(d.vwap, d.mean("v", 80), 17), 20))
    denominator = k.rank(k.decay_linear(k.delta(d.c * 0.5 + d.vwap * 0.5, 3), 16))
    return k.safe_div(numerator, denominator)


def gtja_126(d: _Data) -> Any:
    return (d.c + d.h + d.l) / 3.0


def gtja_127(d: _Data) -> Any:
    maximum = k.ts_max(d.c, 12)
    ratio = k.safe_div((d.c - maximum) * 100.0, maximum)
    return k.sqrt(k.mean(ratio * ratio, 12))


def gtja_128(d: _Data) -> Any:
    typical = (d.h + d.l + d.c) / 3.0
    previous = k.delay(typical, 1)
    traded = typical * d.v
    usable = k.valid(typical, previous, d.v)
    up, down = _up_down(d, traded, k.gt(typical, previous), k.lt(typical, previous), usable, 14)
    return 100.0 - k.safe_div(k.const(100.0), 1.0 + k.safe_div(up, down))


def gtja_129(d: _Data) -> Any:
    change = d.change()
    return k.ts_sum(k.mask(k.select(k.lt(change, 0.0), _neg(change), 0.0), k.notna(change)), 12)


def gtja_130(d: _Data) -> Any:
    numerator = k.rank(k.decay_linear(k.correlation((d.h + d.l) / 2.0, d.mean("v", 40), 9), 10))
    denominator = k.rank(k.decay_linear(k.correlation(k.rank(d.vwap), k.rank(d.v), 7), 3))
    return k.safe_div(numerator, denominator)


def gtja_131(d: _Data) -> Any:
    base = k.rank(k.delta(d.vwap, 1))
    exponent = k.ts_rank(k.correlation(d.c, d.mean("v", 50), 18), 18)
    return k.power(base, exponent)


def gtja_132(d: _Data) -> Any:
    return d.mean("amt", 20)


def gtja_133(d: _Data) -> Any:
    return (20.0 - highday(d.h, 20)) / 20.0 * 100.0 - (20.0 - lowday(d.l, 20)) / 20.0 * 100.0


def gtja_134(d: _Data) -> Any:
    previous = k.delay(d.c, 12)
    return k.safe_div(d.c - previous, previous) * d.v


def gtja_135(d: _Data) -> Any:
    return k.sma_cn(k.delay(k.safe_div(d.c, k.delay(d.c, 20)), 1), 20, 1)


def gtja_136(d: _Data) -> Any:
    return _neg(k.rank(k.delta(d.ret, 3))) * k.correlation(d.o, d.v, 10)


def gtja_137(d: _Data) -> Any:
    a, b, c, gap, numerator = d.gap_parts()
    denominator = k.select(k.all_of(k.gt(b, c), k.gt(b, a)), b + a / 2.0 + gap / 4.0, c + gap / 4.0)
    denominator = k.select(k.all_of(k.gt(a, b), k.gt(a, c)), a + b / 2.0 + gap / 4.0, denominator)
    return k.safe_div(numerator, denominator) * k.emax(a, b)


def gtja_138(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.l * 0.7 + d.vwap * 0.3, 3), 20))
    inner = k.correlation(k.ts_rank(d.l, 8), k.ts_rank(d.mean("v", 60), 17), 5)
    right = k.ts_rank(k.decay_linear(k.ts_rank(inner, 19), 16), 7)
    return _neg(left - right)


def gtja_139(d: _Data) -> Any:
    return _neg(k.correlation(d.o, d.v, 10))


def gtja_140(d: _Data) -> Any:
    left = k.rank(k.decay_linear((k.rank(d.o) + k.rank(d.l)) - (k.rank(d.h) + k.rank(d.c)), 8))
    right = k.ts_rank(
        k.decay_linear(k.correlation(k.ts_rank(d.c, 8), k.ts_rank(d.mean("v", 60), 20), 8), 7), 3
    )
    return k.emin(left, right)


def gtja_141(d: _Data) -> Any:
    return _neg(k.rank(k.correlation(k.rank(d.h), k.rank(d.mean("v", 15)), 9)))


def gtja_142(d: _Data) -> Any:
    return (
        _neg(k.rank(k.ts_rank(d.c, 10)))
        * k.rank(k.delta(k.delta(d.c, 1), 1))
        * k.rank(k.ts_rank(k.safe_div(d.v, d.mean("v", 20)), 5))
    )


def gtja_143(d: _Data) -> Any:
    returns = k.safe_div(d.c - d.prev_c, d.prev_c)
    return k.recursive(
        returns, lambda value, previous: k.select(k.gt(value, 0.0), previous * (value + 1.0), previous), initial=1.0
    )


def gtja_144(d: _Data) -> Any:
    value = k.safe_div(k.absolute(d.ret), d.amt)
    condition = k.lt(d.c, d.prev_c)
    usable = k.valid(value, d.c, d.prev_c)
    return k.safe_div(
        k.sumif(value, 20, condition, valid_rows=usable), k.count(condition, 20, valid_rows=usable)
    )


def gtja_145(d: _Data) -> Any:
    return k.safe_div(d.mean("v", 9) - d.mean("v", 26), d.mean("v", 12)) * 100.0


def gtja_146(d: _Data) -> Any:
    smoothed = k.sma_cn(d.ret, 61, 2)
    deviation = d.ret - smoothed
    return k.safe_div(k.mean(deviation, 20) * deviation, k.sma_cn(deviation * deviation, 60, 2))


def gtja_147(d: _Data) -> Any:
    return sequence_regbeta(d.mean("c", 12), 12)


def gtja_148(d: _Data) -> Any:
    left = k.rank(k.correlation(d.o, k.ts_sum(d.mean("v", 60), 9), 6))
    right = k.rank(d.o - k.ts_min(d.o, 14))
    return _conditional(k.lt(left, right), -1.0, 0.0, left, right)


def gtja_149(d: _Data) -> Any:
    previous = k.delay(d.bc, 1)
    benchmark_return = k.safe_div(d.bc - previous, previous)
    condition = k.all_of(k.lt(benchmark_return, 0.0), k.valid(d.ret, benchmark_return))
    return filtered_regbeta(d.ret, benchmark_return, condition, FILTERED_REGRESSION_DAYS)


def gtja_150(d: _Data) -> Any:
    return (d.c + d.h + d.l) / 3.0 * d.v


def gtja_151(d: _Data) -> Any:
    return k.sma_cn(d.c - k.delay(d.c, 20), 20, 1)


def gtja_152(d: _Data) -> Any:
    inner = k.delay(k.sma_cn(k.delay(k.safe_div(d.c, k.delay(d.c, 9)), 1), 9, 1), 1)
    return k.sma_cn(k.mean(inner, 12) - k.mean(inner, 26), 9, 1)


def gtja_153(d: _Data) -> Any:
    return (d.mean("c", 3) + d.mean("c", 6) + d.mean("c", 12) + d.mean("c", 24)) / 4.0


def gtja_154(d: _Data) -> Any:
    left = d.vwap - k.ts_min(d.vwap, 16)
    right = k.correlation(d.vwap, d.mean("v", 180), 18)
    return _conditional(k.lt(left, right), 1.0, 0.0, left, right)


def gtja_155(d: _Data) -> Any:
    difference = k.sma_cn(d.v, 13, 2) - k.sma_cn(d.v, 27, 2)
    return difference - k.sma_cn(difference, 10, 2)


def gtja_156(d: _Data) -> Any:
    left = k.rank(k.decay_linear(k.delta(d.vwap, 5), 3))
    mixed = d.o * 0.15 + d.l * 0.85
    right = k.rank(k.decay_linear(_neg(k.safe_div(k.delta(mixed, 2), mixed)), 3))
    return _neg(k.emax(left, right))


def gtja_157(d: _Data) -> Any:
    inner = _neg(k.rank(k.delta(d.c - 1.0, 5)))
    first = k.ts_min(
        k.product(k.rank(k.rank(k.scale(k.log(k.ts_sum(k.ts_min(k.rank(k.rank(inner)), 2), 1))))), 1), 5
    )
    return first + k.ts_rank(k.delay(_neg(d.ret), 6), 5)


def gtja_158(d: _Data) -> Any:
    smooth = k.sma_cn(d.c, 15, 2)
    return k.safe_div((d.h - smooth) - (d.l - smooth), d.c)


def gtja_159(d: _Data) -> Any:
    floor = k.emin(d.l, d.prev_c)
    spread = k.emax(d.h, d.prev_c) - floor

    def term(window: int, weight: int) -> Any:
        return k.safe_div(d.c - k.ts_sum(floor, window), k.ts_sum(spread, window)) * float(weight)

    return (term(6, 12 * 24) + term(12, 6 * 24) + term(24, 6 * 24)) * 100.0 / float(6 * 12 + 6 * 24 + 12 * 24)


def gtja_160(d: _Data) -> Any:
    return _volatility_sma(d, rising=False)


def gtja_161(d: _Data) -> Any:
    return k.mean(d.true_range(), 12)


def gtja_162(d: _Data) -> Any:
    rsi = d.rsi(d.change(), 12)
    lowest = k.ts_min(rsi, 12)
    return k.safe_div(rsi - lowest, k.ts_max(rsi, 12) - lowest)


def gtja_163(d: _Data) -> Any:
    return k.rank(_neg(d.ret) * d.mean("v", 20) * d.vwap * (d.h - d.c))


def gtja_164(d: _Data) -> Any:
    change = d.change()
    reciprocal = k.mask(k.select(k.gt(change, 0.0), k.safe_div(1.0, change), 1.0), k.notna(change))
    value = k.safe_div((reciprocal - k.ts_min(reciprocal, 12)) * 100.0, d.h - d.l)
    return k.sma_cn(value, 13, 2)


def gtja_165(d: _Data) -> Any:
    return d.accumulated_deviation(48)


def gtja_166(d: _Data) -> Any:
    centered = d.ret - k.mean(d.ret, 20)
    numerator = k.ts_sum(k.power(centered, 3.0), 20) * (-20 * (19**1.5))
    denominator = k.power(k.ts_sum(centered * centered, 20), 1.5) * float(19 * 18)
    return k.safe_div(numerator, denominator)


def gtja_167(d: _Data) -> Any:
    return k.ts_sum(k.clip(d.change(), lower=0.0), 12)


def gtja_168(d: _Data) -> Any:
    return _neg(k.safe_div(d.v, d.mean("v", 20)))


def gtja_169(d: _Data) -> Any:
    inner = k.delay(k.sma_cn(d.c - k.delay(d.c, 1), 9, 1), 1)
    return k.sma_cn(k.mean(inner, 12) - k.mean(inner, 26), 10, 1)


def gtja_170(d: _Data) -> Any:
    first = k.safe_div(k.rank(k.safe_div(1.0, d.c)) * d.v, d.mean("v", 20))
    second = k.safe_div(d.h * k.rank(d.h - d.c), d.mean("h", 5))
    return first * second - k.rank(d.vwap - k.delay(d.vwap, 5))


def gtja_171(d: _Data) -> Any:
    numerator = _neg(d.l - d.c) * k.power(d.o, 5.0)
    denominator = (d.c - d.h) * k.power(d.c, 5.0)
    return k.safe_div(numerator, denominator)


def gtja_172(d: _Data) -> Any:
    return d.adx()


def gtja_173(d: _Data) -> Any:
    first = k.sma_cn(d.c, 13, 2)
    second = k.sma_cn(first, 13, 2)
    log_first = k.sma_cn(k.log(d.c), 13, 2)
    return first * 3.0 - second * 2.0 + k.sma_cn(k.sma_cn(log_first, 13, 2), 13, 2)


def gtja_174(d: _Data) -> Any:
    return _volatility_sma(d, rising=True)


def gtja_175(d: _Data) -> Any:
    return k.mean(d.true_range(), 6)


def gtja_176(d: _Data) -> Any:
    return k.correlation(k.rank(d.stochastic(12, percent=False)), k.rank(d.v), 6)


def gtja_177(d: _Data) -> Any:
    return (20.0 - highday(d.h, 20)) / 20.0 * 100.0


def gtja_178(d: _Data) -> Any:
    return d.ret * d.v


def gtja_179(d: _Data) -> Any:
    return k.rank(k.correlation(d.vwap, d.v, 4)) * k.rank(
        k.correlation(k.rank(d.l), k.rank(d.mean("v", 50)), 12)
    )


def gtja_180(d: _Data) -> Any:
    change = k.delta(d.c, 7)
    active = _neg(k.ts_rank(k.absolute(change), 60)) * k.sign(change)
    average_volume = d.mean("v", 20)
    value = k.select(k.gt(d.v, average_volume), active, _neg(d.v))
    return k.mask(value, k.valid(active, d.v, average_volume))


def gtja_181(d: _Data) -> Any:
    centered_return = d.ret - k.mean(d.ret, 20)
    centered_benchmark = d.bc - k.mean(d.bc, 20)
    numerator = k.ts_sum(centered_return - centered_benchmark * centered_benchmark, 20)
    denominator = k.ts_sum(k.power(centered_benchmark, 3.0), 20)
    return k.safe_div(numerator, denominator)


def gtja_182(d: _Data) -> Any:
    same = k.any_of(
        k.all_of(k.gt(d.c, d.o), k.gt(d.bc, d.bo)),
        k.all_of(k.lt(d.c, d.o), k.lt(d.bc, d.bo)),
    )
    return k.count(same, 20, valid_rows=k.valid(d.c, d.o)) / 20.0


def gtja_183(d: _Data) -> Any:
    return d.accumulated_deviation(24)


def gtja_184(d: _Data) -> Any:
    return k.rank(k.correlation(k.delay(d.o - d.c, 1), d.c, 200)) + k.rank(d.o - d.c)


def gtja_185(d: _Data) -> Any:
    return k.rank(_neg(k.power(1.0 - k.safe_div(d.o, d.c), 2.0)))


def gtja_186(d: _Data) -> Any:
    adx = d.adx()
    return (adx + k.delay(adx, 6)) / 2.0


def gtja_187(d: _Data) -> Any:
    return k.ts_sum(d.dtm(), 20)


def gtja_188(d: _Data) -> Any:
    price_range = d.h - d.l
    smooth = k.sma_cn(price_range, 11, 2)
    return k.safe_div(price_range - smooth, smooth) * 100.0


def gtja_189(d: _Data) -> Any:
    return k.mean(k.absolute(d.c - d.mean("c", 6)), 6)


def gtja_190(d: _Data) -> Any:
    daily = k.safe_div(d.c, d.prev_c) - 1.0
    threshold = k.power(k.safe_div(d.c, k.delay(d.c, 19)), 1.0 / 20.0) - 1.0
    squared = (daily - threshold) * (daily - threshold)
    above = k.gt(daily, threshold)
    below = k.lt(daily, threshold)
    usable = k.valid(daily, threshold, squared)
    numerator = (k.count(above, 20, valid_rows=usable) - 1.0) * k.sumif(squared, 20, below, valid_rows=usable)
    denominator = k.count(below, 20, valid_rows=usable) * k.sumif(squared, 20, above, valid_rows=usable)
    return k.log(k.safe_div(numerator, denominator))


def gtja_191(d: _Data) -> Any:
    return k.correlation(d.mean("v", 20), d.l, 5) + (d.h + d.l) / 2.0 - d.c


GTJA191_FORMULAS: dict[str, Callable[[_Data], Any]] = {name: globals()[name] for name in GTJA191_NAMES}


# ---------------------------------------------------------------------------------------------------- library
class GTJA191Library(FactorLibrary):
    """The GTJA191 catalog as one KunQuant graph."""

    family = "gtja191"
    required_inputs = ("open", "high", "low", "close", "volume", "amount", "vwap")
    reference_inputs = ("benchmark_open", "benchmark_close")
    strict_float = True

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=name,
                source_name=f"alpha{ordinal:03d}",
                family=self.family,
                ordinal=ordinal,
                formula=f"Guotai Junan Alpha191 #{ordinal:03d}",
                max_lookback=252,
                implementation=f"rquant.factors.libraries.gtja191.{name}",
                catalog_version=3,
            )
            for name in GTJA191_NAMES
            for ordinal in (int(name.rsplit("_", 1)[1]),)
        )

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        data = _Data(inputs)
        return {spec.source_name: k.replace_inf(GTJA191_FORMULAS[spec.canonical_name](data)) for spec in self.specs}

    def validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        """Refuse incomplete amount, VWAP or benchmark data, which would silently blank or bias some factors."""
        observed = np.isfinite(np.asarray(inputs["close"], dtype=float))
        for field in ("amount", "vwap"):
            missing = observed & ~np.isfinite(np.asarray(inputs[field], dtype=float))
            if missing.any():
                raise GTJA191DataError(
                    f"gtja191 requires complete point-in-time {field}; missing {int(missing.sum())} observed market rows"
                )
        required_dates = observed.any(axis=1)
        incomplete = [
            field
            for field in self.reference_inputs
            if not np.isfinite(np.asarray(inputs[field], dtype=float)[required_dates]).all()
        ]
        if incomplete:
            raise GTJA191DataError(f"gtja191 requires complete point-in-time external fields: {', '.join(incomplete)}")
