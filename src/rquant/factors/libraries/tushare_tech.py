"""The technical indicators of Tushare's ``stk_factor_pro`` endpoint, re-implemented as a daily panel factor set.

Tushare publishes, per stock and day, about thirty classic Chinese-terminal indicators (ASI, BBI, BIAS, BOLL, BRAR,
CCI, CR, DMA, DMI, DPO, EMA, EMV, EXPMA, KDJ, KTN, MA, MACD, MASS, MFI, MTM, OBV, PSY, ROC, RSI, TAQ, TRIX, VR, WR,
XSII) with the parameters listed in its documentation, plus streak counts. The endpoint itself needs more Tushare
points than this account has, so every indicator is computed here from the standard TongDaXin formula with
Tushare's parameters. Tushare's values are not used.

* Indicators run on each stock's own trading days: rows without a complete open, high, low, close and volume quote
  are removed first and stay missing, so a suspension does not reset an EMA or shorten a window.
* ``EMA(X, N)`` has weight ``2 / (N + 1)`` and ``SMA(X, N, M)`` weight ``M / N``; both start at the first known value,
  and the first ``lookback`` quotes of every stock are left missing so early values are not published unsettled.
* Prices are RQuant's backward-adjusted canonical prices (Tushare's ``_hfq`` variant), whose level is arbitrary per
  stock. Outputs in price units are published scale-free: lines (MA, EMA, bands, channels) as ``line / close - 1``
  and differences or ranges (ATR, MACD, MTM, DPO, DMA, ASI) as ``value / close``. OBV is published as its 20-day
  change over the volume traded in those days.
* BBI uses the standard 3, 6, 12 and 24 day averages. Outputs that are an exact affine function of another published
  output are left out: ``boll_mid`` (the 20-day MA) and ``xsii_td2``/``xsii_td4`` (fixed multiples of TD1 and TD3).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary

Panel = pd.DataFrame
Series = pd.Series
FIELDS = ("open", "high", "low", "close", "volume")
FLOW_DAYS = 20
INSTRUMENT_BLOCK = 64
MOVING_AVERAGES = (5, 10, 20, 30, 60, 90, 250)

# (source name, description, leading quotes left missing). Order is the stable public column order.
TUSHARE_TECH: tuple[tuple[str, str, int], ...] = (
    ("asi", "ASI(26, 10): 26-day sum of the swing index, over close", 26),
    ("asit", "ASI(26, 10): 10-day MA of ASI, over close", 35),
    ("atr", "ATR(20): 20-day MA of the true range, over close", 20),
    ("bbi", "BBI: mean of the 3, 6, 12 and 24-day MAs, over close - 1", 23),
    ("bias1", "BIAS(6): (close - MA6) / MA6 * 100", 5),
    ("bias2", "BIAS(12): (close - MA12) / MA12 * 100", 11),
    ("bias3", "BIAS(24): (close - MA24) / MA24 * 100", 23),
    ("boll_upper", "BOLL(20, 2): MA20 + 2 std, over close - 1", 19),
    ("boll_lower", "BOLL(20, 2): MA20 - 2 std, over close - 1", 19),
    ("brar_ar", "BRAR(26): AR = sum(high - open) / sum(open - low) * 100", 25),
    ("brar_br", "BRAR(26): BR = sum(max(0, high - prev close)) / sum(max(0, prev close - low)) * 100", 26),
    ("cci", "CCI(14): (TYP - MA(TYP)) / (0.015 AVEDEV(TYP))", 13),
    ("cr", "CR(20): sum(max(0, high - prev mid)) / sum(max(0, prev mid - low)) * 100", 20),
    ("dfma_dif", "DMA(10, 50, 10): MA10 - MA50, over close", 49),
    ("dfma_difma", "DMA(10, 50, 10): 10-day MA of DIF, over close", 58),
    ("dmi_pdi", "DMI(14, 6): +DI", 14),
    ("dmi_mdi", "DMI(14, 6): -DI", 14),
    ("dmi_adx", "DMI(14, 6): 6-day MA of |+DI - -DI| / (+DI + -DI) * 100", 19),
    ("dmi_adxr", "DMI(14, 6): (ADX + ADX 6 days ago) / 2", 25),
    ("dpo", "DPO(20, 10, 6): close - MA20 ten days ago, over close", 29),
    ("madpo", "DPO(20, 10, 6): 6-day MA of DPO, over close", 34),
    *((f"ema_{n}", f"EMA({n}) of close, over close - 1", n - 1) for n in MOVING_AVERAGES),
    ("emv", "EMV(14, 9): 14-day MA of the volume-scaled mid-point move", 28),
    ("maemv", "EMV(14, 9): 9-day MA of EMV", 36),
    ("expma_12", "EXPMA(12): EMA12 of close, over close - 1", 11),
    ("expma_50", "EXPMA(50): EMA50 of close, over close - 1", 49),
    ("kdj_k", "KDJ(9, 3, 3): K = SMA(RSV, 3, 1)", 10),
    ("kdj_d", "KDJ(9, 3, 3): D = SMA(K, 3, 1)", 12),
    ("kdj", "KDJ(9, 3, 3): J = 3 K - 2 D", 12),
    ("ktn_upper", "KTN(20, 10): EMA20 of TYP + 2 EMA10 of TR, over close - 1", 19),
    ("ktn_mid", "KTN(20, 10): EMA20 of TYP, over close - 1", 19),
    ("ktn_down", "KTN(20, 10): EMA20 of TYP - 2 EMA10 of TR, over close - 1", 19),
    *((f"ma_{n}", f"MA({n}) of close, over close - 1", n - 1) for n in MOVING_AVERAGES),
    ("macd_dif", "MACD(12, 26, 9): EMA12 - EMA26, over close", 25),
    ("macd_dea", "MACD(12, 26, 9): EMA9 of DIF, over close", 33),
    ("macd", "MACD(12, 26, 9): 2 (DIF - DEA), over close", 33),
    ("mass", "MASS(9, 25, 6): 25-day sum of MA9(high - low) / MA9(MA9(high - low))", 40),
    ("ma_mass", "MASS(9, 25, 6): 6-day MA of MASS", 45),
    ("mfi", "MFI(14): money-flow index of TYP * volume", 14),
    ("mtm", "MTM(12, 6): close - close 12 days ago, over close", 12),
    ("mtmma", "MTM(12, 6): 6-day MA of MTM, over close", 17),
    ("obv", f"OBV: {FLOW_DAYS}-day change of on-balance volume over the volume of those days", FLOW_DAYS),
    ("psy", "PSY(12, 6): share of up days over 12 days * 100", 12),
    ("psyma", "PSY(12, 6): 6-day MA of PSY", 17),
    ("roc", "ROC(12, 6): 12-day rate of change * 100", 12),
    ("maroc", "ROC(12, 6): 6-day MA of ROC", 17),
    ("rsi_6", "RSI(6): SMA(max(dC, 0), 6, 1) / SMA(|dC|, 6, 1) * 100", 6),
    ("rsi_12", "RSI(12)", 12),
    ("rsi_24", "RSI(24)", 24),
    ("taq_up", "TAQ(20): 20-day highest high, over close - 1", 19),
    ("taq_mid", "TAQ(20): mid of the 20-day channel, over close - 1", 19),
    ("taq_down", "TAQ(20): 20-day lowest low, over close - 1", 19),
    ("trix", "TRIX(12, 20): one-day % change of the triple EMA12", 34),
    ("trma", "TRIX(12, 20): 20-day MA of TRIX", 53),
    ("vr", "VR(26): (2 up volume + flat volume) / (2 down volume + flat volume) * 100", 26),
    ("wr", "WR(10): (10-day high - close) / (10-day high - 10-day low) * 100", 9),
    ("wr1", "WR(6)", 5),
    ("xsii_td1", "XSII(102, 7): TD1 = 1.02 MA5 of (2 close + high + low) / 4, over close - 1", 4),
    ("xsii_td3", "XSII(102, 7): TD3 = 1.07 DMA(close, CC), over close - 1", 19),
    ("updays", "consecutive up days (close above the previous close)", 1),
    ("downdays", "consecutive down days (close below the previous close)", 1),
    ("topdays", "quotes since a high above today's (today's high is the highest of that many quotes)", 1),
    ("lowdays", "quotes since a low below today's (today's low is the lowest of that many quotes)", 1),
)
TUSHARE_TECH_NAMES = tuple(name for name, _, _ in TUSHARE_TECH)
LOOKBACK = {name: lookback for name, _, lookback in TUSHARE_TECH}


class TushareTechLibrary(PanelFactorLibrary):
    """Tushare stk_factor_pro technical indicators, evaluated per stock over its own trading days."""

    family = "tushare_tech"
    required_inputs = FIELDS

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"tst_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback + 1,
                implementation=f"rquant.factors.libraries.tushare_tech.stock_indicators.{name}",
                catalog_version=1,
            )
            for ordinal, (name, description, lookback) in enumerate(TUSHARE_TECH, 1)
        )

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        close = inputs["close"]
        buffers = {name: np.empty(close.shape) for name in TUSHARE_TECH_NAMES}
        self._calculate_columns(inputs, lambda name, columns, values: buffers[name].__setitem__(
            (slice(None), columns), values))
        return {name: Panel(buffers[name], index=close.index, columns=close.columns) for name in TUSHARE_TECH_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        canonical = {spec.source_name: spec.canonical_name for spec in self.specs}

        def store(name: str, columns: slice, values: np.ndarray) -> None:
            output_buffers[canonical[name]][:, columns] = values

        self._calculate_columns(inputs, store)

    def _calculate_columns(self, inputs: Mapping[str, Any], store: Callable[[str, slice, np.ndarray], None]) -> None:
        """Compute blocks of instruments at a time, so writes to a date-major store stay mostly contiguous."""
        panels = {field: np.asarray(inputs[field], dtype=float) for field in FIELDS}
        dates, stocks = panels["close"].shape
        for first in range(0, stocks, INSTRUMENT_BLOCK):
            columns = slice(first, min(first + INSTRUMENT_BLOCK, stocks))
            width = columns.stop - columns.start
            block = {name: np.full((dates, width), np.nan) for name in TUSHARE_TECH_NAMES}
            for offset in range(width):
                column = first + offset
                quoted = np.isfinite(np.column_stack([panels[field][:, column] for field in FIELDS])).all(axis=1)
                rows = np.flatnonzero(quoted)
                if not len(rows):
                    continue
                arrays = {field: panels[field][rows, column] for field in FIELDS}
                for name, values in stock_indicators(arrays).items():
                    block[name][rows, offset] = values
            for name in TUSHARE_TECH_NAMES:
                store(name, columns, block[name])


# ---------------------------------------------------------------------------------------------- formula operators
def ref(x: Series, n: int) -> Series:
    return x.shift(n)


def ma(x: Series, n: int) -> Series:
    return x.rolling(n, min_periods=n).mean()


def total(x: Series, n: int) -> Series:
    return x.rolling(n, min_periods=n).sum()


def hhv(x: Series, n: int) -> Series:
    return x.rolling(n, min_periods=n).max()


def llv(x: Series, n: int) -> Series:
    return x.rolling(n, min_periods=n).min()


def ema(x: Series, n: int) -> Series:
    return x.ewm(alpha=2.0 / (n + 1.0), adjust=False).mean()


def sma(x: Series, n: int, m: int) -> Series:
    return x.ewm(alpha=m / n, adjust=False).mean()


def avedev(x: Series, n: int) -> Series:
    """Mean absolute deviation from the window mean."""
    values = x.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= n:
        windows = np.lib.stride_tricks.sliding_window_view(values, n)
        out[n - 1 :] = np.abs(windows - windows.mean(axis=1, keepdims=True)).mean(axis=1)
    return Series(out, index=x.index)


def dma(x: Series, weight: Series) -> Series:
    """Dynamic moving average ``Y = A X + (1 - A) Y'`` with a per-row weight; starts at the first known row."""
    values = x.to_numpy(dtype=float)
    weights = weight.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    previous = np.nan
    for row in range(len(values)):
        if np.isfinite(weights[row]) and np.isfinite(values[row]):
            blended = weights[row] * values[row] + (1.0 - weights[row]) * previous
            previous = values[row] if np.isnan(previous) else blended
        out[row] = previous
    return Series(out, index=x.index)


def streak(condition: np.ndarray) -> np.ndarray:
    """Length of the run of true values ending on every row."""
    out = np.zeros(len(condition))
    run = 0.0
    for row, value in enumerate(condition):
        run = run + 1.0 if value else 0.0
        out[row] = run
    return out


def range_days(values: np.ndarray, higher: bool) -> np.ndarray:
    """Quotes back to the last value above (or below) today's, or since the first quote when there is none."""
    out = np.zeros(len(values))
    stack: list[int] = []
    for row, value in enumerate(values):
        while stack and ((values[stack[-1]] <= value) if higher else (values[stack[-1]] >= value)):
            stack.pop()
        out[row] = row - stack[-1] - 1 if stack else row
        stack.append(row)
    return out


def stock_indicators(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every published indicator for one stock's consecutive quotes."""
    o, h, l, c, v = (Series(arrays[field], dtype=float) for field in FIELDS)  # noqa: E741
    lc = ref(c, 1)
    out: dict[str, Series] = {}

    aa = (h - lc).abs()
    bb = (l - lc).abs()
    cc = (h - ref(l, 1)).abs()
    dd = (lc - ref(o, 1)).abs()
    swing_range = np.where((aa > bb) & (aa > cc), aa + bb / 2 + dd / 4,
                           np.where((bb > cc) & (bb > aa), bb + aa / 2 + dd / 4, cc + dd / 4))
    x = c - lc + (c - o) / 2 + lc - ref(o, 1)
    si = 16 * x / Series(swing_range).where(lambda r: r > 0) * np.maximum(aa, bb)
    asi = total(si, 26)
    out["asi"] = asi / c
    out["asit"] = ma(asi, 10) / c

    true_range = pd.concat([h - l, (lc - h).abs(), (lc - l).abs()], axis=1).max(axis=1, skipna=False)
    out["atr"] = ma(true_range, 20) / c
    out["bbi"] = (ma(c, 3) + ma(c, 6) + ma(c, 12) + ma(c, 24)) / 4 / c - 1
    for name, n in (("bias1", 6), ("bias2", 12), ("bias3", 24)):
        mean = ma(c, n)
        out[name] = (c - mean) / mean * 100
    mid = ma(c, 20)
    spread = c.rolling(20, min_periods=20).std(ddof=0)
    out["boll_upper"] = (mid + 2 * spread) / c - 1
    out["boll_lower"] = (mid - 2 * spread) / c - 1
    out["brar_ar"] = total(h - o, 26) / total(o - l, 26) * 100
    out["brar_br"] = total((h - lc).clip(lower=0), 26) / total((lc - l).clip(lower=0), 26) * 100

    typical = (h + l + c) / 3
    out["cci"] = (typical - ma(typical, 14)) / (0.015 * avedev(typical, 14))
    prior_mid = ref(h + l, 1) / 2
    out["cr"] = total((h - prior_mid).clip(lower=0), 20) / total((prior_mid - l).clip(lower=0), 20) * 100
    dif = ma(c, 10) - ma(c, 50)
    out["dfma_dif"] = dif / c
    out["dfma_difma"] = ma(dif, 10) / c

    moving_range = total(true_range, 14)
    up_move = h - ref(h, 1)
    down_move = ref(l, 1) - l
    plus = total(up_move.where((up_move > 0) & (up_move > down_move), 0.0).where(up_move.notna()), 14)
    minus = total(down_move.where((down_move > 0) & (down_move > up_move), 0.0).where(down_move.notna()), 14)
    pdi = plus * 100 / moving_range
    mdi = minus * 100 / moving_range
    adx = ma((mdi - pdi).abs() / (mdi + pdi) * 100, 6)
    out["dmi_pdi"] = pdi
    out["dmi_mdi"] = mdi
    out["dmi_adx"] = adx
    out["dmi_adxr"] = (adx + ref(adx, 6)) / 2
    dpo = c - ref(ma(c, 20), 10)
    out["dpo"] = dpo / c
    out["madpo"] = ma(dpo, 6) / c
    for n in MOVING_AVERAGES:
        out[f"ema_{n}"] = ema(c, n) / c - 1

    hl = h + l
    volume_ratio = ma(v, 14) / v
    mid_move = 100 * (hl - ref(hl, 1)) / hl
    emv = ma(mid_move * volume_ratio * (h - l) / ma(h - l, 14), 14)
    out["emv"] = emv
    out["maemv"] = ma(emv, 9)
    out["expma_12"] = ema(c, 12) / c - 1
    out["expma_50"] = ema(c, 50) / c - 1

    rsv = (c - llv(l, 9)) / (hhv(h, 9) - llv(l, 9)) * 100
    k = sma(rsv, 3, 1)
    d = sma(k, 3, 1)
    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj"] = 3 * k - 2 * d
    keltner = ema(typical, 20)
    keltner_range = ema(true_range, 10)
    out["ktn_upper"] = (keltner + 2 * keltner_range) / c - 1
    out["ktn_mid"] = keltner / c - 1
    out["ktn_down"] = (keltner - 2 * keltner_range) / c - 1
    for n in MOVING_AVERAGES:
        out[f"ma_{n}"] = ma(c, n) / c - 1

    macd_dif = ema(c, 12) - ema(c, 26)
    macd_dea = ema(macd_dif, 9)
    out["macd_dif"] = macd_dif / c
    out["macd_dea"] = macd_dea / c
    out["macd"] = 2 * (macd_dif - macd_dea) / c
    range_ma = ma(h - l, 9)
    mass = total(range_ma / ma(range_ma, 9), 25)
    out["mass"] = mass
    out["ma_mass"] = ma(mass, 6)
    flow = typical * v
    rising = typical > ref(typical, 1)
    falling = typical < ref(typical, 1)
    known = ref(typical, 1).notna()
    flow_ratio = total(flow.where(rising, 0.0).where(known), 14) / total(flow.where(falling, 0.0).where(known), 14)
    out["mfi"] = 100 - 100 / (1 + flow_ratio)
    momentum = c - ref(c, 12)
    out["mtm"] = momentum / c
    out["mtmma"] = ma(momentum, 6) / c

    change = c - lc
    direction = np.sign(change).where(change.notna())
    obv = (direction * v).fillna(0.0).cumsum()
    traded = total(v, FLOW_DAYS)
    out["obv"] = (obv - ref(obv, FLOW_DAYS)) / traded
    up_days = (change > 0).astype(float).where(change.notna())
    psy = total(up_days, 12) / 12 * 100
    out["psy"] = psy
    out["psyma"] = ma(psy, 6)
    roc = 100 * (c - ref(c, 12)) / ref(c, 12)
    out["roc"] = roc
    out["maroc"] = ma(roc, 6)
    for n in (6, 12, 24):
        out[f"rsi_{n}"] = sma(change.clip(lower=0), n, 1) / sma(change.abs(), n, 1) * 100

    top = hhv(h, 20)
    bottom = llv(l, 20)
    out["taq_up"] = top / c - 1
    out["taq_mid"] = (top + bottom) / 2 / c - 1
    out["taq_down"] = bottom / c - 1
    triple = ema(ema(ema(c, 12), 12), 12)
    trix = (triple - ref(triple, 1)) / ref(triple, 1) * 100
    out["trix"] = trix
    out["trma"] = ma(trix, 20)
    up_volume = total(v.where(change > 0, 0.0).where(change.notna()), 26)
    down_volume = total(v.where(change < 0, 0.0).where(change.notna()), 26)
    flat_volume = total(v.where(change == 0, 0.0).where(change.notna()), 26)
    out["vr"] = 100 * (2 * up_volume + flat_volume) / (2 * down_volume + flat_volume)
    for name, n in (("wr", 10), ("wr1", 6)):
        out[name] = 100 * (hhv(h, n) - c) / (hhv(h, n) - llv(l, n))

    weighted = (2 * c + h + l) / 4
    out["xsii_td1"] = ma(weighted, 5) * 102 / 100 / c - 1
    ma20 = ma(c, 20)
    out["xsii_td3"] = (1 + 7 / 100) * dma(c, (weighted - ma20).abs() / ma20) / c - 1

    out["updays"] = Series(streak((change > 0).to_numpy()))
    out["downdays"] = Series(streak((change < 0).to_numpy()))
    out["topdays"] = Series(range_days(h.to_numpy(), higher=True))
    out["lowdays"] = Series(range_days(l.to_numpy(), higher=False))

    count = len(c)
    result: dict[str, np.ndarray] = {}
    for name in TUSHARE_TECH_NAMES:
        values = np.asarray(out[name], dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        values[: min(count, LOOKBACK[name])] = np.nan
        result[name] = values
    return result
