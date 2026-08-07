# ruff: noqa: E501, F821, F841, UP034
"""RQuant-owned WorldQuant Alpha101 formulas executed by KunQuant.

The 82 formulas available in KunQuant 0.1.11 were moved into this module
under Apache-2.0. RQuant adds the 19 formulas absent from that release and
uses point-in-time industry neutralization where the definitions require it.
All 101 formula implementations are now repository-owned and fingerprinted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rquant.errors import DependencyError, FactorContractError
from rquant.factors.extensions.kunquant import group_neutralize, kunquant_symbols, minimum
from rquant.factors.libraries.base import FactorLibrary, FactorSpec

CATALOG_VERSION = 3
TS_RANK_SEMANTICS = "percentile_v1"
ALPHA101_UPSTREAM_MISSING = frozenset({48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100})


def _load_kunquant_symbols() -> None:
    """Load KunQuant's public operators lazily without importing predefined formulas."""
    try:
        from KunQuant import Op, ops
    except ImportError as exc:
        raise DependencyError("KunQuant==0.1.11 is required to build Alpha101") from exc
    for module in (Op, ops):
        for name in dir(module):
            if not name.startswith("_"):
                globals()[name] = getattr(module, name)


class _BuiltinAlpha101Data:
    def __init__(
        self,
        open: OpBase,
        close: OpBase = None,
        high: OpBase = None,
        low: OpBase = None,
        volume: OpBase = None,
        amount: OpBase = None,
        vwap: OpBase = None,
    ) -> None:
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        if vwap is None:
            self.vwap = Div(self.amount, AddConst(self.volume, 0.0000001))
        else:
            self.vwap = vwap
        self.returns = returns(close)


def stddev(v: OpBase, window: int) -> OpBase:
    return WindowedStddev(v, window)


def returns(v: OpBase) -> OpBase:
    prev1 = BackRef(v, 1)
    return SubConst(Div(v, prev1), 1.0)


def ts_argmax(v: OpBase, window: int) -> OpBase:
    return TsArgMax(v, window)


def ts_argmin(v: OpBase, window: int) -> OpBase:
    return TsArgMin(v, window)


def ts_rank(v: OpBase, window: int) -> OpBase:
    return TsRank(v, window) / float(window)


def ts_sum(v: OpBase, window: int) -> OpBase:
    return WindowedSum(v, window)


def ts_min(v: OpBase, window: int) -> OpBase:
    return WindowedMin(v, window)


def ts_max(v: OpBase, window: int) -> OpBase:
    return WindowedMax(v, window)


def correlation(v1: OpBase, v2: OpBase, window: int, no_optimization: bool = False) -> OpBase:
    ret = WindowedCorrelation(v1, window, v2)
    if no_optimization:
        ret.attrs["no_fast_stat"] = True
    return ret


def delta(v1: OpBase, window: int = 1) -> OpBase:
    return Sub(v1, BackRef(v1, window))


def rank(v: OpBase) -> OpBase:
    return Rank(v)


def sign(v: OpBase) -> OpBase:
    return Sign(v)


def covariance(v: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCovariance(v, window, v2)


def sma(v: OpBase, window: int, no_optimization: bool = False) -> OpBase:
    ret = WindowedAvg(v, window)
    if no_optimization:
        ret.attrs["no_fast_stat"] = True
    return ret


def bool_to_10(v: OpBase) -> OpBase:
    return Select(v, ConstantOp(1), ConstantOp(0))


def scale(value: OpBase) -> OpBase:
    return Scale(value)


def delay(value: OpBase, window: int) -> OpBase:
    return BackRef(value, window)


def decay_linear(value: OpBase, window: int) -> OpBase:
    return DecayLinear(value, window)


def alpha001(d: _BuiltinAlpha101Data):
    inner = d.close
    cond = LessThanConst(d.returns, 0.0)
    sel = Select(cond, stddev(d.returns, 20), d.close)
    sel = Mul(sel, sel)
    return Rank(ts_argmax(sel, 5))


def alpha002(d: _BuiltinAlpha101Data):
    v = MulConst(WindowedCorrelation(Rank(delta(Log(d.volume), 2)), 6, Rank(Div(Sub(d.close, d.open), d.open))), -1)
    return SetInfOrNanToValue(v)


def alpha003(d: _BuiltinAlpha101Data):
    df = MulConst(correlation(Rank(d.open), Rank(d.volume), 10), -1)
    return SetInfOrNanToValue(df)


def alpha004(d: _BuiltinAlpha101Data):
    df = MulConst(ts_rank(Rank(d.low), 9), -1)
    return df


def alpha005(d: _BuiltinAlpha101Data):
    v1 = Rank(Sub(d.open, DivConst(WindowedSum(d.vwap, 10), 10)))
    v2 = MulConst(Abs(Rank(Sub(d.close, d.vwap))), -1)
    return Mul(v1, v2)


def alpha006(d: _BuiltinAlpha101Data):
    dopen = d.open
    vol = d.volume
    v1 = MulConst(WindowedCorrelation(dopen, 10, vol), -1)
    return SetInfOrNanToValue(v1)


def alpha007(d: _BuiltinAlpha101Data):
    adv20 = WindowedAvg(d.volume, 20)
    alpha = MulConst(Mul(ts_rank(Abs(delta(d.close, 7)), 60), sign(delta(d.close, 7))), -1)
    return Select(GreaterEqual(adv20, d.volume), ConstantOp(-1), alpha)


def alpha008(d: _BuiltinAlpha101Data):
    v = rank(
        Sub(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), BackRef(Mul(ts_sum(d.open, 5), ts_sum(d.returns, 5)), 10))
    )
    return MulConst(v, -1)


def alpha009(d: _BuiltinAlpha101Data):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 5), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 5), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    return alpha


def alpha010(d: _BuiltinAlpha101Data):
    delta_close = delta(d.close, 1)
    cond_1 = GreaterThan(ts_min(delta_close, 4), ConstantOp(0))
    cond_2 = LessThan(ts_max(delta_close, 4), ConstantOp(0))
    alpha = MulConst(delta_close, -1)
    alpha = Select(Or(cond_1, cond_2), delta_close, alpha)
    return alpha


def alpha011(d: _BuiltinAlpha101Data):
    v = (rank(ts_max((d.vwap - d.close), 3)) + rank(ts_min((d.vwap - d.close), 3))) * rank(delta(d.volume, 3))
    return v


# Alpha#12	 (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha012(d: _BuiltinAlpha101Data):
    v = sign(delta(d.volume, 1)) * (-1 * delta(d.close, 1))
    return v


def alpha013(d: _BuiltinAlpha101Data):
    # alpha013 has rank(cov(rank(X), rank(Y))). Output of cov seems to have very similar results
    # like 1e-6 and 0. Thus the rank result will be different from pandas's reference
    return -1 * rank(covariance(rank(d.close), rank(d.volume), 5))


def alpha014(d: _BuiltinAlpha101Data):
    df = SetInfOrNanToValue(correlation(d.open, d.volume, 10))
    return -1 * (rank(delta(d.returns, 3)) * df)


def alpha015(d: _BuiltinAlpha101Data):
    # due to corr on Rank data, the rank result will be different from pandas's reference
    df = SetInfOrNanToValue(correlation(rank(d.high), rank(d.volume), 3, no_optimization=True))
    return -1 * ts_sum(rank(df), 3)


def alpha016(d: _BuiltinAlpha101Data):
    return -1 * rank(covariance(rank(d.high), rank(d.volume), 5))


def alpha017(d: _BuiltinAlpha101Data):
    adv20 = WindowedAvg(d.volume, 20)
    return -1 * (
        rank(ts_rank(d.close, 10))
        * rank(delta(delta(d.close, 1), 1))
        * rank(ts_rank(SetInfOrNanToValue(d.volume / adv20), 5))
    )


def alpha018(d: _BuiltinAlpha101Data):
    df = correlation(d.close, d.open, 10)
    df = SetInfOrNanToValue(df)
    return -1 * (rank((stddev(Abs((d.close - d.open)), 5) + (d.close - d.open)) + df))


def alpha019(d: _BuiltinAlpha101Data):
    return (-1 * sign((d.close - delay(d.close, 7)) + delta(d.close, 7))) * (1 + rank(1 + ts_sum(d.returns, 250)))


# Alpha#20	 (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
def alpha020(d: _BuiltinAlpha101Data):
    return -1 * (rank(d.open - delay(d.high, 1)) * rank(d.open - delay(d.close, 1)) * rank(d.open - delay(d.low, 1)))


def alpha021(d: _BuiltinAlpha101Data):
    # d = (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))
    # c = ((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8)))
    # b = (c ? 1 : d)
    # a = (((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2))
    # (a? (-1 * 1) : b)
    c = WindowedAvg(d.close, 2) < WindowedAvg(d.close, 8) - stddev(d.close, 8)
    adv20 = WindowedAvg(d.volume, 20)
    dd = d.volume / adv20 >= 1
    a = WindowedAvg(d.close, 8) + stddev(d.close, 8) < WindowedAvg(d.close, 2)
    out = Select(~a & (c | dd), ConstantOp(1), ConstantOp(-1))
    return out


def alpha022(self: _BuiltinAlpha101Data):
    df = correlation(self.high, self.volume, 5)
    df = SetInfOrNanToValue(df)
    return -1 * delta(df, 5) * rank(stddev(self.close, 20))


# Alpha#23	 (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
def alpha023(self: _BuiltinAlpha101Data):
    cond = sma(self.high, 20) < self.high
    alpha = -1 * SetInfOrNanToValue(delta(self.high, 2))
    return Select(cond, alpha, ConstantOp(0))


def alpha024(self: _BuiltinAlpha101Data):
    cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
    alpha = -1 * delta(self.close, 3)
    return Select(cond, -1 * (self.close - ts_min(self.close, 100)), alpha)


def alpha025(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))


def alpha026(self: _BuiltinAlpha101Data):
    df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5, no_optimization=True)
    df = SetInfOrNanToValue(df)
    return -1 * ts_max(df, 3)


def alpha027(self: _BuiltinAlpha101Data):
    alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2, no_optimization=True) / 2.0))
    return Select(alpha > 0.5, ConstantOp(-1), ConstantOp(1))


# Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
def alpha028(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    df = correlation(adv20, self.low, 5)
    df = SetInfOrNanToValue(df)
    return scale(((df + ((self.high + self.low) / 2)) - self.close))


# Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
def alpha029(self: _BuiltinAlpha101Data):
    return ts_min(rank(rank(scale(Log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) + ts_rank(
        delay((-1 * self.returns), 6), 5
    )


# Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
def alpha030(self: _BuiltinAlpha101Data):
    delta_close = delta(self.close, 1)
    inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
    return (1.0 - rank(inner)) * SetInfOrNanToValue(ts_sum(self.volume, 5) / ts_sum(self.volume, 20), 1)


# Alpha#31	 ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
def alpha031(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    df = correlation(adv20, self.low, 12)
    df = SetInfOrNanToValue(df)
    p1 = rank(rank(rank(DecayLinear((-1 * rank(rank(delta(self.close, 10)))), 10))))
    p2 = rank((-1 * delta(self.close, 3)))
    p3 = sign(scale(df))
    return p1 + p2 + p3


# Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
def alpha032(self):
    return scale(((sma(self.close, 7) / 7) - self.close)) + (
        20 * scale(correlation(self.vwap, delay(self.close, 5), 230))
    )


def alpha033(self: _BuiltinAlpha101Data):
    return rank((self.open / self.close) - 1)


def alpha034(self: _BuiltinAlpha101Data):
    inner = stddev(self.returns, 2) / stddev(self.returns, 5)
    inner = SetInfOrNanToValue(inner, 1.0)
    return rank(2 - rank(inner) - rank(delta(self.close, 1)))


def alpha035(self: _BuiltinAlpha101Data):
    return (ts_rank(self.volume, 32) * (1 - ts_rank(self.close + self.high - self.low, 16))) * (
        1 - ts_rank(self.returns, 32)
    )


def alpha036(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    return (
        (
            (
                (2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15)))
                + (0.7 * rank((self.open - self.close)))
            )
            + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))
        )
        + rank(Abs(correlation(self.vwap, adv20, 6)))
    ) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open))))


def alpha037(self: _BuiltinAlpha101Data):
    return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)


# Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
def alpha038(self: _BuiltinAlpha101Data):
    inner = self.close / self.open
    inner = SetInfOrNanToValue(inner, 1.0)
    return -1 * rank(ts_rank(self.open, 10)) * rank(inner)


# Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
def alpha039(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    return (-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20), 9))))) * (
        1 + rank(sma(self.returns, 250))
    )


# Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
def alpha040(self: _BuiltinAlpha101Data):
    return -1 * rank(stddev(self.high, 10)) * SetInfOrNanToValue(correlation(self.high, self.volume, 10), 1.0)


# Alpha#41	 (((high * low)^0.5) - vwap)
def alpha041(self: _BuiltinAlpha101Data):
    return Pow((self.high * self.low), ConstantOp(0.5)) - self.vwap


# Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
def alpha042(self: _BuiltinAlpha101Data):
    return rank((self.vwap - self.close)) / rank((self.vwap + self.close))


# Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
def alpha043(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    return ts_rank(SetInfOrNanToValue(self.volume / adv20), 20) * ts_rank((-1 * delta(self.close, 7)), 8)


# Alpha#44	 (-1 * correlation(high, rank(volume), 5))
def alpha044(self: _BuiltinAlpha101Data):
    df = correlation(self.high, rank(self.volume), 5)
    df = SetInfOrNanToValue(df)
    return -1 * df


# Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
def alpha045(self: _BuiltinAlpha101Data):
    df = correlation(self.close, self.volume, 2)
    df = SetInfOrNanToValue(df)
    return -1 * (
        rank(sma(delay(self.close, 5), 20))
        * df
        * rank(SetInfOrNanToValue(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2), 1))
    )


# Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
def alpha046(self: _BuiltinAlpha101Data):
    inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
    alpha = -1 * delta(self.close, 1)
    alpha = Select(inner < 0, ConstantOp(1), alpha)
    alpha = Select(inner > 0.25, ConstantOp(-1), alpha)
    # alpha[inner < 0] = 1
    # alpha[inner > 0.25] = -1
    return alpha


# Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
def alpha047(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    return SetInfOrNanToValue(
        (
            (rank((1 / self.close)) * self.volume / adv20)
            * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) / 5))
        )
        - rank((self.vwap - delay(self.vwap, 5)))
    )


# Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha049(self: _BuiltinAlpha101Data):
    inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
    alpha = -1 * delta(self.close)
    # alpha[inner < -0.1] = 1
    alpha = Select(inner < -0.1, ConstantOp(1), alpha)
    return alpha


# Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
def alpha050(self: _BuiltinAlpha101Data):
    df = SetInfOrNanToValue(correlation(rank(self.volume), rank(self.vwap), 5))
    return -1 * ts_max(rank(df), 5)
    # return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))


# Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha051(self: _BuiltinAlpha101Data):
    inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
    alpha = -1 * delta(self.close)
    alpha = Select(inner < -0.05, ConstantOp(1), alpha)
    return alpha


# Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
def alpha052(self: _BuiltinAlpha101Data):
    return (
        (-1 * delta(ts_min(self.low, 5), 5)) * rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))
    ) * ts_rank(self.volume, 5)


# Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
def alpha053(self: _BuiltinAlpha101Data):
    inner = self.close - self.low
    inner = Select(Equals(inner, ConstantOp(0)), ConstantOp(0.0001), inner)
    return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)


# Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha054(self: _BuiltinAlpha101Data):
    inner = self.low - self.high
    inner = Select(Equals(inner, ConstantOp(0)), ConstantOp(0.0001), inner)
    return -1 * (self.low - self.close) * (Pow(self.open, ConstantOp(5))) / (inner * Pow(self.close, ConstantOp(5)))


def alpha055(self: _BuiltinAlpha101Data):
    divisor = ts_max(self.high, 12) - ts_min(self.low, 12)
    divisor = Select(Equals(divisor, ConstantOp(0)), ConstantOp(0.0001), divisor)
    inner = (self.close - ts_min(self.low, 12)) / (divisor)
    df = correlation(rank(inner), rank(self.volume), 6)
    return -1 * SetInfOrNanToValue(df)


def alpha057(self: _BuiltinAlpha101Data):
    return 0 - (1 * ((self.close - self.vwap) / DecayLinear(rank(ts_argmax(self.close, 30)), 2)))


# Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
def alpha060(self: _BuiltinAlpha101Data):
    divisor = self.high - self.low
    divisor = Select(Equals(divisor, ConstantOp(0)), ConstantOp(0.0001), divisor)
    inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
    return 0 - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))


# Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
def alpha061(self: _BuiltinAlpha101Data):
    adv180 = sma(self.volume, 180)
    return bool_to_10(rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18)))


# Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
def alpha062(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    v1 = rank(correlation(self.vwap, sma(adv20, 22), 10))
    v2 = rank(self.open) + rank(self.open)
    v3 = rank(((self.high + self.low) / 2)) + rank(self.high)
    v4 = bool_to_10(v2 < v3)
    v5 = bool_to_10(v1 < rank(v4))
    return v5 * -1


# Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
def alpha064(self: _BuiltinAlpha101Data):
    adv120 = sma(self.volume, 120)
    a = rank(correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), sma(adv120, 13), 17))
    b = rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 4))
    c = bool_to_10(a < b)
    return c * -1


# Alpha#65	 ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
def alpha065(self: _BuiltinAlpha101Data):
    adv60 = sma(self.volume, 60)
    a = rank(correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60, 9), 6))
    b = rank((self.open - ts_min(self.open, 14)))
    return bool_to_10(a < b) * -1


# Alpha#66	 ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
def alpha066(self: _BuiltinAlpha101Data):
    return (
        rank(decay_linear(delta(self.vwap, 4), 7))
        + ts_rank(
            decay_linear(
                SetInfOrNanToValue(
                    (((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap)
                    / (self.open - ((self.high + self.low) / 2))
                ),
                11,
            ),
            7,
        )
    ) * -1


# Alpha#68	 ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
def alpha068(self: _BuiltinAlpha101Data):
    adv15 = sma(self.volume, 15)
    a = ts_rank(correlation(rank(self.high), rank(adv15), 9), 14)
    b = rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))
    return bool_to_10(a < b) * -1


# Alpha#71	 max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
def alpha071(self: _BuiltinAlpha101Data):
    adv180 = sma(self.volume, 180)
    p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
    inner = Pow(rank(((self.low + self.open) - (self.vwap + self.vwap))), ConstantOp(2))
    p2 = ts_rank(decay_linear(inner, 16), 4)
    return Max(p1, p2)
    # return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))


# Alpha#72	 (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
def alpha072(self: _BuiltinAlpha101Data):
    adv40 = sma(self.volume, 40)
    a = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) + 0.0001
    b = rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)) + 0.0001
    return a / b


# Alpha#73	 (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
def alpha073(self: _BuiltinAlpha101Data):
    p1 = rank(decay_linear(delta(self.vwap, 5), 3))
    p2 = ts_rank(
        decay_linear(
            (
                (
                    delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2)
                    / ((self.open * 0.147155) + (self.low * (1 - 0.147155)))
                )
                * -1
            ),
            3,
        ),
        17,
    )
    return -1 * Max(p1, p2)


# Alpha#74	 ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
def alpha074(self: _BuiltinAlpha101Data):
    adv30 = sma(self.volume, 30)
    a = rank(correlation(self.close, sma(adv30, 37), 15))
    b = rank(
        correlation(
            rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11, no_optimization=True
        )
    )
    return bool_to_10(a < b) * -1


# Alpha#75	 (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
def alpha075(self: _BuiltinAlpha101Data):
    adv50 = sma(self.volume, 50)
    return bool_to_10(rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50), 12)))


def alpha077(self: _BuiltinAlpha101Data):
    adv40 = sma(self.volume, 40)
    p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
    p2 = rank(decay_linear(SetInfOrNanToValue(correlation(((self.high + self.low) / 2), adv40, 3), 1), 6))
    return Min(p1, p2)


# Alpha#78	 (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
def alpha078(self: _BuiltinAlpha101Data):
    adv40 = sma(self.volume, 40)
    a = rank(correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7))
    b = rank(SetInfOrNanToValue(correlation(rank(self.vwap), rank(self.volume), 6)))
    return Pow(a, b)


# Alpha#81	 ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
def alpha081(self: _BuiltinAlpha101Data):
    adv10 = sma(self.volume, 10)
    inner = Pow(rank(correlation(self.vwap, ts_sum(adv10, 50), 8)), ConstantOp(4))
    a = rank(Log(WindowedProduct(rank(inner), 15)))
    b = rank(correlation(rank(self.vwap), rank(self.volume), 5))
    return bool_to_10(a < b) * -1


# Alpha#83	 ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
def alpha083(self: _BuiltinAlpha101Data):
    return (rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (
        ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)
    )


# Alpha#84	 SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
def alpha084(self: _BuiltinAlpha101Data):
    return Pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))


# Alpha#85	 (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
def alpha085(self: _BuiltinAlpha101Data):
    adv30 = sma(self.volume, 30)
    base = rank(SetInfOrNanToValue(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10), 1))
    expo = rank(
        SetInfOrNanToValue(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7), 1)
    )
    return Pow(base, expo)


def alpha086(self: _BuiltinAlpha101Data):
    adv20 = sma(self.volume, 20)
    a = ts_rank(correlation(self.close, sma(adv20, 15), 6), 20)
    b = rank(((self.open + self.close) - (self.vwap + self.open)))
    return bool_to_10(a < b) * -1


def alpha088(self: _BuiltinAlpha101Data):
    adv60 = sma(self.volume, 60)
    p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
    p2 = ts_rank(decay_linear(SetInfOrNanToValue(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8)), 7), 3)
    return Min(p2, p1)


def alpha092(self: _BuiltinAlpha101Data):
    adv30 = sma(self.volume, 30)
    p1 = ts_rank(decay_linear(bool_to_10((((self.high + self.low) / 2) + self.close) < (self.low + self.open)), 15), 19)
    p2 = ts_rank(decay_linear(SetInfOrNanToValue(correlation(rank(self.low), rank(adv30), 8)), 7), 7)
    return Min(p2, p1)


def alpha094(self: _BuiltinAlpha101Data):
    adv60 = sma(self.volume, 60)
    base = rank((self.vwap - ts_min(self.vwap, 12)))
    expo = ts_rank(SetInfOrNanToValue(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18)), 3)
    return Pow(base, expo) * -1


# Alpha#95	 (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
def alpha095(self: _BuiltinAlpha101Data):
    adv40 = sma(self.volume, 40)
    return bool_to_10(
        rank((self.open - ts_min(self.open, 12)))
        < ts_rank(Pow(rank(correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13)), ConstantOp(5)), 12)
    )


def alpha096(self: _BuiltinAlpha101Data):
    adv60 = sma(self.volume, 60)
    p1 = ts_rank(decay_linear(SetInfOrNanToValue(correlation(rank(self.vwap), rank(self.volume), 4)), 4), 8)
    p2 = ts_rank(
        decay_linear(ts_argmax(SetInfOrNanToValue(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4)), 13), 14),
        13,
    )
    return -1 * Max(p1, p2)


def alpha098(self: _BuiltinAlpha101Data):
    adv5 = sma(self.volume, 5)
    adv15 = sma(self.volume, 15)
    return rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5), 7)) - rank(
        decay_linear(ts_rank(ts_argmin(SetInfOrNanToValue(correlation(rank(self.open), rank(adv15), 21)), 9), 7), 8)
    )


def alpha099(self: _BuiltinAlpha101Data):
    adv60 = sma(self.volume, 60)
    return (
        bool_to_10(
            rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9))
            < rank(correlation(self.low, self.volume, 6))
        )
        * -1
    )


def alpha101(self: _BuiltinAlpha101Data):
    return (self.close - self.open) / ((self.high - self.low) + 0.001)


_BUILTIN_ALPHA101_FUNCTIONS = [
    alpha001,
    alpha002,
    alpha003,
    alpha004,
    alpha005,
    alpha006,
    alpha007,
    alpha008,
    alpha009,
    alpha010,
    alpha011,
    alpha012,
    alpha013,
    alpha014,
    alpha015,
    alpha016,
    alpha017,
    alpha018,
    alpha019,
    alpha020,
    alpha021,
    alpha022,
    alpha023,
    alpha024,
    alpha025,
    alpha026,
    alpha027,
    alpha028,
    alpha029,
    alpha030,
    alpha031,
    alpha032,
    alpha033,
    alpha034,
    alpha035,
    alpha036,
    alpha037,
    alpha038,
    alpha039,
    alpha040,
    alpha041,
    alpha042,
    alpha043,
    alpha044,
    alpha045,
    alpha046,
    alpha047,
    alpha049,
    alpha050,
    alpha051,
    alpha052,
    alpha053,
    alpha054,
    alpha055,
    alpha057,
    alpha060,
    alpha061,
    alpha062,
    alpha064,
    alpha065,
    alpha066,
    alpha068,
    alpha071,
    alpha072,
    alpha073,
    alpha074,
    alpha075,
    alpha077,
    alpha078,
    alpha081,
    alpha083,
    alpha084,
    alpha085,
    alpha086,
    alpha088,
    alpha092,
    alpha094,
    alpha095,
    alpha096,
    alpha098,
    alpha099,
    alpha101,
]

_BUILTIN_ALPHA101 = {function.__name__: function for function in _BUILTIN_ALPHA101_FUNCTIONS}


@dataclass(frozen=True)
class Alpha101Data:
    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any
    amount: Any
    vwap: Any
    cap: Any
    sector: Any
    industry: Any
    subindustry: Any


class WorldQuantAlpha101Library(FactorLibrary):
    family = "wq_alpha101"
    required_inputs = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "cap",
        "sector",
        "industry",
        "subindustry",
    )

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"a101_{ordinal:03d}",
                source_name=f"alpha{ordinal:03d}",
                family=self.family,
                ordinal=ordinal,
                formula=f"WorldQuant Formulaic Alpha #{ordinal:03d}",
                max_lookback=250,
                implementation=(
                    f"rquant.factors.libraries.alpha101._build_missing[alpha{ordinal:03d}]"
                    if ordinal in ALPHA101_UPSTREAM_MISSING
                    else f"rquant.factors.libraries.alpha101.alpha{ordinal:03d}"
                ),
                catalog_version=CATALOG_VERSION,
            )
            for ordinal in range(1, 102)
        )

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        _load_kunquant_symbols()
        builtin_data = _BuiltinAlpha101Data(
            open=inputs["open"],
            close=inputs["close"],
            high=inputs["high"],
            low=inputs["low"],
            volume=inputs["volume"],
            amount=inputs["amount"],
            vwap=inputs["vwap"],
        )
        extended = Alpha101Data(**{name: inputs[name] for name in self.required_inputs})
        extended_outputs = _build_missing(extended)
        outputs: dict[str, Any] = {}
        for spec in self.specs:
            builtin = _BUILTIN_ALPHA101.get(spec.source_name)
            if builtin is not None:
                outputs[spec.source_name] = builtin(builtin_data)
            elif spec.source_name in extended_outputs:
                outputs[spec.source_name] = extended_outputs[spec.source_name]
            else:
                raise FactorContractError(f"No local implementation for {spec.source_name}")
        return outputs


def _build_missing(data: Alpha101Data) -> dict[str, Any]:
    symbols = kunquant_symbols(
        "Abs",
        "BackRef",
        "DecayLinear",
        "Max",
        "Pow",
        "Rank",
        "Scale",
        "Select",
        "Sign",
        "TsArgMin",
        "TsRank",
        "WindowedAvg",
        "WindowedCorrelation",
        "WindowedMax",
        "WindowedSum",
    )
    ConstantOp = symbols["ConstantOp"]
    Abs = symbols["Abs"]
    BackRef = symbols["BackRef"]
    DecayLinear = symbols["DecayLinear"]
    Max = symbols["Max"]
    Pow = symbols["Pow"]
    Rank = symbols["Rank"]
    Scale = symbols["Scale"]
    Select = symbols["Select"]
    Sign = symbols["Sign"]
    TsArgMin = symbols["TsArgMin"]
    TsRank = symbols["TsRank"]
    WindowedAvg = symbols["WindowedAvg"]
    WindowedCorrelation = symbols["WindowedCorrelation"]
    WindowedMax = symbols["WindowedMax"]
    WindowedSum = symbols["WindowedSum"]

    def delta(value: Any, window: int = 1) -> Any:
        return value - BackRef(value, window)

    def corr(left: Any, right: Any, window: int) -> Any:
        return WindowedCorrelation(left, window, right)

    def adv(window: int) -> Any:
        return WindowedAvg(data.volume, window)

    def ts_rank(value: Any, window: int) -> Any:
        return TsRank(value, window) / float(window)

    def decay(value: Any, window: int) -> Any:
        return DecayLinear(value, window)

    def rank(value: Any) -> Any:
        return Rank(value)

    def power(base: Any, exponent: Any) -> Any:
        return Pow(base, exponent)

    def bool10(value: Any) -> Any:
        return Select(value, ConstantOp(1), ConstantOp(0))

    returns = data.close / BackRef(data.close, 1) - 1

    a048_inner = corr(delta(data.close), delta(BackRef(data.close, 1)), 250) * delta(data.close) / data.close
    a048 = group_neutralize(a048_inner, data.subindustry) / WindowedSum(
        power(delta(data.close) / BackRef(data.close, 1), ConstantOp(2)), 250
    )
    a056 = 0 - (rank(WindowedSum(returns, 10) / WindowedSum(WindowedSum(returns, 2), 3)) * rank(returns * data.cap))
    a058 = 0 - ts_rank(decay(corr(group_neutralize(data.vwap, data.sector), data.volume, 4), 8), 6)
    a059 = 0 - ts_rank(decay(corr(group_neutralize(data.vwap, data.industry), data.volume, 4), 16), 8)
    a063 = 0 - (
        rank(decay(delta(group_neutralize(data.close, data.industry), 2), 8))
        - rank(decay(corr(data.vwap * 0.318108 + data.open * 0.681892, WindowedSum(adv(180), 38), 14), 12))
    )
    a067 = 0 - power(
        rank(data.high - WindowedMax(data.high, 2)),
        rank(
            corr(
                group_neutralize(data.vwap, data.sector),
                group_neutralize(adv(20), data.subindustry),
                6,
            )
        ),
    )
    a069 = 0 - power(
        rank(WindowedMax(delta(group_neutralize(data.vwap, data.industry), 3), 5)),
        ts_rank(corr(data.close * 0.490655 + data.vwap * 0.509345, adv(20), 5), 9),
    )
    a070 = 0 - power(
        rank(delta(data.vwap, 1)),
        ts_rank(corr(group_neutralize(data.close, data.industry), adv(50), 18), 18),
    )
    a076 = 0 - Max(
        rank(decay(delta(data.vwap, 1), 12)),
        ts_rank(decay(ts_rank(corr(group_neutralize(data.low, data.sector), adv(81), 8), 20), 17), 19),
    )
    a079 = bool10(
        rank(delta(group_neutralize(data.close * 0.60733 + data.open * 0.39267, data.sector), 1))
        < rank(corr(ts_rank(data.vwap, 4), ts_rank(adv(150), 9), 15))
    )
    a080 = 0 - power(
        rank(Sign(delta(group_neutralize(data.open * 0.868128 + data.high * 0.131872, data.industry), 4))),
        ts_rank(corr(data.high, adv(10), 5), 6),
    )
    a082 = 0 - minimum(
        rank(decay(delta(data.open, 1), 15)),
        ts_rank(decay(corr(group_neutralize(data.volume, data.sector), data.open, 17), 7), 13),
    )
    a087 = 0 - Max(
        rank(decay(delta(data.close * 0.369701 + data.vwap * 0.630299, 2), 3)),
        ts_rank(decay(Abs(corr(group_neutralize(adv(81), data.industry), data.close, 13)), 5), 14),
    )
    a089 = ts_rank(decay(corr(data.low, adv(10), 7), 6), 4) - ts_rank(
        decay(delta(group_neutralize(data.vwap, data.industry), 3), 10), 15
    )
    a090 = 0 - power(
        rank(data.close - WindowedMax(data.close, 5)),
        ts_rank(corr(group_neutralize(adv(40), data.subindustry), data.low, 5), 3),
    )
    a091 = 0 - (
        ts_rank(decay(decay(corr(group_neutralize(data.close, data.industry), data.volume, 10), 16), 4), 5)
        - rank(decay(corr(data.vwap, adv(30), 4), 3))
    )
    a093 = ts_rank(decay(corr(group_neutralize(data.vwap, data.industry), adv(81), 17), 20), 8) / rank(
        decay(delta(data.close * 0.524434 + data.vwap * 0.475566, 3), 16)
    )
    a097 = rank(
        decay(delta(group_neutralize(data.low * 0.721001 + data.vwap * 0.278999, data.industry), 3), 20)
    ) - ts_rank(decay(ts_rank(corr(ts_rank(data.low, 8), ts_rank(adv(60), 17), 5), 19), 16), 7)
    adv20 = adv(20)
    ranked_pressure = rank(
        (((data.close - data.low) - (data.high - data.close)) / (data.high - data.low)) * data.volume
    )
    term1 = 1.5 * Scale(group_neutralize(group_neutralize(ranked_pressure, data.subindustry), data.subindustry))
    term2 = Scale(
        group_neutralize(
            corr(data.close, rank(adv20), 5) - rank(TsArgMin(data.close, 30)),
            data.subindustry,
        )
    )
    a100 = 0 - ((term1 - term2) * (data.volume / adv20))

    return {
        "alpha048": a048,
        "alpha056": a056,
        "alpha058": a058,
        "alpha059": a059,
        "alpha063": a063,
        "alpha067": a067,
        "alpha069": a069,
        "alpha070": a070,
        "alpha076": a076,
        "alpha079": a079,
        "alpha080": a080,
        "alpha082": a082,
        "alpha087": a087,
        "alpha089": a089,
        "alpha090": a090,
        "alpha091": a091,
        "alpha093": a093,
        "alpha097": a097,
        "alpha100": a100,
    }
