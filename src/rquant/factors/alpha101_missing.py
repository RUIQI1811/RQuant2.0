"""KunQuant implementations for the 19 WorldQuant Alpha101 formulas absent upstream.

Window constants follow the integer approximations used by KunQuant's bundled Alpha101 implementation.
Industry inputs are point-in-time numeric IDs. Missing values remain missing; they are never assigned to a
synthetic market group.
"""

from __future__ import annotations

from typing import Any


def _kunquant_symbols() -> dict[str, Any]:
    try:
        from KunQuant import Op, ops
        from KunQuant.Op import ConstantOp
    except ImportError as exc:
        from rquant.errors import DependencyError

        raise DependencyError("KunQuant==0.1.11 is required to build Alpha101") from exc
    names = (
        "Abs",
        "BackRef",
        "DecayLinear",
        "GenericCrossSectionalOp",
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
    symbols = {"ConstantOp": ConstantOp}
    for name in names:
        symbols[name] = getattr(ops, name) if hasattr(ops, name) else getattr(Op, name)
    return symbols


def make_group_neutralize_class() -> type:
    symbols = _kunquant_symbols()
    base = symbols["GenericCrossSectionalOp"]

    class GroupNeutralize(base):
        def __init__(self, value: Any, group_id: Any) -> None:
            super().__init__([value, group_id], None)

        def generate_head(self) -> str:
            # KunQuant places this string inside the generated stage function. Its shared header already includes
            # <vector>, <cmath> and <limits>, so only local reusable storage may be declared here.
            return "std::vector<T> group_sums;\nstd::vector<size_t> group_counts;\n"

        def generate_body(self) -> str:
            return r"""
            size_t max_group = 0;
            for (size_t i = 0; i < num_stocks; ++i) {
                const T group = input_1[i];
                if (std::isfinite(group) && group >= T(0)) {
                    max_group = std::max(max_group, static_cast<size_t>(group));
                }
            }
            group_sums.assign(max_group + 1, T(0));
            group_counts.assign(max_group + 1, 0);
            for (size_t i = 0; i < num_stocks; ++i) {
                const T value = input_0[i];
                const T group = input_1[i];
                if (!std::isfinite(value) || !std::isfinite(group) || group < T(0)) continue;
                const size_t key = static_cast<size_t>(group);
                group_sums[key] += value;
                group_counts[key] += 1;
            }
            for (size_t i = 0; i < num_stocks; ++i) {
                const T value = input_0[i];
                const T group = input_1[i];
                if (!std::isfinite(value) || !std::isfinite(group) || group < T(0)) {
                    output_0[i] = std::numeric_limits<T>::quiet_NaN();
                    continue;
                }
                const size_t key = static_cast<size_t>(group);
                output_0[i] = group_counts[key] == 0
                    ? std::numeric_limits<T>::quiet_NaN()
                    : value - group_sums[key] / static_cast<T>(group_counts[key]);
            }
            """

    GroupNeutralize.__name__ = "RQuantGroupNeutralize"
    return GroupNeutralize


class ExtendedAllData:
    def __init__(
        self,
        *,
        open: Any,
        close: Any,
        high: Any,
        low: Any,
        volume: Any,
        amount: Any,
        vwap: Any,
        cap: Any,
        sector: Any,
        industry: Any,
        subindustry: Any,
    ) -> None:
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        self.vwap = vwap
        self.cap = cap
        self.sector = sector
        self.industry = industry
        self.subindustry = subindustry


def build_missing(data: ExtendedAllData) -> dict[str, Any]:
    s = _kunquant_symbols()
    ConstantOp = s["ConstantOp"]
    Abs = s["Abs"]
    BackRef = s["BackRef"]
    DecayLinear = s["DecayLinear"]
    Max = s["Max"]
    Pow = s["Pow"]
    Rank = s["Rank"]
    Scale = s["Scale"]
    Select = s["Select"]
    Sign = s["Sign"]
    TsArgMin = s["TsArgMin"]
    TsRank = s["TsRank"]
    WindowedAvg = s["WindowedAvg"]
    WindowedCorrelation = s["WindowedCorrelation"]
    WindowedMax = s["WindowedMax"]
    WindowedSum = s["WindowedSum"]
    Neutralize = make_group_neutralize_class()

    def delta(value: Any, window: int = 1) -> Any:
        return value - BackRef(value, window)

    def corr(left: Any, right: Any, window: int) -> Any:
        return WindowedCorrelation(left, window, right)

    def adv(window: int) -> Any:
        return WindowedAvg(data.volume, window)

    def ts_rank(value: Any, window: int) -> Any:
        # WorldQuant formulas combine time-series ranks with cross-sectional percentile ranks. KunQuant's
        # TsRank returns the raw rolling rank in [1, window], so normalize it to the compatible (0, 1] scale.
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
    a048 = Neutralize(a048_inner, data.subindustry) / WindowedSum(
        power(delta(data.close) / BackRef(data.close, 1), ConstantOp(2)), 250
    )
    a056 = 0 - (rank(WindowedSum(returns, 10) / WindowedSum(WindowedSum(returns, 2), 3)) * rank(returns * data.cap))
    a058 = 0 - ts_rank(decay(corr(Neutralize(data.vwap, data.sector), data.volume, 4), 8), 6)
    a059 = 0 - ts_rank(decay(corr(Neutralize(data.vwap, data.industry), data.volume, 4), 16), 8)
    a063 = 0 - (
        rank(decay(delta(Neutralize(data.close, data.industry), 2), 8))
        - rank(decay(corr(data.vwap * 0.318108 + data.open * 0.681892, WindowedSum(adv(180), 38), 14), 12))
    )
    a067 = 0 - power(
        rank(data.high - WindowedMax(data.high, 2)),
        rank(corr(Neutralize(data.vwap, data.sector), Neutralize(adv(20), data.subindustry), 6)),
    )
    a069 = 0 - power(
        rank(WindowedMax(delta(Neutralize(data.vwap, data.industry), 3), 5)),
        ts_rank(corr(data.close * 0.490655 + data.vwap * 0.509345, adv(20), 5), 9),
    )
    a070 = 0 - power(
        rank(delta(data.vwap, 1)),
        ts_rank(corr(Neutralize(data.close, data.industry), adv(50), 18), 18),
    )
    a076 = 0 - Max(
        rank(decay(delta(data.vwap, 1), 12)),
        ts_rank(decay(ts_rank(corr(Neutralize(data.low, data.sector), adv(81), 8), 20), 17), 19),
    )
    a079 = bool10(
        rank(delta(Neutralize(data.close * 0.60733 + data.open * 0.39267, data.sector), 1))
        < rank(corr(ts_rank(data.vwap, 4), ts_rank(adv(150), 9), 15))
    )
    a080 = 0 - power(
        rank(Sign(delta(Neutralize(data.open * 0.868128 + data.high * 0.131872, data.industry), 4))),
        ts_rank(corr(data.high, adv(10), 5), 6),
    )
    a082 = 0 - min_op(
        rank(decay(delta(data.open, 1), 15)),
        ts_rank(decay(corr(Neutralize(data.volume, data.sector), data.open, 17), 7), 13),
        s,
    )
    a087 = 0 - Max(
        rank(decay(delta(data.close * 0.369701 + data.vwap * 0.630299, 2), 3)),
        ts_rank(decay(Abs(corr(Neutralize(adv(81), data.industry), data.close, 13)), 5), 14),
    )
    a089 = ts_rank(decay(corr(data.low, adv(10), 7), 6), 4) - ts_rank(
        decay(delta(Neutralize(data.vwap, data.industry), 3), 10), 15
    )
    a090 = 0 - power(
        rank(data.close - WindowedMax(data.close, 5)),
        ts_rank(corr(Neutralize(adv(40), data.subindustry), data.low, 5), 3),
    )
    a091 = 0 - (
        ts_rank(decay(decay(corr(Neutralize(data.close, data.industry), data.volume, 10), 16), 4), 5)
        - rank(decay(corr(data.vwap, adv(30), 4), 3))
    )
    a093 = ts_rank(decay(corr(Neutralize(data.vwap, data.industry), adv(81), 17), 20), 8) / rank(
        decay(delta(data.close * 0.524434 + data.vwap * 0.475566, 3), 16)
    )
    a097 = rank(decay(delta(Neutralize(data.low * 0.721001 + data.vwap * 0.278999, data.industry), 3), 20)) - ts_rank(
        decay(ts_rank(corr(ts_rank(data.low, 8), ts_rank(adv(60), 17), 5), 19), 16), 7
    )
    adv20 = adv(20)
    ranked_pressure = rank(
        (((data.close - data.low) - (data.high - data.close)) / (data.high - data.low)) * data.volume
    )
    term1 = 1.5 * Scale(Neutralize(Neutralize(ranked_pressure, data.subindustry), data.subindustry))
    term2 = Scale(Neutralize(corr(data.close, rank(adv20), 5) - rank(TsArgMin(data.close, 30)), data.subindustry))
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


def min_op(left: Any, right: Any, symbols: dict[str, Any]) -> Any:
    try:
        from KunQuant.ops import Min
    except ImportError as exc:
        from rquant.errors import DependencyError

        raise DependencyError("KunQuant==0.1.11 is required to build Alpha101") from exc
    return Min(left, right)
