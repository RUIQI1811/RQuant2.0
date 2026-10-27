"""The technical indicators of Tushare's ``stk_factor_pro`` endpoint, re-implemented as a daily panel factor set.

Tushare publishes, per stock and day, about thirty classic Chinese-terminal indicators (ASI, BBI, BIAS, BOLL, BRAR,
CCI, CR, DMA, DMI, DPO, EMA, EMV, EXPMA, KDJ, KTN, MA, MACD, MASS, MFI, MTM, OBV, PSY, ROC, RSI, TAQ, TRIX, VR, WR,
XSII) with the parameters listed in its documentation, plus streak counts. The endpoint itself needs more Tushare
points than this account has, so every indicator is computed here from the standard TongDaXin formula with
Tushare's parameters. Tushare's values are not used. The indicators run in KunQuant as C++ series kernels
(``CPP_PRELUDE``); the original pandas implementation is kept as the parity reference in ``tests/panel_reference``.

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

from collections.abc import Mapping
from typing import Any

from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.libraries.base import FactorLibrary, FactorSpec

FIELDS = ("open", "high", "low", "close", "volume")
FLOW_DAYS = 20
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


# C++ helpers over one stock's consecutive quotes. Every operator keeps the pandas semantics of the reference
# implementation (tests/panel_reference/tushare_tech.py): rolling windows need ``n`` known values, ``ewm`` is pandas'
# ``adjust=False`` exponential mean that carries its value over missing rows, comparisons with a missing value are
# false, and divisions follow IEEE arithmetic until the published value is checked for finiteness.
CPP_PRELUDE = r"""
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace rq_tst {
using V = std::vector<double>;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

inline V filled(size_t n, double value) { return V(n, value); }

template <class F> inline V map1(const V &a, F f) {
    V r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = f(a[i]);
    return r;
}
template <class F> inline V map2(const V &a, const V &b, F f) {
    V r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = f(a[i], b[i]);
    return r;
}
template <class F> inline V map3(const V &a, const V &b, const V &c, F f) {
    V r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = f(a[i], b[i], c[i]);
    return r;
}
inline V add(const V &a, const V &b) { return map2(a, b, [](double x, double y) { return x + y; }); }
inline V sub(const V &a, const V &b) { return map2(a, b, [](double x, double y) { return x - y; }); }
inline V mul(const V &a, const V &b) { return map2(a, b, [](double x, double y) { return x * y; }); }
inline V div(const V &a, const V &b) { return map2(a, b, [](double x, double y) { return x / y; }); }
inline V scale(const V &a, double k) { return map1(a, [k](double x) { return x * k; }); }
inline V offset(const V &a, double k) { return map1(a, [k](double x) { return x + k; }); }
inline V absv(const V &a) { return map1(a, [](double x) { return std::fabs(x); }); }
inline V clip0(const V &a) { return map1(a, [](double x) { return x != x ? x : (x < 0.0 ? 0.0 : x); }); }

inline V ref(const V &x, int n) {
    V r(x.size(), NaN);
    for (size_t i = n; i < x.size(); ++i) r[i] = x[i - n];
    return r;
}
// Rolling sum over a complete window of n known values.
inline V total(const V &x, int n) {
    V r(x.size(), NaN);
    for (size_t i = n - 1; i < x.size() && n > 0; ++i) {
        double s = 0.0;
        bool ok = true;
        for (size_t j = i + 1 - n; j <= i; ++j) {
            if (x[j] != x[j]) { ok = false; break; }
            s += x[j];
        }
        if (ok) r[i] = s;
    }
    return r;
}
inline V ma(const V &x, int n) { return scale(total(x, n), 1.0 / n); }
inline V extreme(const V &x, int n, bool maximum) {
    V r(x.size(), NaN);
    for (size_t i = n - 1; i < x.size(); ++i) {
        double e = x[i + 1 - n];
        bool ok = e == e;
        for (size_t j = i + 2 - n; j <= i && ok; ++j) {
            if (x[j] != x[j]) ok = false;
            else if (maximum ? x[j] > e : x[j] < e) e = x[j];
        }
        if (ok) r[i] = e;
    }
    return r;
}
inline V hhv(const V &x, int n) { return extreme(x, n, true); }
inline V llv(const V &x, int n) { return extreme(x, n, false); }
inline V std0(const V &x, int n) {
    V r(x.size(), NaN);
    for (size_t i = n - 1; i < x.size(); ++i) {
        double s = 0.0;
        bool ok = true;
        for (size_t j = i + 1 - n; j <= i; ++j) {
            if (x[j] != x[j]) { ok = false; break; }
            s += x[j];
        }
        if (!ok) continue;
        const double m = s / n;
        double q = 0.0;
        for (size_t j = i + 1 - n; j <= i; ++j) q += (x[j] - m) * (x[j] - m);
        r[i] = std::sqrt(q / n);
    }
    return r;
}
inline V avedev(const V &x, int n) {
    V r(x.size(), NaN);
    for (size_t i = n - 1; i < x.size(); ++i) {
        double s = 0.0;
        for (size_t j = i + 1 - n; j <= i; ++j) s += x[j];
        const double m = s / n;
        double d = 0.0;
        for (size_t j = i + 1 - n; j <= i; ++j) d += std::fabs(x[j] - m);
        r[i] = d / n;
    }
    return r;
}
// pandas ``ewm(alpha=alpha, adjust=False).mean()``: starts at the first known value and carries over missing rows.
inline V ewm(const V &x, double alpha) {
    V r(x.size(), NaN);
    if (x.empty()) return r;
    double weighted = x[0];
    double old_wt = 1.0;
    const double factor = 1.0 - alpha;
    r[0] = weighted;
    for (size_t i = 1; i < x.size(); ++i) {
        const double cur = x[i];
        const bool observation = cur == cur;
        if (weighted == weighted) {
            old_wt *= factor;
            if (observation) {
                if (weighted != cur) weighted = (old_wt * weighted + alpha * cur) / (old_wt + alpha);
                old_wt = 1.0;
            }
        } else if (observation) {
            weighted = cur;
        }
        r[i] = weighted;
    }
    return r;
}
inline V ema(const V &x, int n) { return ewm(x, 2.0 / (n + 1.0)); }
inline V sma(const V &x, int n, int m) { return ewm(x, (double)m / n); }
inline V dma(const V &x, const V &weight) {
    V r(x.size(), NaN);
    double previous = NaN;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::isfinite(weight[i]) && std::isfinite(x[i])) {
            const double blended = weight[i] * x[i] + (1.0 - weight[i]) * previous;
            previous = previous != previous ? x[i] : blended;
        }
        r[i] = previous;
    }
    return r;
}
inline V streak(const V &condition) {
    V r(condition.size());
    double run = 0.0;
    for (size_t i = 0; i < condition.size(); ++i) {
        run = condition[i] > 0.5 ? run + 1.0 : 0.0;
        r[i] = run;
    }
    return r;
}
inline V range_days(const V &x, bool higher) {
    V r(x.size());
    std::vector<size_t> stack;
    for (size_t i = 0; i < x.size(); ++i) {
        while (!stack.empty() && (higher ? x[stack.back()] <= x[i] : x[stack.back()] >= x[i])) stack.pop_back();
        r[i] = stack.empty() ? (double)i : (double)(i - stack.back() - 1);
        stack.push_back(i);
    }
    return r;
}
inline V true_range(const V &h, const V &l, const V &lc) {
    return map3(h, l, lc, [](double hi, double lo, double pc) {
        const double a = hi - lo, b = std::fabs(pc - hi), c = std::fabs(pc - lo);
        if (a != a || b != b || c != c) return NaN;
        return std::max(a, std::max(b, c));
    });
}
inline V over_close(const V &x, const V &c) { return div(x, c); }
inline V rel_close(const V &x, const V &c) { return offset(div(x, c), -1.0); }

// The published series of one indicator for a stock's quotes.
inline V indicator(const std::string &name, const V &o, const V &h, const V &l, const V &c, const V &v) {
    const size_t m = c.size();
    const V lc = ref(c, 1);
    const V change = sub(c, lc);
    if (name == "asi" || name == "asit") {
        const V aa = absv(sub(h, lc)), bb = absv(sub(l, lc));
        const V cc = absv(sub(h, ref(l, 1))), dd = absv(sub(lc, ref(o, 1)));
        V si(m, NaN);
        const V po = ref(o, 1);
        for (size_t i = 0; i < m; ++i) {
            const double r = (aa[i] > bb[i] && aa[i] > cc[i]) ? aa[i] + bb[i] / 2 + dd[i] / 4
                             : ((bb[i] > cc[i] && bb[i] > aa[i]) ? bb[i] + aa[i] / 2 + dd[i] / 4 : cc[i] + dd[i] / 4);
            const double x = c[i] - lc[i] + (c[i] - o[i]) / 2 + lc[i] - po[i];
            const double range = r > 0 ? r : NaN;
            const double larger = (aa[i] != aa[i] || bb[i] != bb[i]) ? NaN : std::max(aa[i], bb[i]);
            si[i] = 16 * x / range * larger;
        }
        const V asi = total(si, 26);
        return name == "asi" ? over_close(asi, c) : over_close(ma(asi, 10), c);
    }
    const V tr = true_range(h, l, lc);
    if (name == "atr") return over_close(ma(tr, 20), c);
    if (name == "bbi") return rel_close(scale(add(add(ma(c, 3), ma(c, 6)), add(ma(c, 12), ma(c, 24))), 0.25), c);
    if (name == "bias1" || name == "bias2" || name == "bias3") {
        const int n = name == "bias1" ? 6 : (name == "bias2" ? 12 : 24);
        const V mean = ma(c, n);
        return scale(div(sub(c, mean), mean), 100.0);
    }
    if (name == "boll_upper" || name == "boll_lower") {
        const V mid = ma(c, 20), spread = std0(c, 20);
        return rel_close(name == "boll_upper" ? add(mid, scale(spread, 2)) : sub(mid, scale(spread, 2)), c);
    }
    if (name == "brar_ar") return scale(div(total(sub(h, o), 26), total(sub(o, l), 26)), 100.0);
    if (name == "brar_br") return scale(div(total(clip0(sub(h, lc)), 26), total(clip0(sub(lc, l)), 26)), 100.0);
    const V typical = map3(h, l, c, [](double hi, double lo, double cl) { return (hi + lo + cl) / 3.0; });
    if (name == "cci") return div(sub(typical, ma(typical, 14)), scale(avedev(typical, 14), 0.015));
    if (name == "cr") {
        const V mid = scale(ref(add(h, l), 1), 0.5);
        return scale(div(total(clip0(sub(h, mid)), 20), total(clip0(sub(mid, l)), 20)), 100.0);
    }
    if (name == "dfma_dif" || name == "dfma_difma") {
        const V dif = sub(ma(c, 10), ma(c, 50));
        return over_close(name == "dfma_dif" ? dif : ma(dif, 10), c);
    }
    if (name.rfind("dmi_", 0) == 0) {
        const V moving = total(tr, 14);
        const V up = sub(h, ref(h, 1)), down = sub(ref(l, 1), l);
        auto directional = [](double a, double b) { return a != a ? NaN : ((a > 0 && a > b) ? a : 0.0); };
        const V plus = total(map2(up, down, directional), 14);
        const V minus = total(map2(down, up, directional), 14);
        const V pdi = div(scale(plus, 100.0), moving), mdi = div(scale(minus, 100.0), moving);
        if (name == "dmi_pdi") return pdi;
        if (name == "dmi_mdi") return mdi;
        const V adx = ma(scale(div(absv(sub(mdi, pdi)), add(mdi, pdi)), 100.0), 6);
        if (name == "dmi_adx") return adx;
        return scale(add(adx, ref(adx, 6)), 0.5);
    }
    if (name == "dpo" || name == "madpo") {
        const V dpo = sub(c, ref(ma(c, 20), 10));
        return over_close(name == "dpo" ? dpo : ma(dpo, 6), c);
    }
    if (name.rfind("ema_", 0) == 0) return rel_close(ema(c, std::stoi(name.substr(4))), c);
    if (name == "emv" || name == "maemv") {
        const V hl = add(h, l), range = sub(h, l);
        const V ratio = div(ma(v, 14), v);
        const V mid = div(scale(sub(hl, ref(hl, 1)), 100.0), hl);
        const V emv = ma(div(mul(mul(mid, ratio), range), ma(range, 14)), 14);
        return name == "emv" ? emv : ma(emv, 9);
    }
    if (name == "expma_12") return rel_close(ema(c, 12), c);
    if (name == "expma_50") return rel_close(ema(c, 50), c);
    if (name == "kdj_k" || name == "kdj_d" || name == "kdj") {
        const V low = llv(l, 9);
        const V rsv = scale(div(sub(c, low), sub(hhv(h, 9), low)), 100.0);
        const V k = sma(rsv, 3, 1);
        if (name == "kdj_k") return k;
        const V d = sma(k, 3, 1);
        if (name == "kdj_d") return d;
        return sub(scale(k, 3.0), scale(d, 2.0));
    }
    if (name.rfind("ktn_", 0) == 0) {
        const V mid = ema(typical, 20), width = scale(ema(tr, 10), 2.0);
        if (name == "ktn_mid") return rel_close(mid, c);
        return rel_close(name == "ktn_upper" ? add(mid, width) : sub(mid, width), c);
    }
    if (name.rfind("ma_", 0) == 0 && name != "ma_mass") return rel_close(ma(c, std::stoi(name.substr(3))), c);
    if (name == "macd_dif" || name == "macd_dea" || name == "macd") {
        const V dif = sub(ema(c, 12), ema(c, 26));
        if (name == "macd_dif") return over_close(dif, c);
        const V dea = ema(dif, 9);
        if (name == "macd_dea") return over_close(dea, c);
        return over_close(scale(sub(dif, dea), 2.0), c);
    }
    if (name == "mass" || name == "ma_mass") {
        const V range = ma(sub(h, l), 9);
        const V mass = total(div(range, ma(range, 9)), 25);
        return name == "mass" ? mass : ma(mass, 6);
    }
    if (name == "mfi") {
        const V previous = ref(typical, 1);
        V in(m), out(m);
        for (size_t i = 0; i < m; ++i) {
            const double flow = typical[i] * v[i];
            const bool known = previous[i] == previous[i];
            in[i] = known ? (typical[i] > previous[i] ? flow : 0.0) : NaN;
            out[i] = known ? (typical[i] < previous[i] ? flow : 0.0) : NaN;
        }
        const V ratio = div(total(in, 14), total(out, 14));
        return map1(ratio, [](double r) { return 100.0 - 100.0 / (1.0 + r); });
    }
    if (name == "mtm" || name == "mtmma") {
        const V momentum = sub(c, ref(c, 12));
        return over_close(name == "mtm" ? momentum : ma(momentum, 6), c);
    }
    if (name == "obv") {
        V line(m);
        double running = 0.0;
        for (size_t i = 0; i < m; ++i) {
            const double d = change[i];
            if (d == d) running += (d > 0 ? 1.0 : (d < 0 ? -1.0 : 0.0)) * v[i];
            line[i] = running;
        }
        return div(sub(line, ref(line, 20)), total(v, 20));
    }
    if (name == "psy" || name == "psyma") {
        const V up = map1(change, [](double d) { return d != d ? NaN : (d > 0 ? 1.0 : 0.0); });
        const V psy = scale(total(up, 12), 100.0 / 12.0);
        return name == "psy" ? psy : ma(psy, 6);
    }
    if (name == "roc" || name == "maroc") {
        const V base = ref(c, 12);
        const V roc = div(scale(sub(c, base), 100.0), base);
        return name == "roc" ? roc : ma(roc, 6);
    }
    if (name.rfind("rsi_", 0) == 0) {
        const int n = std::stoi(name.substr(4));
        return scale(div(sma(clip0(change), n, 1), sma(absv(change), n, 1)), 100.0);
    }
    if (name.rfind("taq_", 0) == 0) {
        const V top = hhv(h, 20), bottom = llv(l, 20);
        if (name == "taq_up") return rel_close(top, c);
        if (name == "taq_down") return rel_close(bottom, c);
        return rel_close(scale(add(top, bottom), 0.5), c);
    }
    if (name == "trix" || name == "trma") {
        const V triple = ema(ema(ema(c, 12), 12), 12);
        const V previous = ref(triple, 1);
        const V trix = div(scale(sub(triple, previous), 100.0), previous);
        return name == "trix" ? trix : ma(trix, 20);
    }
    if (name == "vr") {
        auto part = [&](int sign) {
            return map2(change, v, [sign](double d, double vol) {
                if (d != d) return NaN;
                return (sign > 0 ? d > 0 : (sign < 0 ? d < 0 : d == 0)) ? vol : 0.0;
            });
        };
        const V up = total(part(1), 26), down = total(part(-1), 26), flat = total(part(0), 26);
        return div(scale(add(scale(up, 2.0), flat), 100.0), add(scale(down, 2.0), flat));
    }
    if (name == "wr" || name == "wr1") {
        const int n = name == "wr" ? 10 : 6;
        const V top = hhv(h, n);
        return div(scale(sub(top, c), 100.0), sub(top, llv(l, n)));
    }
    const V weighted = scale(add(add(scale(c, 2.0), h), l), 0.25);
    if (name == "xsii_td1") return rel_close(scale(ma(weighted, 5), 102.0 / 100.0), c);
    if (name == "xsii_td3") {
        const V ma20 = ma(c, 20);
        return rel_close(scale(dma(c, div(absv(sub(weighted, ma20)), ma20)), 1.0 + 7.0 / 100.0), c);
    }
    if (name == "updays") return streak(map1(change, [](double d) { return d > 0 ? 1.0 : 0.0; }));
    if (name == "downdays") return streak(map1(change, [](double d) { return d < 0 ? 1.0 : 0.0; }));
    if (name == "topdays") return range_days(h, true);
    if (name == "lowdays") return range_days(l, false);
    return V(m, NaN);
}
}  // namespace rq_tst
"""


class TushareTechLibrary(FactorLibrary):
    """Tushare stk_factor_pro technical indicators, evaluated in KunQuant over every stock's own trading days."""

    family = "tushare_tech"
    required_inputs = FIELDS
    quote_fields = FIELDS
    strict_float = True
    cpp_prelude = CPP_PRELUDE

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
                implementation=f"rquant.factors.libraries.tushare_tech.CPP_PRELUDE.{name}",
                catalog_version=2,
            )
            for ordinal, (name, description, lookback) in enumerate(TUSHARE_TECH, 1)
        )

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        fields = [inputs[field] for field in FIELDS]
        return {name: k.series_kernel(fields, _kernel(name)) for name in TUSHARE_TECH_NAMES}


def _kernel(name: str) -> str:
    """C++ body publishing one indicator over a stock's quotes, which the engine has moved to the top rows."""
    return f"""
        int m = 0;
        while (m < n && std::isfinite(x0[m]) && std::isfinite(x1[m]) && std::isfinite(x2[m]) && std::isfinite(x3[m])
               && std::isfinite(x4[m])) ++m;
        if (m > 0) {{
            const rq_tst::V o(x0, x0 + m), h(x1, x1 + m), l(x2, x2 + m), c(x3, x3 + m), v(x4, x4 + m);
            const rq_tst::V values = rq_tst::indicator("{name}", o, h, l, c, v);
            for (int t = 0; t < m; ++t) {{
                out[t] = t < {LOOKBACK[name]} || !std::isfinite(values[t]) ? rq_tst::NaN : values[t];
            }}
        }}"""
