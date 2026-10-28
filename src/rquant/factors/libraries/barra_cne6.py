"""Barra China Equity Model (CNE6) style descriptors evaluated as daily point-in-time panels.

The catalog follows the descriptor definitions MSCI published for the CNE6 model (size, beta, residual volatility,
liquidity, momentum, long-term reversal, value, earnings yield, leverage, earnings variability, earnings quality,
profitability, investment quality, growth and dividend yield). This module is an independent implementation from
those public definitions: no MSCI code or data is used and inputs come from Tushare. It publishes the descriptors,
not the composite style factors, so a model can weight them itself.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days and a year 252.
* Returns are close-to-close returns of RQuant's backward-adjusted prices; the day after a suspension has none. The
  market return is the value-weighted all-A series of ``reference/market_factors.parquet``. Returns are not reduced
  by a risk-free rate: within the trailing windows used here a constant rate only moves the intercepts.
* Exponentially weighted windows give the newest row weight 1 and halve it every ``half-life`` rows. Lagged
  descriptors (STREV, RSTR, LTRSTR, LTHALPHA) average the raw value over the documented lags.
* Financial statements are the first-reported values of ``reference/fundamentals.parquet`` on the quarter grid of
  ``china_chars``: a quarter is usable once it and every earlier quarter are announced. Flow-to-price and flow-to-assets
  ratios use trailing-twelve-month flows; five-year descriptors use the last five fiscal-year (Q4) values and need at
  least three of them.
* The two cross-sectional descriptors (MIDCAP, INDMOM) use only the research universe's members of that day (the
  CSI 300 by default), so later index members do not shape earlier cross-sections.
* Every output is missing on days the stock has no quote.
* Statement formulas run in a KunQuant graph over fiscal quarters and reach the daily graph from their first usable
  announcement day; exponentially weighted windows, regressions and the cross-sectional steps are per-stock or
  per-day C++ kernels. The original pandas implementation is the parity reference in ``tests/panel_reference``.

Left out: the analyst-forecast descriptors EPIBS, EGRLF, ETOPF_STD and DPIBS (Tushare's forecast endpoint allows one
call per minute at this account level).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from rquant.errors import DataContractError
from rquant.factors import statements
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.extensions.kunquant_oap import g
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.factors.libraries.china_chars import CPP_PRELUDE as CNC_PRELUDE
from rquant.io import sha256_file, stable_hash

MONTH = 21
YEAR = 252
LONG_WINDOW = 1040
LONG_LAG = 273
FISCAL_YEARS = 5
CNE6_DAILY_BASIC = {"turnover_rate": 0.01, "dv_ttm": 0.01}

# (source name, description, maximum lookback in trading days). Order is the stable public column order.
CNE6_DESCRIPTORS: tuple[tuple[str, str, int], ...] = (
    # Size
    ("lncap", "LNCAP: log total market value", 1),
    ("midcap", "MIDCAP: cube of standardised LNCAP, orthogonal to it, winsorised and standardised", 1),
    # Beta and residual volatility
    ("hbeta", "HBETA: market beta, 252 days, half-life 63", YEAR),
    ("hsigma", "HSIGMA: residual volatility of the HBETA regression", YEAR),
    ("dastd", "DASTD: daily return volatility, 252 days, half-life 42", YEAR),
    ("cmra", "CMRA: range of cumulative log returns over the last 1..12 months", YEAR),
    # Liquidity
    ("stom", "STOM: log one-month share turnover", MONTH),
    ("stoq", "STOQ: log average monthly turnover over three months", 3 * MONTH),
    ("stoa", "STOA: log average monthly turnover over twelve months", YEAR),
    ("atvr", "ATVR: turnover summed over 252 days, half-life 63", YEAR),
    # Momentum
    ("strev", "STREV: log return over 21 days, half-life 5, averaged over lags 1..3", MONTH + 3),
    ("season", "SEASON: average next-month return in each of the prior five years", 5 * YEAR),
    ("indmom", "INDMOM: sqrt-cap-weighted industry relative strength minus the stock's, 126 days, half-life 21",
     126),
    ("rstr", "RSTR: log return over 252 days, half-life 126, averaged over lags 11..21", YEAR + 21),
    # Long-term reversal
    ("ltrstr", "LTRSTR: minus the log return over 1040 days, half-life 260, averaged over lags 273..283",
     LONG_WINDOW + LONG_LAG + 10),
    ("lthalpha", "LTHALPHA: minus the market-model alpha over 1040 days, half-life 260, averaged over lags 273..283",
     LONG_WINDOW + LONG_LAG + 10),
    # Value and earnings yield
    ("btop", "BTOP: common equity over total market value", 1),
    ("etop", "ETOP: trailing net income to parent over total market value", 1),
    ("cetop", "CETOP: trailing operating cash flow over total market value", 1),
    ("em", "EM: trailing EBIT over enterprise value", 1),
    # Leverage
    ("mlev", "MLEV: market value plus preferred equity and long-term debt, over market value", 1),
    ("blev", "BLEV: common equity plus preferred equity and long-term debt, over common equity", 1),
    ("dtoa", "DTOA: total liabilities over total assets", 1),
    # Earnings variability
    ("vsal", "VSAL: std over mean of the last five fiscal-year revenues", 1),
    ("vern", "VERN: std over mean absolute value of the last five fiscal-year net incomes to parent", 1),
    ("vflo", "VFLO: std over mean absolute value of the last five fiscal-year net cash changes", 1),
    # Earnings quality
    ("abs", "ABS: minus balance-sheet accruals (change in net operating assets less D&A) over total assets", 1),
    ("acf", "ACF: minus cash-flow accruals (net income less operating and investing cash flow, plus D&A) over total "
     "assets", 1),
    # Profitability
    ("ato", "ATO: trailing revenue over total assets", 1),
    ("gp", "GP: trailing gross profit over total assets", 1),
    ("gpm", "GPM: trailing gross profit over trailing revenue", 1),
    ("roa", "ROA: trailing net income to parent over total assets", 1),
    # Investment quality
    ("agro", "AGRO: minus the five-year slope of fiscal-year total assets over their mean", 1),
    ("igro", "IGRO: minus the five-year slope of fiscal-year total shares over their mean", 1),
    ("cxgro", "CXGRO: minus the five-year slope of fiscal-year capital expenditure over its mean", 1),
    # Growth
    ("egro", "EGRO: five-year slope of fiscal-year earnings per share over its mean absolute value", 1),
    ("sgro", "SGRO: five-year slope of fiscal-year revenue per share over its mean", 1),
    # Dividend yield
    ("dtop", "DTOP: trailing twelve-month dividend yield", 1),
)
CNE6_NAMES = tuple(name for name, _, _ in CNE6_DESCRIPTORS)

STATEMENT_FIELDS = (
    "revenue",
    "oper_cost",
    "fin_exp",
    "total_profit",
    "n_income_attr_p",
    "n_cashflow_act",
    "n_cashflow_inv_act",
    "n_incr_cash_cash_equ",
    "c_pay_acq_const_fiolta",
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
    "total_share",
    "money_cap",
    "total_assets",
    "total_liab",
    "st_borr",
    "non_cur_liab_due_1y",
    "lt_borr",
    "bond_payable",
    "oth_eqt_tools_p_shr",
    "total_hldr_eqy_exc_min_int",
)
# Descriptors published unchanged from a statement-graph result (``q_`` latest quarter, ``y_`` latest fiscal year).
STATEMENT_FACTORS = {
    **{name: f"q_{name}" for name in ("blev", "dtoa", "abs", "acf", "ato", "gp", "gpm", "roa")},
    **{name: f"y_{name}" for name in ("vsal", "vern", "vflo", "agro", "igro", "cxgro", "egro", "sgro")},
}
# Statement results the daily graph combines with the market value.
STATEMENT_INPUTS = ("q_book", "q_earnings_ttm", "q_cash_ttm", "q_ebit_ttm", "q_net_debt", "q_other_capital")
DAILY_REFERENCE = ("basic_turnover_rate", "basic_dv_ttm", "mkt", "universe")

# Exponentially weighted window sums, kept per stock with the recursion of the pandas reference:
# ``S_t = d S_{t-1} + x_t - d^window x_{t-window}``; counts use ``d = 1``.
CPP_PRELUDE = r"""
#ifndef RQUANT_CNE6_CPP_HELPERS
#define RQUANT_CNE6_CPP_HELPERS
namespace rq_cne6 {
struct EwSum {
    double decay = 1.0, tail = 1.0, running = 0.0;
    int window = 1;
    std::vector<double> history;
    void setup(int w, double half_life) {
        window = w;
        decay = half_life > 0 ? std::pow(0.5, 1.0 / half_life) : 1.0;
        tail = std::pow(decay, (double)w);
        running = 0.0;
        history.clear();
    }
    double push(double x) {
        history.push_back(x);
        const size_t row = history.size() - 1;
        running = decay * running + x;
        if (row >= (size_t)window) running = running - tail * history[row - window];
        return running;
    }
};
}  // namespace rq_cne6
#endif
"""

# Weighted market regression: sums of the weight, x, y, xx, xy and yy plus the plain count of known pairs.
_FIT_MEMBERS = "rq_cne6::EwSum w, x, y, xx, xy, yy, n; bool ready;"
_FIT_SETUP = (
    "if (!s.ready) {{ s.w.setup({window}, {half_life}); s.x.setup({window}, {half_life}); "
    "s.y.setup({window}, {half_life}); s.xx.setup({window}, {half_life}); s.xy.setup({window}, {half_life}); "
    "s.yy.setup({window}, {half_life}); s.n.setup({window}, 0.0); s.ready = true; }}"
)
_FIT_STEP = """
    const bool known = std::isfinite(x0) && std::isfinite(x1);
    const double ry = known ? x0 : 0.0, rx = known ? x1 : 0.0, rw = known ? 1.0 : 0.0;
    const double sw = s.w.push(rw), sx = s.x.push(rx), sy = s.y.push(ry);
    const double sxx = s.xx.push(rx * rx), sxy = s.xy.push(rx * ry), syy = s.yy.push(ry * ry);
    const double count = s.n.push(rw);
    const double mean_x = sx / sw, mean_y = sy / sw;
    const double var_x = sxx / sw - mean_x * mean_x, var_y = syy / sw - mean_y * mean_y;
    const double cov = sxy / sw - mean_x * mean_y;
    const double beta = cov / var_x, alpha = mean_y - beta * mean_x, residual = var_y - beta * cov;
    const bool valid = count > {minimum} - 0.5 && var_x > 1e-16;
    y = NAN;
    if (valid) {{ {publish} }}
"""
_FIT_PUBLISH = {
    "beta": "y = beta;",
    "alpha": "y = alpha;",
    "resid_std": "if (residual > 0) y = std::sqrt(std::max(residual, 0.0));",
}


def ew_market_fit(returns, market, window: int, half_life: float, minimum: int, output: str):
    step = _FIT_SETUP.format(window=window, half_life=half_life) + _FIT_STEP.format(
        minimum=minimum, publish=_FIT_PUBLISH[output]
    )
    return op.state([returns, market], step, _FIT_MEMBERS, "s.ready = false;")


def ew_log_return(log_returns, window: int, half_life: float, minimum: int):
    """Exponentially weighted sum of the known log returns, given at least ``minimum`` of them in the window."""
    step = f"""
    if (!s.ready) {{ s.r.setup({window}, {half_life}); s.n.setup({window}, 0.0); s.ready = true; }}
    const bool known = std::isfinite(x0);
    const double total = s.r.push(known ? x0 : 0.0), count = s.n.push(known ? 1.0 : 0.0);
    y = count > {minimum} - 0.5 ? total : NAN;"""
    return op.state([log_returns], step, "rq_cne6::EwSum r, n; bool ready;", "s.ready = false;")


def lag_average(value, first: int, last: int):
    """Mean of ``value`` lagged ``first`` through ``last`` rows; missing unless every lag is known."""
    return sum(lag(value, periods) for periods in range(first, last + 1)) / float(last - first + 1)


def five_year_slope(values, *, absolute_mean: bool = False):
    """OLS slope of the last five fiscal-year values on time over their mean (or mean absolute value)."""
    stacked = [lag(values, 4 * (FISCAL_YEARS - 1 - k)) for k in range(FISCAL_YEARS)]
    known = [g.isfinite(value) for value in stacked]
    count = sum(known)
    times = [float(k + 1) for k in range(FISCAL_YEARS)]
    mean_t = sum(g.where(f, t, 0.0) for f, t in zip(known, times, strict=True)) / count
    mean_v = sum(g.where(f, v, 0.0) for f, v in zip(known, stacked, strict=True)) / count
    centered = [g.where(f, t - mean_t, 0.0) for f, t in zip(known, times, strict=True)]
    covariance = sum(c * g.where(f, v - mean_v, 0.0) for c, f, v in zip(centered, known, stacked, strict=True))
    slope = covariance / sum(c * c for c in centered)
    scale = sum(g.where(f, g.abs(v), 0.0) for f, v in zip(known, stacked, strict=True)) / count if absolute_mean \
        else mean_v
    return g.where(count >= 3, ratio(slope, scale), g.nan)


def five_year_variation(values):
    """Sample std of the last five fiscal-year values over their mean absolute value; at least three years."""
    stacked = [lag(values, 4 * k) for k in range(FISCAL_YEARS)]
    known = [g.isfinite(value) for value in stacked]
    count = sum(known)
    mean = sum(g.where(f, v, 0.0) for f, v in zip(known, stacked, strict=True)) / count
    squares = sum(g.where(f, (v - mean) * (v - mean), 0.0) for f, v in zip(known, stacked, strict=True))
    std = g.sqrt(squares / (count - 1))
    scale = sum(g.where(f, g.abs(v), 0.0) for f, v in zip(known, stacked, strict=True)) / count
    return g.where(count >= 3, ratio(std, scale), g.nan)


def positive(value):
    return g.where(value > 0, value, g.nan)


def lag(value, periods: int):
    return op.shift(value, periods)


def ratio(numerator, denominator):
    return op.ratio(numerator, denominator)


def _statement_graph() -> dict[str, Any]:
    """Statement formulas over fiscal quarters (rows) x instruments."""
    quarter = op.input_("fiscal_quarter")

    def value(name: str):
        return op.input_(name)

    def single(name: str):
        ytd = value(name)
        return g.where(quarter == 0, ytd, ytd - lag(ytd, 1))

    def ttm(name: str):
        flow = single(name)
        return flow + lag(flow, 1) + lag(flow, 2) + lag(flow, 3)

    def annual(name: str):
        return g.where(quarter == 3, value(name), g.nan)

    def fiscal_year(node):
        return g.where(quarter == 3, node, g.nan)

    assets = value("total_assets")
    equity = value("total_hldr_eqy_exc_min_int") - value("oth_eqt_tools_p_shr")
    long_term_debt = value("lt_borr") + value("bond_payable")
    total_debt = value("st_borr") + value("non_cur_liab_due_1y") + long_term_debt
    depreciation = ttm("depr_fa_coga_dpba") + ttm("amort_intang_assets") + ttm("lt_amort_deferred_exp")
    revenue = ttm("revenue")
    gross = revenue - ttm("oper_cost")
    shares = fiscal_year(value("total_share"))

    def per_share(flow: str):
        return ratio(annual(flow), positive(shares))

    operating = (assets - value("money_cap")) - (value("total_liab") - total_debt)
    out = {
        "q_book": equity,
        "q_earnings_ttm": ttm("n_income_attr_p"),
        "q_cash_ttm": ttm("n_cashflow_act"),
        "q_ebit_ttm": ttm("total_profit") + ttm("fin_exp"),
        "q_net_debt": total_debt - value("money_cap"),
        "q_other_capital": value("oth_eqt_tools_p_shr") + long_term_debt,
        "q_blev": ratio(equity + value("oth_eqt_tools_p_shr") + long_term_debt, positive(equity)),
        "q_dtoa": ratio(value("total_liab"), assets),
        "y_vsal": five_year_variation(annual("revenue")),
        "y_vern": five_year_variation(annual("n_income_attr_p")),
        "y_vflo": five_year_variation(annual("n_incr_cash_cash_equ")),
        "q_abs": -ratio(operating - lag(operating, 4) - depreciation, assets),
        "q_acf": -ratio(
            ttm("n_income_attr_p") - (ttm("n_cashflow_act") + ttm("n_cashflow_inv_act")) + depreciation, assets
        ),
        "q_ato": ratio(revenue, assets),
        "q_gp": ratio(gross, assets),
        "q_gpm": ratio(gross, revenue),
        "q_roa": ratio(ttm("n_income_attr_p"), assets),
        "y_agro": -five_year_slope(fiscal_year(assets)),
        "y_igro": -five_year_slope(fiscal_year(value("total_share"))),
        "y_cxgro": -five_year_slope(annual("c_pay_acq_const_fiolta")),
        "y_egro": five_year_slope(per_share("n_income_attr_p"), absolute_mean=True),
        "y_sgro": five_year_slope(per_share("revenue")),
    }
    return out


def midcap(cap, universe):
    """MIDCAP over the day's universe members: inputs are the positive cap and the membership flag."""
    code = """
    const size_t n = num_stocks;
    std::vector<double> lc(n), size(n, NAN), cube(n, NAN), residual(n, NAN), clipped(n, NAN);
    for (size_t i = 0; i < n; ++i) lc[i] = std::log(input_0[i]);
    auto standardise = [&](const std::vector<double>& v, std::vector<double>& out) {
        double wsum = 0, vsum = 0; std::vector<double> member;
        for (size_t i = 0; i < n; ++i)
            if (input_1[i] > 0 && std::isfinite(v[i]) && std::isfinite(input_0[i])) {
                member.push_back(v[i]); wsum += input_0[i]; vsum += v[i] * input_0[i];
            }
        std::fill(out.begin(), out.end(), NAN);
        if (member.size() < 3) return;
        const double mean = vsum / wsum;
        double plain = 0; for (double z : member) plain += z; plain /= member.size();
        double squares = 0; for (double z : member) squares += (z - plain) * (z - plain);
        const double sd = std::sqrt(squares / (member.size() - 1));
        if (sd > 0) for (size_t i = 0; i < n; ++i) out[i] = (v[i] - mean) / sd;
    };
    standardise(lc, size);
    for (size_t i = 0; i < n; ++i) cube[i] = size[i] * size[i] * size[i];
    {
        double sw = 0, sz = 0, sy = 0; size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            const double w = std::sqrt(input_0[i]);
            if (input_1[i] > 0 && std::isfinite(cube[i]) && std::isfinite(w)) { ++count; sw += w; sz += w * size[i];
                sy += w * cube[i]; }
        }
        if (count >= 3) {
            const double mz = sz / sw, my = sy / sw;
            double szz = 0, szy = 0;
            for (size_t i = 0; i < n; ++i) {
                const double w = std::sqrt(input_0[i]);
                if (input_1[i] > 0 && std::isfinite(cube[i]) && std::isfinite(w)) {
                    szz += w * (size[i] - mz) * (size[i] - mz); szy += w * (size[i] - mz) * (cube[i] - my);
                }
            }
            const double slope = szy / szz, intercept = my - slope * mz;
            for (size_t i = 0; i < n; ++i) residual[i] = cube[i] - intercept - slope * size[i];
        }
    }
    {
        std::vector<double> member;
        for (size_t i = 0; i < n; ++i) if (input_1[i] > 0 && std::isfinite(residual[i])) member.push_back(residual[i]);
        if (member.size() >= 3) {
            double mean = 0; for (double z : member) mean += z; mean /= member.size();
            double squares = 0; for (double z : member) squares += (z - mean) * (z - mean);
            const double sd = std::sqrt(squares / (member.size() - 1));
            const double low = mean - 3 * sd, high = mean + 3 * sd;
            for (size_t i = 0; i < n; ++i)
                clipped[i] = std::isnan(residual[i]) ? NAN : std::min(high, std::max(low, residual[i]));
        }
    }
    std::vector<double> result(n, NAN);
    standardise(clipped, result);
    for (size_t i = 0; i < n; ++i) output_0[i] = result[i];
    """
    return op.cross(code, cap, universe)


def _daily_graph() -> dict[str, Any]:
    close = op.input_("close")
    trading = g.isfinite(close)
    price = op.ffill(close)
    returns = close / lag(close, 1) - 1.0
    log_returns = op.scalar("x0 > -1.0 ? std::log1p(x0) : NAN", returns)
    cap = positive(op.input_("cap"))
    market = op.input_("mkt")
    universe = op.input_("universe")
    turnover = op.input_("basic_turnover_rate")

    def to_price(name: str):
        return ratio(op.input_(name), cap)

    def log_turnover(window: int, minimum: int, months: int):
        total = op.trailing(turnover, window, minimum, "sum") / float(months)
        return g.log(positive(total))

    dastd_code = """
    if (!s.ready) { s.w.setup(252, 42); s.r.setup(252, 42); s.rr.setup(252, 42); s.n.setup(252, 0.0);
        s.ready = true; }
    const bool known = std::isfinite(x0);
    const double r = known ? x0 : 0.0, weight = known ? 1.0 : 0.0;
    const double sw = s.w.push(weight), sr = s.r.push(r), srr = s.rr.push(r * r), count = s.n.push(weight);
    const double mean = sr / sw, variance = srr / sw - mean * mean;
    y = count > 62.5 && variance > 0 ? std::sqrt(std::max(variance, 0.0)) : NAN;"""
    atvr_code = """
    if (!s.ready) { s.v.setup(252, 63); s.n.setup(252, 0.0); s.ready = true; }
    const bool known = std::isfinite(x0);
    const double total = s.v.push(known ? x0 : 0.0), count = s.n.push(known ? 1.0 : 0.0);
    y = count >= 120 ? total : NAN;"""
    cmra_code = """
    const bool known = std::isfinite(x0);
    const double last = s.prefix.empty() ? 0.0 : s.prefix.back();
    const double seen = s.counts.empty() ? 0.0 : s.counts.back();
    s.prefix.push_back(last + (known ? x0 : 0.0));
    s.counts.push_back(seen + (known ? 1.0 : 0.0));
    const int rows = (int)s.prefix.size();
    y = NAN;
    if (rows >= 252) {
        const double observed = s.counts[rows - 1] - (rows > 252 ? s.counts[rows - 253] : 0.0);
        if (observed >= 200) {
            double high = -INFINITY, low = INFINITY;
            for (int months = 1; months <= 12; ++months) {
                const int days = 21 * months;
                const double z = s.prefix[rows - 1] - (rows > days ? s.prefix[rows - 1 - days] : 0.0);
                high = std::max(high, z); low = std::min(low, z);
            }
            y = std::log1p(high) - std::log1p(low);
        }
    }"""
    season_terms = [lag(price, YEAR * k - MONTH) / lag(price, YEAR * k) - 1.0 for k in range(1, 6)]
    season_known = [g.isfinite(term) for term in season_terms]
    season_count = sum(season_known)
    season = sum(g.where(f, t, 0.0) for f, t in zip(season_known, season_terms, strict=True)) / season_count
    strength = ew_log_return(log_returns, 126, 21, 60)
    industry = op.group_stat(strength, op.input_("sector"), universe, "mean", g.sqrt(cap))
    dividend = op.input_("basic_dv_ttm")
    out = {
        "lncap": g.log(cap),
        "midcap": midcap(cap, universe),
        "hbeta": ew_market_fit(returns, market, YEAR, 63, 63, "beta"),
        "hsigma": ew_market_fit(returns, market, YEAR, 63, 63, "resid_std"),
        "dastd": op.state([returns], dastd_code, "rq_cne6::EwSum w, r, rr, n; bool ready;", "s.ready = false;"),
        "cmra": op.state([log_returns], cmra_code, "std::vector<double> prefix, counts;"),
        "stom": log_turnover(MONTH, 15, 1),
        "stoq": log_turnover(3 * MONTH, 40, 3),
        "stoa": log_turnover(YEAR, 120, 12),
        "atvr": op.state([turnover], atvr_code, "rq_cne6::EwSum v, n; bool ready;", "s.ready = false;"),
        "strev": lag_average(ew_log_return(log_returns, MONTH, 5, 15), 1, 3),
        "season": g.where(season_count >= 2, season, g.nan),
        "indmom": industry - strength,
        "rstr": lag_average(ew_log_return(log_returns, YEAR, 126, 120), 11, 21),
        "ltrstr": -lag_average(ew_log_return(log_returns, LONG_WINDOW, 260, 500), LONG_LAG, LONG_LAG + 10),
        "lthalpha": -lag_average(
            ew_market_fit(returns, market, LONG_WINDOW, 260, 500, "alpha"), LONG_LAG, LONG_LAG + 10
        ),
        "btop": to_price("q_book"),
        "etop": to_price("q_earnings_ttm"),
        "cetop": to_price("q_cash_ttm"),
        "em": ratio(op.input_("q_ebit_ttm"), positive(cap + op.input_("q_net_debt"))),
        "mlev": ratio(cap + op.input_("q_other_capital"), cap),
        "dtop": g.where(
            g.isfinite(turnover) | g.isfinite(dividend), g.where(g.isfinite(dividend), dividend, 0.0), g.nan
        ),
    }
    return {name: g.where(g.isfinite(value) & trading, value, g.nan) for name, value in out.items()}


class BarraCNE6Library(FactorLibrary):
    """Barra CNE6 style descriptors, re-dated to daily point-in-time panels and computed in KunQuant."""

    family = "barra_cne6"
    required_inputs = ("close", "cap", "sector")
    reference_inputs = (*DAILY_REFERENCE, *STATEMENT_INPUTS)
    statement_fields = STATEMENT_FIELDS
    statement_outputs = (*STATEMENT_INPUTS, *STATEMENT_FACTORS.values())
    statement_factors = STATEMENT_FACTORS
    strict_float = True
    cpp_prelude = CNC_PRELUDE + op.CPP_PRELUDE + CPP_PRELUDE

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"cne6_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.barra_cne6.{name}",
                catalog_version=2,
            )
            for ordinal, (name, description, lookback) in enumerate(CNE6_DESCRIPTORS, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Also pin the statement alignment and the shared expression operators, which live in other modules."""
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "statements": inspect.getsource(statements),
                "compiled_prelude": self.cpp_prelude,
            }
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.reference import load_reference_manifest

        canonical_root = Path(canonical_root)
        root = canonical_root / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "barra_cne6 needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in (*daily_files, "market_factors.parquet", "fundamentals.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        wanted = [("instrument", "in", list(instruments))]
        columns = ["instrument", "datetime", *CNE6_DAILY_BASIC]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in CNE6_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale
        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        inputs["mkt"] = factors["mkt"].reindex(dates)
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]
        # Cross-sectional descriptors use the research universe's members of each day, which the engine passes in.
        hashes = {
            "barra_cne6_reference": manifest["fingerprint"],
            "barra_cne6_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "barra_cne6_market_factors": files["market_factors.parquet"],
            "barra_cne6_fundamentals": files["fundamentals.parquet"],
        }
        return inputs, hashes

    def build_statements(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        graph = _statement_graph()
        return {name: op.lower(graph[name], inputs, memo) for name in self.statement_outputs}

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        graph = _daily_graph()
        return {name: op.lower(graph[name], inputs, memo) for name in CNE6_NAMES if name not in STATEMENT_FACTORS}
