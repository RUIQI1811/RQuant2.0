"""China A-share equity characteristics evaluated as daily point-in-time panels.

The catalog follows the characteristic definitions published by Quantactix/ChinaAShareEquityCharacteristics (monthly,
CSMAR-based, after Hou, Xue and Zhang's anomaly list). This module is an independent re-implementation: no upstream
code is used, inputs come from Tushare, and every characteristic is re-dated to the trading day on which it is known.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days, a quarter 63 and a year 252.
* Market return, SMB and HML come from ``reference/market_factors.parquet`` (value-weighted over all A shares). Daily
  returns are not reduced by a risk-free rate; over the 120-day windows used here that only moves the intercept.
* Financial statements are first-reported values. A fiscal quarter becomes usable on the first trading day on or after
  its announcement, and only once every earlier reported quarter is also out, so lagged quarters are never newer
  than the quarter that anchors them. Income and cash-flow items are converted from year-to-date to single quarters.
* Flow-to-price ratios (ep, cfp, ocfp, em) use trailing-twelve-month flows against the current total market value.
* Every output is missing on days the stock has no quote.

Upstream characteristics that cannot be built from Tushare data at this account level are not in the catalog:
employee counts (hn, lfe), advertising expense (adm, gad), CH-4 factor loadings (mkt_beta, vmg_beta, smb_beta,
pmo_beta), q-factor idiosyncratic moments (idvq, idsq) and the unfinished tail-risk measure (tail).

KunQuant computes every formula in two graphs. The statement graph (``build_statements``) runs over fiscal quarters
x instruments; the engine publishes each of its results on the trading days its quarter is the latest known one
(``rquant.factors.statements``). The daily graph (``build``) combines those results with prices and market data; its
rolling statistics and regressions are C++ kernels (``CPP_PRELUDE``) over each stock's whole series. The pandas
reference in ``tests/panel_reference`` is the original implementation.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors import statements
from rquant.factors.extensions import kunquant_panel as k
from rquant.factors.extensions.kunquant import kunquant_symbols
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.io import sha256_file, stable_hash

MONTH = 21
QUARTER = 63
HALF_YEAR = 120
YEAR = 252

# (source name, description, maximum lookback in trading days). Order is the stable public column order.
CHINA_CHARS: tuple[tuple[str, str, int], ...] = (
    # Liquidity
    ("size", "log float market value", 1),
    ("size3", "cube of log float market value", 1),
    ("turnm", "log one-month float-share turnover", MONTH),
    ("turnq", "log average monthly turnover over one quarter", QUARTER),
    ("turna", "log average monthly turnover over one year", YEAR),
    ("vturn", "std of daily turnover, 120 days", HALF_YEAR),
    ("cvturn", "coefficient of variation of daily turnover, 120 days", HALF_YEAR),
    ("abturn", "one-month over one-year average daily turnover", 250),
    ("dtvm", "log one-month traded value", MONTH),
    ("dtvq", "log average monthly traded value over one quarter", QUARTER),
    ("dtva", "log average monthly traded value over one year", YEAR),
    ("vdtv", "std of daily traded value, 120 days", HALF_YEAR),
    ("cvd", "coefficient of variation of daily traded value, 120 days", HALF_YEAR),
    ("ami", "Amihud illiquidity |r|/value, one month", MONTH),
    # Risk
    ("idvc", "CAPM residual volatility, 120 days", HALF_YEAR),
    ("idvff", "Fama-French three-factor residual volatility, 120 days", HALF_YEAR),
    ("tv", "total return volatility, 120 days", HALF_YEAR),
    ("idsff", "Fama-French three-factor residual skewness, 120 days", HALF_YEAR),
    ("idsc", "CAPM residual skewness, 120 days", HALF_YEAR),
    ("ts", "total return skewness, 120 days", HALF_YEAR),
    ("cs", "Harvey-Siddique co-skewness, 120 days", HALF_YEAR),
    ("dbeta", "downside market beta, 240 days", 240),
    ("betafp", "Frazzini-Pedersen beta, 120 days", HALF_YEAR + 2),
    ("betadm", "Dimson beta with lead and lag market returns, 120 days", HALF_YEAR + 2),
    ("beta", "CAPM market beta, 120 days", HALF_YEAR),
    # Past returns
    ("m1", "one-month return", MONTH),
    ("m11", "return from month t-12 to t-1", YEAR),
    ("m60", "return from month t-60 to t-12", 5 * YEAR),
    ("m6", "return from month t-6 to t-1", 6 * MONTH),
    ("m3", "return from month t-3 to t-1", 3 * MONTH),
    ("indmom", "value-weighted six-month return of the SW level-1 industry", 6 * MONTH),
    ("m24", "return from month t-36 to t-12", 3 * YEAR),
    ("mchg", "momentum change: months t-6..t-1 minus t-12..t-7", YEAR),
    ("im12", "FF3 residual momentum, months t-12..t-1 of a 720-day fit", 720),
    ("im6", "FF3 residual momentum, months t-6..t-1 of a 720-day fit", 720),
    ("w52", "price over its 240-day high", 240),
    ("mdr", "average of the five largest daily returns, one month", MONTH),
    ("pr", "unadjusted share price", 1),
    ("abr", "abnormal return from day -2 to +1 around the latest earnings announcement", 4),
    ("season", "average same-calendar-month return over the prior five years", 5 * YEAR),
    # Profitability
    ("roe", "quarterly net income to parent over lagged book equity", 1),
    ("droe", "four-quarter change in roe", 1),
    ("roa", "quarterly net income over lagged total assets", 1),
    ("droa", "four-quarter change in roa", 1),
    ("rna", "quarterly operating profit over lagged net operating assets", 1),
    ("pm", "quarterly operating profit over revenue", 1),
    ("ato", "quarterly revenue over lagged net operating assets", 1),
    ("ct", "quarterly revenue over lagged total assets", 1),
    ("gpa", "annual gross profit over total assets", 1),
    ("gpla", "quarterly gross profit over lagged total assets", 1),
    ("ope", "annual operating profit (revenue - costs - SG&A - finance) over book equity", 1),
    ("ople", "quarterly operating profit over lagged book equity", 1),
    ("opa", "annual operating profit over total assets", 1),
    ("opla", "quarterly operating profit over lagged total assets", 1),
    ("tbi", "quarterly pretax income over net income", 1),
    ("bl", "total assets over book equity", 1),
    ("sg", "annual revenue growth", 1),
    ("sgq", "quarterly revenue growth over four quarters", 1),
    ("fscore", "Piotroski F-score from quarterly statements", 1),
    ("oscore", "Ohlson O-score from quarterly and trailing statements", 1),
    # Value
    ("bm", "book equity over total market value", 1),
    ("dm", "total liabilities over total market value", 1),
    ("am", "total assets over total market value", 1),
    ("ep", "trailing net income to parent over total market value", 1),
    ("cfp", "trailing net income plus depreciation and amortisation over total market value", 1),
    ("sr", "weighted five-year annual revenue growth", 1),
    ("em", "enterprise value over trailing EBITDA", 1),
    ("sp", "trailing revenue over total market value", 1),
    ("ocfp", "trailing operating cash flow over total market value", 1),
    ("de", "annual total liabilities over book equity", 1),
    ("ebp", "enterprise book-to-price", 1),
    ("ndp", "net debt over total market value", 1),
    # Investment
    ("ag", "four-quarter total asset growth", 1),
    ("dpia", "change in fixed assets plus inventory over lagged assets", 1),
    ("noa", "net operating assets over lagged total assets", 1),
    ("dnoa", "change in net operating assets over lagged total assets", 1),
    ("ig", "annual capital expenditure growth", 1),
    ("cei", "one-year composite equity issuance", YEAR),
    ("cdi", "five-year log growth of total liabilities", 1),
    ("ivg", "annual inventory growth", 1),
    ("ivchg", "annual inventory change over average total assets", 1),
    ("oacc", "annual net income minus operating cash flow over lagged assets", 1),
    ("tacc", "annual total accruals over lagged assets", 1),
    ("dwc", "change in net non-cash working capital over lagged assets", 1),
    ("dcoa", "change in current operating assets over lagged assets", 1),
    ("dcol", "change in current operating liabilities over lagged assets", 1),
    ("dnco", "change in net non-current operating assets over lagged assets", 1),
    ("dnca", "change in non-current operating assets over lagged assets", 1),
    ("dncl", "change in non-current operating liabilities over lagged assets", 1),
    ("dfin", "change in net financial assets over lagged assets", 1),
    ("dbe", "change in book equity over lagged assets", 1),
    # Other
    ("rdm", "quarterly R&D expense over total market value", 1),
    ("rds", "quarterly R&D expense over revenue", 1),
    ("ol", "annual operating costs over total assets", 1),
    ("age", "months since listing", 1),
    ("dsi", "% change in revenue minus % change in inventory", 1),
    ("dsa", "% change in revenue minus % change in receivables", 1),
    ("dgs", "% change in gross profit minus % change in revenue", 1),
    ("dss", "% change in revenue minus % change in SG&A", 1),
    ("etr", "effective tax rate change times earnings change over market value", 1),
    ("tan", "tangibility: weighted liquid and fixed assets over total assets", 1),
    ("vcf", "volatility of quarterly operating cash flow over revenue, 16 quarters", 1),
    ("cta", "cash over total assets", 1),
    ("esm", "earnings smoothness, 16 quarters", 1),
    ("ala", "asset liquidity over lagged total assets", 1),
    ("alm", "asset liquidity over market value of assets", 1),
    ("sue", "standardized unexpected quarterly earnings", 1),
    ("rs", "standardized unexpected quarterly revenue", 1),
    ("tes", "four-quarter change in tax expense over lagged assets", 1),
)
CHINA_CHARS_NAMES = tuple(name for name, _, _ in CHINA_CHARS)
CHINA_CHARS_DAILY_BASIC = {"turnover_rate": 0.01, "circ_mv": 10_000.0, "close": 1.0}
MARKET_FACTORS = ("mkt", "smb", "hml")
# Statement columns the statement graph reads.
STATEMENT_FIELDS = (
    "revenue",
    "oper_cost",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "rd_exp",
    "operate_profit",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "n_cashflow_act",
    "c_pay_acq_const_fiolta",
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
    "total_share",
    "money_cap",
    "trad_asset",
    "accounts_receiv",
    "inventories",
    "total_cur_assets",
    "lt_eqt_invest",
    "fix_assets",
    "intan_assets",
    "goodwill",
    "total_assets",
    "st_borr",
    "non_cur_liab_due_1y",
    "total_cur_liab",
    "lt_borr",
    "bond_payable",
    "total_liab",
    "minority_int",
    "oth_eqt_tools_p_shr",
    "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int",
)
# Statement-graph results. ``q_`` results reach the daily graph from the latest known quarter, ``y_`` results from
# the latest known fiscal year.
STATEMENT_OUTPUTS = (
    "q_roe",
    "q_droe",
    "q_roa",
    "q_droa",
    "q_rna",
    "q_pm",
    "q_ato",
    "q_ct",
    "y_gpa",
    "q_gpla",
    "y_ope",
    "q_ople",
    "y_opa",
    "q_opla",
    "q_tbi",
    "q_bl",
    "y_sg",
    "q_sgq",
    "q_fscore",
    "q_oscore",
    "q_book",
    "q_total_liab",
    "q_total_assets",
    "q_earnings_ttm",
    "q_cash_earnings_ttm",
    "y_sr",
    "q_enterprise_debt",
    "q_ebitda_ttm",
    "q_revenue_ttm",
    "q_operating_cash_ttm",
    "y_de",
    "q_net_debt",
    "q_ag",
    "q_dpia",
    "q_noa",
    "q_dnoa",
    "y_ig",
    "y_cdi",
    "y_ivg",
    "y_ivchg",
    "y_oacc",
    "y_tacc",
    "y_dwc",
    "y_dcoa",
    "y_dcol",
    "y_dnco",
    "y_dnca",
    "y_dncl",
    "y_dfin",
    "y_dbe",
    "q_rd",
    "q_rds",
    "y_ol",
    "y_dsi",
    "y_dsa",
    "y_dgs",
    "y_dss",
    "y_etr_rate_change",
    "y_earnings_change",
    "q_tan",
    "q_vcf",
    "q_cta",
    "q_esm",
    "q_ala",
    "q_asset_liquidity",
    "q_debt_value",
    "q_sue",
    "q_rs",
    "q_tes",
)
# Characteristics published unchanged from one statement-graph result.
STATEMENT_FACTORS = {
    name: result
    for result in STATEMENT_OUTPUTS
    if (name := result[2:]) in {spec[0] for spec in CHINA_CHARS}
}
# Characteristics that divide a statement-graph result by the total market value.
OVER_MARKET_VALUE = {
    "bm": "q_book",
    "dm": "q_total_liab",
    "am": "q_total_assets",
    "ep": "q_earnings_ttm",
    "cfp": "q_cash_earnings_ttm",
    "sp": "q_revenue_ttm",
    "ocfp": "q_operating_cash_ttm",
    "ndp": "q_net_debt",
    "rdm": "q_rd",
}
DAILY_REFERENCE = (
    *(f"basic_{field}" for field in CHINA_CHARS_DAILY_BASIC),
    *MARKET_FACTORS,
    "announced",
    "calendar_day",
    "listed_day",
)

CPP_PRELUDE = r"""
#ifndef RQUANT_CHINA_CHARS_CPP_HELPERS
#define RQUANT_CHINA_CHARS_CPP_HELPERS
namespace rq_cnc {

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
enum Statistic { kSum, kMean, kStd, kSkew, kMax };

// close[t] / close[t - 1] - 1, missing around a missing close.
static inline std::vector<double> simple_returns(int n, const double *close) {
    std::vector<double> r(n, kNaN);
    for (int t = 1; t < n; ++t) r[t] = close[t] / close[t - 1] - 1.0;
    return r;
}

// The latest finite value at or before every row.
static inline std::vector<double> forward_filled(int n, const double *values) {
    std::vector<double> filled(n, kNaN);
    double last = kNaN;
    for (int t = 0; t < n; ++t) {
        if (std::isfinite(values[t])) last = values[t];
        filled[t] = last;
    }
    return filled;
}

// pandas ``safe_div``: missing for a denominator within 1e-12 of zero and for an infinite result.
static inline double safe_div(double numerator, double denominator) {
    if (!(std::fabs(denominator) >= 1e-12)) return kNaN;
    const double result = numerator / denominator;
    return std::isinf(result) ? kNaN : result;
}

// pandas ``rolling(window, min_periods=minimum)`` statistics of the finite values among the ``window`` rows ending
// at ``t``: sum, mean, sample standard deviation, bias-corrected skewness or maximum. A window of one repeated value
// has a standard deviation and skewness of exactly 0.
static inline double rolling(const double *values, int t, int window, int minimum, Statistic statistic) {
    int count = 0;
    double total = 0.0, largest = -std::numeric_limits<double>::infinity(), first = kNaN;
    bool same = true;
    for (int i = t - window + 1 < 0 ? 0 : t - window + 1; i <= t; ++i) {
        const double value = values[i];
        if (!std::isfinite(value)) continue;
        if (count == 0) first = value; else same = same && value == first;
        ++count;
        total += value;
        if (value > largest) largest = value;
    }
    if (count == 0 || count < minimum) return kNaN;
    if (statistic == kSum) return total;
    if (statistic == kMax) return largest;
    const double mean = total / count;
    if (statistic == kMean) return mean;
    if (count < (statistic == kStd ? 2 : 3)) return kNaN;
    if (same) return 0.0;
    double squares = 0.0, cubes = 0.0;
    for (int i = t - window + 1 < 0 ? 0 : t - window + 1; i <= t; ++i) {
        if (!std::isfinite(values[i])) continue;
        const double deviation = values[i] - mean;
        squares += deviation * deviation;
        cubes += deviation * deviation * deviation;
    }
    if (statistic == kStd) return std::sqrt(squares / (count - 1));
    const double variance = squares / count, spread = std::sqrt(variance);
    if (variance <= 1e-14) return kNaN;
    return std::sqrt(count * (count - 1.0)) * (cubes / count) / ((count - 2.0) * spread * spread * spread);
}

// Solve the k x k system ``a x = b`` by Gaussian elimination with partial pivoting; false when singular.
static inline bool solve(int k, std::vector<double> a, std::vector<double> b, double *x) {
    for (int c = 0; c < k; ++c) {
        int pivot = c;
        for (int r = c + 1; r < k; ++r) if (std::fabs(a[r * k + c]) > std::fabs(a[pivot * k + c])) pivot = r;
        if (a[pivot * k + c] == 0.0) return false;
        if (pivot != c) {
            for (int j = 0; j < k; ++j) std::swap(a[c * k + j], a[pivot * k + j]);
            std::swap(b[c], b[pivot]);
        }
        for (int r = c + 1; r < k; ++r) {
            const double factor = a[r * k + c] / a[c * k + c];
            for (int j = c; j < k; ++j) a[r * k + j] -= factor * a[c * k + j];
            b[r] -= factor * b[c];
        }
    }
    for (int r = k - 1; r >= 0; --r) {
        double value = b[r];
        for (int j = r + 1; j < k; ++j) value -= a[r * k + j] * x[j];
        x[r] = value / a[r * k + r];
    }
    return true;
}

enum Fit { kBeta, kSlopes, kResidualStd, kResidualSkew, kCoskew, kResidualMomentum };

// Rolling OLS of ``y`` on a constant and the ``x`` columns over trailing ``window`` rows, dropping rows where ``y`` or
// a regressor is missing, published on the window's last row. ``x[0]`` is the market for co-skewness. The residual
// momentum is the mean over standard deviation of the full-window residuals in window positions [first, last).
// Window sums come from prefix sums of cross-products up to third order.
static inline void rolling_regression(int n, const double *y, const std::vector<const double *> &x, int window,
                                      int minimum, Fit fit, int first, int last, double *out) {
    const int k = static_cast<int>(x.size()) + 1;
    if (n < window) return;
    std::vector<double> design(static_cast<size_t>(n) * k, 0.0), weight(n, 0.0), value(n, 0.0);
    for (int t = 0; t < n; ++t) {
        bool ok = true;
        for (int j = 1; j < k; ++j) ok = ok && std::isfinite(x[j - 1][t]);
        if (!ok) continue;
        design[static_cast<size_t>(t) * k] = 1.0;
        for (int j = 1; j < k; ++j) design[static_cast<size_t>(t) * k + j] = x[j - 1][t];
        if (std::isfinite(y[t])) { weight[t] = 1.0; value[t] = y[t]; }
    }
    const size_t k2 = static_cast<size_t>(k) * k, k3 = k2 * k;
    std::vector<double> pn(n + 1, 0.0), py(n + 1, 0.0), pyy(n + 1, 0.0), pyyy(n + 1, 0.0);
    std::vector<double> px((n + 1) * static_cast<size_t>(k), 0.0), pxy(px.size(), 0.0), pxyy(px.size(), 0.0);
    std::vector<double> pxx((n + 1) * k2, 0.0), pxxy(pxx.size(), 0.0), pxxx((n + 1) * k3, 0.0);
    for (int t = 0; t < n; ++t) {
        const double w = weight[t], v = value[t];
        const double *d = &design[static_cast<size_t>(t) * k];
        pn[t + 1] = pn[t] + w;
        py[t + 1] = py[t] + v;
        pyy[t + 1] = pyy[t] + v * v;
        pyyy[t + 1] = pyyy[t] + std::pow(v, 3.0);
        for (int a = 0; a < k; ++a) {
            const size_t ia = static_cast<size_t>(t) * k + a;
            px[ia + k] = px[ia] + d[a] * w;
            pxy[ia + k] = pxy[ia] + d[a] * v;
            pxyy[ia + k] = pxyy[ia] + d[a] * (v * v);
            for (int b = 0; b < k; ++b) {
                const size_t iab = t * k2 + a * k + b;
                const double outer = d[a] * d[b];
                pxx[iab + k2] = pxx[iab] + outer * w;
                pxxy[iab + k2] = pxxy[iab] + outer * v;
                for (int c = 0; c < k; ++c) {
                    const size_t iabc = t * k3 + (a * k + b) * k + c;
                    pxxx[iabc + k3] = pxxx[iabc] + outer * d[c] * w;
                }
            }
        }
    }
    std::vector<double> gram(k2), sxy(k), beta(k);
    // Range sums over rows [lo, hi) of the prefix arrays with ``size`` values per row.
    auto range = [](const std::vector<double> &prefix, size_t size, int lo, int hi, size_t at) {
        return prefix[hi * size + at] - prefix[lo * size + at];
    };
    auto linear = [&](const std::vector<double> &prefix, int lo, int hi) {
        double total = 0.0;
        for (int a = 0; a < k; ++a) total += beta[a] * range(prefix, k, lo, hi, a);
        return total;
    };
    auto quadratic = [&](const std::vector<double> &prefix, int lo, int hi) {
        double total = 0.0;
        for (int a = 0; a < k; ++a) {
            double row = 0.0;
            for (int b = 0; b < k; ++b) row += range(prefix, k2, lo, hi, a * k + b) * beta[b];
            total += beta[a] * row;
        }
        return total;
    };
    for (int t = window - 1; t < n; ++t) {
        const int lo = t + 1 - window, hi = t + 1;
        const double count = pn[hi] - pn[lo];
        if (!(count >= minimum)) continue;
        for (int a = 0; a < k; ++a) {
            sxy[a] = range(pxy, k, lo, hi, a);
            for (int b = 0; b < k; ++b) gram[a * k + b] = range(pxx, k2, lo, hi, a * k + b);
        }
        if (!solve(k, gram, sxy, beta.data())) continue;
        double result = kNaN;
        if (fit == kBeta) {
            result = beta[1];
        } else if (fit == kSlopes) {
            result = 0.0;
            for (int a = 1; a < k; ++a) result += beta[a];
        } else if (fit == kResidualMomentum) {
            const int start = lo + first, stop = lo + last;
            const double part_n = pn[stop] - pn[start];
            const double part_e = py[stop] - py[start] - linear(px, start, stop);
            const double part_ee = pyy[stop] - pyy[start] - 2.0 * linear(pxy, start, stop)
                + quadratic(pxx, start, stop);
            const double mean = part_e / part_n;
            const double spread = std::sqrt((part_ee - part_n * (mean * mean)) / (part_n - 1));
            if (part_n >= (last - first) / 2 && spread > 0) result = mean / spread;
        } else {
            const double syy = pyy[hi] - pyy[lo];
            const double squared = syy - 2.0 * linear(pxy, lo, hi) + quadratic(pxx, lo, hi);
            // Prefix-sum rounding can leave a tiny negative or zero residual sum of squares for an exact fit.
            if (squared > 1e-14 * (syy > 1e-300 ? syy : 1e-300)) {
                const double variance = squared / count;
                if (fit == kResidualStd) {
                    result = count > k ? std::sqrt(squared / (count - k)) : kNaN;
                } else if (fit == kResidualSkew) {
                    double cubic = 0.0;
                    for (int a = 0; a < k; ++a)
                        for (int b = 0; b < k; ++b)
                            for (int c = 0; c < k; ++c)
                                cubic += beta[a] * beta[b] * beta[c] * range(pxxx, k3, lo, hi, (a * k + b) * k + c);
                    const double cubed = pyyy[hi] - pyyy[lo] - 3.0 * linear(pxyy, lo, hi)
                        + 3.0 * quadratic(pxxy, lo, hi) - cubic;
                    result = (cubed / count) / std::pow(variance, 1.5);
                } else {
                    // Residual moments against the market: sum(e m), sum(e m^2) and the centered market moments.
                    double e_m = range(pxy, k, lo, hi, 1), e_mm = range(pxxy, k2, lo, hi, k + 1);
                    for (int a = 0; a < k; ++a) {
                        e_m -= beta[a] * range(pxx, k2, lo, hi, a * k + 1);
                        e_mm -= beta[a] * range(pxxx, k3, lo, hi, (a * k + 1) * k + 1);
                    }
                    const double sum_e = py[hi] - py[lo] - linear(px, lo, hi);
                    const double market_mean = range(px, k, lo, hi, 1) / count;
                    const double centered = range(pxx, k2, lo, hi, k + 1) - count * (market_mean * market_mean);
                    const double co_moment = e_mm - 2.0 * market_mean * e_m + market_mean * market_mean * sum_e;
                    result = (co_moment / count) / (std::sqrt(variance) * centered / count);
                }
            }
        }
        out[t] = result;
    }
}

}  // namespace rq_cnc
#endif
"""


class ChinaCharsLibrary(FactorLibrary):
    """China A-share equity characteristics, re-dated to daily point-in-time panels and computed in KunQuant."""

    family = "china_chars"
    required_inputs = ("close", "amount", "cap", "sector")
    reference_inputs = (
        *DAILY_REFERENCE,
        *(name for name in STATEMENT_OUTPUTS if name not in STATEMENT_FACTORS.values()),
    )
    statement_fields = STATEMENT_FIELDS
    statement_outputs = STATEMENT_OUTPUTS
    statement_factors = STATEMENT_FACTORS
    strict_float = True
    cpp_prelude = CPP_PRELUDE

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"cnc_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=description,
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.china_chars.{name}",
                catalog_version=3,
            )
            for ordinal, (name, description, lookback) in enumerate(CHINA_CHARS, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Also pin the statement alignment, which lives in another module."""
        return stable_hash(
            {"module": super().implementation_fingerprint, "statements": inspect.getsource(statements)}
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.canonical import ts_code_to_qlib
        from rquant.data.reference import load_reference_manifest

        root = Path(canonical_root) / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "china_chars needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        for name, expected in files.items():
            if sha256_file(root / name) != expected:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        columns = ["instrument", "datetime", *CHINA_CHARS_DAILY_BASIC]
        daily = pd.concat([pd.read_parquet(root / name, columns=columns) for name in daily_files], ignore_index=True)
        daily = daily[daily["instrument"].isin(set(instruments))]
        inputs: dict[str, Any] = {}
        for field, scale in CHINA_CHARS_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale

        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        for name in MARKET_FACTORS:
            inputs[name] = factors[name].reindex(dates)

        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        basic = pd.read_parquet(root / "stock_basic.parquet", columns=["ts_code", "list_date"])
        stock_basic_hash = sha256_file(root / "stock_basic.parquet")
        basic = basic.assign(instrument=basic["ts_code"].map(ts_code_to_qlib)).drop_duplicates("instrument")
        listed = pd.to_datetime(basic.set_index("instrument")["list_date"], format="%Y%m%d", errors="coerce")
        inputs["list_date"] = listed.reindex(instruments)
        hashes = {
            "china_chars_reference": manifest["fingerprint"],
            "china_chars_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "china_chars_market_factors": files["market_factors.parquet"],
            "china_chars_fundamentals": files["fundamentals.parquet"],
            "china_chars_stock_basic": stock_basic_hash,
        }
        return self.align_reference_inputs(inputs, dates, instruments), hashes

    def align_reference_inputs(self, inputs: Mapping[str, Any], dates: Any, instruments: Any) -> dict[str, Any]:
        """Add the calendar inputs of the daily graph: announcement events and the day numbers of dates and listings.

        ``announced`` is 1 on the trading day after each statement announcement (the first day its earnings
        reaction is complete), ``calendar_day`` numbers the dates and ``listed_day`` each instrument's listing date.
        """
        dates, instruments = pd.DatetimeIndex(dates), pd.Index(instruments)
        aligned = dict(inputs)
        frame = inputs["fundamentals"]
        published = np.searchsorted(dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy()) + 1
        columns = instruments.get_indexer(frame["instrument"])
        keep = (columns >= 0) & (published < len(dates))
        announced = np.full((len(dates), len(instruments)), np.nan)
        announced[published[keep], columns[keep]] = 1.0
        aligned["announced"] = announced
        epoch = np.datetime64("1970-01-01", "D")
        aligned["calendar_day"] = pd.Series((dates.to_numpy().astype("datetime64[D]") - epoch).astype(float),
                                            index=dates)
        listed = pd.to_datetime(pd.Series(inputs["list_date"]).reindex(instruments)).to_numpy().astype("datetime64[D]")
        listed_day = np.where(np.isnat(listed), np.nan, (listed - epoch).astype(float))
        aligned["listed_day"] = np.ascontiguousarray(np.broadcast_to(listed_day, (len(dates), len(instruments))))
        return aligned

    def build_statements(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return _statement_graph(inputs)

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = _daily_graph(inputs)
        trading = k.notna(inputs["close"])
        return {
            name: k.mask(k.replace_inf(outputs[name]), trading)
            for name in CHINA_CHARS_NAMES
            if name not in STATEMENT_FACTORS
        }


# ---------------------------------------------------------------------------------------------------- statements
def _statement_graph(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Statement formulas over fiscal quarters (rows) x instruments."""
    fiscal_quarter = inputs["fiscal_quarter"]
    singles: dict[str, Any] = {}

    def value(name: str) -> Any:
        return inputs[name]

    def lag(node: Any, periods: int) -> Any:
        return k.delay(node, periods)

    def single(name: str) -> Any:
        """A year-to-date flow as single-quarter values."""
        if name not in singles:
            ytd = value(name)
            singles[name] = k.select(k.eq(fiscal_quarter, 0.0), ytd, ytd - lag(ytd, 1))
        return singles[name]

    def ttm(name: str) -> Any:
        flow = single(name)
        return flow + lag(flow, 1) + lag(flow, 2) + lag(flow, 3)

    def annual(name: str) -> Any:
        """Fiscal-year totals (Q4 year-to-date) at Q4 rows."""
        return k.mask(value(name), k.eq(fiscal_quarter, 3.0))

    def growth(current: Any, previous: Any) -> Any:
        """Relative change, defined only for a positive base."""
        return k.safe_div(current - previous, k.mask(previous, k.gt(previous, 0.0)))

    ratio = k.safe_div
    assets = value("total_assets")
    book = value("total_hldr_eqy_exc_min_int") - value("oth_eqt_tools_p_shr")
    debt = value("st_borr") + value("non_cur_liab_due_1y") + value("lt_borr") + value("bond_payable")
    net_operating_assets = value("total_hldr_eqy_inc_min_int") + debt - value("money_cap") - value("trad_asset")

    def operating_profit(flow: Callable[[str], Any]) -> Any:
        return flow("revenue") - flow("oper_cost") - flow("sell_exp") - flow("admin_exp") - flow("fin_exp")

    def lagged_assets_change(node: Any) -> Any:
        return ratio(node - lag(node, 4), lag(assets, 4))

    def change_gap(first: Any, second: Any) -> Any:
        return growth(first, lag(first, 4)) - growth(second, lag(second, 4))

    def surprise(node: Any) -> Any:
        change = node - lag(node, 4)
        return ratio(change, k.rolling_nan_std(change, 8, 6))

    out: dict[str, Any] = {}
    # Profitability
    roe = ratio(single("n_income_attr_p"), lag(book, 1))
    roa = ratio(single("n_income"), lag(assets, 1))
    out["q_roe"] = roe
    out["q_droe"] = roe - lag(roe, 4)
    out["q_roa"] = roa
    out["q_droa"] = roa - lag(roa, 4)
    out["q_rna"] = ratio(single("operate_profit"), lag(net_operating_assets, 1))
    out["q_pm"] = ratio(single("operate_profit"), single("revenue"))
    out["q_ato"] = ratio(single("revenue"), lag(net_operating_assets, 1))
    out["q_ct"] = ratio(single("revenue"), lag(assets, 1))
    out["y_gpa"] = ratio(annual("revenue") - annual("oper_cost"), assets)
    out["q_gpla"] = ratio(single("revenue") - single("oper_cost"), lag(assets, 1))
    out["y_ope"] = ratio(operating_profit(annual), book)
    out["q_ople"] = ratio(operating_profit(single), lag(book, 1))
    out["y_opa"] = ratio(operating_profit(annual), assets)
    out["q_opla"] = ratio(operating_profit(single), lag(assets, 1))
    out["q_tbi"] = ratio(single("total_profit"), single("n_income"))
    out["q_bl"] = ratio(assets, book)
    out["y_sg"] = growth(annual("revenue"), lag(annual("revenue"), 4))
    out["q_sgq"] = growth(single("revenue"), lag(single("revenue"), 4))
    out["q_fscore"] = _fscore(value, single, lag)
    out["q_oscore"] = _oscore(value, ttm, lag)
    # Value
    depreciation = ttm("depr_fa_coga_dpba") + ttm("amort_intang_assets") + ttm("lt_amort_deferred_exp")
    revenue = annual("revenue")
    out["q_book"] = book
    out["q_total_liab"] = value("total_liab")
    out["q_total_assets"] = assets
    out["q_earnings_ttm"] = ttm("n_income_attr_p")
    out["q_cash_earnings_ttm"] = ttm("n_income") + depreciation
    sales_growth = k.const(0.0)
    for year in range(1, 6):
        sales_growth = sales_growth + (6 - year) * growth(lag(revenue, 4 * (year - 1)), lag(revenue, 4 * year))
    out["y_sr"] = sales_growth / 15.0
    out["q_enterprise_debt"] = (
        debt + value("oth_eqt_tools_p_shr") + value("minority_int") - value("money_cap") - value("trad_asset")
    )
    out["q_ebitda_ttm"] = ttm("total_profit") + ttm("fin_exp") + depreciation
    out["q_revenue_ttm"] = ttm("revenue")
    out["q_operating_cash_ttm"] = ttm("n_cashflow_act")
    out["y_de"] = ratio(value("total_liab"), book)
    out["q_net_debt"] = debt + value("oth_eqt_tools_p_shr") - value("money_cap") - value("trad_asset")
    # Investment
    out["q_ag"] = growth(assets, lag(assets, 4))
    out["q_dpia"] = lagged_assets_change(value("fix_assets") + value("inventories"))
    out["q_noa"] = ratio(net_operating_assets, lag(assets, 4))
    out["q_dnoa"] = lagged_assets_change(net_operating_assets)
    capex = annual("c_pay_acq_const_fiolta")
    out["y_ig"] = growth(capex, lag(capex, 4))
    liabilities = value("total_liab")
    base = lag(liabilities, 20)
    out["y_cdi"] = k.log(ratio(liabilities, k.mask(base, k.gt(base, 0.0))))
    inventory = value("inventories")
    out["y_ivg"] = growth(inventory, lag(inventory, 4))
    out["y_ivchg"] = ratio(inventory - lag(inventory, 4), (assets + lag(assets, 4)) / 2.0)
    out["y_oacc"] = ratio(annual("n_income") - annual("n_cashflow_act"), lag(assets, 4))
    current_assets = value("total_cur_assets") - value("money_cap") - value("trad_asset")
    current_liabilities = value("total_cur_liab") - value("st_borr") - value("non_cur_liab_due_1y")
    non_current_assets = value("total_assets") - value("total_cur_assets") - value("lt_eqt_invest")
    non_current_liabilities = (
        value("total_liab") - value("total_cur_liab") - value("lt_borr") - value("bond_payable")
    )
    financial = value("trad_asset") + value("lt_eqt_invest") - debt - value("oth_eqt_tools_p_shr")
    working = current_assets - current_liabilities
    total = working + (non_current_assets - non_current_liabilities) + financial
    out["y_tacc"] = lagged_assets_change(total)
    out["y_dwc"] = lagged_assets_change(current_assets - current_liabilities)
    out["y_dcoa"] = lagged_assets_change(current_assets)
    out["y_dcol"] = lagged_assets_change(current_liabilities)
    out["y_dnco"] = lagged_assets_change(non_current_assets - non_current_liabilities)
    out["y_dnca"] = lagged_assets_change(non_current_assets)
    out["y_dncl"] = lagged_assets_change(non_current_liabilities)
    out["y_dfin"] = lagged_assets_change(financial)
    out["y_dbe"] = lagged_assets_change(book)
    # Other
    out["q_rd"] = single("rd_exp")
    out["q_rds"] = ratio(single("rd_exp"), single("revenue"))
    out["y_ol"] = ratio(annual("oper_cost") + annual("sell_exp") + annual("admin_exp"), assets)
    out["y_dsi"] = change_gap(annual("revenue"), value("inventories"))
    out["y_dsa"] = change_gap(annual("revenue"), value("accounts_receiv"))
    out["y_dgs"] = change_gap(annual("revenue") - annual("oper_cost"), annual("revenue"))
    out["y_dss"] = change_gap(annual("revenue"), annual("sell_exp") + annual("admin_exp"))
    rate = ratio(annual("income_tax"), annual("total_profit"))
    out["y_etr_rate_change"] = rate - k.nan_mean(*(lag(rate, 4 * years) for years in (1, 2, 3)))
    income = annual("n_income_attr_p")
    out["y_earnings_change"] = income - lag(income, 4)
    tangible = (
        value("money_cap")
        + 0.715 * value("accounts_receiv")
        + 0.547 * value("inventories")
        + 0.535 * value("fix_assets")
    )
    out["q_tan"] = ratio(tangible, assets)
    out["q_vcf"] = k.rolling_nan_std(ratio(single("n_cashflow_act"), single("revenue")), 16, 8)
    out["q_cta"] = ratio(value("money_cap"), assets)
    earnings = k.rolling_nan_std(ratio(single("n_income"), lag(assets, 1)), 16, 8)
    cash = k.rolling_nan_std(ratio(single("n_cashflow_act"), lag(assets, 1)), 16, 8)
    out["q_esm"] = ratio(earnings, cash)
    current = value("total_cur_assets")
    money = value("money_cap")
    liquidity = money + 0.75 * (current - money) + 0.5 * (assets - current - value("goodwill") - value("intan_assets"))
    out["q_ala"] = ratio(liquidity, lag(assets, 1))
    out["q_asset_liquidity"] = liquidity
    out["q_debt_value"] = assets - book
    out["q_sue"] = surprise(single("n_income_attr_p"))
    out["q_rs"] = surprise(single("revenue"))
    tax = single("income_tax")
    out["q_tes"] = ratio(tax - lag(tax, 4), lag(assets, 4))
    return {name: out[name] for name in STATEMENT_OUTPUTS}


def _fscore(value: Callable[[str], Any], single: Callable[[str], Any], lag: Callable[[Any, int], Any]) -> Any:
    """Piotroski F-score: nine binary signals, missing unless every signal's inputs are known."""
    ratio = k.safe_div
    assets = value("total_assets")
    income = single("n_income")
    cash_flow = single("n_cashflow_act")
    revenue = single("revenue")
    roa = ratio(income, lag(assets, 1))
    leverage = ratio(value("lt_borr") + value("bond_payable"), assets)
    current = ratio(value("total_cur_assets"), value("total_cur_liab"))
    margin = ratio(revenue - single("oper_cost"), revenue)
    turnover = ratio(revenue, lag(assets, 1))
    shares = value("total_share")
    components = (
        (roa, k.gt(roa, 0.0)),
        (cash_flow, k.gt(cash_flow, 0.0)),
        (roa - lag(roa, 4), k.gt(roa - lag(roa, 4), 0.0)),
        (cash_flow - income, k.gt(cash_flow, income)),
        (leverage - lag(leverage, 4), k.lt(leverage, lag(leverage, 4))),
        (current - lag(current, 4), k.gt(current, lag(current, 4))),
        (shares - lag(shares, 4), k.le(shares, lag(shares, 4))),
        (margin - lag(margin, 4), k.gt(margin, lag(margin, 4))),
        (turnover - lag(turnover, 4), k.gt(turnover, lag(turnover, 4))),
    )
    score = k.indicator(components[0][1])
    for _, signal in components[1:]:
        score = score + k.indicator(signal)
    return k.mask(score, k.valid(*(component for component, _ in components)))


def _oscore(value: Callable[[str], Any], ttm: Callable[[str], Any], lag: Callable[[Any, int], Any]) -> Any:
    """Ohlson O-score from the latest quarter's balance sheet and trailing-twelve-month flows."""
    ratio = k.safe_div
    assets = value("total_assets")
    liabilities = value("total_liab")
    current_assets = value("total_cur_assets")
    current_liabilities = value("total_cur_liab")
    income = ttm("n_income")
    previous = lag(income, 4)
    log_assets = k.log(k.mask(assets, k.gt(assets, 0.0)))
    insolvent = k.mask(k.indicator(k.gt(liabilities, assets)), k.valid(liabilities - assets))
    two_losses = k.mask(
        k.indicator(k.all_of(k.lt(income, 0.0), k.lt(previous, 0.0))), k.valid(income + previous)
    )
    return (
        k.const(-1.32)
        - 0.407 * log_assets
        + 6.03 * ratio(liabilities, assets)
        - 1.43 * ratio(current_assets - current_liabilities, assets)
        + 0.076 * ratio(current_liabilities, current_assets)
        - 1.72 * insolvent
        - 2.37 * ratio(income, assets)
        - 1.83 * ratio(ttm("n_cashflow_act"), liabilities)
        + 0.285 * two_losses
        - 0.521 * ratio(income - previous, k.absolute(income) + k.absolute(previous))
    )


# ---------------------------------------------------------------------------------------------------- daily graph
def _daily_graph(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Every characteristic outside ``STATEMENT_FACTORS``, before the trading-day mask."""
    close, amount, cap = inputs["close"], inputs["amount"], inputs["cap"]
    turnover, float_value = inputs["basic_turnover_rate"], inputs["basic_circ_mv"]
    market = [inputs[name] for name in MARKET_FACTORS]
    positive_cap = k.mask(cap, k.gt(cap, 0.0))

    def to_price(node: Any) -> Any:
        return k.safe_div(node, positive_cap)

    def rolling(node: Any, window: int, minimum: int, statistic: str) -> Any:
        return k.series_kernel(
            [node],
            f"for (int t = 0; t < n; ++t) out[t] = rq_cnc::rolling(x0, t, {window}, {minimum}, rq_cnc::{statistic});",
        )

    def monthly_log(node: Any, window: int, minimum: int) -> Any:
        mean = rolling(node, window, minimum, "kMean")
        return k.log(k.mask(mean * float(MONTH), k.gt(mean, 0.0)))

    def on_returns(statement: str) -> Any:
        """A kernel over the stock's daily returns ``r`` (from ``x0``, the close) and further inputs ``x1, ...``."""
        return f"const std::vector<double> r = rq_cnc::simple_returns(n, x0);\n{statement}"

    def returns_rolling(window: int, minimum: int, statistic: str) -> Any:
        return k.series_kernel(
            [close],
            on_returns(
                f"for (int t = 0; t < n; ++t) out[t] = rq_cnc::rolling(r.data(), t, {window}, {minimum}, "
                f"rq_cnc::{statistic});"
            ),
        )

    def regression(regressors: list[Any], window: int, minimum: int, fit: str, first: int = 0, last: int = 0) -> Any:
        columns = ", ".join(f"x{index}" for index in range(1, len(regressors) + 1))
        return k.series_kernel(
            [close, *regressors],
            on_returns(
                f"rq_cnc::rolling_regression(n, r.data(), {{{columns}}}, {window}, {minimum}, rq_cnc::{fit}, {first}, "
                f"{last}, out);"
            ),
        )

    def momentum(start: int, end: int) -> Any:
        """Return from ``start`` trading days ago to ``end`` trading days ago, on the forward-filled close."""
        return k.series_kernel(
            [close],
            "const std::vector<double> price = rq_cnc::forward_filled(n, x0);\n"
            f"for (int t = {start}; t < n; ++t) out[t] = price[t - {end}] / price[t - {start}] - 1.0;",
        )

    out: dict[str, Any] = {}
    # Liquidity
    out["size"] = k.log(k.mask(float_value, k.gt(float_value, 0.0)))
    out["size3"] = k.power(out["size"], 3.0)
    out["turnm"] = monthly_log(turnover, MONTH, 15)
    out["turnq"] = monthly_log(turnover, QUARTER, 40)
    out["turna"] = monthly_log(turnover, YEAR, 120)
    out["vturn"] = rolling(turnover, HALF_YEAR, 40, "kStd")
    out["cvturn"] = k.safe_div(out["vturn"], rolling(turnover, HALF_YEAR, 40, "kMean"))
    out["abturn"] = k.safe_div(rolling(turnover, MONTH, 7, "kMean"), rolling(turnover, 250, 85, "kMean"))
    out["dtvm"] = monthly_log(amount, MONTH, 15)
    out["dtvq"] = monthly_log(amount, QUARTER, 40)
    out["dtva"] = monthly_log(amount, YEAR, 120)
    out["vdtv"] = rolling(amount, HALF_YEAR, 40, "kStd")
    out["cvd"] = k.safe_div(out["vdtv"], rolling(amount, HALF_YEAR, 40, "kMean"))
    out["ami"] = k.series_kernel(
        [close, amount],
        on_returns(
            "std::vector<double> illiquidity(n, rq_cnc::kNaN);\n"
            "for (int t = 0; t < n; ++t) illiquidity[t] = x1[t] > 0 ? rq_cnc::safe_div(std::fabs(r[t]), x1[t]) "
            ": rq_cnc::kNaN;\n"
            f"for (int t = 0; t < n; ++t) out[t] = rq_cnc::rolling(illiquidity.data(), t, {MONTH}, 15, rq_cnc::kMean);"
        ),
    )
    # Risk
    out["idvc"] = regression(market[:1], HALF_YEAR, 40, "kResidualStd")
    out["idvff"] = regression(market, HALF_YEAR, 40, "kResidualStd")
    out["tv"] = returns_rolling(HALF_YEAR, 40, "kStd")
    out["idsff"] = regression(market, HALF_YEAR, 40, "kResidualSkew")
    out["idsc"] = regression(market[:1], HALF_YEAR, 40, "kResidualSkew")
    out["ts"] = returns_rolling(HALF_YEAR, 40, "kSkew")
    out["cs"] = regression(market[:1], HALF_YEAR, 40, "kCoskew")
    out["dbeta"] = k.series_kernel(
        [close, market[0]],
        on_returns(
            """
            // Beta on the days the market return is below its 240-day mean.
            const int window = 240, minimum = 50;
            for (int t = window - 1; t < n; ++t) {
                double market_total = 0.0;
                int market_count = 0;
                for (int i = t - window + 1; i <= t; ++i)
                    if (std::isfinite(x1[i])) { market_total += x1[i]; ++market_count; }
                const double market_mean = market_total / market_count;
                double count = 0.0, sum_r = 0.0, sum_m = 0.0, sum_rm = 0.0, sum_mm = 0.0;
                for (int i = t - window + 1; i <= t; ++i) {
                    if (!(x1[i] < market_mean) || !std::isfinite(r[i])) continue;
                    count += 1.0;
                    sum_r += r[i];
                    sum_m += x1[i];
                    sum_rm += r[i] * x1[i];
                    sum_mm += x1[i] * x1[i];
                }
                const double mean_r = sum_r / count, mean_m = sum_m / count;
                const double covariance = sum_rm / count - mean_r * mean_m;
                const double variance = sum_mm / count - mean_m * mean_m;
                if (count >= minimum && variance > 0) out[t] = covariance / variance;
            }"""
        ),
    )
    out["betafp"] = k.series_kernel(
        [close, market[0]],
        on_returns(
            f"""
            // Frazzini-Pedersen: correlation of 3-day log returns times the ratio of daily log-return volatilities.
            const int window = {HALF_YEAR}, minimum = 40;
            std::vector<double> stock(n), index(n), index_on_stock(n), stock3(n, rq_cnc::kNaN), index3(n, rq_cnc::kNaN);
            for (int t = 0; t < n; ++t) {{
                stock[t] = std::log1p(r[t]);
                index[t] = std::log1p(x1[t]);
                index_on_stock[t] = std::isfinite(stock[t]) ? index[t] : rq_cnc::kNaN;
            }}
            for (int t = 2; t < n; ++t) {{
                stock3[t] = stock[t - 2] + stock[t - 1] + stock[t];
                index3[t] = index[t - 2] + index[t - 1] + index[t];
            }}
            for (int t = 0; t < n; ++t) {{
                const double ratio = rq_cnc::safe_div(rq_cnc::rolling(stock.data(), t, window, minimum, rq_cnc::kStd),
                    rq_cnc::rolling(index_on_stock.data(), t, window, minimum, rq_cnc::kStd));
                double count = 0.0, sum_x = 0.0, sum_y = 0.0;
                for (int i = t - window + 1 < 0 ? 0 : t - window + 1; i <= t; ++i) {{
                    if (!std::isfinite(stock3[i]) || !std::isfinite(index3[i])) continue;
                    count += 1.0;
                    sum_x += stock3[i];
                    sum_y += index3[i];
                }}
                if (count < minimum) continue;
                const double mean_x = sum_x / count, mean_y = sum_y / count;
                double xy = 0.0, xx = 0.0, yy = 0.0;
                for (int i = t - window + 1 < 0 ? 0 : t - window + 1; i <= t; ++i) {{
                    if (!std::isfinite(stock3[i]) || !std::isfinite(index3[i])) continue;
                    xy += (stock3[i] - mean_x) * (index3[i] - mean_y);
                    xx += (stock3[i] - mean_x) * (stock3[i] - mean_x);
                    yy += (index3[i] - mean_y) * (index3[i] - mean_y);
                }}
                const double correlation = xy / std::sqrt(xx * yy);
                if (std::isnan(correlation)) continue;
                out[t] = std::min(1.0, std::max(-1.0, correlation)) * ratio;
            }}"""
        ),
    )
    out["betadm"] = k.series_kernel(
        [close, market[0]],
        on_returns(
            f"""
            // Dimson beta: slopes on the lagged, current and next-day market return. A window ending on day d uses
            // the market return of d + 1, so its estimate is published on d + 1.
            std::vector<double> lagged(n, rq_cnc::kNaN), lead(n, rq_cnc::kNaN), fit(n, rq_cnc::kNaN);
            for (int t = 1; t < n; ++t) lagged[t] = x1[t - 1];
            for (int t = 0; t + 1 < n; ++t) lead[t] = x1[t + 1];
            rq_cnc::rolling_regression(n, r.data(), {{lagged.data(), x1, lead.data()}}, {HALF_YEAR}, 40,
                                       rq_cnc::kSlopes, 0, 0, fit.data());
            for (int t = 1; t < n; ++t) out[t] = fit[t - 1];"""
        ),
    )
    out["beta"] = regression(market[:1], HALF_YEAR, 40, "kBeta")
    # Past returns
    out["m1"] = momentum(MONTH, 0)
    out["m11"] = momentum(YEAR, MONTH)
    out["m60"] = momentum(5 * YEAR, YEAR)
    out["m6"] = momentum(6 * MONTH, MONTH)
    out["m3"] = momentum(3 * MONTH, MONTH)
    out["indmom"] = _industry_mean(momentum(6 * MONTH, 0), float_value, inputs["sector"])
    out["m24"] = momentum(3 * YEAR, YEAR)
    out["mchg"] = momentum(6 * MONTH, MONTH) - momentum(YEAR, 6 * MONTH)
    residual_window = 720
    out["im12"] = regression(market, residual_window, 360, "kResidualMomentum", residual_window - YEAR,
                             residual_window - MONTH)
    out["im6"] = regression(market, residual_window, 360, "kResidualMomentum", residual_window - 6 * MONTH,
                            residual_window - MONTH)
    out["w52"] = k.series_kernel(
        [close],
        "const std::vector<double> price = rq_cnc::forward_filled(n, x0);\n"
        "for (int t = 0; t < n; ++t) out[t] = rq_cnc::safe_div(price[t], "
        "rq_cnc::rolling(price.data(), t, 240, 120, rq_cnc::kMax));",
    )
    out["mdr"] = k.series_kernel(
        [close],
        on_returns(
            f"""
            // Mean of the five largest daily returns in the month.
            std::vector<double> window;
            for (int t = {MONTH} - 1; t < n; ++t) {{
                window.clear();
                for (int i = t - {MONTH} + 1; i <= t; ++i) if (std::isfinite(r[i])) window.push_back(r[i]);
                if (window.size() < 15) continue;
                std::sort(window.begin(), window.end(), [](double a, double b) {{ return a > b; }});
                double total = 0.0;
                for (int i = 0; i < 5; ++i) total += window[i];
                out[t] = total / 5;
            }}"""
        ),
    )
    out["pr"] = inputs["basic_close"]
    out["abr"] = k.series_kernel(
        [close, market[0], inputs["announced"]],
        on_returns(
            """
            // Abnormal return over days -2..+1 around the latest announcement, published on day +1 and carried forward.
            std::vector<double> abnormal(n);
            for (int t = 0; t < n; ++t) abnormal[t] = r[t] - x1[t];
            double latest = rq_cnc::kNaN;
            for (int t = 0; t < n; ++t) {
                if (x2[t] == 1.0) {
                    const double total = rq_cnc::rolling(abnormal.data(), t, 4, 3, rq_cnc::kSum);
                    if (std::isfinite(total)) latest = total;
                }
                out[t] = latest;
            }"""
        ),
    )
    out["season"] = k.series_kernel(
        [close],
        f"""
        // Mean same-calendar-month return over the prior five years, with at least three of them.
        const std::vector<double> price = rq_cnc::forward_filled(n, x0);
        std::vector<double> monthly(n, rq_cnc::kNaN);
        for (int t = {MONTH}; t < n; ++t) monthly[t] = price[t] / price[t - {MONTH}] - 1.0;
        for (int t = 0; t < n; ++t) {{
            double total = 0.0;
            int count = 0;
            for (int years = 1; years <= 5; ++years) {{
                const int at = t - ({YEAR} * years - {MONTH});
                if (at >= 0 && std::isfinite(monthly[at])) {{ total += monthly[at]; ++count; }}
            }}
            if (count >= 3) out[t] = total / count;
        }}""",
    )
    # Statement-based characteristics that involve the market value (the others are ``STATEMENT_FACTORS``)
    for name, result in OVER_MARKET_VALUE.items():
        out[name] = to_price(inputs[result])
    ebitda = inputs["q_ebitda_ttm"]
    out["em"] = k.safe_div(cap + inputs["q_enterprise_debt"], k.mask(ebitda, k.gt(ebitda, 0.0)))
    out["ebp"] = k.safe_div(inputs["q_net_debt"] + inputs["q_book"], inputs["q_net_debt"] + cap)
    out["cei"] = k.series_kernel(
        [close, cap],
        f"""
        // One-year log growth of total market value minus log growth of the price.
        const std::vector<double> price = rq_cnc::forward_filled(n, x0), value = rq_cnc::forward_filled(n, x1);
        for (int t = {YEAR}; t < n; ++t)
            out[t] = std::log(rq_cnc::safe_div(value[t], value[t - {YEAR}]))
                - std::log(rq_cnc::safe_div(price[t], price[t - {YEAR}]));""",
    )
    months = (inputs["calendar_day"] - inputs["listed_day"]) / 30.4375
    out["age"] = k.mask(months, k.ge(months, 0.0))
    out["etr"] = inputs["y_etr_rate_change"] * to_price(inputs["y_earnings_change"])
    market_assets = inputs["q_debt_value"] + cap
    out["alm"] = k.safe_div(inputs["q_asset_liquidity"], k.mask(market_assets, k.gt(market_assets, 0.0)))
    return out


def _industry_mean(value: Any, weight: Any, sector: Any) -> Any:
    """Weighted mean of ``value`` over each day's members of the stock's sector (weights positive, values known)."""
    return _industry_mean_operator()(value, weight, sector)


@lru_cache(maxsize=1)
def _industry_mean_operator() -> type:
    base = kunquant_symbols("GenericCrossSectionalOp")["GenericCrossSectionalOp"]

    class IndustryMean(base):
        def __init__(self, value: Any, weight: Any, sector: Any) -> None:
            super().__init__([value, weight, sector], None)

        def generate_head(self) -> str:
            return "std::vector<T> rq_codes, rq_weighted, rq_weights;\n"

        def generate_body(self) -> str:
            return r"""
            rq_codes.clear();
            rq_weighted.clear();
            rq_weights.clear();
            for (size_t i = 0; i < num_stocks; ++i) {
                const T code = input_2[i];
                if (std::isnan(code)) continue;
                const size_t group = std::find(rq_codes.begin(), rq_codes.end(), code) - rq_codes.begin();
                if (group == rq_codes.size()) {
                    rq_codes.push_back(code);
                    rq_weighted.push_back(0);
                    rq_weights.push_back(0);
                }
                const T value = input_0[i], weight = input_1[i];
                if (std::isnan(value) || !(weight > 0)) continue;
                rq_weighted[group] += value * weight;
                rq_weights[group] += weight;
            }
            for (size_t i = 0; i < num_stocks; ++i) {
                const T code = input_2[i];
                const size_t group = std::find(rq_codes.begin(), rq_codes.end(), code) - rq_codes.begin();
                output_0[i] = group == rq_codes.size() ? std::numeric_limits<T>::quiet_NaN()
                                                       : rq_weighted[group] / rq_weights[group];
            }
            """

    IndustryMean.__name__ = "RQuantIndustryMean"
    return IndustryMean
