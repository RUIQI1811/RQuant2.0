"""The Jensen-Kelly-Pedersen (JKP) global factor characteristics, evaluated as daily point-in-time A-share panels.

The catalog follows the 153 characteristics of "Is There a Replication Crisis in Finance?" (Jensen, Kelly and Pedersen,
Journal of Finance 2023) as listed in the authors' ``factor_details`` table, with the constructions of their public
code base. This module is an independent re-implementation: no upstream code or data is used, inputs come from
Tushare, and every characteristic is re-dated to the trading day on which it is known.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days and a year 252; JKP's monthly
  windows (``_21d``, ``_126d``, ``_252d``, ``_1260d``, ``ret_x_y``, ``seas_*``) become the same number of trading days
  or of 21-day blocks ending on the row.
* Returns are close-to-close returns of RQuant's backward-adjusted prices; the market, SMB and HML returns come from
  ``reference/market_factors.parquet``. Returns are not reduced by a risk-free rate.
* Accounting characteristics use the first-reported statements of ``reference/fundamentals.parquet`` on the quarter
  grid of ``china_chars``: a quarter is usable once it and every earlier quarter are announced. Flows are
  trailing-twelve-month sums of single quarters (JKP's ``_x`` annual items, refreshed quarterly), ``q`` items are the
  latest single quarter, balance-sheet items the latest quarter end, and JKP's 12-month lags are four quarters.
  Compustat items map to Tushare lines as documented in ``ITEMS``; preferred stock is ``oth_eqt_tools_p_shr``, and
  share buybacks, which Chinese cash-flow statements do not report separately, are zero.
* Dividends are the trailing-twelve-month dividend yield of daily_basic times the total market value.
* The cross-sectional composites (mispricing and quality-minus-junk) rank only the research universe's members of the
  day (the CSI 300 by default), so later index members do not shape earlier cross-sections.
* Every output is missing on days the stock has no quote.
* Statement formulas run in a KunQuant graph over fiscal quarters and reach the daily graph from their first usable
  announcement day; windows, regressions and cross-sectional ranks are per-stock or per-day C++ kernels. The original
  pandas implementation is the parity reference in ``tests/panel_reference``.

Left out because Tushare data at this account level cannot build them: employee counts (``emp_gr1``,
``sale_emp_gr1``), the q-factor model moments (``ivol_hxz4_21d``, ``iskew_hxz4_21d``), seasonality beyond ten years
(``seas_11_15*``, ``seas_16_20*``, before the data starts), and ``eqpo_me``, which equals ``div12m_me`` without
buybacks.
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
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.extensions import kunquant_oap_regression as regression_op
from rquant.factors.extensions.kunquant_oap import g
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.factors.libraries.china_chars import CPP_PRELUDE as CNC_PRELUDE
from rquant.io import sha256_file, stable_hash

MONTH = 21
YEAR = 252
JKP_DAILY_BASIC = {"turnover_rate": 0.01, "close": 1.0, "total_share": 10_000.0, "dv_ttm": 0.01}
MARKET_FACTORS = ("mkt", "smb", "hml")
DISCOUNT = 0.12

# Compustat item: Tushare statement lines summed into it.
ITEMS: dict[str, tuple[str, ...]] = {
    "at": ("total_assets",),
    "act": ("total_cur_assets",),
    "lct": ("total_cur_liab",),
    "lt": ("total_liab",),
    "che": ("money_cap", "trad_asset"),
    "ivst": ("trad_asset",),
    "ivao": ("lt_eqt_invest", "fa_avail_for_sale", "htm_invest"),
    "rect": ("accounts_receiv", "notes_receiv"),
    "invt": ("inventories",),
    "ppegt": ("fix_assets", "cip"),
    "ppent": ("fix_assets",),
    "intan": ("intan_assets", "goodwill"),
    "nca": ("total_nca",),
    "ncl": ("total_ncl",),
    "dlc": ("st_borr", "non_cur_liab_due_1y"),
    "dltt": ("lt_borr", "bond_payable"),
    "ap": ("acct_payable", "notes_payable"),
    "txp": ("taxes_payable",),
    "txditc": ("defer_tax_liab",),
    "pstk": ("oth_eqt_tools_p_shr",),
    "seq": ("total_hldr_eqy_exc_min_int",),
    "mib": ("minority_int",),
    "re": ("surplus_rese", "undistr_porfit"),
}
FLOWS: dict[str, tuple[str, ...]] = {
    "sale": ("revenue",),
    "cogs": ("oper_cost",),
    "xsga": ("sell_exp", "admin_exp", "rd_exp"),
    "xrd": ("rd_exp",),
    "dp": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp"),
    "xint": ("fin_exp_int_exp",),
    "txt": ("income_tax",),
    "pi": ("total_profit",),
    "ni": ("n_income_attr_p",),
    "oancf": ("n_cashflow_act",),
    "capx": ("c_pay_acq_const_fiolta",),
    "sstk": ("c_recp_cap_contrib",),
    "dltis": ("c_recp_borrow", "proc_issue_bonds"),
    "dltr": ("c_prepay_amt_borr",),
    "fincf": ("n_cash_flows_fnc_act",),
}
# Statement lines that mean "none reported" when blank, keyed by the line that shows the statement exists.
ZERO_FILL = {
    "total_assets": ("money_cap", "trad_asset", "lt_eqt_invest", "fa_avail_for_sale", "htm_invest",
                     "accounts_receiv", "notes_receiv", "inventories", "fix_assets", "cip", "intan_assets",
                     "goodwill", "st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable", "acct_payable",
                     "notes_payable", "taxes_payable", "defer_tax_liab", "oth_eqt_tools_p_shr", "minority_int",
                     "surplus_rese", "undistr_porfit"),
    "revenue": ("sell_exp", "admin_exp", "rd_exp", "fin_exp_int_exp", "oper_cost"),
    "n_cashflow_act": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp", "c_pay_acq_const_fiolta",
                       "c_recp_cap_contrib", "c_recp_borrow", "proc_issue_bonds", "c_prepay_amt_borr",
                       "n_cash_flows_fnc_act"),
}
STATEMENT_LINES = tuple(sorted({line for lines in (*ITEMS.values(), *FLOWS.values()) for line in lines}))

# (source name, JKP theme, description, maximum lookback in trading days). Order is the stable public column order.
JKP_CHARS: tuple[tuple[str, str, str, int], ...] = (
    # Momentum
    ("niq_su", "momentum", "standardized earnings surprise over the prior eight quarters", 1),
    ("ret_6_1", "momentum", "return from day -126 to -21", 126),
    ("ret_12_1", "momentum", "return from day -252 to -21", YEAR),
    ("saleq_su", "momentum", "standardized revenue surprise over the prior eight quarters", 1),
    ("tax_gr1a", "momentum", "one-year change in tax expense over assets", 1),
    ("ni_inc8q", "momentum", "consecutive quarters (of the last eight) with year-on-year earnings increases", 1),
    ("prc_highprc_252d", "momentum", "price over its 252-day high", YEAR),
    ("resff3_6_1", "momentum", "FF3 residual momentum, months t-6..t-1 of a 756-day fit", 756),
    ("resff3_12_1", "momentum", "FF3 residual momentum, months t-12..t-1 of a 756-day fit", 756),
    # Value
    ("be_me", "value", "book equity over market equity", 1),
    ("debt_me", "value", "debt over market equity", 1),
    ("at_me", "value", "assets over market equity", 1),
    ("ret_60_12", "value", "return from day -1260 to -252", 5 * YEAR),
    ("ni_me", "value", "trailing earnings over market equity", 1),
    ("fcf_me", "value", "trailing free cash flow over market equity", 1),
    ("div12m_me", "value", "trailing twelve-month dividends over market equity", 1),
    ("eqnpo_me", "value", "dividends minus net equity issuance, over market equity", 1),
    ("sale_gr3", "value", "three-year sales growth", 1),
    ("sale_gr1", "value", "one-year sales growth", 1),
    ("ebitda_mev", "value", "EBITDA over enterprise value", 1),
    ("sale_me", "value", "sales over market equity", 1),
    ("ocf_me", "value", "operating cash flow over market equity", 1),
    ("ival_me", "value", "residual-income intrinsic value over market equity", 1),
    ("bev_mev", "value", "book enterprise value over enterprise value", 1),
    ("netdebt_me", "value", "net debt over market equity", 1),
    ("eq_dur", "value", "equity duration from ten-year AR(1) projections of ROE and growth", 1),
    # Investment
    ("capex_abn", "investment", "capex-to-sales over its average one to three years ago, minus 1", 1),
    ("at_gr1", "investment", "one-year asset growth", 1),
    ("ppeinv_gr1a", "investment", "one-year change in PP&E plus inventory over lagged assets", 1),
    ("noa_at", "investment", "net operating assets over lagged assets", 1),
    ("noa_gr1a", "investment", "one-year change in net operating assets over lagged assets", 1),
    ("lnoa_gr1a", "investment", "one-year change in long-term net operating assets over current plus lagged assets", 1),
    ("capx_gr1", "investment", "one-year capex growth", 1),
    ("capx_gr2", "investment", "two-year capex growth", 1),
    ("capx_gr3", "investment", "three-year capex growth", 1),
    ("chcsho_12m", "investment", "252-day change in total shares", YEAR),
    ("eqnpo_12m", "investment", "252-day log return minus log change in market equity", YEAR),
    ("debt_gr3", "investment", "three-year debt growth", 1),
    ("inv_gr1", "investment", "one-year inventory growth", 1),
    ("inv_gr1a", "investment", "one-year change in inventory over assets", 1),
    ("oaccruals_at", "investment", "operating accruals over assets", 1),
    ("taccruals_at", "investment", "total accruals over assets", 1),
    ("cowc_gr1a", "investment", "one-year change in current operating working capital over assets", 1),
    ("coa_gr1a", "investment", "one-year change in current operating assets over assets", 1),
    ("col_gr1a", "investment", "one-year change in current operating liabilities over assets", 1),
    ("nncoa_gr1a", "investment", "one-year change in net non-current operating assets over assets", 1),
    ("ncoa_gr1a", "investment", "one-year change in non-current operating assets over assets", 1),
    ("ncol_gr1a", "investment", "one-year change in non-current operating liabilities over assets", 1),
    ("nfna_gr1a", "investment", "one-year change in net financial assets over assets", 1),
    ("sti_gr1a", "investment", "one-year change in short-term investments over assets", 1),
    ("lti_gr1a", "investment", "one-year change in long-term investments over assets", 1),
    ("fnl_gr1a", "investment", "one-year change in financial liabilities over assets", 1),
    ("be_gr1a", "investment", "one-year change in book equity over assets", 1),
    ("oaccruals_ni", "investment", "operating accruals over absolute net income", 1),
    ("taccruals_ni", "investment", "total accruals over absolute net income", 1),
    ("netis_at", "investment", "net equity plus debt issuance over assets", 1),
    ("eqnetis_at", "investment", "net equity issuance over assets", 1),
    ("dbnetis_at", "investment", "net debt issuance over assets", 1),
    # Profitability
    ("niq_be", "profitability", "quarterly earnings over book equity a quarter earlier", 1),
    ("niq_be_chg1", "profitability", "one-year change in niq_be", 1),
    ("niq_at", "profitability", "quarterly earnings over assets a quarter earlier", 1),
    ("niq_at_chg1", "profitability", "one-year change in niq_at", 1),
    ("ebit_bev", "profitability", "EBIT over book enterprise value", 1),
    ("ebit_sale", "profitability", "EBIT over sales", 1),
    ("sale_bev", "profitability", "sales over book enterprise value", 1),
    ("at_turnover", "profitability", "sales over average assets", 1),
    ("gp_at", "profitability", "gross profit over assets", 1),
    ("gp_atl1", "profitability", "gross profit over lagged assets", 1),
    ("ope_be", "profitability", "operating profit after interest over book equity", 1),
    ("ope_bel1", "profitability", "operating profit after interest over lagged book equity", 1),
    ("op_at", "profitability", "operating profit before R&D over assets", 1),
    ("op_atl1", "profitability", "operating profit before R&D over lagged assets", 1),
    ("cop_at", "profitability", "cash-based operating profit over assets", 1),
    ("cop_atl1", "profitability", "cash-based operating profit over lagged assets", 1),
    ("f_score", "profitability", "Piotroski F-score", 1),
    ("o_score", "profitability", "Ohlson O-score", 1),
    ("z_score", "profitability", "Altman Z-score", 1),
    ("pi_nix", "profitability", "pretax income over net income", 1),
    ("at_be", "profitability", "assets over book equity", 1),
    ("saleq_gr1", "profitability", "quarterly sales growth over four quarters", 1),
    # Intangibles
    ("rd_me", "intangibles", "R&D expense over market equity", 1),
    ("rd_sale", "intangibles", "R&D expense over sales", 1),
    ("opex_at", "intangibles", "operating expenses over assets", 1),
    ("rd5_at", "intangibles", "five-year depreciated R&D capital over assets", 1),
    ("age", "intangibles", "months since listing", 1),
    ("dsale_dinv", "intangibles", "% change in sales minus % change in inventory (vs two-year average)", 1),
    ("dsale_drec", "intangibles", "% change in sales minus % change in receivables", 1),
    ("dgp_dsale", "intangibles", "% change in gross profit minus % change in sales", 1),
    ("dsale_dsga", "intangibles", "% change in sales minus % change in SG&A", 1),
    ("tangibility", "intangibles", "weighted liquid and fixed assets over assets", 1),
    ("kz_index", "intangibles", "Kaplan-Zingales index", 1),
    ("ocfq_saleq_std", "intangibles", "std of quarterly operating cash flow over sales, 16 quarters", 1),
    ("cash_at", "intangibles", "cash and short-term investments over assets", 1),
    ("ni_ar1", "intangibles", "AR(1) slope of earnings over assets, five annual pairs", 1),
    ("ni_ivol", "intangibles", "residual std of the earnings AR(1)", 1),
    ("earnings_variability", "intangibles", "std of ROA over std of cash ROA, five years", 1),
    ("aliq_at", "intangibles", "asset liquidity over lagged assets", 1),
    ("aliq_mat", "intangibles", "asset liquidity over market assets a year earlier", YEAR),
    ("seas_1_1an", "intangibles", "return in the same month a year ago", YEAR),
    ("seas_1_1na", "intangibles", "average return in the other months of the last year", YEAR),
    ("seas_2_5an", "intangibles", "average same-month return two to five years ago", 5 * YEAR),
    ("seas_2_5na", "intangibles", "average other-month return two to five years ago", 5 * YEAR),
    ("seas_6_10an", "intangibles", "average same-month return six to ten years ago", 10 * YEAR),
    ("seas_6_10na", "intangibles", "average other-month return six to ten years ago", 10 * YEAR),
    # Trading frictions
    ("market_equity", "frictions", "total market value", 1),
    ("ivol_ff3_21d", "frictions", "FF3 residual volatility, 21 days", MONTH),
    ("ivol_capm_252d", "frictions", "CAPM residual volatility, 252 days", YEAR),
    ("ivol_capm_21d", "frictions", "CAPM residual volatility, 21 days", MONTH),
    ("rvol_21d", "frictions", "return volatility, 21 days", MONTH),
    ("beta_60m", "frictions", "market beta from sixty 21-day returns", 60 * MONTH),
    ("betabab_1260d", "frictions", "Frazzini-Pedersen beta: 1260-day correlation times 252-day volatility ratio",
     5 * YEAR + 2),
    ("beta_dimson_21d", "frictions", "Dimson beta with lead and lag market returns, 21 days", MONTH + 1),
    ("turnover_126d", "frictions", "average daily turnover, 126 days", 126),
    ("turnover_var_126d", "frictions", "coefficient of variation of daily turnover, 126 days", 126),
    ("dolvol_126d", "frictions", "average traded value, 126 days", 126),
    ("dolvol_var_126d", "frictions", "coefficient of variation of traded value, 126 days", 126),
    ("prc", "frictions", "unadjusted share price", 1),
    ("ami_126d", "frictions", "Amihud illiquidity |r| / value * 1e6, 126 days", 126),
    ("zero_trades_21d", "frictions", "days without trading, turnover tiebreaker, 21 days", MONTH),
    ("zero_trades_126d", "frictions", "days without trading, turnover tiebreaker, 126 days scaled to 21", 126),
    ("zero_trades_252d", "frictions", "days without trading, turnover tiebreaker, 252 days scaled to 21", YEAR),
    ("rmax1_21d", "frictions", "largest daily return, 21 days", MONTH),
    ("rskew_21d", "frictions", "return skewness, 21 days", MONTH),
    ("iskew_capm_21d", "frictions", "CAPM residual skewness, 21 days", MONTH),
    ("iskew_ff3_21d", "frictions", "FF3 residual skewness, 21 days", MONTH),
    ("coskew_21d", "frictions", "co-skewness with the market, 21 days", MONTH),
    ("ret_1_0", "frictions", "return over the last 21 days", MONTH),
    ("betadown_252d", "frictions", "market beta on days the market falls, 252 days", YEAR),
    ("bidaskhl_21d", "frictions", "Corwin-Schultz high-low bid-ask spread, 21 days", MONTH + 1),
    # New
    ("ret_3_1", "new", "return from day -63 to -21", 3 * MONTH),
    ("ret_9_1", "new", "return from day -189 to -21", 9 * MONTH),
    ("ret_12_7", "new", "return from day -252 to -126", YEAR),
    ("corr_1260d", "new", "correlation of three-day returns with the market, 1260 days", 5 * YEAR + 2),
    ("rmax5_21d", "new", "average of the five largest daily returns, 21 days", MONTH),
    ("rmax5_rvol_21d", "new", "rmax5_21d over rvol_21d", MONTH),
    ("ni_be", "new", "trailing earnings over book equity", 1),
    ("ocf_at", "new", "operating cash flow over assets", 1),
    ("ocf_at_chg1", "new", "one-year change in ocf_at", 1),
    ("mispricing_perf", "new", "Stambaugh-Yuan performance mispricing score (percentile mean)", YEAR),
    ("mispricing_mgmt", "new", "Stambaugh-Yuan management mispricing score (percentile mean)", YEAR),
    ("qmj", "new", "quality minus junk composite", 5 * YEAR + 2),
    ("qmj_prof", "new", "quality minus junk: profitability", 1),
    ("qmj_growth", "new", "quality minus junk: five-year profitability growth", 1),
    ("qmj_safety", "new", "quality minus junk: safety", 5 * YEAR + 2),
)
JKP_NAMES = tuple(entry[0] for entry in JKP_CHARS)

DAILY_REFERENCE = (
    *(f"basic_{field}" for field in JKP_DAILY_BASIC),
    *MARKET_FACTORS,
    "universe",
    "calendar_day",
    "listed_day",
)

# Per-stock C++ helpers of the daily graph; every helper keeps the pandas reference's windows and missing values.
CPP_PRELUDE = r"""
#ifndef RQUANT_JKP_CPP_HELPERS
#define RQUANT_JKP_CPP_HELPERS
namespace rq_jkp {
// Per-day percentile rank (or rank z-score) of ``value`` among the finite universe members of the day.
// The cross-sectional op passes buffer accessors, so the arguments are generic indexables.
template <class Values, class Universe, class Out>
static inline void cross_rank(size_t n, Values &&values, Universe &&universe, bool z_score, Out &&out) {
    std::vector<double> members;
    for (size_t i = 0; i < n; ++i) if (universe[i] > 0 && std::isfinite(values[i])) members.push_back(values[i]);
    for (size_t i = 0; i < n; ++i) out[i] = NAN;
    if (members.size() < 3) return;
    std::sort(members.begin(), members.end());
    const double count = (double)members.size();
    double mean = 0, squares = 0;
    if (z_score) {
        for (size_t r = 1; r <= members.size(); ++r) mean += (double)r;
        mean /= count;
        for (size_t r = 1; r <= members.size(); ++r) squares += ((double)r - mean) * ((double)r - mean);
    }
    const double sd = std::sqrt(squares / (count - 1));
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        const double lower = (double)(std::lower_bound(members.begin(), members.end(), values[i]) - members.begin());
        const double upper = (double)(std::upper_bound(members.begin(), members.end(), values[i]) - members.begin());
        const double rank = (lower + upper + 1.0) / 2.0;
        out[i] = z_score ? (rank - mean) / sd : rank / count;
    }
}
}  // namespace rq_jkp
#endif
"""


# ---------------------------------------------------------------------------------------------- expression helpers
def lag(value, periods: int):
    return op.shift(value, periods)


def ratio(numerator, denominator):
    return op.ratio(numerator, denominator)


def positive(value):
    return g.where(value > 0, value, g.nan)


def growth(current, previous):
    """Relative change, defined only for a positive base."""
    return ratio(current, positive(previous)) - 1.0


def finite_mean(values: list, minimum: int):
    """Mean of the finite values (``statements.nan_mean``), given at least ``minimum`` of them."""
    known = [g.isfinite(value) for value in values]
    count = sum(known)
    mean = sum(g.where(f, v, 0.0) for f, v in zip(known, values, strict=True)) / count
    return g.where(count >= minimum, mean, g.nan)


def cross_rank(values, universe, *, z_score: bool):
    flag = "true" if z_score else "false"
    return op.cross(f"rq_jkp::cross_rank(num_stocks, input_0, input_1, {flag}, output_0);", values, universe)


def corwin_schultz(high, low):
    """Daily Corwin-Schultz spread from today's and yesterday's high and low; negative estimates are zero."""
    single = g.log(high / low) ** 2
    beta = single + lag(single, 1)
    two_high = g.fmax(high, lag(high, 1))
    two_low = g.fmin(low, lag(low, 1))
    gamma = g.where(g.isfinite(lag(high, 1)), g.log(two_high / two_low) ** 2, g.nan)
    denominator = 3.0 - 2.0 * 2.0**0.5
    alpha = (g.sqrt(2.0 * beta) - g.sqrt(beta)) / denominator - g.sqrt(gamma / denominator)
    spread = 2.0 * (g.exp(alpha) - 1.0) / (1.0 + g.exp(alpha))
    return g.where(g.isfinite(spread), op.scalar("std::max(x0, 0.0)", spread), g.nan)


class JKPGraph:
    """Build statement and daily expressions mirroring the pandas reference's ``JKPChars``; nothing is evaluated."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self.statements: dict[str, Any] = {}
        self.quarter = op.input_("fiscal_quarter")
        self.close = op.input_("close")
        self.trading = g.isfinite(self.close)
        self.price = op.ffill(self.close)
        self.returns = self.close / lag(self.close, 1) - 1.0
        self.universe = op.input_("universe")

    # ------------------------------------------------------------------ infrastructure
    def calculate(self, name: str):
        def build():
            value = getattr(self, f"_c_{name}")()
            return g.where(g.isfinite(value) & self.trading, value, g.nan)

        return self._memo(f"out:{name}", build)

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _q(self, values):
        """Publish a quarter-grid value on every trading day from the latest usable quarter."""
        key = f"q:{id(values)}"
        if key not in self._cache:
            name = f"q_jkp_{len(self.statements):03d}"
            self.statements[name] = values
            self._cache[key] = (op.input_(name), values)
        return self._cache[key][0]

    def _input(self, name: str):
        return op.input_(name)

    # ------------------------------------------------------------------ quarter items
    def value(self, line: str):
        return op.input_(line)

    def single(self, line: str):
        ytd = self.value(line)
        return self._memo(f"single:{line}", lambda: g.where(self.quarter == 0, ytd, ytd - lag(ytd, 1)))

    def ttm(self, line: str):
        def build():
            flow = self.single(line)
            return flow + lag(flow, 1) + lag(flow, 2) + lag(flow, 3)

        return self._memo(f"ttm:{line}", build)

    def x(self, name: str):
        return self._memo(f"x:{name}", lambda: self._item(name))

    def _item(self, name: str):
        if name in ITEMS:
            values = sum(self.value(line) for line in ITEMS[name])
            return g.where(values >= 0, values, g.nan) if name == "at" else values
        if name in FLOWS:
            return sum(self.ttm(line) for line in FLOWS[name])
        return getattr(self, f"_x_{name}")()

    def quarterly(self, line: str):
        return self.single(line)

    def _x_debt(self):
        return self.x("dltt") + self.x("dlc")

    def _x_gp(self):
        return self.x("sale") - self.x("cogs")

    def _x_opex(self):
        return self.x("cogs") + self.x("xsga")

    def _x_ebitda(self):
        return self.x("sale") - self.x("opex")

    def _x_ebit(self):
        return self.x("ebitda") - self.x("dp")

    def _x_op(self):
        return self.x("ebitda") + self.x("xrd")

    def _x_ope(self):
        return self.x("ebitda") - self.x("xint")

    def _x_coa(self):
        return self.x("act") - self.x("che")

    def _x_col(self):
        return self.x("lct") - self.x("dlc")

    def _x_cowc(self):
        return self.x("coa") - self.x("col")

    def _x_ncoa(self):
        return self.x("at") - self.x("act") - self.x("ivao")

    def _x_ncol(self):
        return self.x("lt") - self.x("lct") - self.x("dltt")

    def _x_nncoa(self):
        return self.x("ncoa") - self.x("ncol")

    def _x_noa(self):
        return (self.x("coa") + self.x("ncoa")) - (self.x("col") + self.x("ncol"))

    def _x_lnoa(self):
        return (self.x("nca") - self.x("ivao")) - (self.x("ncl") - self.x("dltt")) + self.x("dp")

    def _x_fna(self):
        return self.x("ivst") + self.x("ivao")

    def _x_fnl(self):
        return self.x("debt") + self.x("pstk")

    def _x_nfna(self):
        return self.x("fna") - self.x("fnl")

    def _x_be(self):
        return positive(self.x("seq") + self.x("txditc") - self.x("pstk"))

    def _x_bev(self):
        return positive(self.x("debt") - self.x("che") + self.x("seq") + self.x("mib"))

    def _x_ppeinv(self):
        return self.x("ppegt") + self.x("invt")

    def _x_oacc(self):
        return self.x("ni") - self.x("oancf")

    def _x_tacc(self):
        nfna = self.x("nfna")
        return self.x("oacc") + nfna - lag(nfna, 4)

    def _x_cop(self):
        return self.x("ebitda") + self.x("xrd") - self.x("oacc")

    def _x_fcf(self):
        return self.x("oancf") - self.x("capx")

    def _x_dbnetis(self):
        return self.x("dltis") - self.x("dltr")

    def _x_netis(self):
        return self.x("sstk") + self.x("dbnetis")

    def _x_aliq(self):
        return self.x("che") + 0.75 * self.x("coa") + 0.5 * (self.x("at") - self.x("act") - self.x("intan"))

    # ------------------------------------------------------------------ quarter-space operators
    def over_at(self, values, *, lagged: bool = False):
        assets = self.x("at")
        return ratio(values, positive(lag(assets, 4) if lagged else assets))

    def change_over_at(self, name: str):
        values = self.x(name)
        return self._q(ratio(values - lag(values, 4), positive(self.x("at"))))

    def change_to_expected(self, values):
        base = (lag(values, 4) + lag(values, 8)) / 2.0
        return ratio(values, positive(base)) - 1.0

    def standardized_unexpected(self, values):
        change = values - lag(values, 4)
        history = [lag(change, k) for k in range(1, 9)]
        known = [g.isfinite(value) for value in history]
        count = sum(known)
        mean = sum(g.where(f, v, 0.0) for f, v in zip(known, history, strict=True)) / count
        squares = sum(g.where(f, (v - mean) * (v - mean), 0.0) for f, v in zip(known, history, strict=True))
        std = g.sqrt(squares / (count - 1))
        return g.where(count >= 6, ratio(change - mean, std), g.nan)

    # ------------------------------------------------------------------ daily building blocks
    def me(self):
        return self._memo("me", lambda: positive(self._input("cap")))

    def dividends(self):
        def build():
            dividend_yield = self._input("basic_dv_ttm")
            known = g.isfinite(dividend_yield) | g.isfinite(self._input("basic_turnover_rate"))
            return g.where(known, g.where(g.isfinite(dividend_yield), dividend_yield, 0.0) * self.me(), g.nan)

        return self._memo("dividends", build)

    def mev(self):
        def build():
            value = self.me() + self._q(self.x("debt") - self.x("che") + self.x("pstk") + self.x("mib"))
            return positive(value)

        return self._memo("mev", build)

    def _momentum(self, start: int, end: int):
        return lag(self.price, end) / lag(self.price, start) - 1.0

    def _monthly_block(self, k: int):
        """The 21-day return ending ``21 k`` rows ago."""
        return lag(self.price, MONTH * k) / lag(self.price, MONTH * (k + 1)) - 1.0

    def _regression(self, regressors: tuple[str, ...], window: int, minimum: int, kind: str, **slices: int):
        key = f"fit:{regressors}:{window}:{minimum}:{kind}:{slices}"
        xs = [self._input(name) for name in regressors]
        return self._memo(key, lambda: op.regression(self.returns, xs, window, minimum, kind, **slices))

    def _turnover(self):
        return self._input("basic_turnover_rate")

    def _variation(self, value, window: int, minimum: int):
        return ratio(op.trailing(value, window, minimum, "std"), op.trailing(value, window, minimum, "mean"))

    # ------------------------------------------------------------------ momentum
    def _c_niq_su(self):
        return self._q(self.standardized_unexpected(self.quarterly("n_income_attr_p")))

    def _c_ret_6_1(self):
        return self._momentum(6 * MONTH, MONTH)

    def _c_ret_12_1(self):
        return self._momentum(YEAR, MONTH)

    def _c_saleq_su(self):
        return self._q(self.standardized_unexpected(self.quarterly("revenue")))

    def _c_tax_gr1a(self):
        return self.change_over_at("txt")

    def _c_ni_inc8q(self):
        earnings = self.quarterly("n_income_attr_p")
        increases = []
        for k in range(8):
            current, previous = lag(earnings, k), lag(earnings, k + 4)
            known = g.isfinite(current) & g.isfinite(previous)
            increases.append(g.where(known, current > previous, g.nan))
        all_known = sum(g.isfinite(value) for value in increases) == 8
        run, product = 0.0, 1.0
        for value in increases:
            product = product * g.where(g.isfinite(value), value, 0.0)
            run = run + product
        return self._q(g.where(all_known, run, g.nan))

    def _c_prc_highprc_252d(self):
        return ratio(self.price, op.trailing(self.price, YEAR, 120, "max"))

    def _c_resff3_6_1(self):
        window = 756
        return self._regression(MARKET_FACTORS, window, 378, "kResidualMomentum", first=window - 6 * MONTH,
                                last=window - MONTH)

    def _c_resff3_12_1(self):
        window = 756
        return self._regression(MARKET_FACTORS, window, 378, "kResidualMomentum", first=window - YEAR,
                                last=window - MONTH)

    # ------------------------------------------------------------------ value
    def _to_me(self, values):
        return ratio(self._q(values), self.me())

    def _c_be_me(self):
        return self._to_me(self.x("be"))

    def _c_debt_me(self):
        return self._to_me(self.x("debt"))

    def _c_at_me(self):
        return self._to_me(self.x("at"))

    def _c_ret_60_12(self):
        return self._momentum(5 * YEAR, YEAR)

    def _c_ni_me(self):
        return self._to_me(self.x("ni"))

    def _c_fcf_me(self):
        return self._to_me(self.x("fcf"))

    def _c_div12m_me(self):
        return ratio(self.dividends(), self.me())

    def _c_eqnpo_me(self):
        return ratio(self.dividends() - self._q(self.x("sstk")), self.me())

    def _c_sale_gr3(self):
        sale = self.x("sale")
        return self._q(growth(sale, lag(sale, 12)))

    def _c_sale_gr1(self):
        sale = self.x("sale")
        return self._q(growth(sale, lag(sale, 4)))

    def _c_ebitda_mev(self):
        return ratio(self._q(self.x("ebitda")), self.mev())

    def _c_sale_me(self):
        return self._to_me(self.x("sale"))

    def _c_ocf_me(self):
        return self._to_me(self.x("oancf"))

    def _c_ival_me(self):
        be = self._q(self.x("be"))
        lagged_be = self._q(lag(self.x("be"), 4))
        income = self._q(self.x("ni"))
        assets = self._q(self.x("at"))
        dividends = self.dividends()
        payout = g.where(income > 0, ratio(dividends, positive(income)), ratio(dividends, assets * 0.06))
        roe = ratio(income, positive((be + lagged_be) / 2.0))
        next_be = (1.0 + (1.0 - payout) * roe) * be
        excess = roe - DISCOUNT
        value = be + excess / (1.0 + DISCOUNT) * be + excess / ((1.0 + DISCOUNT) * DISCOUNT) * next_be
        return ratio(positive(value), self.me())

    def _c_bev_mev(self):
        return ratio(self._q(self.x("bev")), self.mev())

    def _c_netdebt_me(self):
        return self._to_me(self.x("debt") - self.x("che"))

    def _c_eq_dur(self):
        horizon, roe_mean, roe_ar1, g_mean, g_ar1 = 10, 0.12, 0.57, 0.06, 0.24
        be = self.x("be")
        sale = self.x("sale")
        roe = ratio(self.x("ni"), g.where(lag(be, 4) > 1, lag(be, 4), g.nan))
        growth_rate = ratio(sale, g.where(lag(sale, 4) > 1, lag(sale, 4), g.nan)) - 1.0
        book = be
        weighted, plain, negative = 0.0, 0.0, op.as_expr(0.0)
        for t in range(1, horizon + 1):
            growth_rate = g_mean * (1 - g_ar1) + g_ar1 * growth_rate
            roe = roe_mean * (1 - roe_ar1) + roe_ar1 * roe
            dividend = book * (roe - growth_rate)
            book = book * (1 + growth_rate)
            negative = g.where((negative > 0) | (book < 0), 1.0, 0.0)
            weighted = weighted + t * dividend / (1 + DISCOUNT) ** t
            plain = plain + dividend / (1 + DISCOUNT) ** t
        constant = horizon + (1 + DISCOUNT) / DISCOUNT
        me = self.me()
        duration = ratio(self._q(weighted), me) + constant * ratio(me - self._q(plain), me)
        flagged = self._q(g.where(g.isfinite(be), negative, g.nan))
        return g.where((flagged == 1.0) | ~(duration > 0), g.nan, duration)

    # ------------------------------------------------------------------ investment
    def _c_capex_abn(self):
        intensity = ratio(self.x("capx"), positive(self.x("sale")))
        base = (lag(intensity, 4) + lag(intensity, 8) + lag(intensity, 12)) / 3.0
        return self._q(ratio(intensity, base) - 1.0)

    def _c_at_gr1(self):
        assets = self.x("at")
        return self._q(growth(assets, lag(assets, 4)))

    def _c_ppeinv_gr1a(self):
        values = self.x("ppeinv")
        return self._q(self.over_at(values - lag(values, 4), lagged=True))

    def _c_noa_at(self):
        return self._q(self.over_at(self.x("noa"), lagged=True))

    def _c_noa_gr1a(self):
        values = self.x("noa")
        return self._q(self.over_at(values - lag(values, 4), lagged=True))

    def _c_lnoa_gr1a(self):
        values = self.x("lnoa")
        assets = self.x("at")
        return self._q(ratio(values - lag(values, 4), positive(assets + lag(assets, 4))))

    def _capex_growth(self, quarters: int):
        capex = self.x("capx")
        return self._q(growth(capex, lag(capex, quarters)))

    def _c_capx_gr1(self):
        return self._capex_growth(4)

    def _c_capx_gr2(self):
        return self._capex_growth(8)

    def _c_capx_gr3(self):
        return self._capex_growth(12)

    def _c_chcsho_12m(self):
        shares = op.ffill(self._input("basic_total_share"))
        return ratio(shares, positive(lag(shares, YEAR))) - 1.0

    def _c_eqnpo_12m(self):
        me = op.ffill(self.me())
        return g.log(ratio(self.price, lag(self.price, YEAR))) - g.log(ratio(me, lag(me, YEAR)))

    def _c_debt_gr3(self):
        debt = self.x("debt")
        return self._q(growth(debt, lag(debt, 12)))

    def _c_inv_gr1(self):
        inventory = self.x("invt")
        return self._q(growth(inventory, lag(inventory, 4)))

    def _c_inv_gr1a(self):
        return self.change_over_at("invt")

    def _c_oaccruals_at(self):
        return self._q(self.over_at(self.x("oacc")))

    def _c_taccruals_at(self):
        return self._q(self.over_at(self.x("tacc")))

    def _c_cowc_gr1a(self):
        return self.change_over_at("cowc")

    def _c_coa_gr1a(self):
        return self.change_over_at("coa")

    def _c_col_gr1a(self):
        return self.change_over_at("col")

    def _c_nncoa_gr1a(self):
        return self.change_over_at("nncoa")

    def _c_ncoa_gr1a(self):
        return self.change_over_at("ncoa")

    def _c_ncol_gr1a(self):
        return self.change_over_at("ncol")

    def _c_nfna_gr1a(self):
        return self.change_over_at("nfna")

    def _c_sti_gr1a(self):
        return self.change_over_at("ivst")

    def _c_lti_gr1a(self):
        return self.change_over_at("ivao")

    def _c_fnl_gr1a(self):
        return self.change_over_at("fnl")

    def _c_be_gr1a(self):
        return self.change_over_at("be")

    def _c_oaccruals_ni(self):
        return self._q(ratio(self.x("oacc"), g.abs(self.x("ni"))))

    def _c_taccruals_ni(self):
        return self._q(ratio(self.x("tacc"), g.abs(self.x("ni"))))

    def _c_netis_at(self):
        return self._q(self.over_at(self.x("netis")))

    def _c_eqnetis_at(self):
        return self._q(self.over_at(self.x("sstk")))

    def _c_dbnetis_at(self):
        return self._q(self.over_at(self.x("dbnetis")))

    # ------------------------------------------------------------------ profitability
    def _niq_be(self):
        return self._memo("niq_be", lambda: ratio(self.quarterly("n_income_attr_p"), positive(lag(self.x("be"), 1))))

    def _niq_at(self):
        return self._memo("niq_at", lambda: ratio(self.quarterly("n_income_attr_p"), positive(lag(self.x("at"), 1))))

    def _c_niq_be(self):
        return self._q(self._niq_be())

    def _c_niq_be_chg1(self):
        values = self._niq_be()
        return self._q(values - lag(values, 4))

    def _c_niq_at(self):
        return self._q(self._niq_at())

    def _c_niq_at_chg1(self):
        values = self._niq_at()
        return self._q(values - lag(values, 4))

    def _c_ebit_bev(self):
        return self._q(ratio(self.x("ebit"), self.x("bev")))

    def _c_ebit_sale(self):
        return self._q(ratio(self.x("ebit"), positive(self.x("sale"))))

    def _c_sale_bev(self):
        return self._q(ratio(self.x("sale"), self.x("bev")))

    def _c_at_turnover(self):
        assets = self.x("at")
        return self._q(ratio(self.x("sale"), positive((assets + lag(assets, 4)) / 2.0)))

    def _c_gp_at(self):
        return self._q(self.over_at(self.x("gp")))

    def _c_gp_atl1(self):
        return self._q(self.over_at(self.x("gp"), lagged=True))

    def _c_ope_be(self):
        return self._q(ratio(self.x("ope"), self.x("be")))

    def _c_ope_bel1(self):
        return self._q(ratio(self.x("ope"), lag(self.x("be"), 4)))

    def _c_op_at(self):
        return self._q(self.over_at(self.x("op")))

    def _c_op_atl1(self):
        return self._q(self.over_at(self.x("op"), lagged=True))

    def _c_cop_at(self):
        return self._q(self.over_at(self.x("cop")))

    def _c_cop_atl1(self):
        return self._q(self.over_at(self.x("cop"), lagged=True))

    def _c_f_score(self):
        assets = self.x("at")
        lagged_assets = positive(lag(assets, 4))
        roa = ratio(self.x("ni"), lagged_assets)
        croa = ratio(self.x("oancf"), lagged_assets)
        leverage = ratio(self.x("dltt"), positive(assets))
        current = ratio(self.x("act"), positive(self.x("lct")))
        margin = ratio(self.x("gp"), positive(self.x("sale")))
        sale = self.x("sale")
        turnover = ratio(sale, lagged_assets) - ratio(lag(sale, 4), positive(lag(assets, 8)))
        sstk = self.x("sstk")
        signals = [
            roa > 0,
            croa > 0,
            roa - lag(roa, 4) > 0,
            croa - roa > 0,
            current - lag(current, 4) > 0,
            margin - lag(margin, 4) > 0,
            turnover > 0,
            leverage - lag(leverage, 4) < 0,
            g.where(g.isfinite(sstk), sstk, 0.0) == 0,
        ]
        inputs = (roa, croa, lag(roa, 4), current, lag(current, 4), margin, lag(margin, 4), turnover, leverage,
                  lag(leverage, 4))
        known = sum(g.isfinite(value) for value in inputs) == float(len(inputs))
        score = sum(g.where(signal, 1.0, 0.0) for signal in signals)
        return self._q(g.where(known, score, g.nan))

    def _o_score(self):
        def build():
            assets = self.x("at")
            liabilities = self.x("lt")
            income = self.x("ni")
            previous = lag(income, 4)
            log_assets = g.log(positive(assets))
            negative_equity = g.where(g.isfinite(liabilities - assets), liabilities > assets, g.nan)
            negative_earnings = g.where(g.isfinite(income + previous), (income < 0) & (previous < 0), g.nan)
            return (
                -1.32
                - 0.407 * log_assets
                + 6.03 * ratio(self.x("debt"), positive(assets))
                - 1.43 * ratio(self.x("act") - self.x("lct"), positive(assets))
                + 0.076 * ratio(self.x("lct"), positive(self.x("act")))
                - 1.72 * negative_equity
                - 2.37 * ratio(income, positive(assets))
                - 1.83 * ratio(self.x("pi") + self.x("dp"), positive(liabilities))
                + 0.285 * negative_earnings
                - 0.52 * ratio(income - previous, g.abs(income) + g.abs(previous))
            )

        return self._memo("o_score", build)

    def _c_o_score(self):
        return self._q(self._o_score())

    def _c_z_score(self):
        assets = positive(self.x("at"))
        quarterly = (
            1.2 * ratio(self.x("act") - self.x("lct"), assets)
            + 1.4 * ratio(self.x("re"), assets)
            + 3.3 * ratio(self.x("ebitda"), assets)
            + 1.0 * ratio(self.x("sale"), assets)
        )
        return self._q(quarterly) + 0.6 * ratio(self.me(), self._q(positive(self.x("lt"))))

    def _c_pi_nix(self):
        return self._q(ratio(positive(self.x("pi")), positive(self.x("ni"))))

    def _c_at_be(self):
        return self._q(ratio(self.x("at"), self.x("be")))

    def _c_saleq_gr1(self):
        sales = self.quarterly("revenue")
        return self._q(growth(sales, lag(sales, 4)))

    # ------------------------------------------------------------------ intangibles
    def _c_rd_me(self):
        return self._to_me(self.x("xrd"))

    def _c_rd_sale(self):
        return self._q(ratio(self.x("xrd"), positive(self.x("sale"))))

    def _c_opex_at(self):
        return self._q(self.over_at(self.x("opex")))

    def _c_rd5_at(self):
        rd = self.x("xrd")
        capital = rd + 0.8 * lag(rd, 4) + 0.6 * lag(rd, 8) + 0.4 * lag(rd, 12) + 0.2 * lag(rd, 16)
        return self._q(self.over_at(capital))

    def _c_age(self):
        days = self._input("calendar_day") - self._input("listed_day")
        months = days / 30.4375
        return g.where(months >= 0, months, g.nan)

    def _expected_gap(self, first, second):
        return self._q(self.change_to_expected(first) - self.change_to_expected(second))

    def _c_dsale_dinv(self):
        return self._expected_gap(self.x("sale"), self.x("invt"))

    def _c_dsale_drec(self):
        return self._expected_gap(self.x("sale"), self.x("rect"))

    def _c_dgp_dsale(self):
        return self._expected_gap(self.x("gp"), self.x("sale"))

    def _c_dsale_dsga(self):
        return self._expected_gap(self.x("sale"), self.x("xsga"))

    def _c_tangibility(self):
        tangible = self.x("che") + 0.715 * self.x("rect") + 0.547 * self.x("invt") + 0.535 * self.x("ppegt")
        return self._q(ratio(tangible, self.x("at")))

    def _c_kz_index(self):
        plant = self._q(positive(lag(self.x("ppent"), 4)))
        assets = self._q(positive(self.x("at")))
        debt = self.x("debt")
        cash_flow = ratio(self._q(self.x("ni") + self.x("dp")), plant)
        dividends = ratio(self.dividends(), plant)
        cash = ratio(self._q(self.x("che")), plant)
        tobin = ratio(assets + self.me() - self._q(self.x("be")), assets)
        leverage = self._q(ratio(debt, debt + self.x("seq")))
        return -1.002 * cash_flow + 0.283 * tobin + 3.139 * leverage - 39.368 * dividends - 1.315 * cash

    def _c_ocfq_saleq_std(self):
        values = ratio(self.quarterly("n_cashflow_act"), positive(self.quarterly("revenue")))
        return self._q(op.trailing(values, 16, 8, "std"))

    def _c_cash_at(self):
        return self._q(self.over_at(self.x("che")))

    def _earnings_ar1(self):
        def build():
            roa = ratio(self.x("ni"), self.x("at"))
            ys = [lag(roa, 4 * k) for k in range(5)]
            xs = [lag(roa, 4 * k + 4) for k in range(5)]
            known = sum(g.isfinite(y) & g.isfinite(x) for x, y in zip(xs, ys, strict=True)) == 5.0
            mean_x, mean_y = sum(xs) / 5.0, sum(ys) / 5.0
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sum(
                (x - mean_x) * (x - mean_x) for x in xs
            )
            intercept = mean_y - slope * mean_x
            residuals = [y - intercept - slope * x for x, y in zip(xs, ys, strict=True)]
            ivol = g.sqrt(sum(r * r for r in residuals) / 3.0)
            return {"ar1": g.where(known, slope, g.nan), "ivol": g.where(known, ivol, g.nan)}

        return self._memo("earnings_ar1", build)

    def _c_ni_ar1(self):
        return self._q(self._earnings_ar1()["ar1"])

    def _c_ni_ivol(self):
        return self._q(self._earnings_ar1()["ivol"])

    def _c_earnings_variability(self):
        assets = self.x("at")
        roa = [lag(ratio(self.x("ni"), assets), 4 * k) for k in range(5)]
        croa = [lag(ratio(self.x("oancf"), assets), 4 * k) for k in range(5)]
        known = sum(g.isfinite(a) & g.isfinite(b) for a, b in zip(roa, croa, strict=True)) == 5.0

        def sample_std(values):
            mean = sum(values) / 5.0
            return g.sqrt(sum((v - mean) * (v - mean) for v in values) / 4.0)

        return self._q(g.where(known, ratio(sample_std(roa), sample_std(croa)), g.nan))

    def _c_aliq_at(self):
        return self._q(self.over_at(self.x("aliq"), lagged=True))

    def _c_aliq_mat(self):
        market_assets = self._q(self.x("at") - self.x("be")) + self.me()
        return ratio(self._q(self.x("aliq")), positive(lag(market_assets, YEAR)))

    def _seasonality(self, first_year: int, last_year: int, annual: bool):
        """Mean 21-day return of the same month (or of the other months) ``first_year`` to ``last_year`` back.

        Month ``k`` is the 21-day block ending ``21 k`` rows ago; month 11 is the coming month a year earlier. Every
        month of the span must be known.
        """
        months = [k for k in range((first_year - 1) * 12, last_year * 12) if (k % 12 == 11) == annual]
        listed = ", ".join(str(k) for k in months)
        code = f"""
        s.history.push_back(x0);
        const int t = (int)s.history.size() - 1;
        static const int months[] = {{{listed}}};
        double total = 0.0;
        for (int k : months) {{
            const int end = t - 21 * k, start = t - 21 * (k + 1);
            total += start < 0 ? NAN : s.history[end] / s.history[start] - 1.0;
        }}
        y = total / {float(len(months))};"""
        return op.state([self.price], code, "std::vector<double> history;")

    def _c_seas_1_1an(self):
        return self._seasonality(1, 1, True)

    def _c_seas_1_1na(self):
        return self._seasonality(1, 1, False)

    def _c_seas_2_5an(self):
        return self._seasonality(2, 5, True)

    def _c_seas_2_5na(self):
        return self._seasonality(2, 5, False)

    def _c_seas_6_10an(self):
        return self._seasonality(6, 10, True)

    def _c_seas_6_10na(self):
        return self._seasonality(6, 10, False)

    # ------------------------------------------------------------------ trading frictions
    def _c_market_equity(self):
        return self.me()

    def _c_ivol_ff3_21d(self):
        return self._regression(MARKET_FACTORS, MONTH, 15, "kResidualStd")

    def _c_ivol_capm_252d(self):
        return self._regression(("mkt",), YEAR, 120, "kResidualStd")

    def _c_ivol_capm_21d(self):
        return self._regression(("mkt",), MONTH, 15, "kResidualStd")

    def _c_rvol_21d(self):
        return self._memo("rvol_21d", lambda: op.trailing(self.returns, MONTH, 15, "std"))

    def _c_beta_60m(self):
        stock = self._monthly_block(0)
        market_log = op.scalar("std::log1p(x0)", self._input("mkt"))
        monthly_market = op.scalar("std::expm1(x0)", op.trailing(market_log, MONTH, MONTH, "sum"))
        code = """
        s.y.push_back(x0); s.x.push_back(x1);
        const int t = (int)s.y.size() - 1;
        double sx = 0, sy = 0, sxx = 0, sxy = 0, count = 0;
        for (int k = 0; k < 60; ++k) {
            const int i = t - 21 * k;
            if (i < 0) break;
            if (std::isfinite(s.y[i]) && std::isfinite(s.x[i])) {
                sx += s.x[i]; sy += s.y[i]; sxx += s.x[i] * s.x[i]; sxy += s.x[i] * s.y[i]; count += 1;
            }
        }
        const double covariance = sxy - sx * sy / count, variance = sxx - sx * sx / count;
        y = count >= 36 && variance > 0 ? covariance / variance : NAN;"""
        return op.state([stock, monthly_market], code, "std::vector<double> x, y;")

    def _fp_parts(self):
        def build():
            log_returns = op.scalar("std::log1p(x0)", self.returns)
            market = op.scalar("std::log1p(x0)", self._input("mkt"))
            sigma_stock = op.trailing(log_returns, YEAR, 120, "std")
            sigma_market = op.trailing(g.where(g.isfinite(log_returns), market, g.nan), YEAR, 120, "std")
            stock_3d = op.trailing(log_returns, 3, 3, "sum")
            market_3d = op.trailing(market, 3, 3, "sum")
            correlation = g.clip(op.correlation(stock_3d, market_3d, 5 * YEAR, 750), -1.0, 1.0)
            return correlation, ratio(sigma_stock, sigma_market)

        return self._memo("frazzini_pedersen", build)

    def _c_betabab_1260d(self):
        correlation, volatility = self._fp_parts()
        return correlation * volatility

    def _c_beta_dimson_21d(self):
        market = self._input("mkt")
        # The reference fits the window ending on day d with lag, same-day and lead market returns and publishes it on
        # d + 1; on row t that is the window of returns ending t - 1 against market returns ending t - 2, t - 1 and t.
        regressors = [lag(market, 2), lag(market, 1), market]
        fit = op.regression(lag(self.returns, 1), regressors, MONTH, 15, "kSlopes")
        # The reference's first window ends on row 20, so its first estimate is published on row 21.
        row = op.state([self.close], "y = s.row; ++s.row;", "int row;", "s.row = 0;")
        return g.where(row >= MONTH, fit, g.nan)

    def _c_turnover_126d(self):
        return op.trailing(self._turnover(), 126, 60, "mean")

    def _c_turnover_var_126d(self):
        return self._variation(self._turnover(), 126, 60)

    def _c_dolvol_126d(self):
        return op.trailing(self._input("amount"), 126, 60, "mean")

    def _c_dolvol_var_126d(self):
        return self._variation(self._input("amount"), 126, 60)

    def _c_prc(self):
        return self._input("basic_close")

    def _c_ami_126d(self):
        amount = self._input("amount")
        illiquidity = ratio(g.abs(self.returns), positive(amount))
        return op.trailing(illiquidity, 126, 60, "mean") * 1e6

    def _zero_trades(self, window: int):
        quoted = g.where(g.isfinite(self.close), 1.0, 0.0)
        listed = op.state(
            [self.close],
            f"if (s.first < 0 && std::isfinite(x0)) s.first = s.row; y = s.first >= 0 && s.row - s.first >= "
            f"{window - 1} ? 1.0 : 0.0; ++s.row;",
            "int first; int row;",
            "s.first = -1; s.row = 0;",
        )
        missing = float(window) - op.trailing(quoted, window, window, "sum")
        turnover = op.trailing(self._turnover(), window, 1, "mean")
        result = (missing + (1.0 / turnover) / 480_000.0) * MONTH / float(window)
        return g.where((listed > 0) & g.isfinite(turnover) & (turnover > 0), result, g.nan)

    def _c_zero_trades_21d(self):
        return self._zero_trades(MONTH)

    def _c_zero_trades_126d(self):
        return self._zero_trades(126)

    def _c_zero_trades_252d(self):
        return self._zero_trades(YEAR)

    def _c_rmax1_21d(self):
        return op.trailing(self.returns, MONTH, 15, "max")

    def _c_rskew_21d(self):
        return op.trailing(self.returns, MONTH, 15, "skew")

    def _c_iskew_capm_21d(self):
        return self._regression(("mkt",), MONTH, 15, "kResidualSkew")

    def _c_iskew_ff3_21d(self):
        return self._regression(MARKET_FACTORS, MONTH, 15, "kResidualSkew")

    def _c_coskew_21d(self):
        return self._regression(("mkt",), MONTH, 15, "kCoskew")

    def _c_ret_1_0(self):
        return self._momentum(MONTH, 0)

    def _c_betadown_252d(self):
        code = """
        s.y.push_back(x0); s.x.push_back(x1);
        const int t = (int)s.y.size() - 1;
        y = NAN;
        if (t >= 251) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0, count = 0;
            for (int i = t - 251; i <= t; ++i) {
                const bool down = std::isfinite(s.x[i]) && s.x[i] < 0;
                if (down && std::isfinite(s.y[i])) {
                    sx += s.x[i]; sy += s.y[i]; sxx += s.x[i] * s.x[i]; sxy += s.x[i] * s.y[i]; count += 1;
                }
            }
            const double variance = sxx - sx * sx / count;
            if (count >= 60 && variance > 0) y = (sxy - sx * sy / count) / variance;
        }"""
        return op.state([self.returns, self._input("mkt")], code, "std::vector<double> x, y;")

    def _c_bidaskhl_21d(self):
        spread = corwin_schultz(self._input("high"), self._input("low"))
        return op.trailing(spread, MONTH, 15, "mean")

    # ------------------------------------------------------------------ new
    def _c_ret_3_1(self):
        return self._momentum(3 * MONTH, MONTH)

    def _c_ret_9_1(self):
        return self._momentum(9 * MONTH, MONTH)

    def _c_ret_12_7(self):
        return self._momentum(YEAR, 6 * MONTH)

    def _c_corr_1260d(self):
        return self._fp_parts()[0]

    def _rmax5(self):
        code = """
        s.history.push_back(x0);
        const int t = (int)s.history.size() - 1;
        y = NAN;
        if (t >= 20) {
            std::vector<double> v;
            for (int i = t - 20; i <= t; ++i) if (std::isfinite(s.history[i])) v.push_back(s.history[i]);
            if (v.size() >= 15) {
                std::sort(v.begin(), v.end(), std::greater<double>());
                double total = 0; for (int j = 0; j < 5; ++j) total += v[j];
                y = total / 5.0;
            }
        }"""
        return self._memo("rmax5", lambda: op.state([self.returns], code, "std::vector<double> history;"))

    def _c_rmax5_21d(self):
        return self._rmax5()

    def _c_rmax5_rvol_21d(self):
        return ratio(self._rmax5(), self._c_rvol_21d())

    def _c_ni_be(self):
        return self._q(ratio(self.x("ni"), self.x("be")))

    def _ocf_at(self):
        return self._memo("ocf_at", lambda: self.over_at(self.x("oancf")))

    def _c_ocf_at(self):
        return self._q(self._ocf_at())

    def _c_ocf_at_chg1(self):
        values = self._ocf_at()
        return self._q(values - lag(values, 4))

    def _ranked(self, name: str, *, z_score: bool, sign: float = 1.0):
        return cross_rank(sign * self.calculate(name), self.universe, z_score=z_score)

    def _c_mispricing_perf(self):
        parts = [self._ranked("o_score", z_score=False, sign=-1.0), self._ranked("ret_12_1", z_score=False),
                 self._ranked("gp_at", z_score=False), self._ranked("niq_at", z_score=False)]
        return finite_mean(parts, 3)

    def _c_mispricing_mgmt(self):
        parts = [self._ranked(name, z_score=False, sign=-1.0)
                 for name in ("chcsho_12m", "oaccruals_at", "noa_at", "at_gr1", "ppeinv_gr1a")]
        parts.append(self._ranked("eqnpo_12m", z_score=False))
        return finite_mean(parts, 3)

    def _quality_panel(self, values):
        panel = self._q(values)
        return cross_rank(g.where(self.trading, panel, g.nan), self.universe, z_score=True)

    def _qmj_parts(self):
        def build():
            assets, be, sale = self.x("at"), self.x("be"), self.x("sale")
            ratios = {
                "gpoa": self.over_at(self.x("gp")),
                "roe": ratio(self.x("ni"), be),
                "roa": self.over_at(self.x("ni")),
                "cfoa": self.over_at(self.x("oancf")),
                "gmar": ratio(self.x("gp"), positive(sale)),
            }
            profitability = [self._quality_panel(values) for values in ratios.values()]
            profitability.append(self._quality_panel(-self.over_at(self.x("oacc"))))
            growth_parts = [self._quality_panel(values - lag(values, 20)) for values in ratios.values()]
            roe_quarterly = ratio(self.quarterly("n_income_attr_p"), positive(be))
            safety = [
                cross_rank(-self.calculate("betabab_1260d"), self.universe, z_score=True),
                self._quality_panel(-ratio(self.x("debt"), positive(assets))),
                self._quality_panel(-self._o_score()),
                cross_rank(self.calculate("z_score"), self.universe, z_score=True),
                self._quality_panel(-op.trailing(roe_quarterly, 20, 12, "std")),
            ]
            scores = {}
            for key, parts in (("prof", profitability), ("growth", growth_parts), ("safety", safety)):
                scores[key] = cross_rank(finite_mean(parts, 3), self.universe, z_score=True)
            total = scores["prof"] + scores["growth"] + scores["safety"]
            scores["qmj"] = cross_rank(total, self.universe, z_score=True)
            return scores

        return self._memo("qmj", build)

    def _c_qmj(self):
        return self._qmj_parts()["qmj"]

    def _c_qmj_prof(self):
        return self._qmj_parts()["prof"]

    def _c_qmj_growth(self):
        return self._qmj_parts()["growth"]

    def _c_qmj_safety(self):
        return self._qmj_parts()["safety"]


@lru_cache(maxsize=1)
def _definitions() -> tuple[dict[str, Any], dict[str, Any]]:
    """Daily outputs by characteristic and the statement-graph results they read."""
    graph = JKPGraph()
    outputs = {name: graph.calculate(name) for name in JKP_NAMES}
    return outputs, dict(graph.statements)


class JKP153Library(FactorLibrary):
    """Jensen-Kelly-Pedersen characteristics, re-dated to daily point-in-time panels and computed in KunQuant."""

    family = "jkp153"
    required_inputs = ("high", "low", "close", "amount", "cap")
    statement_fields = STATEMENT_LINES
    strict_float = True
    compile_split_source = 64
    cpp_prelude = CNC_PRELUDE + op.CPP_PRELUDE + regression_op.CPP_PRELUDE + CPP_PRELUDE

    @property
    def statement_outputs(self) -> tuple[str, ...]:
        return tuple(_definitions()[1])

    @property
    def reference_inputs(self) -> tuple[str, ...]:
        return (*DAILY_REFERENCE, *self.statement_outputs)

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"jkp_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=f"{theme}: {description}",
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.jkp153.JKPGraph.{name}",
                catalog_version=2,
            )
            for ordinal, (name, theme, description, lookback) in enumerate(JKP_CHARS, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Also pin the statement alignment and the compiled helpers, which live in other modules."""
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "statements": inspect.getsource(statements),
                "compiled_prelude": self.cpp_prelude,
            }
        )

    def prepare_statements(self, fundamentals: Any) -> Any:
        return prepare_fundamentals(fundamentals)

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.canonical import ts_code_to_qlib
        from rquant.data.reference import load_reference_manifest

        canonical_root = Path(canonical_root)
        root = canonical_root / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "jkp153 needs financial statements; run `rquant data sync-fundamentals` and "
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
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        missing = sorted(set(STATEMENT_LINES) - set(fundamentals.columns))
        if missing:
            raise DataContractError(
                f"Fundamental reference lacks jkp153 statement items {missing[:5]}; run `rquant data "
                "sync-fundamentals` and `rquant data build-reference` again"
            )

        wanted = [("instrument", "in", list(instruments))]
        columns = ["instrument", "datetime", *JKP_DAILY_BASIC]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in JKP_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale
        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        for name in MARKET_FACTORS:
            inputs[name] = factors[name].reindex(dates)
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        basic = pd.read_parquet(root / "stock_basic.parquet", columns=["ts_code", "list_date"])
        stock_basic_hash = sha256_file(root / "stock_basic.parquet")
        basic = basic.assign(instrument=basic["ts_code"].map(ts_code_to_qlib)).drop_duplicates("instrument")
        listed = pd.to_datetime(basic.set_index("instrument")["list_date"], format="%Y%m%d", errors="coerce")
        inputs["list_date"] = listed.reindex(instruments)

        hashes = {
            "jkp153_reference": manifest["fingerprint"],
            "jkp153_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "jkp153_market_factors": files["market_factors.parquet"],
            "jkp153_fundamentals": files["fundamentals.parquet"],
            "jkp153_stock_basic": stock_basic_hash,
        }
        return self.align_reference_inputs(inputs, dates, instruments), hashes

    def align_reference_inputs(self, inputs: Mapping[str, Any], dates: Any, instruments: Any) -> dict[str, Any]:
        """Add the day numbers of the dates and of each instrument's listing, which ``age`` reads."""
        dates, instruments = pd.DatetimeIndex(dates), pd.Index(instruments)
        aligned = dict(inputs)
        epoch = np.datetime64("1970-01-01", "D")
        aligned["calendar_day"] = pd.Series((dates.to_numpy().astype("datetime64[D]") - epoch).astype(float),
                                            index=dates)
        listed = pd.to_datetime(pd.Series(inputs["list_date"]).reindex(instruments)).to_numpy().astype("datetime64[D]")
        listed_day = np.where(np.isnat(listed), np.nan, (listed - epoch).astype(float))
        aligned["listed_day"] = np.ascontiguousarray(np.broadcast_to(listed_day, (len(dates), len(instruments))))
        return aligned

    def build_statements(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[1].items()}

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        memo: dict[int, Any] = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[0].items()}


def prepare_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill statement lines a published statement leaves blank."""
    frame = frame.copy()
    for anchor, lines in ZERO_FILL.items():
        published = frame[anchor].notna()
        for line in lines:
            frame.loc[published & frame[line].isna(), line] = 0.0
    return frame
