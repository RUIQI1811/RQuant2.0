"""Open Source Asset Pricing predictors (Chen and Zimmermann), evaluated as daily point-in-time A-share panels.

The catalog follows the 212 "Predictor" signals documented in the Open Source Asset Pricing ``SignalDoc`` (Chen and
Zimmermann, "Open Source Cross-Sectional Asset Pricing", Critical Finance Review 2022; openassetpricing.com). The
upstream code builds monthly signals from CRSP, Compustat, IBES, OptionMetrics and 13F data for US stocks. This module
is an independent re-implementation from the published signal definitions: no upstream code or data is used, inputs
come from Tushare, and every signal is re-dated to the trading day on which it is known. The 114 "Placebo" signals
are not part of the set.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. A month is 21 trading days and a year 252. Monthly-return
  signals (Beta, Coskewness, std_turn, VolumeTrend, ResidualMomentum, ...) are run on non-overlapping 21-day blocks
  ending on the row, so they update every day.
* Returns are close-to-close returns of RQuant's backward-adjusted prices and are not reduced by a risk-free rate.
  Market, SMB and HML returns come from ``reference/market_factors.parquet`` (value-weighted over all A shares).
* Accounting signals use the first-reported statements of ``reference/fundamentals.parquet`` on the quarter grid of
  ``china_chars``: a quarter becomes usable on the first trading day on or after its announcement, once every earlier
  quarter is also out. Compustat annual flows become trailing-twelve-month sums of single quarters, balance-sheet
  items the latest quarter end, quarterly items the latest single quarter, and "one year ago" is four quarters ago.
  Compustat items map to Tushare lines as listed in ``ITEMS``.
* Market value is daily total market value; dividends are the trailing-twelve-month dividend yield of daily_basic
  times the total market value (the smallest of the last 15 days, because Tushare double-counts a payment for a few
  days around its ex-date), and a payment is due when that trailing amount changed 11 to 12 months ago. Share
  counts that must be split-adjusted (ShareIss1Y, ShareIss5Y, PS) are total shares times the unadjusted over the
  adjusted close.
* Cross-sectional steps (quintile flags, industry means and medians, cross-sectional regressions, Herfindahl indices)
  use only the stocks that are in the CSI 300 on that day, so later index members do not leak into earlier
  cross-sections. Industries are the Shenwan level-1 sectors of the canonical bars.
* US sample screens (price below $5, exchange or share-code filters, SIC-based exclusions of financials or utilities)
  are not applied. Restrictions that define a signal are kept (FirmAgeMom: youngest quintile; MomVol: top volume
  tercile; the quintile flags AccrualsBM and MomRev). US-specific constants are replaced by their A-share
  counterparts (statutory tax rate 33% before 2008, 25% after; ShareVol uses CSI 300 turnover terciles instead of the
  US 5% and 10% thresholds).
* Every output is missing on days the stock has no quote.

Signals that Tushare data at this account level cannot build are listed in ``OAP_SKIPPED`` with the reason.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS
from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.panel_operators import rolling_regression
from rquant.factors.statements import QuarterGrid, lag, nan_mean, ratio, rolling_quarter_std
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
MONTH = 21
QUARTER = 63
YEAR = 252
OAP_DAILY_BASIC = {"close": 1.0, "turnover_rate": 0.01, "total_share": 10_000.0, "dv_ttm": 0.01}
MARKET_FACTORS = ("mkt", "smb", "hml")
DIVIDEND_SPIKE_DAYS = 15
# Tushare reports R&D expense as its own income-statement line from 2018; afterwards a blank line means none.
RD_LINE_YEAR = 2018
TREND_WINDOWS = (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000)

# Compustat item: Tushare statement lines summed into it. Flow items are read as trailing-twelve-month sums, single
# quarters or fiscal-year totals; the rest are balance-sheet values at the quarter end.
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
    "ppe": ("fix_assets",),
    "intan": ("intan_assets", "goodwill"),
    "dlc": ("st_borr", "non_cur_liab_due_1y"),
    "dltt": ("lt_borr", "bond_payable"),
    "ap": ("acct_payable", "notes_payable"),
    "txp": ("taxes_payable",),
    "txditc": ("defer_tax_liab",),
    "pstk": ("oth_eqt_tools_p_shr",),
    "seq": ("total_hldr_eqy_exc_min_int",),
    "mib": ("minority_int",),
    "prepaid": ("prepayment",),
    "aco": ("prepayment", "oth_receiv"),
    "nca": ("total_nca",),
    "ncl": ("total_ncl",),
    # Flows.
    "sale": ("revenue",),
    "cogs": ("oper_cost",),
    "xsga": ("sell_exp", "admin_exp", "rd_exp"),
    "xrd": ("rd_exp",),
    "xint": ("fin_exp_int_exp",),
    "ib": ("n_income_attr_p",),
    "ni": ("n_income",),
    "pi": ("total_profit",),
    "txt": ("income_tax",),
    "oancf": ("n_cashflow_act",),
    "ivncf": ("n_cashflow_inv_act",),
    "fincf": ("n_cash_flows_fnc_act",),
    "capx": ("c_pay_acq_const_fiolta",),
    "dp": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp"),
    "sstk": ("c_recp_cap_contrib",),
    "dltis": ("c_recp_borrow", "proc_issue_bonds"),
    "dltr": ("c_prepay_amt_borr",),
}
# R&D is a separate income-statement line only from 2018; before that it sits inside administrative expenses, so the
# SG&A sum treats a blank R&D line as zero while the R&D signals keep it missing.
OPTIONAL_LINES = ("rd_exp",)

# Statement lines that mean "none reported" once the statement itself exists, keyed by the line that shows the
# statement was published for that quarter.
ZERO_FILL = {
    "total_assets": (
        "money_cap", "trad_asset", "lt_eqt_invest", "fa_avail_for_sale", "htm_invest", "accounts_receiv",
        "notes_receiv", "inventories", "fix_assets", "intan_assets", "goodwill", "st_borr", "non_cur_liab_due_1y",
        "lt_borr", "bond_payable", "acct_payable", "notes_payable", "taxes_payable", "defer_tax_liab",
        "oth_eqt_tools_p_shr", "minority_int", "prepayment", "oth_receiv",
    ),
    "revenue": ("sell_exp", "admin_exp", "fin_exp_int_exp"),
    "n_cashflow_act": (
        "depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp", "c_recp_cap_contrib", "c_recp_borrow",
        "proc_issue_bonds", "c_prepay_amt_borr", "c_pay_acq_const_fiolta",
    ),
}

# (OAP acronym, data category, description, maximum lookback in trading days). Order is the stable column order.
OAP: tuple[tuple[str, str, str, int], ...] = (
    # ------------------------------------------------------------------ price
    ("Size", "price", "log total market value", 1),
    ("Price", "price", "log unadjusted close", 1),
    ("STreversal", "price", "one-month return", MONTH),
    ("Mom6m", "price", "return from month t-6 to t-1", 6 * MONTH),
    ("Mom12m", "price", "return from month t-12 to t-1", YEAR),
    ("IntMom", "price", "return from month t-12 to t-6", YEAR),
    ("MRreversal", "price", "return from month t-18 to t-13", 18 * MONTH),
    ("LRreversal", "price", "return from month t-36 to t-13", 36 * MONTH),
    ("Mom12mOffSeason", "price", "average monthly return over lags 1-11", YEAR),
    ("MomSeasonShort", "price", "return of the same calendar month one year ago", YEAR),
    ("MomSeason", "price", "average same-month return in years 2-5", 5 * YEAR),
    ("MomOffSeason", "price", "average other-month return in years 2-5", 5 * YEAR),
    ("MomSeason06YrPlus", "price", "average same-month return in years 6-10", 10 * YEAR),
    ("MomOffSeason06YrPlus", "price", "average other-month return in years 6-10", 10 * YEAR),
    ("MaxRet", "price", "largest daily return over one month", MONTH),
    ("ReturnSkew", "price", "skewness of daily returns over one month", MONTH),
    ("ReturnSkew3F", "price", "skewness of FF3 residuals over one month", MONTH),
    ("IdioVol3F", "price", "volatility of FF3 residuals over one month", MONTH),
    ("RealizedVol", "price", "volatility of daily returns over one month", MONTH),
    ("IdioVolAHT", "price", "volatility of CAPM residuals over one year", YEAR),
    ("Beta", "price", "CAPM beta from 60 monthly returns", 60 * MONTH),
    ("BetaFP", "price", "Frazzini-Pedersen beta: 5-year 3-day correlation times 1-year volatility ratio", 5 * YEAR),
    ("BetaTailRisk", "price", "beta on the Kelly-Jiang tail-risk index, 60 months", 61 * MONTH),
    ("CoskewACX", "price", "coskewness of daily log returns over one year", YEAR),
    ("Coskewness", "price", "coskewness of monthly log returns over 60 months", 60 * MONTH),
    ("PriceDelayRsq", "price", "1 - R2 without / R2 with four lagged market returns, one year", YEAR + 4),
    ("PriceDelaySlope", "price", "lag-weighted share of the lagged market betas, one year", YEAR + 4),
    ("PriceDelayTstat", "price", "lag-weighted share of the lagged market-beta t-statistics, one year", YEAR + 4),
    ("ResidualMomentum", "price", "mean over std of FF3 residuals in months t-12..t-2 of a 36-month fit", 36 * MONTH),
    ("TrendFactor", "price", "Han-Zhou-Zhu trend: moving-average ratios times 12-month average loadings", 1000 + YEAR),
    ("High52", "price", "price over its 52-week high", YEAR),
    ("IndMom", "price", "value-weighted six-month return of the Shenwan industry", 6 * MONTH),
    ("IndRetBig", "price", "one-month return of the largest 30% of the industry, for the other 70%", MONTH),
    ("FirmAgeMom", "price", "six-month momentum of the youngest age quintile", 6 * MONTH),
    ("MomRev", "price", "1 if high Mom6m and low Mom36m quintile, 0 if the reverse", 36 * MONTH),
    ("MomVol", "price", "Mom6m decile within the top turnover tercile", 6 * MONTH),
    # ------------------------------------------------------------------ trading
    ("DolVol", "trading", "log traded value of the month before last", 2 * MONTH),
    ("Illiquidity", "trading", "Amihud |return| over traded value, one year", YEAR),
    ("ShareVol", "trading", "1 if three-month turnover in the top tercile, 0 if in the bottom", QUARTER),
    ("std_turn", "trading", "std of monthly turnover over 36 months", 36 * MONTH),
    ("VolSD", "trading", "std of monthly volume over 36 months", 36 * MONTH),
    ("VolMkt", "trading", "average monthly traded value over one year over market value", YEAR),
    ("VolumeTrend", "trading", "60-month time trend of monthly volume over its mean", 60 * MONTH),
    ("zerotrade1M", "trading", "Liu zero-trading days plus turnover tie-breaker, one month", MONTH),
    ("zerotrade6M", "trading", "Liu zero-trading days plus turnover tie-breaker, six months", 6 * MONTH),
    ("zerotrade12M", "trading", "Liu zero-trading days plus turnover tie-breaker, twelve months", YEAR),
    ("BidAskSpread", "trading", "Corwin-Schultz high-low spread, one month", MONTH + 1),
    # ------------------------------------------------------------------ valuation
    ("AM", "accounting", "total assets over market value", 1),
    ("BM", "accounting", "log tangible book equity over market value at the quarter end", 1),
    ("BMdec", "accounting", "book equity over market value of the last December", YEAR),
    ("BookLeverage", "accounting", "total assets over book equity plus deferred taxes", 1),
    ("Leverage", "accounting", "total liabilities over market value", 1),
    ("EP", "accounting", "trailing net income over market value, positive only", 1),
    ("CF", "accounting", "trailing net income plus depreciation over market value", 1),
    ("cfp", "accounting", "trailing operating cash flow over market value", 1),
    ("SP", "accounting", "trailing revenue over market value", 1),
    ("EBM", "accounting", "enterprise book-to-price", 1),
    ("BPEBM", "accounting", "book-to-price minus enterprise book-to-price", 1),
    ("NetDebtPrice", "accounting", "net debt over market value", 1),
    ("EntMult", "accounting", "enterprise value over trailing EBITDA", 1),
    ("CashProd", "accounting", "market value minus total assets, over cash", 1),
    ("PayoutYield", "accounting", "trailing dividends over market value six months ago", 6 * MONTH),
    ("NetPayoutYield", "accounting", "trailing dividends minus equity issuance over market value six months ago",
     6 * MONTH),
    ("EquityDuration", "accounting", "Dechow-Sloan-Soliman implied equity duration", 1),
    ("IntanBM", "accounting", "residual of 5-year return on lagged log BM and its book return", 5 * YEAR),
    ("IntanCFP", "accounting", "residual of 5-year return on lagged log CF/P and its cash-flow return", 5 * YEAR),
    ("IntanEP", "accounting", "residual of 5-year return on lagged log E/P and its earnings return", 5 * YEAR),
    ("IntanSP", "accounting", "residual of 5-year return on lagged log S/P and its sales return", 5 * YEAR),
    ("Frontier", "accounting", "residual of log BM on firm characteristics and industry", 1),
    ("VarCF", "accounting", "variance of monthly cash-flow-to-price over 60 months", 60 * MONTH),
    # ------------------------------------------------------------------ profitability
    ("Cash", "accounting", "cash and short-term investments over total assets", 1),
    ("GP", "accounting", "trailing gross profit over total assets", 1),
    ("OperProf", "accounting", "trailing revenue - COGS - SG&A - interest over book equity", 1),
    ("OperProfRD", "accounting", "trailing revenue - COGS - (SG&A - R&D) over total assets", 1),
    ("CBOperProf", "accounting", "cash-based operating profitability over total assets", 1),
    ("RoE", "accounting", "trailing net income over book equity", 1),
    ("roaq", "accounting", "quarterly net income over lagged total assets", 1),
    ("OPLeverage", "accounting", "trailing SG&A plus COGS over total assets", 1),
    ("tang", "accounting", "tangibility: weighted liquid and fixed assets over total assets", 1),
    ("Tax", "accounting", "income tax over statutory rate times net income", 1),
    ("PS", "accounting", "Piotroski F-score", YEAR),
    ("MS", "accounting", "Mohanram G-score against industry medians", 1),
    ("OScore", "accounting", "Ohlson O-score", 1),
    ("EarningsConsistency", "accounting", "average annual earnings growth over four years", 1),
    ("EarningsSurprise", "accounting", "quarterly earnings change minus drift, over its std", 1),
    ("RevenueSurprise", "accounting", "quarterly revenue change minus drift, over its std", 1),
    ("NumEarnIncrease", "accounting", "consecutive quarters of year-on-year earnings increases, up to 8", 1),
    ("EarnSupBig", "accounting", "earnings surprise of the largest 30% of the industry, for the other 70%", 1),
    ("MeanRankRevGrowth", "accounting", "weighted average of five yearly revenue-growth ranks", 4 * YEAR),
    ("GrSaleToGrInv", "accounting", "revenue growth minus inventory growth", 1),
    ("GrSaleToGrOverhead", "accounting", "revenue growth minus SG&A growth", 1),
    ("ChAssetTurnover", "accounting", "one-year change in revenue over net operating assets", 1),
    ("OrgCap", "accounting", "organisational capital over assets, industry-standardised", 1),
    # ------------------------------------------------------------------ investment and accruals
    ("AssetGrowth", "accounting", "one-year total asset growth", 1),
    ("ChEQ", "accounting", "book equity over book equity a year ago", 1),
    ("ChInv", "accounting", "one-year inventory change over average total assets", 1),
    ("InvGrowth", "accounting", "one-year inventory growth", 1),
    ("InvestPPEInv", "accounting", "one-year change in PP&E plus inventory over lagged assets", 1),
    ("dNoa", "accounting", "one-year change in net operating assets over lagged assets", 1),
    ("NOA", "accounting", "net operating assets over lagged total assets", 1),
    ("ChNWC", "accounting", "one-year change in net working capital over assets", 1),
    ("ChNNCOA", "accounting", "one-year change in net non-current operating assets over assets", 1),
    ("DelCOA", "accounting", "change in current operating assets over average assets", 1),
    ("DelCOL", "accounting", "change in current operating liabilities over average assets", 1),
    ("DelEqu", "accounting", "change in book equity over average assets", 1),
    ("DelFINL", "accounting", "change in financial liabilities over average assets", 1),
    ("DelLTI", "accounting", "change in long-term investments over average assets", 1),
    ("DelNetFin", "accounting", "change in net financial assets over average assets", 1),
    ("GrLTNOA", "accounting", "growth in long-term net operating assets minus accruals", 1),
    ("Accruals", "accounting", "Sloan working-capital accruals over average assets", 1),
    ("AbnormalAccruals", "accounting", "residual of accruals on the Jones-model regressors", 1),
    ("AccrualsBM", "accounting", "1 if high accruals and low BM quintile, 0 if the reverse", 1),
    ("PctAcc", "accounting", "net income minus operating cash flow over |net income|", 1),
    ("PctTotAcc", "accounting", "total accruals over |net income|", 1),
    ("TotalAccruals", "accounting", "total accruals over lagged assets", 1),
    ("ChTax", "accounting", "four-quarter change in quarterly tax over lagged assets", 1),
    ("Investment", "accounting", "capex over revenue relative to its three prior years", 1),
    ("ChInvIA", "accounting", "capex growth minus its industry mean", 1),
    ("grcapx", "accounting", "two-year capex growth", 1),
    ("grcapx3y", "accounting", "capex over its three prior-year average", 1),
    ("CompositeDebtIssuance", "accounting", "five-year log growth of debt", 1),
    ("CompEquIss", "accounting", "five-year log growth of market value minus log return", 5 * YEAR),
    ("ShareIss1Y", "accounting", "split-adjusted share growth from month t-18 to t-6", 18 * MONTH),
    ("ShareIss5Y", "accounting", "five-year split-adjusted share growth", 5 * YEAR),
    ("NetEquityFinance", "accounting", "trailing equity issuance over average assets", 1),
    ("NetDebtFinance", "accounting", "trailing debt issuance minus repayment over average assets", 1),
    ("XFIN", "accounting", "net external financing over total assets", 1),
    ("DebtIssuance", "accounting", "1 if any trailing debt issuance", 1),
    # ------------------------------------------------------------------ R&D and other
    ("RD", "accounting", "trailing R&D over market value", 1),
    ("RDcap", "accounting", "five-year weighted R&D capital over total assets", 1),
    ("SurpriseRD", "accounting", "1 if R&D intensity is positive and R&D grew more than 5%", 1),
    ("Herf", "other", "three-year average industry Herfindahl of revenue", 2 * YEAR),
    ("HerfBE", "other", "three-year average industry Herfindahl of book equity", 2 * YEAR),
    ("HerfAsset", "other", "three-year average industry Herfindahl of total assets", 2 * YEAR),
    ("FirmAge", "other", "months since listing", 1),
    # ------------------------------------------------------------------ events
    ("AnnouncementReturn", "event", "market-adjusted return from day -1 to +2 around the latest statement", 3),
    ("IndIPO", "event", "1 if listed 6 to 36 months ago", 1),
    ("RDIPO", "event", "1 if IndIPO and no R&D", 1),
    ("DivInit", "event", "1 if a dividend in the past six months after none in the prior eighteen", 2 * YEAR),
    ("DivOmit", "event", "1 if a two-year payer stopped paying in the past two months", 2 * YEAR),
    ("DivSeason", "event", "1 if a payer's dividend changed 11-12 months ago, 0 for other payers", YEAR),
    ("DivYieldST", "event", "binned trailing dividend yield when a payment is due within a month, payers only",
     YEAR),
)
OAP_NAMES = tuple(name for name, _, _, _ in OAP)

# OAP predictors left out, with the reason. Together with ``OAP_NAMES`` this is the full 212-signal predictor list.
_ANALYST = "needs IBES analyst forecasts or recommendations, which Tushare does not provide at this account level"
_HOLDINGS = "needs 13F institutional holdings; Chinese fund holdings are semi-annual and not synced"
_OPTIONS = "needs single-stock option prices; A shares have no listed single-stock options"
OAP_SKIPPED: dict[str, str] = {
    **{name: _ANALYST for name in (
        "AnalystRevision", "ChangeInRecommendation", "ChForecastAccrual", "ChNAnalyst", "CredRatDG",
        "EarningsForecastDisparity", "ExclExp", "FEPS", "fgr5yrLag", "ForecastDispersion", "Recomm_ShortInterest",
        "REV6", "sfe", "AnalystValue", "AOP", "ConsRecomm", "DownRecomm", "UpRecomm", "EarningsStreak",
        "PredictedFE")},
    **{name: _HOLDINGS for name in (
        "Activism1", "Activism2", "DelBreadth", "RIO_Disp", "RIO_MB", "RIO_Turnover", "RIO_Volatility",
        "IO_ShortInterest")},
    **{name: _OPTIONS for name in (
        "CPVolSpread", "dVolCall", "dVolPut", "dCPVolSpread", "OptionVolume1", "OptionVolume2", "RIVolSpread",
        "SmileSlope", "skew1")},
    "BetaLiquidityPS": "needs the Pastor-Stambaugh aggregate liquidity innovation series",
    "betaVIX": "needs a daily implied-volatility index; China's iVIX stopped in 2018",
    "CitationsRD": "needs patent citation data",
    "PatentsRD": "needs patent counts",
    "ConvDebt": "needs convertible-debt or deferred-charge items that the Tushare statements do not split out",
    "CustomerMomentum": "needs customer links from segment disclosures",
    "iomom_cust": "needs input-output tables matched to industries",
    "iomom_supp": "needs input-output tables matched to industries",
    "retConglomerate": "needs business-segment data",
    "sinAlgo": "needs business-segment industry codes",
    "Governance": "needs the Gompers-Ishii-Metrick governance index",
    "ExchSwitch": "A shares do not switch listing exchanges",
    "Spinoff": "needs a spin-off event file",
    "AgeIPO": "needs company founding years, which are not synced",
    "hire": "needs historical employee counts",
    "AdExp": "needs advertising expense, which Chinese income statements do not report",
    "GrAdExp": "needs advertising expense, which Chinese income statements do not report",
    "BrandInvest": "needs advertising expense, which Chinese income statements do not report",
    "OrderBacklog": "needs order backlog",
    "OrderBacklogChg": "needs order backlog",
    "FR": "needs pension plan assets and obligations",
    "realestate": "needs the buildings and land breakdown of PP&E",
    "RDS": "needs other-comprehensive-income components of the equity change",
    "RDAbility": "needs eight years of R&D; Tushare reports R&D as its own line only from 2018",
    "DelDRC": "needs deferred revenue; advance receipts were reclassified to contract liabilities in 2018-2020",
    "ShareRepurchase": "needs share repurchases, which Chinese cash-flow statements do not report separately",
    "ShortInterest": "needs securities-lending balances (margin_detail), which are not synced",
    "Mom6mJunk": "needs credit ratings",
    "MomSeason11YrPlus": "needs 11-15 years of prices; canonical prices start in 2008, so it is empty before 2020",
    "MomOffSeason11YrPlus": "needs 11-15 years of prices; canonical prices start in 2008, so it is empty before 2020",
    "MomSeason16YrPlus": "needs 16-20 years of prices; canonical prices start in 2008, so it is empty before 2025",
    "MomOffSeason16YrPlus": "needs 16-20 years of prices; canonical prices start in 2008, so it is empty before 2025",
    "ProbInformedTrading": "needs signed intraday trades to estimate PIN",
}
# Cached intermediates and the last signal, in catalog order, that reads them.
RELEASE_AFTER = {
    "IdioVol3F": ("ff3_month",),
    "IdioVolAHT": ("capm_year",),
    "BetaTailRisk": ("tail_index",),
    "PriceDelayTstat": ("price_delay",),
    "ResidualMomentum": ("monthly_returns",),
    "MomRev": ("mom36",),
    "MomVol": ("mom6",),
    "EarnSupBig": ("earnings_surprise", "big"),
}
REQUIRED_LINES = tuple(sorted({line for lines in ITEMS.values() for line in lines} - set(OPTIONAL_LINES)))


class OAPLibrary(PanelFactorLibrary):
    """Open Source Asset Pricing predictors re-implemented on Tushare data."""

    family = "oap"
    required_inputs = ("high", "low", "close", "volume", "amount", "cap", "sector")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"oap_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=f"[{category}] {description}",
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.oap.OAPSignals.{name}",
                catalog_version=1,
            )
            for ordinal, (name, category, description, lookback) in enumerate(OAP, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Also pin the shared statement grid and the statement fields, which live in other modules."""
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "quarter_grid": inspect.getsource(importlib.import_module("rquant.factors.statements")),
                "panel_operators": inspect.getsource(importlib.import_module("rquant.factors.panel_operators")),
                "statement_fields": {name: list(fields) for name, fields in FUNDAMENTAL_ENDPOINT_FIELDS.items()},
            }
        )

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        from rquant.data.canonical import ts_code_to_qlib
        from rquant.data.reference import load_reference_manifest

        canonical_root = Path(canonical_root)
        root = canonical_root / "reference"
        manifest = load_reference_manifest(canonical_root)
        files = manifest["files"]
        if "fundamentals.parquet" not in files:
            raise DataContractError(
                "oap needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        missing = sorted(set(REQUIRED_LINES) - set(fundamentals.columns))
        if missing:
            raise DataContractError(
                f"Fundamental reference lacks oap statement items {missing[:5]}; run `rquant data "
                "sync-fundamentals` and `rquant data build-reference` again"
            )
        for name in ("fundamentals.parquet", "market_factors.parquet"):
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")

        dates = pd.DatetimeIndex(dates)
        years = set(dates.year)
        daily_files = sorted(
            name for name in files if name.startswith("daily_basic/") and int(name.split("=")[1][:4]) in years
        )
        for name in daily_files:
            if sha256_file(root / name) != files[name]:
                raise DataContractError(f"Reference file differs from its manifest: {root / name}")
        columns = ["instrument", "datetime", *OAP_DAILY_BASIC]
        # Reading only the panel's instruments keeps the all-A daily_basic partitions out of memory.
        wanted = [("instrument", "in", list(instruments))]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in OAP_DAILY_BASIC.items():
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

        # Cross-sectional operations use the research universe's members of each day, which the engine passes in.
        hashes = {
            "oap_reference": manifest["fingerprint"],
            "oap_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "oap_market_factors": files["market_factors.parquet"],
            "oap_fundamentals": files["fundamentals.parquet"],
            "oap_stock_basic": stock_basic_hash,
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = OAPSignals(inputs)
        return {name: calculator.calculate(name) for name in OAP_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = OAPSignals(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)
            calculator.release(spec.source_name)


# ---------------------------------------------------------------------------------------------- array helpers
def prepare_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill lines a published statement leaves blank."""
    frame = fundamentals.copy()
    for anchor, fields in ZERO_FILL.items():
        published = frame[anchor].notna() if anchor in frame.columns else pd.Series(False, index=frame.index)
        for field in fields:
            if field not in frame.columns:
                frame[field] = np.nan
            frame[field] = frame[field].where(frame[field].notna() | ~published, 0.0)
    return frame


def positive(values: Any) -> Any:
    return np.where(values > 0, values, np.nan)


def safe_log(values: Any) -> Any:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(positive(values))


def finite_or(values: np.ndarray, fill: float) -> np.ndarray:
    return np.where(np.isfinite(values), values, fill)


def flag(condition: np.ndarray, known: np.ndarray) -> np.ndarray:
    """1.0 where ``condition`` holds, 0.0 where it does not, missing where the inputs are unknown."""
    return np.where(known, condition.astype(float), np.nan)


def shift_rows(values: np.ndarray, periods: int) -> np.ndarray:
    """Shift rows down so row t holds row t - periods."""
    result = np.full(values.shape, np.nan)
    if 0 < periods < len(values):
        result[periods:] = values[: len(values) - periods]
    elif periods == 0:
        result = values.astype(float, copy=True)
    return result


def trailing(values: np.ndarray, window: int, minimum: int, how: str) -> np.ndarray:
    frame = pd.DataFrame(values).rolling(window, min_periods=minimum)
    return getattr(frame, how)().to_numpy()


def growth(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Relative change, defined only for a positive base."""
    return ratio(current - previous, positive(previous))


def quarter_window(values: np.ndarray, lags: tuple[int, ...], minimum: int) -> np.ndarray:
    """Mean over the given quarter lags."""
    stacked = np.stack([lag(values, k) for k in lags], axis=2)
    mean, count = nan_mean(stacked, 2)
    return np.where(count >= minimum, mean, np.nan)


def monthly(values: np.ndarray, reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``reduce`` to the series of rows t, t-21, t-42, ... for every row t.

    ``reduce`` receives one residue class of rows in time order and returns an array of the same shape whose row k
    depends only on rows up to k, so the result on row t uses non-overlapping 21-day blocks ending on t.
    """
    result = np.full(values.shape, np.nan)
    for residue in range(min(MONTH, len(values))):
        result[residue::MONTH] = reduce(values[residue::MONTH])
    return result


def rolling_coskew(stock: np.ndarray, market: np.ndarray, window: int, minimum: int) -> np.ndarray:
    """E[r~ m~^2] / (sd(r) var(m)) over trailing windows of jointly observed, demeaned stock and market returns."""
    weight = (np.isfinite(stock) & np.isfinite(market)[:, None]).astype(float)
    y = np.where(weight > 0, stock, 0.0)
    m = np.where(weight > 0, np.broadcast_to(finite_or(market, 0.0)[:, None], stock.shape), 0.0)

    def sums(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
        out = np.full(values.shape, np.nan)
        if len(values) >= window:
            out[window - 1 :] = prefix[window:] - prefix[:-window]
        return out

    n = sums(weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_y, mean_m = sums(y) / n, sums(m) / n
        e_ym, e_ymm, e_mm, e_yy = sums(y * m) / n, sums(y * m * m) / n, sums(m * m) / n, sums(y * y) / n
        co_moment = e_ymm - 2.0 * mean_m * e_ym + 2.0 * mean_m**2 * mean_y - mean_y * e_mm
        var_y, var_m = e_yy - mean_y**2, e_mm - mean_m**2
        result = co_moment / (np.sqrt(var_y) * var_m)
    return np.where((n >= minimum) & (var_y > 0) & (var_m > 0), result, np.nan)


def price_delay(stock: np.ndarray, market: np.ndarray, window: int, minimum: int) -> dict[str, np.ndarray]:
    """Hou-Moskowitz price delay from trailing regressions on the market return and its four lags."""
    dates, stocks = stock.shape
    out = {name: np.full((dates, stocks), np.nan) for name in ("rsq", "slope", "tstat")}
    if dates < window:
        return out
    design = np.column_stack([np.ones(dates), *[shift_rows(market[:, None], j)[:, 0] for j in range(5)]])
    design_ok = np.isfinite(design).all(axis=1)
    design = np.where(design_ok[:, None], design, 0.0)
    k = design.shape[1]
    outer = design[:, :, None] * design[:, None, :]
    identity = np.eye(k)
    weights_lag = np.arange(5, dtype=float)
    rows = slice(window - 1, dates)

    def window_sums(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate([np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)])
        return prefix[window:] - prefix[:-window]

    for column in range(stocks):
        series = stock[:, column]
        mask = np.isfinite(series) & design_ok
        if mask.sum() < minimum:
            continue
        weight = mask.astype(float)
        value = np.where(mask, series, 0.0)
        count = window_sums(weight)
        valid = count >= minimum
        if not valid.any():
            continue
        gram = window_sums(outer * weight[:, None, None])
        xy = window_sums(design * value[:, None])
        yy = window_sums(value**2)
        gram[~valid] = identity
        try:
            inverse = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(gram)
        beta = np.einsum("nkl,nl->nk", inverse, xy)
        sse = yy - np.einsum("nk,nk->n", beta, xy)
        restricted_gram = gram[:, :2, :2].copy()
        restricted = np.linalg.solve(restricted_gram, xy[:, :2, None])[..., 0]
        sse_restricted = yy - np.einsum("nk,nk->n", restricted, xy[:, :2])
        with np.errstate(invalid="ignore", divide="ignore"):
            total = yy - xy[:, 0] ** 2 / count
            r2_full = 1.0 - sse / total
            r2_restricted = 1.0 - sse_restricted / total
            sigma2 = sse / (count - k)
            se = np.sqrt(sigma2[:, None] * np.einsum("nkk->nk", inverse))
            t = beta / se
            slopes = beta[:, 1:]
            slope = (slopes[:, 1:] * weights_lag[1:]).sum(axis=1) / slopes.sum(axis=1)
            tstats = t[:, 1:]
            tstat = (tstats[:, 1:] * weights_lag[1:]).sum(axis=1) / tstats.sum(axis=1)
            rsq = 1.0 - r2_restricted / r2_full
        keep = valid & (total > 0) & (sse > 1e-14 * np.maximum(yy, 1e-300))
        out["rsq"][rows, column] = np.where(keep & (r2_full > 0), rsq, np.nan)
        out["slope"][rows, column] = np.where(keep, slope, np.nan)
        out["tstat"][rows, column] = np.where(keep, tstat, np.nan)
    return out


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Two-day Corwin-Schultz spread estimates on the second day, floored at zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        today = np.log(high / low) ** 2
        previous = shift_rows(today, 1)
        beta = today + previous
        gamma = np.log(np.fmax(high, shift_rows(high, 1)) / np.fmin(low, shift_rows(low, 1))) ** 2
        denominator = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator - np.sqrt(gamma / denominator)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.nan)


# ---------------------------------------------------------------------------------------------- cross sections
def cross_rank(values: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Percentile rank among the day's index members; missing elsewhere."""
    masked = np.where(universe & np.isfinite(values), values, np.nan)
    return pd.DataFrame(masked).rank(axis=1, pct=True).to_numpy()


def cross_bucket(values: np.ndarray, universe: np.ndarray, buckets: int) -> np.ndarray:
    """Bucket 1..``buckets`` of the day's index members by value; missing elsewhere."""
    ranks = cross_rank(values, universe)
    return np.where(np.isfinite(ranks), np.clip(np.ceil(ranks * buckets), 1, buckets), np.nan)


def group_stat(
    values: np.ndarray,
    groups: np.ndarray,
    universe: np.ndarray,
    how: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-day industry mean, median or std over index members, assigned to every stock of the industry."""
    result = np.full(values.shape, np.nan)
    eligible = universe & np.isfinite(values) & np.isfinite(groups)
    if weights is not None:
        eligible &= np.isfinite(weights) & (weights > 0)
    codes = np.unique(groups[np.isfinite(groups)])
    for code in codes:
        members = groups == code
        # Only the columns that are ever in the industry take part, which keeps each reduction small.
        columns = np.flatnonzero(members.any(axis=0))
        member = members[:, columns]
        selected = eligible[:, columns] & member
        if not selected.any():
            continue
        block = values[:, columns]
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if how == "median":
                stat = np.nanmedian(np.where(selected, block, np.nan), axis=1)
            elif how == "std":
                stat = np.nanstd(np.where(selected, block, np.nan), axis=1, ddof=1)
            else:
                weight = np.where(selected, 1.0 if weights is None else weights[:, columns], 0.0)
                stat = (np.where(selected, block, 0.0) * weight).sum(axis=1) / weight.sum(axis=1)
        result[:, columns] = np.where(member, stat[:, None], result[:, columns])
    return result


def group_rank(values: np.ndarray, groups: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Percentile rank of index members within their industry on each day."""
    result = np.full(values.shape, np.nan)
    eligible = universe & np.isfinite(values) & np.isfinite(groups)
    for code in np.unique(groups[np.isfinite(groups)]):
        columns = np.flatnonzero((groups == code).any(axis=0))
        selected = eligible[:, columns] & (groups[:, columns] == code)
        ranks = pd.DataFrame(np.where(selected, values[:, columns], np.nan)).rank(axis=1, pct=True).to_numpy()
        result[:, columns] = np.where(selected, ranks, result[:, columns])
    return result


def cross_residual(
    target: np.ndarray,
    regressors: list[np.ndarray],
    universe: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    minimum: int = 30,
) -> np.ndarray:
    """Residual of a daily cross-sectional OLS over index members, with optional industry dummies."""
    result = np.full(target.shape, np.nan)
    stacked = np.stack(regressors, axis=2) if regressors else np.zeros((*target.shape, 0))
    usable = universe & np.isfinite(target) & np.isfinite(stacked).all(axis=2)
    if groups is not None:
        usable &= np.isfinite(groups)
    for row in np.flatnonzero(usable.sum(axis=1) >= minimum):
        columns = np.flatnonzero(usable[row])
        parts = [np.ones((len(columns), 1)), stacked[row, columns]]
        if groups is not None:
            codes = groups[row, columns]
            levels = np.unique(codes)
            parts.append((codes[:, None] == levels[None, 1:]).astype(float))
        design = np.hstack(parts)
        if len(columns) < design.shape[1] + 10:
            continue
        coefficients, *_ = np.linalg.lstsq(design, target[row, columns], rcond=None)
        result[row, columns] = target[row, columns] - design @ coefficients
    return result


class OAPSignals:
    """Calculator with shared intermediates; ``calculate`` returns one signal panel on the calendar grid."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna().to_numpy()
        self.close = close.to_numpy(dtype=float)
        self.price = close.ffill().to_numpy(dtype=float)
        self.returns = self.close / shift_rows(self.close, 1) - 1.0
        self.cap = positive(self._array("cap"))
        self.universe = self._panel("universe").reindex(index=self.dates, columns=self.instruments).eq(True).to_numpy()
        self.sector = self._array("sector")
        self._cache: dict[str, Any] = {}
        self._grid: QuarterGrid | None = None

    def calculate(self, name: str) -> Panel:
        result = np.asarray(getattr(self, f"_s_{name}")(), dtype=float)
        result = np.where(np.isfinite(result) & self.trading, result, np.nan)
        return Panel(result, index=self.dates, columns=self.instruments)

    def release(self, name: str) -> None:
        """Drop intermediates no later signal needs (signals are computed in catalog order)."""
        for key in RELEASE_AFTER.get(name, ()):
            self._cache.pop(key, None)

    # ------------------------------------------------------------------ inputs
    def _panel(self, key: str) -> Any:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"oap input is missing: {key}")
        return value

    def _array(self, key: str) -> np.ndarray:
        return self._panel(key).reindex(index=self.dates, columns=self.instruments).to_numpy(dtype=float)

    def _series(self, key: str) -> np.ndarray:
        return np.asarray(pd.Series(self._panel(key)).reindex(self.dates), dtype=float)

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def grid(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(prepare_fundamentals(self._panel("fundamentals")), self.dates, self.instruments)
        return self._grid

    def _ffill(self, values: np.ndarray) -> np.ndarray:
        return pd.DataFrame(values).ffill().to_numpy()

    @property
    def cap_filled(self) -> np.ndarray:
        return self._memo("cap_filled", lambda: self._ffill(self.cap))

    @property
    def turnover(self) -> np.ndarray:
        return self._memo("turnover", lambda: self._array("basic_turnover_rate"))

    @property
    def volume(self) -> np.ndarray:
        return self._memo("volume", lambda: self._array("volume"))

    @property
    def amount(self) -> np.ndarray:
        return self._memo("amount", lambda: positive(self._array("amount")))

    @property
    def raw_close(self) -> np.ndarray:
        return self._memo("raw_close", lambda: positive(self._array("basic_close")))

    @property
    def adjusted_shares(self) -> np.ndarray:
        """Total shares times unadjusted over adjusted close, which removes share splits and bonus issues."""
        def build() -> np.ndarray:
            shares = positive(self._array("basic_total_share"))
            with np.errstate(invalid="ignore", divide="ignore"):
                return self._ffill(shares * self.raw_close / positive(self.close))
        return self._memo("adjusted_shares", build)

    @property
    def dividends(self) -> np.ndarray:
        """Trailing-twelve-month cash dividends in yuan, zero for a quoted non-payer, carried over gaps.

        Tushare counts a payment twice for a few days around its ex-date (up to about two weeks), so the amount is the
        smallest of the last ``DIVIDEND_SPIKE_DAYS`` rows; a real change shows up that many rows later.
        """
        def build() -> np.ndarray:
            quoted = np.isfinite(self.raw_close)
            dividend_yield = finite_or(self._array("basic_dv_ttm"), 0.0)
            amount = self._ffill(np.where(quoted, dividend_yield * self.cap_filled, np.nan))
            return trailing(amount, DIVIDEND_SPIKE_DAYS, 1, "min")
        return self._memo("dividends", build)

    @property
    def age_months(self) -> np.ndarray:
        def build() -> np.ndarray:
            listed = pd.to_datetime(pd.Series(self._panel("list_date")).reindex(self.instruments))
            days = (self.dates.to_numpy()[:, None] - listed.to_numpy()[None, :]) / np.timedelta64(1, "D")
            return np.where(days >= 0, days / 30.4375, np.nan)
        return self._memo("age_months", build)

    # ------------------------------------------------------------------ returns
    def momentum(self, start: int, end: int) -> np.ndarray:
        """Return from ``start`` trading days ago to ``end`` trading days ago (start > end)."""
        return shift_rows(self.price, end) / shift_rows(self.price, start) - 1.0

    @property
    def monthly_returns(self) -> np.ndarray:
        return self._memo("monthly_returns", lambda: self.momentum(MONTH, 0))

    def _lag_month_mean(self, lags: tuple[int, ...], minimum: int) -> np.ndarray:
        """Mean of the 21-day returns ``j`` months back (j = 1 is the latest month)."""
        total = np.zeros(self.price.shape)
        count = np.zeros(self.price.shape)
        for j in lags:
            value = shift_rows(self.monthly_returns, MONTH * (j - 1))
            known = np.isfinite(value)
            total += np.where(known, value, 0.0)
            count += known
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count >= minimum, total / count, np.nan)

    def _factor_month(self, name: str) -> np.ndarray:
        return pd.Series(self._series(name)).rolling(MONTH, min_periods=15).sum().to_numpy()

    def _regression(self, key: str, regressors: tuple[str, ...], window: int, minimum: int) -> dict:
        def build() -> dict:
            x = np.column_stack([self._series(name) for name in regressors])
            return rolling_regression(self.returns, x, window, minimum)
        return self._memo(key, build)

    # ------------------------------------------------------------------ statements
    def b(self, item: str) -> np.ndarray:
        """Balance-sheet item on the quarter grid."""
        return self._memo(f"b:{item}", lambda: sum(self.grid.value(line) for line in ITEMS[item]))

    def _flow(self, item: str, kind: str) -> np.ndarray:
        def build() -> np.ndarray:
            parts = []
            for line in ITEMS[item]:
                values = getattr(self.grid, kind)(line)
                if line in OPTIONAL_LINES and item != "xrd":
                    values = finite_or(values, 0.0)
                parts.append(values)
            return sum(parts)
        return self._memo(f"{kind}:{item}", build)

    def x(self, item: str) -> np.ndarray:
        """Trailing-twelve-month flow on the quarter grid."""
        return self._flow(item, "ttm")

    def qtr(self, item: str) -> np.ndarray:
        """Single-quarter flow on the quarter grid."""
        return self._flow(item, "single")

    def p(self, values: np.ndarray) -> np.ndarray:
        """Latest usable quarter value on every trading day."""
        return self.grid.panel(values).to_numpy()

    def be(self) -> np.ndarray:
        """Common book equity: equity to parent less other equity instruments (preferred and perpetual)."""
        return self.b("seq") - self.b("pstk")

    def avg_at(self) -> np.ndarray:
        return (self.b("at") + lag(self.b("at"), 4)) / 2.0

    def change_over_avg_at(self, values: np.ndarray) -> np.ndarray:
        return self.p(ratio(values - lag(values, 4), self.avg_at()))

    def to_me(self, values: np.ndarray) -> np.ndarray:
        return self.p(values) / self.cap

    def debt(self) -> np.ndarray:
        return self.b("dltt") + self.b("dlc")

    def noa(self) -> np.ndarray:
        """Operating assets (at - che) less operating liabilities (at - debt - mib - be - pstk)."""
        return self.debt() + self.b("mib") + self.be() + self.b("pstk") - self.b("che")

    def ebitda(self) -> np.ndarray:
        return self.x("pi") + self.x("xint") + self.x("dp")

    def me_at_quarter_end(self) -> np.ndarray:
        """Market value on the last trading day of each fiscal quarter (instruments x quarters)."""
        grid = self.grid
        numbers = np.arange(grid.quarters) + grid.first_quarter
        ends = pd.to_datetime([f"{n // 4}-{(n % 4) * 3 + 3:02d}-01" for n in numbers]) + pd.offsets.MonthEnd(0)
        rows = np.searchsorted(self.dates.to_numpy(), ends.to_numpy(), side="right") - 1
        result = np.full((len(self.instruments), grid.quarters), np.nan)
        known = rows >= 0
        result[:, known] = self.cap_filled[rows[known]].T
        return result

    def surprise(self, values: np.ndarray) -> np.ndarray:
        """Four-quarter change minus its mean over the previous eight quarters, over their std."""
        change = values - lag(values, 4)
        drift = quarter_window(change, tuple(range(1, 9)), 6)
        spread = rolling_quarter_std(lag(change, 1), 8, 6)
        return ratio(change - drift, spread)

    def _members(self, values: np.ndarray) -> np.ndarray:
        return np.where(self.universe, values, np.nan)

    # ------------------------------------------------------------------ price
    def _s_Size(self) -> np.ndarray:
        return safe_log(self.cap)

    def _s_Price(self) -> np.ndarray:
        return safe_log(self.raw_close)

    def _s_STreversal(self) -> np.ndarray:
        return self.momentum(MONTH, 0)

    def _s_Mom6m(self) -> np.ndarray:
        return self._memo("mom6", lambda: self.momentum(6 * MONTH, MONTH))

    def _s_Mom12m(self) -> np.ndarray:
        return self.momentum(YEAR, MONTH)

    def _s_IntMom(self) -> np.ndarray:
        return self.momentum(YEAR, 6 * MONTH)

    def _s_MRreversal(self) -> np.ndarray:
        return self.momentum(18 * MONTH, 13 * MONTH - MONTH)

    def _s_LRreversal(self) -> np.ndarray:
        return self._memo("mom36", lambda: self.momentum(36 * MONTH, 13 * MONTH - MONTH))

    def _s_Mom12mOffSeason(self) -> np.ndarray:
        return self._lag_month_mean(tuple(range(1, 12)), 6)

    def _s_MomSeasonShort(self) -> np.ndarray:
        return self._lag_month_mean((12,), 1)

    def _s_MomSeason(self) -> np.ndarray:
        return self._lag_month_mean((24, 36, 48, 60), 2)

    def _s_MomOffSeason(self) -> np.ndarray:
        return self._lag_month_mean(tuple(j for j in range(13, 61) if j % 12), 22)

    def _s_MomSeason06YrPlus(self) -> np.ndarray:
        return self._lag_month_mean((72, 84, 96, 108, 120), 2)

    def _s_MomOffSeason06YrPlus(self) -> np.ndarray:
        return self._lag_month_mean(tuple(j for j in range(61, 121) if j % 12), 27)

    def _s_MaxRet(self) -> np.ndarray:
        return trailing(self.returns, MONTH, 15, "max")

    def _s_ReturnSkew(self) -> np.ndarray:
        return trailing(self.returns, MONTH, 15, "skew")

    def _ff3_month(self) -> dict:
        return self._regression("ff3_month", MARKET_FACTORS, MONTH, 15)

    def _s_ReturnSkew3F(self) -> np.ndarray:
        return self._ff3_month()["resid_skew"]

    def _s_IdioVol3F(self) -> np.ndarray:
        return self._ff3_month()["resid_std"]

    def _s_RealizedVol(self) -> np.ndarray:
        return trailing(self.returns, MONTH, 15, "std")

    def _s_IdioVolAHT(self) -> np.ndarray:
        return self._regression("capm_year", ("mkt",), YEAR, 100)["resid_std"]

    def _s_Beta(self) -> np.ndarray:
        market = self._factor_month("mkt")
        return monthly(
            np.column_stack([market, self.monthly_returns]),
            lambda block: np.column_stack(
                [np.full(len(block), np.nan), rolling_regression(block[:, 1:], block[:, :1], 60, 20)["beta"]]
            ),
        )[:, 1:]

    def _s_BetaFP(self) -> np.ndarray:
        log_returns = np.log1p(self.returns)
        log_market = np.log1p(self._series("mkt"))
        stock_3d = trailing(log_returns, 3, 3, "sum")
        market_3d = pd.Series(log_market).rolling(3, min_periods=3).sum().to_numpy()
        market_panel = np.where(np.isfinite(stock_3d), market_3d[:, None], np.nan)
        correlation = (
            pd.DataFrame(stock_3d).rolling(5 * YEAR, min_periods=3 * YEAR).corr(pd.DataFrame(market_panel)).to_numpy()
        )
        sigma_stock = trailing(log_returns, YEAR, 120, "std")
        market_on_days = np.where(np.isfinite(log_returns), log_market[:, None], np.nan)
        sigma_market = trailing(market_on_days, YEAR, 120, "std")
        return np.clip(correlation, -1.0, 1.0) * ratio(sigma_stock, sigma_market)

    def tail_index(self) -> np.ndarray:
        """Kelly-Jiang tail index from the pooled daily returns of index members over the trailing month."""
        def build() -> np.ndarray:
            members = self._members(self.returns)
            result = np.full(len(self.dates), np.nan)
            for row in range(MONTH - 1, len(self.dates)):
                pooled = members[row - MONTH + 1 : row + 1].ravel()
                pooled = pooled[np.isfinite(pooled)]
                if len(pooled) < 100:
                    continue
                threshold = np.quantile(pooled, 0.05)
                tail = pooled[pooled < threshold]
                if threshold < 0 and len(tail):
                    result[row] = np.mean(np.log(tail / threshold))
            return result
        return self._memo("tail_index", build)

    def _s_BetaTailRisk(self) -> np.ndarray:
        # Each month's stock return is regressed on the previous month's tail index.
        previous_tail = shift_rows(self.tail_index()[:, None], MONTH)[:, 0]
        return monthly(
            np.column_stack([previous_tail, self.monthly_returns]),
            lambda block: np.column_stack(
                [np.full(len(block), np.nan), rolling_regression(block[:, 1:], block[:, :1], 60, 36)["beta"]]
            ),
        )[:, 1:]

    def _s_CoskewACX(self) -> np.ndarray:
        return rolling_coskew(np.log1p(self.returns), np.log1p(self._series("mkt")), YEAR, 150)

    def _s_Coskewness(self) -> np.ndarray:
        market = np.log1p(self._factor_month("mkt"))
        stock = np.log1p(self.monthly_returns)
        return monthly(
            np.column_stack([market, stock]),
            lambda block: np.column_stack(
                [np.full(len(block), np.nan), rolling_coskew(block[:, 1:], block[:, 0], 60, 24)]
            ),
        )[:, 1:]

    def _price_delay(self) -> dict:
        return self._memo("price_delay", lambda: price_delay(self.returns, self._series("mkt"), YEAR, 150))

    def _s_PriceDelayRsq(self) -> np.ndarray:
        return self._price_delay()["rsq"]

    def _s_PriceDelaySlope(self) -> np.ndarray:
        return self._price_delay()["slope"]

    def _s_PriceDelayTstat(self) -> np.ndarray:
        return self._price_delay()["tstat"]

    def _s_ResidualMomentum(self) -> np.ndarray:
        factors = np.column_stack([self._factor_month(name) for name in MARKET_FACTORS])

        def reduce(block: np.ndarray) -> np.ndarray:
            fit = rolling_regression(block[:, 3:], block[:, :3], 36, 24, residual_slices={"momentum": (24, 35)})
            return np.column_stack([np.full((len(block), 3), np.nan), fit["momentum"]])

        return monthly(np.column_stack([factors, self.monthly_returns]), reduce)[:, 3:]

    def _s_TrendFactor(self) -> np.ndarray:
        price = pd.DataFrame(self.close)
        # Averages of the quotes inside each window; a short suspension does not void the long averages.
        stacked = np.stack(
            [ratio(price.rolling(n, min_periods=int(np.ceil(0.8 * n))).mean().to_numpy(), self.close)
             for n in TREND_WINDOWS],
            axis=2,
        )
        complete = np.isfinite(stacked).all(axis=2)
        lagged_universe = shift_rows(self.universe.astype(float), MONTH) > 0
        target = self.momentum(MONTH, 0)
        k = len(TREND_WINDOWS) + 1
        loadings = np.full((len(self.dates), k), np.nan)
        # Each day's return over the last month is regressed on the averages of a month earlier.
        usable = np.zeros(target.shape, dtype=bool)
        usable[MONTH:] = lagged_universe[MONTH:] & np.isfinite(target[MONTH:]) & complete[:-MONTH]
        for row in np.flatnonzero(usable.sum(axis=1) >= 3 * k):
            columns = np.flatnonzero(usable[row])
            design = np.column_stack([np.ones(len(columns)), stacked[row - MONTH, columns]])
            loadings[row], *_ = np.linalg.lstsq(design, target[row, columns], rcond=None)
        # Expected loadings average the twelve monthly cross-sections ending today.
        history = np.stack([shift_rows(loadings, MONTH * j) for j in range(12)], axis=2)
        expected, count = nan_mean(history, 2)
        expected = np.where(count >= 8, expected, np.nan)
        return np.einsum("tsk,tk->ts", stacked, expected[:, 1:])

    def _s_High52(self) -> np.ndarray:
        return ratio(self.close, trailing(self.close, YEAR, 126, "max"))

    def _s_IndMom(self) -> np.ndarray:
        returns = self.momentum(6 * MONTH, 0)
        return group_stat(returns, self.sector, self.universe, "mean", weights=self.cap)

    def _big_in_industry(self) -> np.ndarray:
        return self._memo("big", lambda: group_rank(self.cap, self.sector, self.universe) > 0.7)

    def _for_small_firms(self, values: np.ndarray) -> np.ndarray:
        big = self._big_in_industry()
        industry = group_stat(values, self.sector, self.universe & big, "mean")
        return np.where(self.universe & ~big, industry, np.nan)

    def _s_IndRetBig(self) -> np.ndarray:
        return self._for_small_firms(self.momentum(MONTH, 0))

    def _s_FirmAgeMom(self) -> np.ndarray:
        age = np.where(self.age_months >= 12, self.age_months, np.nan)
        young = cross_bucket(age, self.universe, 5) == 1
        return np.where(young, self._s_Mom6m(), np.nan)

    def _s_MomRev(self) -> np.ndarray:
        short = cross_bucket(self._s_Mom6m(), self.universe, 5)
        long = cross_bucket(self._s_LRreversal(), self.universe, 5)
        result = np.where((short == 5) & (long == 1), 1.0, np.nan)
        return np.where((short == 1) & (long == 5), 0.0, result)

    def _s_MomVol(self) -> np.ndarray:
        volume = trailing(self.turnover, 6 * MONTH, 5 * MONTH, "mean")
        seasoned = self.age_months >= 24
        high = cross_bucket(np.where(seasoned, volume, np.nan), self.universe, 3) == 3
        decile = cross_bucket(np.where(seasoned, self._s_Mom6m(), np.nan), self.universe, 10)
        return np.where(high, decile, np.nan)

    # ------------------------------------------------------------------ trading
    def _s_DolVol(self) -> np.ndarray:
        return safe_log(shift_rows(trailing(self.amount, MONTH, 15, "sum"), MONTH))

    def _s_Illiquidity(self) -> np.ndarray:
        return trailing(ratio(np.abs(self.returns), self.amount), YEAR, 120, "mean")

    def _s_ShareVol(self) -> np.ndarray:
        bucket = cross_bucket(trailing(self.turnover, QUARTER, 40, "mean"), self.universe, 3)
        return np.where(bucket == 3, 1.0, np.where(bucket == 1, 0.0, np.nan))

    def _monthly_std(self, values: np.ndarray, window: int, minimum: int) -> np.ndarray:
        sums = trailing(values, MONTH, 15, "sum")
        return monthly(sums, lambda block: trailing(block, window, minimum, "std"))

    def _s_std_turn(self) -> np.ndarray:
        return self._monthly_std(self.turnover, 36, 24)

    def _s_VolSD(self) -> np.ndarray:
        return self._monthly_std(self.volume, 36, 24)

    def _s_VolMkt(self) -> np.ndarray:
        return trailing(self.amount, YEAR, 120, "mean") * MONTH / self.cap

    def _s_VolumeTrend(self) -> np.ndarray:
        volume = trailing(self.volume, MONTH, 15, "sum")

        def reduce(block: np.ndarray) -> np.ndarray:
            trend = np.arange(len(block), dtype=float)[:, None]
            slope = rolling_regression(block, trend, 60, 30)["beta"]
            return ratio(slope, trailing(block, 60, 30, "mean"))

        return monthly(volume, reduce)

    def _zerotrade(self, months: int) -> np.ndarray:
        window = months * MONTH
        listed = np.maximum.accumulate(self.trading, axis=0)
        zero_days = trailing((listed & ~self.trading).astype(float), window, window, "sum")
        turnover = trailing(finite_or(self.turnover, 0.0), window, window, "sum")
        return zero_days + ratio(1.0, positive(turnover)) / 480_000.0

    def _s_zerotrade1M(self) -> np.ndarray:
        return self._zerotrade(1)

    def _s_zerotrade6M(self) -> np.ndarray:
        return self._zerotrade(6)

    def _s_zerotrade12M(self) -> np.ndarray:
        return self._zerotrade(12)

    def _s_BidAskSpread(self) -> np.ndarray:
        spread = corwin_schultz(self._array("high"), self._array("low"))
        return trailing(spread, MONTH, 15, "mean")

    # ------------------------------------------------------------------ valuation
    def _s_AM(self) -> np.ndarray:
        return self.to_me(self.b("at"))

    def _s_BM(self) -> np.ndarray:
        tangible = self.be() - self.b("intan")
        return self.p(safe_log(ratio(tangible, self.me_at_quarter_end())))

    def _s_BMdec(self) -> np.ndarray:
        december = pd.to_datetime([f"{year - 1}-12-31" for year in self.dates.year])
        rows = np.searchsorted(self.dates.to_numpy(), december.to_numpy(), side="right") - 1
        columns = np.arange(len(self.instruments))
        me = np.where(rows[:, None] >= 0, self.cap_filled[np.clip(rows, 0, None)[:, None], columns[None, :]], np.nan)
        return ratio(self.p(self.be()), me)

    def _s_BookLeverage(self) -> np.ndarray:
        return self.p(ratio(self.b("at"), positive(self.b("seq") + self.b("txditc"))))

    def _s_Leverage(self) -> np.ndarray:
        return self.to_me(self.b("lt"))

    def _s_EP(self) -> np.ndarray:
        return positive(self.to_me(self.x("ib")))

    def _s_CF(self) -> np.ndarray:
        return self.to_me(self.x("ib") + self.x("dp"))

    def _s_cfp(self) -> np.ndarray:
        return self.to_me(self.x("oancf"))

    def _s_SP(self) -> np.ndarray:
        return self.to_me(self.x("sale"))

    def _net_debt(self) -> np.ndarray:
        return self.p(self.debt() + self.b("pstk") - self.b("che"))

    def _s_EBM(self) -> np.ndarray:
        net_debt = self._net_debt()
        return ratio(self.p(self.be()) - net_debt, self.cap - net_debt)

    def _s_BPEBM(self) -> np.ndarray:
        return ratio(self.p(self.be()), self.cap) - self._s_EBM()

    def _s_NetDebtPrice(self) -> np.ndarray:
        return self._net_debt() / self.cap

    def _s_EntMult(self) -> np.ndarray:
        enterprise = self.cap + self.p(self.debt() - self.b("che"))
        known_book = np.isfinite(self.p(self.be()))
        return np.where(known_book, ratio(enterprise, positive(self.p(self.ebitda()))), np.nan)

    def _s_CashProd(self) -> np.ndarray:
        return ratio(self.cap - self.p(self.b("at")), self.p(self.b("che")))

    def _s_PayoutYield(self) -> np.ndarray:
        return positive(ratio(self.dividends, shift_rows(self.cap_filled, 6 * MONTH)))

    def _s_NetPayoutYield(self) -> np.ndarray:
        payout = self.dividends - self.p(self.x("sstk"))
        result = ratio(payout, shift_rows(self.cap_filled, 6 * MONTH))
        return np.where(result != 0, result, np.nan)

    def _s_EquityDuration(self) -> np.ndarray:
        discount, horizon = 0.12, 10
        book = self.be()
        roe = ratio(self.x("ib"), positive(lag(book, 4)))
        sales_growth = growth(self.x("sale"), lag(self.x("sale"), 4))
        weighted = np.zeros(book.shape)
        present = np.zeros(book.shape)
        for year in range(1, horizon + 1):
            roe = 0.12 + 0.57 * (roe - 0.12)
            sales_growth = 0.06 + 0.24 * (sales_growth - 0.06)
            distribution = book * (roe - sales_growth)
            book = book * (1.0 + sales_growth)
            factor = (1.0 + discount) ** year
            weighted = weighted + year * distribution / factor
            present = present + distribution / factor
        known_book = positive(self.be())
        weighted = np.where(np.isfinite(known_book), weighted, np.nan)
        present = np.where(np.isfinite(known_book), present, np.nan)
        me = self.cap
        return self.p(weighted) / me + (horizon + (1.0 + discount) / discount) * (me - self.p(present)) / me

    def _intangible_return(self, numerator: np.ndarray) -> np.ndarray:
        log_ratio = safe_log(self.to_me(numerator))
        five = 5 * YEAR
        stock_return = safe_log(ratio(self.price, shift_rows(self.price, five)))
        lagged = shift_rows(log_ratio, five)
        accounting_return = log_ratio - lagged + stock_return
        return cross_residual(stock_return, [lagged, accounting_return], self.universe)

    def _s_IntanBM(self) -> np.ndarray:
        return self._intangible_return(self.be())

    def _s_IntanCFP(self) -> np.ndarray:
        return self._intangible_return(self.x("ib") + self.x("dp"))

    def _s_IntanEP(self) -> np.ndarray:
        return self._intangible_return(self.x("ib"))

    def _s_IntanSP(self) -> np.ndarray:
        return self._intangible_return(self.x("sale"))

    def _s_Frontier(self) -> np.ndarray:
        book = positive(self.be())
        assets = self.b("at")
        sales = self.x("sale")
        regressors = [
            self.p(safe_log(book)),
            self.p(ratio(self.b("dltt"), assets)),
            self.p(ratio(self.x("capx"), sales)),
            self.p(finite_or(ratio(self.x("xrd"), sales), 0.0) * np.isfinite(sales)),
            self.p(ratio(self.b("ppe"), assets)),
            self.p(ratio(self.ebitda(), assets)),
        ]
        target = safe_log(self.p(book) / self.cap)
        return cross_residual(target, regressors, self.universe, groups=self.sector)

    def _s_VarCF(self) -> np.ndarray:
        cash_flow_price = self._s_CF()
        return monthly(cash_flow_price, lambda block: trailing(block, 60, 24, "var"))

    # ------------------------------------------------------------------ profitability
    def _s_Cash(self) -> np.ndarray:
        return self.p(ratio(self.b("che"), self.b("at")))

    def _s_GP(self) -> np.ndarray:
        return self.p(ratio(self.x("sale") - self.x("cogs"), self.b("at")))

    def _s_OperProf(self) -> np.ndarray:
        profit = self.x("sale") - self.x("cogs") - self.x("xsga") - self.x("xint")
        return self.p(ratio(profit, positive(self.be())))

    def _operating_profit_rd(self) -> np.ndarray:
        return self.x("sale") - self.x("cogs") - (self.x("xsga") - finite_or(self.x("xrd"), 0.0))

    def _s_OperProfRD(self) -> np.ndarray:
        return self.p(ratio(self._operating_profit_rd(), self.b("at")))

    def _s_CBOperProf(self) -> np.ndarray:
        def change(item: str) -> np.ndarray:
            return self.b(item) - lag(self.b(item), 4)

        cash_profit = (
            self._operating_profit_rd() - change("rect") - change("invt") - change("prepaid") + change("ap")
        )
        return self.p(ratio(cash_profit, self.b("at")))

    def _s_RoE(self) -> np.ndarray:
        return self.p(ratio(self.x("ni"), positive(self.be())))

    def _s_roaq(self) -> np.ndarray:
        return self.p(ratio(self.qtr("ib"), lag(self.b("at"), 1)))

    def _s_OPLeverage(self) -> np.ndarray:
        return self.p(ratio(self.x("xsga") + self.x("cogs"), self.b("at")))

    def _s_tang(self) -> np.ndarray:
        tangible = self.b("che") + 0.715 * self.b("rect") + 0.547 * self.b("invt") + 0.535 * self.b("ppe")
        return self.p(ratio(tangible, self.b("at")))

    def _s_Tax(self) -> np.ndarray:
        grid = self.grid
        years = (np.arange(grid.quarters) + grid.first_quarter) // 4
        rate = np.where(years < 2008, 0.33, 0.25)[None, :]
        taxes, income = self.x("txt"), self.x("ib")
        result = ratio(taxes, rate * income)
        return self.p(np.where((income < 0) & (taxes > 0), 1.0, result))

    def _s_PS(self) -> np.ndarray:
        assets = self.b("at")
        income, cash_flow, sales = self.x("ib"), self.x("oancf"), self.x("sale")
        roa = ratio(income, assets)
        leverage = ratio(self.b("dltt"), assets)
        liquidity = ratio(self.b("act"), self.b("lct"))
        margin = ratio(income + self.x("txt") + self.x("xint"), sales)
        turnover = ratio(sales, assets)
        parts = [
            (income > 0, income),
            (cash_flow > 0, cash_flow),
            (roa > lag(roa, 4), roa - lag(roa, 4)),
            (cash_flow > income, cash_flow - income),
            (leverage < lag(leverage, 4), leverage - lag(leverage, 4)),
            (liquidity > lag(liquidity, 4), liquidity - lag(liquidity, 4)),
            (margin > lag(margin, 4), margin - lag(margin, 4)),
            (turnover > lag(turnover, 4), turnover - lag(turnover, 4)),
        ]
        known = np.all([np.isfinite(value) for _, value in parts], axis=0)
        score = self.p(np.where(known, np.sum([signal.astype(float) for signal, _ in parts], axis=0), np.nan))
        shares = self.adjusted_shares
        previous = shift_rows(shares, YEAR)
        no_issue = flag(shares <= previous * 1.0001, np.isfinite(shares) & np.isfinite(previous))
        return score + no_issue

    def _s_MS(self) -> np.ndarray:
        assets = self.b("at")
        roa = ratio(self.x("ib"), assets)
        cfroa = ratio(self.x("oancf"), assets)
        quarterly_roa = ratio(self.qtr("ib"), lag(assets, 1))
        sales = self.qtr("sale")
        components = [
            self.p(roa),
            self.p(cfroa),
            self.p(-rolling_quarter_std(quarterly_roa, 16, 8)),
            self.p(-rolling_quarter_std(growth(sales, lag(sales, 4)), 16, 8)),
            self.p(ratio(self.x("xrd"), assets)),
            self.p(ratio(self.x("capx"), assets)),
        ]
        score = self.p((cfroa > roa).astype(float))
        for values in components:
            median = group_stat(values, self.sector, self.universe, "median")
            score = score + np.where(np.isfinite(values) & (values > median), 1.0, 0.0)
        return np.where(np.isfinite(components[0]) & np.isfinite(components[1]), score, np.nan)

    def _s_OScore(self) -> np.ndarray:
        assets, liabilities = self.b("at"), self.b("lt")
        current_assets, current_liabilities = self.b("act"), self.b("lct")
        income = self.x("ib")
        previous = lag(income, 4)
        score = (
            -1.32
            - 0.407 * safe_log(assets)
            + 6.03 * ratio(liabilities, assets)
            - 1.43 * ratio(current_assets - current_liabilities, assets)
            + 0.076 * ratio(current_liabilities, current_assets)
            - 1.72 * flag(liabilities > assets, np.isfinite(liabilities - assets))
            - 2.37 * ratio(income, assets)
            - 1.83 * ratio(self.x("oancf"), liabilities)
            + 0.285 * flag((income < 0) & (previous < 0), np.isfinite(income + previous))
            - 0.521 * ratio(income - previous, np.abs(income) + np.abs(previous))
        )
        return self.p(score)

    def _s_EarningsConsistency(self) -> np.ndarray:
        earnings = self.x("ib")
        base = (np.abs(lag(earnings, 4)) + np.abs(lag(earnings, 8))) / 2.0
        change = ratio(earnings - lag(earnings, 4), base)
        change = np.where(np.abs(change) <= 6.0, change, np.nan)
        change = np.where(np.sign(change) == np.sign(lag(change, 4)), change, np.nan)
        return self.p(quarter_window(change, (0, 4, 8, 12), 4))

    def _earnings_surprise(self) -> np.ndarray:
        return self._memo("earnings_surprise", lambda: self.p(self.surprise(self.qtr("ib"))))

    def _s_EarningsSurprise(self) -> np.ndarray:
        return self._earnings_surprise()

    def _s_RevenueSurprise(self) -> np.ndarray:
        return self.p(self.surprise(self.qtr("sale")))

    def _s_NumEarnIncrease(self) -> np.ndarray:
        earnings = self.qtr("ib")
        previous = lag(earnings, 4)
        known = np.isfinite(earnings) & np.isfinite(previous)
        streak = np.full(earnings.shape, np.nan)
        running = np.zeros(earnings.shape[0])
        for column in range(earnings.shape[1]):
            running = np.where(known[:, column] & (earnings[:, column] > previous[:, column]),
                               np.minimum(running + 1.0, 8.0), 0.0)
            streak[:, column] = np.where(known[:, column], running, np.nan)
        return self.p(streak)

    def _s_EarnSupBig(self) -> np.ndarray:
        return self._for_small_firms(self._earnings_surprise())

    def _s_MeanRankRevGrowth(self) -> np.ndarray:
        sales = self.x("sale")
        ranks = cross_rank(self.p(growth(sales, lag(sales, 4))), self.universe)
        total = np.zeros(ranks.shape)
        weights = np.zeros(ranks.shape)
        for year, weight in enumerate((5.0, 4.0, 3.0, 2.0, 1.0)):
            value = shift_rows(ranks, YEAR * year)
            known = np.isfinite(value)
            total += np.where(known, weight * value, 0.0)
            weights += weight * known
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weights >= 12.0, total / weights, np.nan)

    def _growth_over_two_years(self, values: np.ndarray) -> np.ndarray:
        """Growth relative to the average of one and two years ago, or one year ago when two is missing."""
        previous = lag(values, 4)
        base = np.where(np.isfinite(lag(values, 8)), (previous + lag(values, 8)) / 2.0, previous)
        return growth(values, base)

    def _s_GrSaleToGrInv(self) -> np.ndarray:
        return self.p(self._growth_over_two_years(self.x("sale")) - self._growth_over_two_years(self.b("invt")))

    def _s_GrSaleToGrOverhead(self) -> np.ndarray:
        return self.p(self._growth_over_two_years(self.x("sale")) - self._growth_over_two_years(self.x("xsga")))

    def _s_ChAssetTurnover(self) -> np.ndarray:
        noa = self.noa()
        turnover = ratio(self.x("sale"), positive((noa + lag(noa, 4)) / 2.0))
        return self.p(turnover - lag(turnover, 4))

    def _s_OrgCap(self) -> np.ndarray:
        grid = self.grid
        expense = self.grid.annual("sell_exp") + self.grid.annual("admin_exp")
        capital = np.full(expense.shape, np.nan)
        running = np.full(expense.shape[0], np.nan)
        for column in np.flatnonzero(grid.quarter_index == 3):
            current = expense[:, column]
            started = np.isfinite(running)
            running = np.where(started, 0.85 * running + finite_or(current, 0.0), 4.0 * current)
            capital[:, column] = np.where(np.isfinite(current), running, np.nan)
        scaled = self.grid.panel(ratio(capital, self.b("at")), annual=True).to_numpy()
        mean = group_stat(scaled, self.sector, self.universe, "mean")
        spread = group_stat(scaled, self.sector, self.universe, "std")
        return ratio(scaled - mean, spread)

    # ------------------------------------------------------------------ investment and accruals
    def _s_AssetGrowth(self) -> np.ndarray:
        assets = self.b("at")
        return self.p(growth(assets, lag(assets, 4)))

    def _s_ChEQ(self) -> np.ndarray:
        book = positive(self.be())
        return self.p(ratio(book, lag(book, 4)))

    def _s_ChInv(self) -> np.ndarray:
        return self.change_over_avg_at(self.b("invt"))

    def _s_InvGrowth(self) -> np.ndarray:
        inventory = self.b("invt")
        return self.p(growth(inventory, lag(inventory, 4)))

    def _s_InvestPPEInv(self) -> np.ndarray:
        values = self.b("ppe") + self.b("invt")
        return self.p(ratio(values - lag(values, 4), lag(self.b("at"), 4)))

    def _s_dNoa(self) -> np.ndarray:
        noa = self.noa()
        return self.p(ratio(noa - lag(noa, 4), lag(self.b("at"), 4)))

    def _s_NOA(self) -> np.ndarray:
        return self.p(ratio(self.noa(), lag(self.b("at"), 4)))

    def _s_ChNWC(self) -> np.ndarray:
        working = ratio((self.b("act") - self.b("che")) - (self.b("lct") - self.b("dlc")), self.b("at"))
        return self.p(working - lag(working, 4))

    def _s_ChNNCOA(self) -> np.ndarray:
        assets = self.b("at") - self.b("act") - self.b("ivao")
        liabilities = self.b("lt") - self.b("lct") - self.b("dltt")
        non_current = ratio(assets - liabilities, self.b("at"))
        return self.p(non_current - lag(non_current, 4))

    def _s_DelCOA(self) -> np.ndarray:
        return self.change_over_avg_at(self.b("act") - self.b("che"))

    def _s_DelCOL(self) -> np.ndarray:
        return self.change_over_avg_at(self.b("lct") - self.b("dlc"))

    def _s_DelEqu(self) -> np.ndarray:
        return self.change_over_avg_at(self.be())

    def _s_DelFINL(self) -> np.ndarray:
        return self.change_over_avg_at(self.debt() + self.b("pstk"))

    def _s_DelLTI(self) -> np.ndarray:
        return self.change_over_avg_at(self.b("ivao"))

    def _s_DelNetFin(self) -> np.ndarray:
        return self.change_over_avg_at(self.b("ivst") + self.b("ivao") - self.debt() - self.b("pstk"))

    def _s_GrLTNOA(self) -> np.ndarray:
        assets = self.b("at")
        other_assets = self.b("nca") - self.b("ppe") - self.b("intan") - self.b("ivao")
        current_other = self.b("lct") - self.b("dlc") - self.b("ap")
        long_other = self.b("ncl") - self.b("dltt")
        operating = (
            self.b("rect") + self.b("invt") + self.b("ppe") + self.b("aco") + self.b("intan") + other_assets
            - self.b("ap") - current_other - long_other
        )
        noa = ratio(operating, assets)

        def change(values: np.ndarray) -> np.ndarray:
            return values - lag(values, 4)

        accruals = ratio(
            change(self.b("rect")) + change(self.b("invt")) + change(self.b("aco"))
            - (change(self.b("ap")) + change(current_other)) - self.x("dp"),
            self.avg_at(),
        )
        return self.p(noa - lag(noa, 4) - accruals)

    def _accruals(self) -> np.ndarray:
        def change(item: str) -> np.ndarray:
            return self.b(item) - lag(self.b(item), 4)

        working = change("act") - change("che") - (change("lct") - change("dlc") - change("txp"))
        return ratio(working, self.avg_at())

    def _s_Accruals(self) -> np.ndarray:
        return self.p(self._accruals())

    def _s_AbnormalAccruals(self) -> np.ndarray:
        assets = self.b("at")
        lagged = lag(assets, 4)
        accruals = self.p(ratio(self.x("ib") - self.x("oancf"), self.avg_at()))
        regressors = [
            self.p(ratio(1.0, lagged)) * 1e9,
            self.p(ratio(self.x("sale") - lag(self.x("sale"), 4), lagged)),
            self.p(ratio(self.b("ppe"), lagged)),
        ]
        return cross_residual(accruals, regressors, self.universe)

    def _s_AccrualsBM(self) -> np.ndarray:
        book_to_market = ratio(self.p(positive(self.be())), self.cap)
        accruals = np.where(np.isfinite(book_to_market), self.p(self._accruals()), np.nan)
        accrual_bucket = cross_bucket(accruals, self.universe, 5)
        value_bucket = cross_bucket(book_to_market, self.universe, 5)
        result = np.where((accrual_bucket == 5) & (value_bucket == 1), 1.0, np.nan)
        return np.where((accrual_bucket == 1) & (value_bucket == 5), 0.0, result)

    def _s_PctAcc(self) -> np.ndarray:
        income = self.x("ib")
        return self.p(ratio(income - self.x("oancf"), np.abs(income)))

    def _total_accruals(self) -> np.ndarray:
        """Net income less operating, investing and financing cash flows, plus equity issuance, less dividends."""
        flows = self.x("ni") - self.x("oancf") - self.x("ivncf") - self.x("fincf") + self.x("sstk")
        return self.p(flows) - self.dividends

    def _s_PctTotAcc(self) -> np.ndarray:
        return ratio(self._total_accruals(), self.p(np.abs(self.x("ni"))))

    def _s_TotalAccruals(self) -> np.ndarray:
        return ratio(self._total_accruals(), self.p(lag(self.b("at"), 4)))

    def _s_ChTax(self) -> np.ndarray:
        tax = self.qtr("txt")
        return self.p(ratio(tax - lag(tax, 4), lag(self.b("at"), 4)))

    def _s_Investment(self) -> np.ndarray:
        intensity = ratio(self.x("capx"), positive(self.x("sale")))
        return self.p(ratio(intensity, quarter_window(intensity, (4, 8, 12), 3)))

    def _s_ChInvIA(self) -> np.ndarray:
        capex = self.x("capx")
        previous = lag(capex, 4)
        base = np.where(np.isfinite(lag(capex, 8)), (previous + lag(capex, 8)) / 2.0, previous)
        capex_growth = self.p(growth(capex, base))
        return capex_growth - group_stat(capex_growth, self.sector, self.universe, "mean")

    def _s_grcapx(self) -> np.ndarray:
        capex = self.x("capx")
        return self.p(growth(capex, lag(capex, 8)))

    def _s_grcapx3y(self) -> np.ndarray:
        capex = self.x("capx")
        return self.p(ratio(capex, positive(quarter_window(capex, (4, 8, 12), 3))))

    def _s_CompositeDebtIssuance(self) -> np.ndarray:
        debt = self.debt()
        return self.p(safe_log(debt) - safe_log(lag(debt, 20)))

    def _s_CompEquIss(self) -> np.ndarray:
        five = 5 * YEAR
        value_growth = safe_log(ratio(self.cap_filled, shift_rows(self.cap_filled, five)))
        price_growth = safe_log(ratio(self.price, shift_rows(self.price, five)))
        return value_growth - price_growth

    def _s_ShareIss1Y(self) -> np.ndarray:
        shares = self.adjusted_shares
        return ratio(shift_rows(shares, 6 * MONTH), shift_rows(shares, 18 * MONTH)) - 1.0

    def _s_ShareIss5Y(self) -> np.ndarray:
        shares = self.adjusted_shares
        return ratio(shares, shift_rows(shares, 5 * YEAR)) - 1.0

    def _s_NetEquityFinance(self) -> np.ndarray:
        return self.p(ratio(self.x("sstk"), self.avg_at()))

    def _net_debt_issued(self) -> np.ndarray:
        return self.x("dltis") - self.x("dltr")

    def _s_NetDebtFinance(self) -> np.ndarray:
        return self.p(ratio(self._net_debt_issued(), self.avg_at()))

    def _s_XFIN(self) -> np.ndarray:
        financing = self.p(self.x("sstk") + self._net_debt_issued()) - self.dividends
        return ratio(financing, self.p(self.b("at")))

    def _s_DebtIssuance(self) -> np.ndarray:
        issued = self.x("dltis")
        return self.p(flag(issued > 0, np.isfinite(issued)))

    # ------------------------------------------------------------------ R&D and other
    def _s_RD(self) -> np.ndarray:
        return self.to_me(self.x("xrd"))

    def _s_RDcap(self) -> np.ndarray:
        research = self.x("xrd")
        capital = research.copy()
        for years, weight in ((1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)):
            capital = capital + weight * finite_or(lag(research, 4 * years), 0.0)
        return self.p(ratio(capital, self.b("at")))

    def _s_SurpriseRD(self) -> np.ndarray:
        research, assets = self.x("xrd"), self.b("at")
        intensity = ratio(research, assets)
        surprise = (
            (ratio(research, self.x("sale")) > 0)
            & (intensity > 0)
            & (growth(research, lag(research, 4)) > 0.05)
            & (growth(intensity, lag(intensity, 4)) > 0.05)
        )
        return self.p(flag(surprise, np.isfinite(research)))

    def _herfindahl(self, values: np.ndarray) -> np.ndarray:
        values = positive(self.p(values))
        size = self._industry_size(values)
        share = values / (group_stat(values, self.sector, self.universe, "mean") * size)
        index = group_stat(share**2, self.sector, self.universe, "mean") * size
        history = np.stack([shift_rows(index, YEAR * k) for k in range(3)], axis=2)
        return nan_mean(history, 2)[0]

    def _industry_size(self, values: np.ndarray) -> np.ndarray:
        """Number of index members with a value in each stock's industry on each day."""
        known = np.where(np.isfinite(values) & self.universe, 1.0, 0.0)
        result = np.full(values.shape, np.nan)
        for code in np.unique(self.sector[np.isfinite(self.sector)]):
            members = self.sector == code
            size = (known * members).sum(axis=1)
            result = np.where(members, np.where(size > 0, size, np.nan)[:, None], result)
        return result

    def _s_Herf(self) -> np.ndarray:
        return self._herfindahl(self.x("sale"))

    def _s_HerfBE(self) -> np.ndarray:
        return self._herfindahl(self.be())

    def _s_HerfAsset(self) -> np.ndarray:
        return self._herfindahl(self.b("at"))

    def _s_FirmAge(self) -> np.ndarray:
        return self.age_months

    # ------------------------------------------------------------------ events
    def _s_AnnouncementReturn(self) -> np.ndarray:
        frame = self._panel("fundamentals")
        abnormal = self.returns - self._series("mkt")[:, None]
        window_sum = trailing(abnormal, 4, 3, "sum")
        announced = np.searchsorted(self.dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy())
        columns = self.instruments.get_indexer(frame["instrument"])
        # The window ends two trading days after the announcement, when the signal is published.
        published = announced + 2
        keep = (columns >= 0) & (published < len(self.dates))
        events = np.full(window_sum.shape, np.nan)
        events[published[keep], columns[keep]] = window_sum[published[keep], columns[keep]]
        return self._ffill(events)

    def _s_IndIPO(self) -> np.ndarray:
        age = self.age_months
        return flag((age >= 6) & (age <= 36), np.isfinite(age))

    def _s_RDIPO(self) -> np.ndarray:
        grid = self.grid
        research, sales = self.x("xrd"), self.x("sale")
        # A trailing year that lies wholly after R&D became its own line, with a blank line, has no R&D.
        separate = (np.arange(grid.quarters) + grid.first_quarter) >= RD_LINE_YEAR * 4 + 3
        known = np.isfinite(sales) & separate[None, :]
        no_research = self.p(flag(finite_or(research, 0.0) == 0, known))
        return self._s_IndIPO() * no_research

    def _dividend_changes(self) -> np.ndarray:
        """Days the trailing dividend amount changes by more than 5%: a payment enters or a year-old one drops out."""
        def build() -> np.ndarray:
            dividends = self.dividends
            previous = shift_rows(dividends, 1)
            larger = np.fmax(dividends, previous)
            known = np.isfinite(dividends) & np.isfinite(previous) & (larger > 0)
            return known & (np.abs(dividends - previous) > 0.05 * larger)
        return self._memo("dividend_changes", build)

    def _s_DivInit(self) -> np.ndarray:
        dividends = self.dividends
        half, year = shift_rows(dividends, 6 * MONTH), shift_rows(dividends, YEAR)
        return flag((dividends > 0) & (half == 0) & (year == 0), np.isfinite(dividends) & np.isfinite(year))

    def _s_DivOmit(self) -> np.ndarray:
        dividends = self.dividends
        recent, year, two = (shift_rows(dividends, lag_days) for lag_days in (2 * MONTH, YEAR, 2 * YEAR))
        omitted = (dividends == 0) & (recent > 0) & (year > 0) & (two > 0)
        return flag(omitted, np.isfinite(dividends) & np.isfinite(two))

    def _seasonal_payer(self) -> np.ndarray:
        """1 where a dividend changed 11 to 12 months ago, so the annual payment is due within a month."""
        changes = trailing(self._dividend_changes().astype(float), MONTH, 1, "max")
        return shift_rows(changes, YEAR - MONTH) > 0

    def _s_DivSeason(self) -> np.ndarray:
        return np.where(self.dividends > 0, self._seasonal_payer().astype(float), np.nan)

    def _s_DivYieldST(self) -> np.ndarray:
        expected = np.where(self._seasonal_payer(), ratio(self.dividends, self.cap), 0.0)
        bins = np.where(expected <= 0, 0.0, np.where(expected <= 0.005, 1.0, np.where(expected <= 0.01, 2.0, 3.0)))
        return np.where((self.dividends > 0) & np.isfinite(self.cap), bins, np.nan)
