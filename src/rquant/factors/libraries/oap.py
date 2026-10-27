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
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS
from rquant.errors import DataContractError
from rquant.factors.extensions import kunquant_oap as op
from rquant.factors.extensions import kunquant_oap_regression as regression_op
from rquant.factors.extensions.kunquant_oap import (
    cross_bucket,
    cross_rank,
    cross_residual,
    g,
    group_rank,
    group_stat,
    ratio,
)
from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.factors.libraries.china_chars import CPP_PRELUDE as CNC_PRELUDE
from rquant.factors.statements import QuarterGrid
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
        "money_cap",
        "trad_asset",
        "lt_eqt_invest",
        "fa_avail_for_sale",
        "htm_invest",
        "accounts_receiv",
        "notes_receiv",
        "inventories",
        "fix_assets",
        "intan_assets",
        "goodwill",
        "st_borr",
        "non_cur_liab_due_1y",
        "lt_borr",
        "bond_payable",
        "acct_payable",
        "notes_payable",
        "taxes_payable",
        "defer_tax_liab",
        "oth_eqt_tools_p_shr",
        "minority_int",
        "prepayment",
        "oth_receiv",
    ),
    "revenue": ("sell_exp", "admin_exp", "fin_exp_int_exp"),
    "n_cashflow_act": (
        "depr_fa_coga_dpba",
        "amort_intang_assets",
        "lt_amort_deferred_exp",
        "c_recp_cap_contrib",
        "c_recp_borrow",
        "proc_issue_bonds",
        "c_prepay_amt_borr",
        "c_pay_acq_const_fiolta",
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
    (
        "NetPayoutYield",
        "accounting",
        "trailing dividends minus equity issuance over market value six months ago",
        6 * MONTH,
    ),
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
    ("DivYieldST", "event", "binned trailing dividend yield when a payment is due within a month, payers only", YEAR),
)
OAP_NAMES = tuple(name for name, _, _, _ in OAP)

# OAP predictors left out, with the reason. Together with ``OAP_NAMES`` this is the full 212-signal predictor list.
_ANALYST = "needs IBES analyst forecasts or recommendations, which Tushare does not provide at this account level"
_HOLDINGS = "needs 13F institutional holdings; Chinese fund holdings are semi-annual and not synced"
_OPTIONS = "needs single-stock option prices; A shares have no listed single-stock options"
OAP_SKIPPED: dict[str, str] = {
    **{
        name: _ANALYST
        for name in (
            "AnalystRevision",
            "ChangeInRecommendation",
            "ChForecastAccrual",
            "ChNAnalyst",
            "CredRatDG",
            "EarningsForecastDisparity",
            "ExclExp",
            "FEPS",
            "fgr5yrLag",
            "ForecastDispersion",
            "Recomm_ShortInterest",
            "REV6",
            "sfe",
            "AnalystValue",
            "AOP",
            "ConsRecomm",
            "DownRecomm",
            "UpRecomm",
            "EarningsStreak",
            "PredictedFE",
        )
    },
    **{
        name: _HOLDINGS
        for name in (
            "Activism1",
            "Activism2",
            "DelBreadth",
            "RIO_Disp",
            "RIO_MB",
            "RIO_Turnover",
            "RIO_Volatility",
            "IO_ShortInterest",
        )
    },
    **{
        name: _OPTIONS
        for name in (
            "CPVolSpread",
            "dVolCall",
            "dVolPut",
            "dCPVolSpread",
            "OptionVolume1",
            "OptionVolume2",
            "RIVolSpread",
            "SmileSlope",
            "skew1",
        )
    },
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


class OAPLibrary(FactorLibrary):
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
                "compiled_prelude": self.cpp_prelude,
                "shared_operators": inspect.getsource(op.k),
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
        return self.align_reference_inputs(inputs, dates, instruments), hashes

    strict_float = True
    compile_split_source = 64
    cpp_prelude = CNC_PRELUDE + op.CPP_PRELUDE + regression_op.CPP_PRELUDE
    statement_fields = (*sorted({line for lines in ITEMS.values() for line in lines}), "oap_year")

    @property
    def statement_outputs(self):
        return tuple(_definitions()[1])

    @property
    def reference_inputs(self):
        return (*DAILY_REFERENCE, *self.statement_outputs)

    def prepare_statements(self, fundamentals):
        frame = prepare_fundamentals(fundamentals)
        frame["oap_year"] = pd.to_datetime(frame["end_date"]).dt.year
        if "rd_exp" not in frame:
            frame["rd_exp"] = np.nan
        return frame

    def align_reference_inputs(self, inputs, dates, instruments):
        dates, instruments = pd.DatetimeIndex(dates), pd.Index(instruments)
        result = dict(inputs)
        grid = QuarterGrid(self.prepare_statements(inputs["fundamentals"]), dates, instruments)
        numbers = np.arange(grid.quarters) + grid.first_quarter
        ends = pd.to_datetime([f"{n // 4}-{(n % 4) * 3 + 3:02d}-01" for n in numbers]) + pd.offsets.MonthEnd(0)
        rows = np.searchsorted(dates.to_numpy(), ends.to_numpy(), side="right") - 1
        result["quarter_end_row"] = grid.gather(np.broadcast_to(rows[:, None], (grid.quarters, len(instruments))))
        december = pd.to_datetime([f"{year - 1}-12-31" for year in dates.year])
        result["december_row"] = pd.Series(
            np.searchsorted(dates.to_numpy(), december.to_numpy(), side="right") - 1, index=dates
        )
        epoch = np.datetime64("1970-01-01", "D")
        result["calendar_day"] = pd.Series(
            (dates.to_numpy().astype("datetime64[D]") - epoch).astype(float), index=dates
        )
        listed = pd.to_datetime(pd.Series(inputs["list_date"]).reindex(instruments)).to_numpy().astype("datetime64[D]")
        days = np.where(np.isnat(listed), np.nan, (listed - epoch).astype(float))
        result["listed_day"] = np.broadcast_to(days, (len(dates), len(instruments))).copy()
        frame = inputs["fundamentals"]
        rows = np.searchsorted(dates.to_numpy(), pd.to_datetime(frame["ann_date"]).to_numpy()) + 2
        cols = instruments.get_indexer(frame["instrument"])
        keep = (cols >= 0) & (rows < len(dates))
        announced = np.full((len(dates), len(instruments)), np.nan)
        announced[rows[keep], cols[keep]] = 1
        result["announced"] = announced
        return result

    def build_statements(self, inputs):
        memo = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[1].items()}

    def build(self, inputs):
        memo = {}
        return {name: op.lower(expr, inputs, memo) for name, expr in _definitions()[0].items()}


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


DAILY_REFERENCE = (
    *("basic_" + name for name in OAP_DAILY_BASIC),
    *MARKET_FACTORS,
    "universe",
    "calendar_day",
    "listed_day",
    "announced",
    "quarter_end_row",
    "december_row",
)
lag = shift_rows = op.shift
trailing = op.trailing


def rolling_quarter_std(value, window, minimum):
    return trailing(value, window, minimum, "std")


quarter_window = op.mean_lags


def positive(value):
    return g.where(value > 0, value, g.nan)


def safe_log(value):
    return g.log(positive(value))


def finite_or(value, fill):
    return g.where(g.isfinite(value), value, fill)


def flag(condition, known):
    return g.where(known, condition, g.nan)


def growth(current, previous):
    return ratio(current - previous, positive(previous))


def corwin_schultz(high, low):
    today = g.log(high / low) ** 2
    beta = today + shift_rows(today, 1)
    gamma = g.log(g.fmax(high, shift_rows(high, 1)) / g.fmin(low, shift_rows(low, 1))) ** 2
    denominator = 3 - 2 * 2**0.5
    alpha = (g.sqrt(2 * beta) - g.sqrt(beta)) / denominator - g.sqrt(gamma / denominator)
    spread = 2 * (g.exp(alpha) - 1) / (1 + g.exp(alpha))
    return g.where(g.isfinite(spread), g.maximum(spread, 0), g.nan)


class GraphQuarterGrid:
    def value(self, name):
        return op.input_(name)

    @property
    def quarter_index(self):
        return self.value("fiscal_quarter")

    def single(self, name):
        value = self.value(name)
        return g.where(self.quarter_index == 0, value, value - lag(value, 1))

    def ttm(self, name):
        value = self.single(name)
        return value + lag(value, 1) + lag(value, 2) + lag(value, 3)

    def annual(self, name):
        return g.where(self.quarter_index == 3, self.value(name), g.nan)


class OAPSignals:
    """Build statement and daily expressions; this class never evaluates pandas/NumPy formulas."""

    def __init__(self):
        self._cache = {}
        self.grid = GraphQuarterGrid()
        self.statements = {}
        self.close = op.input_("close")
        self.trading = g.isfinite(self.close)
        self.price = op.ffill(self.close)
        self.returns = self.close / shift_rows(self.close, 1) - 1
        self.cap = positive(op.input_("cap"))
        self.universe = op.input_("universe") == 1
        self.sector = op.input_("sector")

    def _array(self, key):
        return op.input_(key)

    def _series(self, key):
        return op.input_(key)

    def _ffill(self, value):
        return op.ffill(value)

    def p(self, value, annual=False):
        name = ("y_" if annual else "q_") + f"oap_{len(self.statements):03d}"
        self.statements[name] = value
        return op.input_(name)

    @property
    def age_months(self):
        days = self._array("calendar_day") - self._array("listed_day")
        return g.where(days >= 0, days / 30.4375, g.nan)

    def _lag_month_mean(self, lags, minimum):
        return op.mean_lags(self.monthly_returns, [MONTH * (j - 1) for j in lags], minimum)

    def _factor_month(self, name):
        return trailing(self._series(name), MONTH, 15, "sum")

    def _regression(self, key, regressors, window, minimum):
        def build():
            xs = [self._series(name) for name in regressors]
            return {
                key: op.regression(self.returns, xs, window, minimum, kind)
                for key, kind in (("resid_std", "kResidualStd"), ("resid_skew", "kResidualSkew"))
            }

        return self._memo(key, build)

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def cap_filled(self) -> g.ndarray:
        return self._memo("cap_filled", lambda: self._ffill(self.cap))

    @property
    def turnover(self) -> g.ndarray:
        return self._memo("turnover", lambda: self._array("basic_turnover_rate"))

    @property
    def volume(self) -> g.ndarray:
        return self._memo("volume", lambda: self._array("volume"))

    @property
    def amount(self) -> g.ndarray:
        return self._memo("amount", lambda: positive(self._array("amount")))

    @property
    def raw_close(self) -> g.ndarray:
        return self._memo("raw_close", lambda: positive(self._array("basic_close")))

    @property
    def adjusted_shares(self) -> g.ndarray:
        """Total shares times unadjusted over adjusted close, which removes share splits and bonus issues."""

        def build() -> g.ndarray:
            shares = positive(self._array("basic_total_share"))
            with g.errstate(invalid="ignore", divide="ignore"):
                return self._ffill(shares * self.raw_close / positive(self.close))

        return self._memo("adjusted_shares", build)

    @property
    def dividends(self) -> g.ndarray:
        """Trailing-twelve-month cash dividends in yuan, zero for a quoted non-payer, carried over gaps.

        Tushare counts a payment twice for a few days around its ex-date (up to about two weeks), so the amount is the
        smallest of the last ``DIVIDEND_SPIKE_DAYS`` rows; a real change shows up that many rows later.
        """

        def build() -> g.ndarray:
            quoted = g.isfinite(self.raw_close)
            dividend_yield = finite_or(self._array("basic_dv_ttm"), 0.0)
            amount = self._ffill(g.where(quoted, dividend_yield * self.cap_filled, g.nan))
            return trailing(amount, DIVIDEND_SPIKE_DAYS, 1, "min")

        return self._memo("dividends", build)

    def momentum(self, start: int, end: int) -> g.ndarray:
        """Return from ``start`` trading days ago to ``end`` trading days ago (start > end)."""
        return shift_rows(self.price, end) / shift_rows(self.price, start) - 1.0

    @property
    def monthly_returns(self) -> g.ndarray:
        return self._memo("monthly_returns", lambda: self.momentum(MONTH, 0))

    def b(self, item: str) -> g.ndarray:
        """Balance-sheet item on the quarter grid."""
        return self._memo(f"b:{item}", lambda: sum(self.grid.value(line) for line in ITEMS[item]))

    def _flow(self, item: str, kind: str) -> g.ndarray:
        def build() -> g.ndarray:
            parts = []
            for line in ITEMS[item]:
                values = getattr(self.grid, kind)(line)
                if line in OPTIONAL_LINES and item != "xrd":
                    values = finite_or(values, 0.0)
                parts.append(values)
            return sum(parts)

        return self._memo(f"{kind}:{item}", build)

    def x(self, item: str) -> g.ndarray:
        """Trailing-twelve-month flow on the quarter grid."""
        return self._flow(item, "ttm")

    def qtr(self, item: str) -> g.ndarray:
        """Single-quarter flow on the quarter grid."""
        return self._flow(item, "single")

    def be(self) -> g.ndarray:
        """Common book equity: equity to parent less other equity instruments (preferred and perpetual)."""
        return self.b("seq") - self.b("pstk")

    def avg_at(self) -> g.ndarray:
        return (self.b("at") + lag(self.b("at"), 4)) / 2.0

    def change_over_avg_at(self, values: g.ndarray) -> g.ndarray:
        return self.p(ratio(values - lag(values, 4), self.avg_at()))

    def to_me(self, values: g.ndarray) -> g.ndarray:
        return self.p(values) / self.cap

    def debt(self) -> g.ndarray:
        return self.b("dltt") + self.b("dlc")

    def noa(self) -> g.ndarray:
        """Operating assets (at - che) less operating liabilities (at - debt - mib - be - pstk)."""
        return self.debt() + self.b("mib") + self.be() + self.b("pstk") - self.b("che")

    def ebitda(self) -> g.ndarray:
        return self.x("pi") + self.x("xint") + self.x("dp")

    def surprise(self, values: g.ndarray) -> g.ndarray:
        """Four-quarter change minus its mean over the previous eight quarters, over their std."""
        change = values - lag(values, 4)
        drift = quarter_window(change, tuple(range(1, 9)), 6)
        spread = rolling_quarter_std(lag(change, 1), 8, 6)
        return ratio(change - drift, spread)

    def _members(self, values: g.ndarray) -> g.ndarray:
        return g.where(self.universe, values, g.nan)

    def _s_Size(self) -> g.ndarray:
        return safe_log(self.cap)

    def _s_Price(self) -> g.ndarray:
        return safe_log(self.raw_close)

    def _s_STreversal(self) -> g.ndarray:
        return self.momentum(MONTH, 0)

    def _s_Mom6m(self) -> g.ndarray:
        return self._memo("mom6", lambda: self.momentum(6 * MONTH, MONTH))

    def _s_Mom12m(self) -> g.ndarray:
        return self.momentum(YEAR, MONTH)

    def _s_IntMom(self) -> g.ndarray:
        return self.momentum(YEAR, 6 * MONTH)

    def _s_MRreversal(self) -> g.ndarray:
        return self.momentum(18 * MONTH, 13 * MONTH - MONTH)

    def _s_LRreversal(self) -> g.ndarray:
        return self._memo("mom36", lambda: self.momentum(36 * MONTH, 13 * MONTH - MONTH))

    def _s_Mom12mOffSeason(self) -> g.ndarray:
        return self._lag_month_mean(tuple(range(1, 12)), 6)

    def _s_MomSeasonShort(self) -> g.ndarray:
        return self._lag_month_mean((12,), 1)

    def _s_MomSeason(self) -> g.ndarray:
        return self._lag_month_mean((24, 36, 48, 60), 2)

    def _s_MomOffSeason(self) -> g.ndarray:
        return self._lag_month_mean(tuple(j for j in range(13, 61) if j % 12), 22)

    def _s_MomSeason06YrPlus(self) -> g.ndarray:
        return self._lag_month_mean((72, 84, 96, 108, 120), 2)

    def _s_MomOffSeason06YrPlus(self) -> g.ndarray:
        return self._lag_month_mean(tuple(j for j in range(61, 121) if j % 12), 27)

    def _s_MaxRet(self) -> g.ndarray:
        return trailing(self.returns, MONTH, 15, "max")

    def _s_ReturnSkew(self) -> g.ndarray:
        return trailing(self.returns, MONTH, 15, "skew")

    def _ff3_month(self) -> dict:
        return self._regression("ff3_month", MARKET_FACTORS, MONTH, 15)

    def _s_ReturnSkew3F(self) -> g.ndarray:
        return self._ff3_month()["resid_skew"]

    def _s_IdioVol3F(self) -> g.ndarray:
        return self._ff3_month()["resid_std"]

    def _s_RealizedVol(self) -> g.ndarray:
        return trailing(self.returns, MONTH, 15, "std")

    def _s_IdioVolAHT(self) -> g.ndarray:
        return self._regression("capm_year", ("mkt",), YEAR, 100)["resid_std"]

    def _s_PriceDelayRsq(self) -> g.ndarray:
        return self._price_delay()["rsq"]

    def _s_PriceDelaySlope(self) -> g.ndarray:
        return self._price_delay()["slope"]

    def _s_PriceDelayTstat(self) -> g.ndarray:
        return self._price_delay()["tstat"]

    def _s_High52(self) -> g.ndarray:
        return ratio(self.close, trailing(self.close, YEAR, 126, "max"))

    def _s_IndMom(self) -> g.ndarray:
        returns = self.momentum(6 * MONTH, 0)
        return group_stat(returns, self.sector, self.universe, "mean", weights=self.cap)

    def _big_in_industry(self) -> g.ndarray:
        return self._memo("big", lambda: group_rank(self.cap, self.sector, self.universe) > 0.7)

    def _for_small_firms(self, values: g.ndarray) -> g.ndarray:
        big = self._big_in_industry()
        industry = group_stat(values, self.sector, self.universe & big, "mean")
        return g.where(self.universe & ~big, industry, g.nan)

    def _s_IndRetBig(self) -> g.ndarray:
        return self._for_small_firms(self.momentum(MONTH, 0))

    def _s_FirmAgeMom(self) -> g.ndarray:
        age = g.where(self.age_months >= 12, self.age_months, g.nan)
        young = cross_bucket(age, self.universe, 5) == 1
        return g.where(young, self._s_Mom6m(), g.nan)

    def _s_MomRev(self) -> g.ndarray:
        short = cross_bucket(self._s_Mom6m(), self.universe, 5)
        long = cross_bucket(self._s_LRreversal(), self.universe, 5)
        result = g.where((short == 5) & (long == 1), 1.0, g.nan)
        return g.where((short == 1) & (long == 5), 0.0, result)

    def _s_MomVol(self) -> g.ndarray:
        volume = trailing(self.turnover, 6 * MONTH, 5 * MONTH, "mean")
        seasoned = self.age_months >= 24
        high = cross_bucket(g.where(seasoned, volume, g.nan), self.universe, 3) == 3
        decile = cross_bucket(g.where(seasoned, self._s_Mom6m(), g.nan), self.universe, 10)
        return g.where(high, decile, g.nan)

    def _s_DolVol(self) -> g.ndarray:
        return safe_log(shift_rows(trailing(self.amount, MONTH, 15, "sum"), MONTH))

    def _s_Illiquidity(self) -> g.ndarray:
        return trailing(ratio(g.abs(self.returns), self.amount), YEAR, 120, "mean")

    def _s_ShareVol(self) -> g.ndarray:
        bucket = cross_bucket(trailing(self.turnover, QUARTER, 40, "mean"), self.universe, 3)
        return g.where(bucket == 3, 1.0, g.where(bucket == 1, 0.0, g.nan))

    def _s_std_turn(self) -> g.ndarray:
        return self._monthly_std(self.turnover, 36, 24)

    def _s_VolSD(self) -> g.ndarray:
        return self._monthly_std(self.volume, 36, 24)

    def _s_VolMkt(self) -> g.ndarray:
        return trailing(self.amount, YEAR, 120, "mean") * MONTH / self.cap

    def _s_zerotrade1M(self) -> g.ndarray:
        return self._zerotrade(1)

    def _s_zerotrade6M(self) -> g.ndarray:
        return self._zerotrade(6)

    def _s_zerotrade12M(self) -> g.ndarray:
        return self._zerotrade(12)

    def _s_BidAskSpread(self) -> g.ndarray:
        spread = corwin_schultz(self._array("high"), self._array("low"))
        return trailing(spread, MONTH, 15, "mean")

    def _s_AM(self) -> g.ndarray:
        return self.to_me(self.b("at"))

    def _s_BookLeverage(self) -> g.ndarray:
        return self.p(ratio(self.b("at"), positive(self.b("seq") + self.b("txditc"))))

    def _s_Leverage(self) -> g.ndarray:
        return self.to_me(self.b("lt"))

    def _s_EP(self) -> g.ndarray:
        return positive(self.to_me(self.x("ib")))

    def _s_CF(self) -> g.ndarray:
        return self.to_me(self.x("ib") + self.x("dp"))

    def _s_cfp(self) -> g.ndarray:
        return self.to_me(self.x("oancf"))

    def _s_SP(self) -> g.ndarray:
        return self.to_me(self.x("sale"))

    def _net_debt(self) -> g.ndarray:
        return self.p(self.debt() + self.b("pstk") - self.b("che"))

    def _s_EBM(self) -> g.ndarray:
        net_debt = self._net_debt()
        return ratio(self.p(self.be()) - net_debt, self.cap - net_debt)

    def _s_BPEBM(self) -> g.ndarray:
        return ratio(self.p(self.be()), self.cap) - self._s_EBM()

    def _s_NetDebtPrice(self) -> g.ndarray:
        return self._net_debt() / self.cap

    def _s_EntMult(self) -> g.ndarray:
        enterprise = self.cap + self.p(self.debt() - self.b("che"))
        known_book = g.isfinite(self.p(self.be()))
        return g.where(known_book, ratio(enterprise, positive(self.p(self.ebitda()))), g.nan)

    def _s_CashProd(self) -> g.ndarray:
        return ratio(self.cap - self.p(self.b("at")), self.p(self.b("che")))

    def _s_PayoutYield(self) -> g.ndarray:
        return positive(ratio(self.dividends, shift_rows(self.cap_filled, 6 * MONTH)))

    def _s_NetPayoutYield(self) -> g.ndarray:
        payout = self.dividends - self.p(self.x("sstk"))
        result = ratio(payout, shift_rows(self.cap_filled, 6 * MONTH))
        return g.where(result != 0, result, g.nan)

    def _s_EquityDuration(self) -> g.ndarray:
        discount, horizon = 0.12, 10
        book = self.be()
        roe = ratio(self.x("ib"), positive(lag(book, 4)))
        sales_growth = growth(self.x("sale"), lag(self.x("sale"), 4))
        weighted = g.zeros(book.shape)
        present = g.zeros(book.shape)
        for year in range(1, horizon + 1):
            roe = 0.12 + 0.57 * (roe - 0.12)
            sales_growth = 0.06 + 0.24 * (sales_growth - 0.06)
            distribution = book * (roe - sales_growth)
            book = book * (1.0 + sales_growth)
            factor = (1.0 + discount) ** year
            weighted = weighted + year * distribution / factor
            present = present + distribution / factor
        known_book = positive(self.be())
        weighted = g.where(g.isfinite(known_book), weighted, g.nan)
        present = g.where(g.isfinite(known_book), present, g.nan)
        me = self.cap
        return self.p(weighted) / me + (horizon + (1.0 + discount) / discount) * (me - self.p(present)) / me

    def _intangible_return(self, numerator: g.ndarray) -> g.ndarray:
        log_ratio = safe_log(self.to_me(numerator))
        five = 5 * YEAR
        stock_return = safe_log(ratio(self.price, shift_rows(self.price, five)))
        lagged = shift_rows(log_ratio, five)
        accounting_return = log_ratio - lagged + stock_return
        return cross_residual(stock_return, [lagged, accounting_return], self.universe)

    def _s_IntanBM(self) -> g.ndarray:
        return self._intangible_return(self.be())

    def _s_IntanCFP(self) -> g.ndarray:
        return self._intangible_return(self.x("ib") + self.x("dp"))

    def _s_IntanEP(self) -> g.ndarray:
        return self._intangible_return(self.x("ib"))

    def _s_IntanSP(self) -> g.ndarray:
        return self._intangible_return(self.x("sale"))

    def _s_Frontier(self) -> g.ndarray:
        book = positive(self.be())
        assets = self.b("at")
        sales = self.x("sale")
        regressors = [
            self.p(safe_log(book)),
            self.p(ratio(self.b("dltt"), assets)),
            self.p(ratio(self.x("capx"), sales)),
            self.p(finite_or(ratio(self.x("xrd"), sales), 0.0) * g.isfinite(sales)),
            self.p(ratio(self.b("ppe"), assets)),
            self.p(ratio(self.ebitda(), assets)),
        ]
        target = safe_log(self.p(book) / self.cap)
        return cross_residual(target, regressors, self.universe, groups=self.sector)

    def _s_Cash(self) -> g.ndarray:
        return self.p(ratio(self.b("che"), self.b("at")))

    def _s_GP(self) -> g.ndarray:
        return self.p(ratio(self.x("sale") - self.x("cogs"), self.b("at")))

    def _s_OperProf(self) -> g.ndarray:
        profit = self.x("sale") - self.x("cogs") - self.x("xsga") - self.x("xint")
        return self.p(ratio(profit, positive(self.be())))

    def _operating_profit_rd(self) -> g.ndarray:
        return self.x("sale") - self.x("cogs") - (self.x("xsga") - finite_or(self.x("xrd"), 0.0))

    def _s_OperProfRD(self) -> g.ndarray:
        return self.p(ratio(self._operating_profit_rd(), self.b("at")))

    def _s_CBOperProf(self) -> g.ndarray:
        def change(item: str) -> g.ndarray:
            return self.b(item) - lag(self.b(item), 4)

        cash_profit = self._operating_profit_rd() - change("rect") - change("invt") - change("prepaid") + change("ap")
        return self.p(ratio(cash_profit, self.b("at")))

    def _s_RoE(self) -> g.ndarray:
        return self.p(ratio(self.x("ni"), positive(self.be())))

    def _s_roaq(self) -> g.ndarray:
        return self.p(ratio(self.qtr("ib"), lag(self.b("at"), 1)))

    def _s_OPLeverage(self) -> g.ndarray:
        return self.p(ratio(self.x("xsga") + self.x("cogs"), self.b("at")))

    def _s_tang(self) -> g.ndarray:
        tangible = self.b("che") + 0.715 * self.b("rect") + 0.547 * self.b("invt") + 0.535 * self.b("ppe")
        return self.p(ratio(tangible, self.b("at")))

    def _s_PS(self) -> g.ndarray:
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
        known = g.all([g.isfinite(value) for _, value in parts], axis=0)
        score = self.p(g.where(known, g.sum([signal.astype(float) for signal, _ in parts], axis=0), g.nan))
        shares = self.adjusted_shares
        previous = shift_rows(shares, YEAR)
        no_issue = flag(shares <= previous * 1.0001, g.isfinite(shares) & g.isfinite(previous))
        return score + no_issue

    def _s_MS(self) -> g.ndarray:
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
            score = score + g.where(g.isfinite(values) & (values > median), 1.0, 0.0)
        return g.where(g.isfinite(components[0]) & g.isfinite(components[1]), score, g.nan)

    def _s_OScore(self) -> g.ndarray:
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
            - 1.72 * flag(liabilities > assets, g.isfinite(liabilities - assets))
            - 2.37 * ratio(income, assets)
            - 1.83 * ratio(self.x("oancf"), liabilities)
            + 0.285 * flag((income < 0) & (previous < 0), g.isfinite(income + previous))
            - 0.521 * ratio(income - previous, g.abs(income) + g.abs(previous))
        )
        return self.p(score)

    def _s_EarningsConsistency(self) -> g.ndarray:
        earnings = self.x("ib")
        base = (g.abs(lag(earnings, 4)) + g.abs(lag(earnings, 8))) / 2.0
        change = ratio(earnings - lag(earnings, 4), base)
        change = g.where(g.abs(change) <= 6.0, change, g.nan)
        change = g.where(g.sign(change) == g.sign(lag(change, 4)), change, g.nan)
        return self.p(quarter_window(change, (0, 4, 8, 12), 4))

    def _earnings_surprise(self) -> g.ndarray:
        return self._memo("earnings_surprise", lambda: self.p(self.surprise(self.qtr("ib"))))

    def _s_EarningsSurprise(self) -> g.ndarray:
        return self._earnings_surprise()

    def _s_RevenueSurprise(self) -> g.ndarray:
        return self.p(self.surprise(self.qtr("sale")))

    def _s_EarnSupBig(self) -> g.ndarray:
        return self._for_small_firms(self._earnings_surprise())

    def _growth_over_two_years(self, values: g.ndarray) -> g.ndarray:
        """Growth relative to the average of one and two years ago, or one year ago when two is missing."""
        previous = lag(values, 4)
        base = g.where(g.isfinite(lag(values, 8)), (previous + lag(values, 8)) / 2.0, previous)
        return growth(values, base)

    def _s_GrSaleToGrInv(self) -> g.ndarray:
        return self.p(self._growth_over_two_years(self.x("sale")) - self._growth_over_two_years(self.b("invt")))

    def _s_GrSaleToGrOverhead(self) -> g.ndarray:
        return self.p(self._growth_over_two_years(self.x("sale")) - self._growth_over_two_years(self.x("xsga")))

    def _s_ChAssetTurnover(self) -> g.ndarray:
        noa = self.noa()
        turnover = ratio(self.x("sale"), positive((noa + lag(noa, 4)) / 2.0))
        return self.p(turnover - lag(turnover, 4))

    def _s_AssetGrowth(self) -> g.ndarray:
        assets = self.b("at")
        return self.p(growth(assets, lag(assets, 4)))

    def _s_ChEQ(self) -> g.ndarray:
        book = positive(self.be())
        return self.p(ratio(book, lag(book, 4)))

    def _s_ChInv(self) -> g.ndarray:
        return self.change_over_avg_at(self.b("invt"))

    def _s_InvGrowth(self) -> g.ndarray:
        inventory = self.b("invt")
        return self.p(growth(inventory, lag(inventory, 4)))

    def _s_InvestPPEInv(self) -> g.ndarray:
        values = self.b("ppe") + self.b("invt")
        return self.p(ratio(values - lag(values, 4), lag(self.b("at"), 4)))

    def _s_dNoa(self) -> g.ndarray:
        noa = self.noa()
        return self.p(ratio(noa - lag(noa, 4), lag(self.b("at"), 4)))

    def _s_NOA(self) -> g.ndarray:
        return self.p(ratio(self.noa(), lag(self.b("at"), 4)))

    def _s_ChNWC(self) -> g.ndarray:
        working = ratio((self.b("act") - self.b("che")) - (self.b("lct") - self.b("dlc")), self.b("at"))
        return self.p(working - lag(working, 4))

    def _s_ChNNCOA(self) -> g.ndarray:
        assets = self.b("at") - self.b("act") - self.b("ivao")
        liabilities = self.b("lt") - self.b("lct") - self.b("dltt")
        non_current = ratio(assets - liabilities, self.b("at"))
        return self.p(non_current - lag(non_current, 4))

    def _s_DelCOA(self) -> g.ndarray:
        return self.change_over_avg_at(self.b("act") - self.b("che"))

    def _s_DelCOL(self) -> g.ndarray:
        return self.change_over_avg_at(self.b("lct") - self.b("dlc"))

    def _s_DelEqu(self) -> g.ndarray:
        return self.change_over_avg_at(self.be())

    def _s_DelFINL(self) -> g.ndarray:
        return self.change_over_avg_at(self.debt() + self.b("pstk"))

    def _s_DelLTI(self) -> g.ndarray:
        return self.change_over_avg_at(self.b("ivao"))

    def _s_DelNetFin(self) -> g.ndarray:
        return self.change_over_avg_at(self.b("ivst") + self.b("ivao") - self.debt() - self.b("pstk"))

    def _s_GrLTNOA(self) -> g.ndarray:
        assets = self.b("at")
        other_assets = self.b("nca") - self.b("ppe") - self.b("intan") - self.b("ivao")
        current_other = self.b("lct") - self.b("dlc") - self.b("ap")
        long_other = self.b("ncl") - self.b("dltt")
        operating = (
            self.b("rect")
            + self.b("invt")
            + self.b("ppe")
            + self.b("aco")
            + self.b("intan")
            + other_assets
            - self.b("ap")
            - current_other
            - long_other
        )
        noa = ratio(operating, assets)

        def change(values: g.ndarray) -> g.ndarray:
            return values - lag(values, 4)

        accruals = ratio(
            change(self.b("rect"))
            + change(self.b("invt"))
            + change(self.b("aco"))
            - (change(self.b("ap")) + change(current_other))
            - self.x("dp"),
            self.avg_at(),
        )
        return self.p(noa - lag(noa, 4) - accruals)

    def _accruals(self) -> g.ndarray:
        def change(item: str) -> g.ndarray:
            return self.b(item) - lag(self.b(item), 4)

        working = change("act") - change("che") - (change("lct") - change("dlc") - change("txp"))
        return ratio(working, self.avg_at())

    def _s_Accruals(self) -> g.ndarray:
        return self.p(self._accruals())

    def _s_AbnormalAccruals(self) -> g.ndarray:
        assets = self.b("at")
        lagged = lag(assets, 4)
        accruals = self.p(ratio(self.x("ib") - self.x("oancf"), self.avg_at()))
        regressors = [
            self.p(ratio(1.0, lagged)) * 1e9,
            self.p(ratio(self.x("sale") - lag(self.x("sale"), 4), lagged)),
            self.p(ratio(self.b("ppe"), lagged)),
        ]
        return cross_residual(accruals, regressors, self.universe)

    def _s_AccrualsBM(self) -> g.ndarray:
        book_to_market = ratio(self.p(positive(self.be())), self.cap)
        accruals = g.where(g.isfinite(book_to_market), self.p(self._accruals()), g.nan)
        accrual_bucket = cross_bucket(accruals, self.universe, 5)
        value_bucket = cross_bucket(book_to_market, self.universe, 5)
        result = g.where((accrual_bucket == 5) & (value_bucket == 1), 1.0, g.nan)
        return g.where((accrual_bucket == 1) & (value_bucket == 5), 0.0, result)

    def _s_PctAcc(self) -> g.ndarray:
        income = self.x("ib")
        return self.p(ratio(income - self.x("oancf"), g.abs(income)))

    def _total_accruals(self) -> g.ndarray:
        """Net income less operating, investing and financing cash flows, plus equity issuance, less dividends."""
        flows = self.x("ni") - self.x("oancf") - self.x("ivncf") - self.x("fincf") + self.x("sstk")
        return self.p(flows) - self.dividends

    def _s_PctTotAcc(self) -> g.ndarray:
        return ratio(self._total_accruals(), self.p(g.abs(self.x("ni"))))

    def _s_TotalAccruals(self) -> g.ndarray:
        return ratio(self._total_accruals(), self.p(lag(self.b("at"), 4)))

    def _s_ChTax(self) -> g.ndarray:
        tax = self.qtr("txt")
        return self.p(ratio(tax - lag(tax, 4), lag(self.b("at"), 4)))

    def _s_Investment(self) -> g.ndarray:
        intensity = ratio(self.x("capx"), positive(self.x("sale")))
        return self.p(ratio(intensity, quarter_window(intensity, (4, 8, 12), 3)))

    def _s_ChInvIA(self) -> g.ndarray:
        capex = self.x("capx")
        previous = lag(capex, 4)
        base = g.where(g.isfinite(lag(capex, 8)), (previous + lag(capex, 8)) / 2.0, previous)
        capex_growth = self.p(growth(capex, base))
        return capex_growth - group_stat(capex_growth, self.sector, self.universe, "mean")

    def _s_grcapx(self) -> g.ndarray:
        capex = self.x("capx")
        return self.p(growth(capex, lag(capex, 8)))

    def _s_grcapx3y(self) -> g.ndarray:
        capex = self.x("capx")
        return self.p(ratio(capex, positive(quarter_window(capex, (4, 8, 12), 3))))

    def _s_CompositeDebtIssuance(self) -> g.ndarray:
        debt = self.debt()
        return self.p(safe_log(debt) - safe_log(lag(debt, 20)))

    def _s_CompEquIss(self) -> g.ndarray:
        five = 5 * YEAR
        value_growth = safe_log(ratio(self.cap_filled, shift_rows(self.cap_filled, five)))
        price_growth = safe_log(ratio(self.price, shift_rows(self.price, five)))
        return value_growth - price_growth

    def _s_ShareIss1Y(self) -> g.ndarray:
        shares = self.adjusted_shares
        return ratio(shift_rows(shares, 6 * MONTH), shift_rows(shares, 18 * MONTH)) - 1.0

    def _s_ShareIss5Y(self) -> g.ndarray:
        shares = self.adjusted_shares
        return ratio(shares, shift_rows(shares, 5 * YEAR)) - 1.0

    def _s_NetEquityFinance(self) -> g.ndarray:
        return self.p(ratio(self.x("sstk"), self.avg_at()))

    def _net_debt_issued(self) -> g.ndarray:
        return self.x("dltis") - self.x("dltr")

    def _s_NetDebtFinance(self) -> g.ndarray:
        return self.p(ratio(self._net_debt_issued(), self.avg_at()))

    def _s_XFIN(self) -> g.ndarray:
        financing = self.p(self.x("sstk") + self._net_debt_issued()) - self.dividends
        return ratio(financing, self.p(self.b("at")))

    def _s_DebtIssuance(self) -> g.ndarray:
        issued = self.x("dltis")
        return self.p(flag(issued > 0, g.isfinite(issued)))

    def _s_RD(self) -> g.ndarray:
        return self.to_me(self.x("xrd"))

    def _s_RDcap(self) -> g.ndarray:
        research = self.x("xrd")
        capital = research.copy()
        for years, weight in ((1, 0.8), (2, 0.6), (3, 0.4), (4, 0.2)):
            capital = capital + weight * finite_or(lag(research, 4 * years), 0.0)
        return self.p(ratio(capital, self.b("at")))

    def _s_SurpriseRD(self) -> g.ndarray:
        research, assets = self.x("xrd"), self.b("at")
        intensity = ratio(research, assets)
        surprise = (
            (ratio(research, self.x("sale")) > 0)
            & (intensity > 0)
            & (growth(research, lag(research, 4)) > 0.05)
            & (growth(intensity, lag(intensity, 4)) > 0.05)
        )
        return self.p(flag(surprise, g.isfinite(research)))

    def _s_Herf(self) -> g.ndarray:
        return self._herfindahl(self.x("sale"))

    def _s_HerfBE(self) -> g.ndarray:
        return self._herfindahl(self.be())

    def _s_HerfAsset(self) -> g.ndarray:
        return self._herfindahl(self.b("at"))

    def _s_FirmAge(self) -> g.ndarray:
        return self.age_months

    def _s_IndIPO(self) -> g.ndarray:
        age = self.age_months
        return flag((age >= 6) & (age <= 36), g.isfinite(age))

    def _dividend_changes(self) -> g.ndarray:
        """Days the trailing dividend amount changes by more than 5%: a payment enters or a year-old one drops out."""

        def build() -> g.ndarray:
            dividends = self.dividends
            previous = shift_rows(dividends, 1)
            larger = g.fmax(dividends, previous)
            known = g.isfinite(dividends) & g.isfinite(previous) & (larger > 0)
            return known & (g.abs(dividends - previous) > 0.05 * larger)

        return self._memo("dividend_changes", build)

    def _s_DivInit(self) -> g.ndarray:
        dividends = self.dividends
        half, year = shift_rows(dividends, 6 * MONTH), shift_rows(dividends, YEAR)
        return flag((dividends > 0) & (half == 0) & (year == 0), g.isfinite(dividends) & g.isfinite(year))

    def _s_DivOmit(self) -> g.ndarray:
        dividends = self.dividends
        recent, year, two = (shift_rows(dividends, lag_days) for lag_days in (2 * MONTH, YEAR, 2 * YEAR))
        omitted = (dividends == 0) & (recent > 0) & (year > 0) & (two > 0)
        return flag(omitted, g.isfinite(dividends) & g.isfinite(two))

    def _seasonal_payer(self) -> g.ndarray:
        """1 where a dividend changed 11 to 12 months ago, so the annual payment is due within a month."""
        changes = trailing(self._dividend_changes().astype(float), MONTH, 1, "max")
        return shift_rows(changes, YEAR - MONTH) > 0

    def _s_DivSeason(self) -> g.ndarray:
        return g.where(self.dividends > 0, self._seasonal_payer().astype(float), g.nan)

    def _s_DivYieldST(self) -> g.ndarray:
        expected = g.where(self._seasonal_payer(), ratio(self.dividends, self.cap), 0.0)
        bins = g.where(expected <= 0, 0.0, g.where(expected <= 0.005, 1.0, g.where(expected <= 0.01, 2.0, 3.0)))
        return g.where((self.dividends > 0) & g.isfinite(self.cap), bins, g.nan)

    def _s_Beta(self):
        return op.regression(self.monthly_returns, [self._factor_month("mkt")], 60, 20, "kBeta", MONTH)

    def _s_BetaFP(self):
        stock = g.log1p(self.returns)
        market = g.log1p(self._series("mkt"))
        s3 = trailing(stock, 3, 3, "sum")
        m3 = trailing(market, 3, 3, "sum")
        corr = op.correlation(s3, g.where(g.isfinite(s3), m3, g.nan), 5 * YEAR, 3 * YEAR)
        return g.clip(corr, -1, 1) * ratio(
            trailing(stock, YEAR, 120, "std"), trailing(g.where(g.isfinite(stock), market, g.nan), YEAR, 120, "std")
        )

    def tail_index(self):
        values = [shift_rows(self._members(self.returns), j) for j in range(MONTH)]
        pushes = "".join(f"if(std::isfinite(input_{j}[i])) pool.push_back(input_{j}[i]);" for j in range(MONTH))
        tail = op.cross(
            f"""
        std::vector<double> pool;for(size_t i=0;i<num_stocks;++i) {{{pushes}}}
        double tail=NAN;
        if(pool.size()>=100) {{
            std::sort(pool.begin(),pool.end());double position=(pool.size()-1)*0.05;size_t lo=(size_t)position;
            double threshold=pool[lo]+(position-lo)*(pool[std::min(lo+1,pool.size()-1)]-pool[lo]);
            double total=0,count=0;
            if(threshold<0) for(double z:pool) if(z<threshold) {{total+=std::log(z/threshold);++count;}}
            if(count>0) tail=total/count;
        }}
        for(size_t i=0;i<num_stocks;++i) output_0[i]=tail;
        """,
            *values,
        )
        row = op.state([self.close], "y=s.row++;", "int row;", "s.row=0;")
        return g.where(row >= MONTH - 1, tail, g.nan)

    def _s_BetaTailRisk(self):
        return op.regression(self.monthly_returns, [shift_rows(self.tail_index(), MONTH)], 60, 36, "kBeta", MONTH)

    def _s_CoskewACX(self):
        return op.coskew(g.log1p(self.returns), g.log1p(self._series("mkt")), YEAR, 150)

    def _s_Coskewness(self):
        return op.coskew(g.log1p(self.monthly_returns), g.log1p(self._factor_month("mkt")), 60, 24, MONTH)

    def _price_delay(self):
        return self._memo(
            "price_delay",
            lambda: {
                name: op.price_delay(self.returns, self._series("mkt"), YEAR, 150, name)
                for name in ("rsq", "slope", "tstat")
            },
        )

    def _s_ResidualMomentum(self):
        return op.regression(
            self.monthly_returns,
            [self._factor_month(name) for name in MARKET_FACTORS],
            36,
            24,
            "kResidualMomentum",
            MONTH,
            24,
            35,
        )

    def _s_TrendFactor(self):
        ratios = [ratio(trailing(self.close, n, (4 * n + 4) // 5, "mean"), self.close) for n in TREND_WINDOWS]
        xs = [shift_rows(value, MONTH) for value in ratios]
        universe = shift_rows(self.universe, MONTH) > 0
        coefficients = [
            op.cross_fit(self.monthly_returns, xs, universe, minimum=36, coefficient=j + 1) for j in range(len(ratios))
        ]
        averaged = [op.mean_lags(value, [MONTH * j for j in range(12)], 8) for value in coefficients]
        return sum(value * coefficient for value, coefficient in zip(ratios, averaged, strict=False))

    def _monthly_std(self, values, window, minimum):
        return trailing(trailing(values, MONTH, 15, "sum"), window, minimum, "std", MONTH)

    def _s_VolumeTrend(self):
        volume = trailing(self.volume, MONTH, 15, "sum")
        trend = op.state([volume], "y=s.index++;", "int index;", "s.index=0;")
        # A residue class advances one observation each 21 trading days.
        slope = op.regression(volume, [trend / MONTH], 60, 30, "kBeta", MONTH)
        return ratio(slope, trailing(volume, 60, 30, "mean", MONTH))

    def _zerotrade(self, months):
        window = months * MONTH
        listed = op.state([self.trading], "s.known=s.known||x0;y=s.known;", "bool known;", "s.known=false;")
        zero = trailing(listed & ~self.trading, window, window, "sum")
        turnover = trailing(finite_or(self.turnover, 0), window, window, "sum")
        return zero + ratio(1, positive(turnover)) / 480000

    def _historical_cap(self, name):
        return op.state(
            [self.cap_filled, self._array(name)],
            "s.v.push_back(x0); y=NAN; if(std::isfinite(x1) && x1>=0 && x1<(double)s.v.size()) y=s.v[(size_t)x1];",
            "std::vector<double> v;",
        )

    def _s_BM(self):
        return safe_log(ratio(self.p(self.be() - self.b("intan")), self._historical_cap("quarter_end_row")))

    def _s_BMdec(self):
        return ratio(self.p(self.be()), self._historical_cap("december_row"))

    def _s_VarCF(self):
        return trailing(self._s_CF(), 60, 24, "var", MONTH)

    def _s_Tax(self):
        rate = g.where(self.grid.value("oap_year") < 2008, 0.33, 0.25)
        taxes, income = self.x("txt"), self.x("ib")
        return self.p(g.where((income < 0) & (taxes > 0), 1, ratio(taxes, rate * income)))

    def _s_NumEarnIncrease(self):
        earnings = self.qtr("ib")
        previous = lag(earnings, 4)
        streak = op.state(
            [earnings, previous],
            "bool known=std::isfinite(x0)&&std::isfinite(x1);"
            "s.count=known&&x0>x1?std::min(s.count+1,8):0;y=known?s.count:NAN;",
            "int count;",
            "s.count=0;",
        )
        return self.p(streak)

    def _s_MeanRankRevGrowth(self):
        sales = self.x("sale")
        ranks = cross_rank(self.p(growth(sales, lag(sales, 4))), self.universe)
        total = op.as_expr(0)
        weights = op.as_expr(0)
        for year, weight in enumerate((5, 4, 3, 2, 1)):
            value = shift_rows(ranks, YEAR * year)
            known = g.isfinite(value)
            total = total + g.where(known, weight * value, 0)
            weights = weights + weight * known
        return g.where(weights >= 12, total / weights, g.nan)

    def _s_OrgCap(self):
        expense = self.grid.annual("sell_exp") + self.grid.annual("admin_exp")
        capital = op.state(
            [expense, self.grid.quarter_index],
            "y=NAN;if(x1==3) {"
            "s.running=std::isfinite(s.running)?0.85*s.running+(std::isfinite(x0)?x0:0):4*x0;"
            "if(std::isfinite(x0)) y=s.running;}",
            "double running;",
            "s.running=NAN;",
        )
        scaled = self.p(ratio(capital, self.b("at")), annual=True)
        return ratio(
            scaled - group_stat(scaled, self.sector, self.universe, "mean"),
            group_stat(scaled, self.sector, self.universe, "std"),
        )

    def _industry_size(self, values):
        return group_stat(values, self.sector, self.universe, "count")

    def _herfindahl(self, values):
        values = positive(self.p(values))
        size = self._industry_size(values)
        share = values / (group_stat(values, self.sector, self.universe, "mean") * size)
        index = group_stat(share**2, self.sector, self.universe, "mean") * size
        return op.mean_lags(index, [0, YEAR, 2 * YEAR], 1)

    def _s_AnnouncementReturn(self):
        abnormal = self.returns - self._series("mkt")
        total = trailing(abnormal, 4, 3, "sum")
        return op.ffill(g.where(self._array("announced") == 1, total, g.nan))

    def _s_RDIPO(self):
        research, sales = self.x("xrd"), self.x("sale")
        separate = (self.grid.value("oap_year") * 4 + self.grid.quarter_index) >= (RD_LINE_YEAR * 4 + 3)
        no_research = self.p(flag(finite_or(research, 0) == 0, g.isfinite(sales) & separate))
        return self._s_IndIPO() * no_research


@lru_cache(maxsize=1)
def _definitions():
    graph = OAPSignals()
    outputs = {}
    for name in OAP_NAMES:
        value = getattr(graph, "_s_" + name)()
        outputs[name] = g.where(g.isfinite(value) & graph.trading, value, g.nan)
    return outputs, graph.statements
