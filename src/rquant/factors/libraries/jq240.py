"""The JoinQuant (聚宽) factor library re-implemented as daily point-in-time panels.

The catalog follows the factor names and published definitions of the JoinQuant factor library (``jqfactor``): the
quality, basic-item, emotion, growth, risk, per-share, technical, momentum and Barra-style categories, about 240
factors in all. This module is an independent re-implementation from those public definitions: no JoinQuant code or
data is used, inputs come from Tushare, and every factor is re-dated to the trading day on which it is known.

Daily adaptation
----------------
* Rows are trading dates and columns are instruments. Price and volume indicators run on each stock's own trading
  days (suspended days are removed first and stay missing), so a gap does not reset an EMA or shorten a window.
* Financial statements are the first-reported values of ``reference/fundamentals.parquet``, usable from the first
  trading day after every earlier quarter is also out (``china_chars.QuarterGrid``). ``_ttm`` items are trailing
  twelve-month sums of single quarters; items without ``_ttm`` that JoinQuant reports per period use the latest single
  quarter; balance-sheet items are the latest quarter end. "Average" balances are the mean of the latest quarter and
  the same quarter a year earlier. Year-on-year growth divides by the absolute value of the base.
* Prices are RQuant's backward-adjusted canonical prices, whose level is arbitrary per stock. JoinQuant already
  publishes moving averages, Bollinger bands and MACD over the close; indicators it leaves in price units (ATR, bull
  and bear power) are divided by the close here as well. ``price_no_fq`` is the unadjusted close from daily_basic.
  Volume is canonical (adjusted) share volume, amount is yuan and turnover is daily_basic float-share turnover in %.
* Style factors follow the Barra CNE5 descriptors JoinQuant documents. Cross-sectional standardisation, ranking and
  orthogonalisation use only the stocks that are in the CSI 300 on that day, so later index members do not leak into
  earlier cross-sections. The market return is the value-weighted all-A series of ``reference/market_factors``.

Left out because Tushare data at this account level cannot build them: ``non_recurring_gain_loss`` and
``adjusted_profit_to_total_profit`` (need the deducted net profit of ``fina_indicator``), and the analyst-forecast
descriptors of the style factors (EPIBS in earnings_yield, EGRLF and EGRSF in growth), whose weights are spread
proportionally over the remaining descriptors.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS
from rquant.errors import DataContractError
from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
from rquant.factors.libraries.china_chars import QuarterGrid, lag, nan_mean, ratio
from rquant.factors.panel_operators import replace_inf
from rquant.io import sha256_file, stable_hash

Panel = pd.DataFrame
Formula = Callable[["JQFactors"], Any]
YEAR = 252
RISK_FREE = 0.04
COLUMN_BLOCK = 32
JQ240_DAILY_BASIC = {"close": 1.0, "turnover_rate": 1.0, "circ_mv": 10_000.0, "total_share": 10_000.0, "pe_ttm": 1.0}

# Missing statement lines that mean "none reported" once the statement itself exists, keyed by the line that shows
# the statement was published for that quarter.
ZERO_FILL = {
    "revenue": (
        "sell_exp", "admin_exp", "fin_exp", "biz_tax_surchg", "assets_impair_loss", "fv_value_chg_gain",
        "invest_income", "ass_invest_income", "non_oper_income", "non_oper_exp",
    ),
    "total_assets": (
        "money_cap", "trad_asset", "notes_receiv", "accounts_receiv", "oth_receiv", "prepayment", "int_receiv",
        "div_receiv", "inventories", "fa_avail_for_sale", "htm_invest", "fix_assets", "cip", "const_materials",
        "intan_assets", "r_and_d", "goodwill", "lt_amor_exp", "st_borr", "trading_fl", "notes_payable",
        "acct_payable", "adv_receipts", "int_payable", "div_payable", "non_cur_liab_due_1y", "lt_borr",
        "bond_payable", "cap_rese", "surplus_rese", "undistr_porfit",
    ),
    "n_cashflow_act": ("depr_fa_coga_dpba", "amort_intang_assets", "lt_amort_deferred_exp"),
}


# ---------------------------------------------------------------------------------------------- formula helpers
def _ttm(name: str) -> Formula:
    return lambda c: c.p(c.g.ttm(name))


def _per_share(values: Callable[[JQFactors], np.ndarray]) -> Formula:
    return lambda c: c.p(values(c)) / c.shares


def _over_cap(values: Callable[[JQFactors], np.ndarray]) -> Formula:
    return lambda c: c.p(values(c)) / c.cap


def _q(values: Callable[[JQFactors], np.ndarray]) -> Formula:
    return lambda c: c.p(values(c))


# (source name, category, description, maximum lookback in trading days, formula). Order is the stable column order.
JQ240: tuple[tuple[str, str, str, int, Formula], ...] = (
    # ------------------------------------------------------------------ quality
    ("net_profit_to_total_operate_revenue_ttm", "quality", "净利润与营业总收入之比: net income TTM / total "
     "operating revenue TTM", 1, _q(lambda c: ratio(c.ttm("n_income"), c.ttm("total_revenue")))),
    ("cfo_to_ev", "quality", "经营现金流与企业价值之比: operating cash flow TTM / (market value + total liabilities "
     "- closing cash and equivalents)", 1, lambda c: c.p(c.ttm("n_cashflow_act")) / (
        c.cap + c.p(c.b("total_liab")) - c.p(c.b("c_cash_equ_end_period")))),
    ("accounts_payable_turnover_days", "quality", "应付账款周转天数: 360 / accounts payable turnover", 1,
     _q(lambda c: ratio(360.0, c.payable_turnover()))),
    ("net_profit_ratio", "quality", "销售净利率: net income TTM / revenue TTM", 1,
     _q(lambda c: ratio(c.ttm("n_income"), c.ttm("revenue")))),
    ("net_non_operating_income_to_total_profit", "quality", "营业外收支净额/利润总额: (non-operating income - "
     "expense) TTM / total profit TTM", 1,
     _q(lambda c: ratio(c.ttm("non_oper_income") - c.ttm("non_oper_exp"), c.ttm("total_profit")))),
    ("fixed_asset_ratio", "quality", "固定资产比率: (fixed assets + construction materials + construction in progress)"
     " / total assets", 1, _q(lambda c: ratio(c.fixed(), c.b("total_assets")))),
    ("account_receivable_turnover_days", "quality", "应收账款周转天数: 360 / receivable turnover", 1,
     _q(lambda c: ratio(360.0, c.receivable_turnover()))),
    ("DEGM", "quality", "毛利率增长: gross margin TTM / gross margin TTM a year earlier - 1", 1,
     _q(lambda c: ratio(c.gross_margin(), lag(c.gross_margin(), 4)) - 1.0)),
    ("sale_expense_to_operating_revenue", "quality", "营业费用与营业总收入之比: selling expense TTM / total "
     "operating revenue TTM", 1, _q(lambda c: ratio(c.ttm("sell_exp"), c.ttm("total_revenue")))),
    ("operating_tax_to_operating_revenue_ratio_ttm", "quality", "销售税金率: business taxes and surcharges TTM / "
     "revenue TTM", 1, _q(lambda c: ratio(c.ttm("biz_tax_surchg"), c.ttm("revenue")))),
    ("inventory_turnover_days", "quality", "存货周转天数: 360 / inventory turnover", 1,
     _q(lambda c: ratio(360.0, c.inventory_turnover()))),
    ("OperatingCycle", "quality", "营业周期: receivable turnover days + inventory turnover days", 1,
     _q(lambda c: ratio(360.0, c.receivable_turnover()) + ratio(360.0, c.inventory_turnover()))),
    ("net_operate_cash_flow_to_operate_income", "quality", "经营现金流与经营活动净收益之比: operating cash flow "
     "TTM / operating net income TTM", 1, _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.operate_net_income()))),
    ("net_operating_cash_flow_coverage", "quality", "净利润现金含量: operating cash flow TTM / net income to "
     "parent TTM", 1, _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.ttm("n_income_attr_p")))),
    ("quick_ratio", "quality", "速动比率: (current assets - inventories) / current liabilities", 1,
     _q(lambda c: ratio(c.b("total_cur_assets") - c.b("inventories"), c.b("total_cur_liab")))),
    ("intangible_asset_ratio", "quality", "无形资产比率: (intangibles + development expenditure + goodwill) / total "
     "assets", 1, _q(lambda c: ratio(c.b("intan_assets") + c.b("r_and_d") + c.b("goodwill"), c.b("total_assets")))),
    ("MLEV", "quality", "市场杠杆: (market value + non-current liabilities) / market value", 1,
     lambda c: (c.cap + c.p(c.b("total_ncl"))) / c.cap),
    ("debt_to_equity_ratio", "quality", "产权比率: total liabilities / equity to parent", 1,
     _q(lambda c: ratio(c.b("total_liab"), c.b("total_hldr_eqy_exc_min_int")))),
    ("super_quick_ratio", "quality", "超速动比率: (cash + trading assets + notes and accounts receivable + other "
     "receivables) / current liabilities", 1, _q(lambda c: ratio(
        c.b("money_cap") + c.b("trad_asset") + c.b("notes_receiv") + c.b("accounts_receiv") + c.other_receivables(),
        c.b("total_cur_liab")))),
    ("inventory_turnover_rate", "quality", "存货周转率: operating cost TTM / average inventories", 1,
     _q(lambda c: c.inventory_turnover())),
    ("long_debt_to_working_capital_ratio", "quality", "长期负债与营运资金比率: non-current liabilities / (current "
     "assets - current liabilities)", 1,
     _q(lambda c: ratio(c.b("total_ncl"), c.b("total_cur_assets") - c.b("total_cur_liab")))),
    ("current_ratio", "quality", "流动比率: current assets / current liabilities", 1,
     _q(lambda c: ratio(c.b("total_cur_assets"), c.b("total_cur_liab")))),
    ("long_debt_to_asset_ratio", "quality", "长期借款与资产总计之比: long-term borrowing / total assets", 1,
     _q(lambda c: ratio(c.b("lt_borr"), c.b("total_assets")))),
    ("gross_income_ratio", "quality", "销售毛利率: (revenue - operating cost) TTM / revenue TTM", 1,
     _q(lambda c: c.gross_margin())),
    ("non_current_asset_ratio", "quality", "非流动资产比率: non-current assets / total assets", 1,
     _q(lambda c: ratio(c.b("total_nca"), c.b("total_assets")))),
    ("total_profit_to_cost_ratio", "quality", "成本费用利润率: total profit TTM / (operating cost + finance + selling"
     " + administrative expense) TTM", 1, _q(lambda c: ratio(c.ttm("total_profit"), c.ttm("oper_cost")
        + c.ttm("fin_exp") + c.ttm("sell_exp") + c.ttm("admin_exp")))),
    ("ROAEBITTTM", "quality", "总资产报酬率: EBIT TTM / average total assets", 1,
     _q(lambda c: ratio(c.ebit(), c.average(c.b("total_assets"))))),
    ("admin_expense_rate", "quality", "管理费用与营业总收入之比: administrative expense TTM / total operating "
     "revenue TTM", 1, _q(lambda c: ratio(c.ttm("admin_exp"), c.ttm("total_revenue")))),
    ("fixed_assets_turnover_rate", "quality", "固定资产周转率: revenue TTM / average fixed assets", 1,
     _q(lambda c: ratio(c.ttm("revenue"), c.average(c.fixed())))),
    ("operating_profit_to_operating_revenue", "quality", "营业利润与营业总收入之比: operating profit TTM / total "
     "operating revenue TTM", 1, _q(lambda c: ratio(c.ttm("operate_profit"), c.ttm("total_revenue")))),
    ("operating_cost_to_operating_revenue_ratio", "quality", "销售成本率: operating cost TTM / revenue TTM", 1,
     _q(lambda c: ratio(c.ttm("oper_cost"), c.ttm("revenue")))),
    ("debt_to_tangible_equity_ratio", "quality", "有形净值债务率: total liabilities / (equity to parent - "
     "intangible assets)", 1, _q(lambda c: ratio(c.b("total_liab"),
                                                c.b("total_hldr_eqy_exc_min_int") - c.b("intan_assets")))),
    ("cash_to_current_liability", "quality", "现金比率: closing cash and equivalents / current liabilities", 1,
     _q(lambda c: ratio(c.b("c_cash_equ_end_period"), c.b("total_cur_liab")))),
    ("financial_expense_rate", "quality", "财务费用与营业总收入之比: finance expense TTM / total operating revenue "
     "TTM", 1, _q(lambda c: ratio(c.ttm("fin_exp"), c.ttm("total_revenue")))),
    ("equity_to_asset_ratio", "quality", "股东权益比率: total equity / total assets", 1,
     _q(lambda c: ratio(c.b("total_hldr_eqy_inc_min_int"), c.b("total_assets")))),
    ("DEGM_8y", "quality", "长期毛利率增长: mean of DEGM over the latest eight years", 1,
     _q(lambda c: c.yearly(ratio(c.gross_margin(), lag(c.gross_margin(), 4)) - 1.0, "mean"))),
    ("maximum_margin", "quality", "最大盈利水平: highest TTM gross margin over the latest eight years", 1,
     _q(lambda c: c.yearly(c.gross_margin(), "max"))),
    ("margin_stability", "quality", "盈利能力稳定性: mean / std of TTM gross margin over the latest eight years", 1,
     _q(lambda c: c.yearly(c.gross_margin(), "stability"))),
    ("roa_ttm", "quality", "资产回报率: net income TTM / average total assets", 1, _q(lambda c: c.roa())),
    ("roa_ttm_8y", "quality", "长期资产回报率: mean of roa_ttm over the latest eight years", 1,
     _q(lambda c: c.yearly(c.roa(), "mean"))),
    ("roe_ttm", "quality", "权益回报率: net income to parent TTM / average equity to parent", 1,
     _q(lambda c: c.roe())),
    ("roe_ttm_8y", "quality", "长期权益回报率: mean of roe_ttm over the latest eight years", 1,
     _q(lambda c: c.yearly(c.roe(), "mean"))),
    ("ACCA", "quality", "现金流资产比和资产回报率之差: (operating cash flow - net income) TTM / total assets", 1,
     _q(lambda c: ratio(c.ttm("n_cashflow_act") - c.ttm("n_income"), c.b("total_assets")))),
    ("GMI", "quality", "毛利率指数: gross margin a year earlier / current gross margin", 1,
     _q(lambda c: ratio(lag(c.gross_margin(), 4), c.gross_margin()))),
    ("LVGI", "quality", "财务杠杆指数: (liabilities / assets) over the same ratio a year earlier", 1,
     _q(lambda c: c.index_yoy(ratio(c.b("total_liab"), c.b("total_assets"))))),
    ("SGAI", "quality", "销售管理费用指数: ((selling + administrative expense) / revenue) TTM over a year earlier", 1,
     _q(lambda c: c.index_yoy(ratio(c.ttm("sell_exp") + c.ttm("admin_exp"), c.ttm("revenue"))))),
    ("SGI", "quality", "营业收入指数: revenue TTM / revenue TTM a year earlier", 1,
     _q(lambda c: c.index_yoy(c.ttm("revenue")))),
    ("DSRI", "quality", "应收账款指数: (receivables / revenue TTM) over the same ratio a year earlier", 1,
     _q(lambda c: c.index_yoy(ratio(c.b("accounts_receiv"), c.ttm("revenue"))))),
    ("AQI", "quality", "资产质量指数: (1 - (current + fixed assets) / total assets) over a year earlier", 1,
     _q(lambda c: c.index_yoy(1.0 - ratio(c.b("total_cur_assets") + c.b("fix_assets"), c.b("total_assets"))))),
    ("invest_income_associates_to_total_profit", "quality", "对联营和合营公司投资收益/利润总额: TTM", 1,
     _q(lambda c: ratio(c.ttm("ass_invest_income"), c.ttm("total_profit")))),
    ("equity_to_fixed_asset_ratio", "quality", "股东权益与固定资产比率: total equity / fixed assets", 1,
     _q(lambda c: ratio(c.b("total_hldr_eqy_inc_min_int"), c.fixed()))),
    ("accounts_payable_turnover_rate", "quality", "应付账款周转率: operating cost TTM / average notes and accounts "
     "payable", 1, _q(lambda c: c.payable_turnover())),
    ("account_receivable_turnover_rate", "quality", "应收账款周转率: revenue TTM / average notes and accounts "
     "receivable", 1, _q(lambda c: c.receivable_turnover())),
    ("current_asset_turnover_rate", "quality", "流动资产周转率: revenue TTM / average current assets", 1,
     _q(lambda c: ratio(c.ttm("revenue"), c.average(c.b("total_cur_assets"))))),
    ("total_asset_turnover_rate", "quality", "总资产周转率: revenue TTM / average total assets", 1,
     _q(lambda c: ratio(c.ttm("revenue"), c.average(c.b("total_assets"))))),
    ("net_operate_cash_flow_to_total_liability", "quality", "经营现金流/负债合计: operating cash flow TTM / total "
     "liabilities", 1, _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.b("total_liab")))),
    ("net_operate_cash_flow_to_net_debt", "quality", "经营现金流/净债务: operating cash flow TTM / net debt", 1,
     _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.net_debt()))),
    ("net_operate_cash_flow_to_asset", "quality", "总资产现金回收率: operating cash flow TTM / total assets", 1,
     _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.b("total_assets")))),
    ("net_operate_cash_flow_to_total_current_liability", "quality", "现金流动负债比: operating cash flow TTM / "
     "current liabilities", 1, _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.b("total_cur_liab")))),
    ("operating_profit_to_total_profit", "quality", "经营活动净收益/利润总额: operating net income TTM / total "
     "profit TTM", 1, _q(lambda c: ratio(c.operate_net_income(), c.ttm("total_profit")))),
    ("goods_sale_and_service_to_revenue_ttm", "quality", "销售商品提供劳务收到的现金与营业收入之比: TTM", 1,
     _q(lambda c: ratio(c.ttm("c_fr_sale_sg"), c.ttm("revenue")))),
    ("rnoa_ttm", "quality", "经营资产回报率: operating profit TTM / average (operating assets - operating "
     "liabilities)", 1, _q(lambda c: ratio(c.ttm("operate_profit"),
                                           c.average(c.operating_assets() - c.operating_liability())))),
    ("cash_rate_of_sales", "quality", "经营活动现金流与营业收入之比: operating cash flow TTM / revenue TTM", 1,
     _q(lambda c: ratio(c.ttm("n_cashflow_act"), c.ttm("revenue")))),
    ("BLEV", "quality", "账面杠杆: (equity to parent + non-current liabilities) / equity to parent", 1,
     _q(lambda c: c.blev())),
    ("debt_to_asset_ratio", "quality", "债务总资产比: total liabilities / total assets", 1,
     _q(lambda c: ratio(c.b("total_liab"), c.b("total_assets")))),
    ("roic_ttm", "quality", "投资资本回报率: net income to parent TTM / average (total assets - current liabilities"
     " + notes payable + short-term borrowing + non-current liabilities due within a year)", 1,
     _q(lambda c: ratio(c.ttm("n_income_attr_p"), c.average(
         c.b("total_assets") - c.b("total_cur_liab") + c.b("notes_payable") + c.b("st_borr")
         + c.b("non_cur_liab_due_1y"))))),
    # ------------------------------------------------------------------ basic items and derivatives
    ("administration_expense_ttm", "basics", "管理费用TTM", 1, _ttm("admin_exp")),
    ("asset_impairment_loss_ttm", "basics", "资产减值损失TTM", 1, _ttm("assets_impair_loss")),
    ("cash_flow_to_price_ratio", "basics", "现金流市值比: operating cash flow TTM / market value", 1,
     _over_cap(lambda c: c.ttm("n_cashflow_act"))),
    ("circulating_market_cap", "basics", "流通市值 (yuan)", 1, lambda c: c.inputs["basic_circ_mv"]),
    ("EBIT", "basics", "息税前利润: total profit + finance expense, TTM", 1, _q(lambda c: c.ebit())),
    ("EBITDA", "basics", "息税折旧摊销前利润: EBIT + depreciation + amortisation, TTM", 1,
     _q(lambda c: c.ebit() + c.ttm("depr_fa_coga_dpba") + c.ttm("amort_intang_assets")
        + c.ttm("lt_amort_deferred_exp"))),
    ("financial_assets", "basics", "金融资产: cash + trading assets + notes, interest and dividends receivable + "
     "available-for-sale + held-to-maturity", 1, _q(lambda c: c.financial_assets())),
    ("financial_expense_ttm", "basics", "财务费用TTM", 1, _ttm("fin_exp")),
    ("financial_liability", "basics", "金融负债: trading liabilities + short-term borrowing + notes, interest and "
     "dividends payable + non-current due within a year + long-term borrowing + bonds", 1,
     _q(lambda c: c.financial_liability())),
    ("goods_sale_and_service_render_cash_ttm", "basics", "销售商品提供劳务收到的现金TTM", 1, _ttm("c_fr_sale_sg")),
    ("gross_profit_ttm", "basics", "毛利TTM: revenue - operating cost", 1,
     _q(lambda c: c.ttm("revenue") - c.ttm("oper_cost"))),
    ("interest_carry_current_liability", "basics", "带息流动负债: short-term borrowing + trading liabilities + "
     "interest payable + non-current liabilities due within a year", 1, _q(lambda c: c.interest_carry_current())),
    ("interest_free_current_liability", "basics", "无息流动负债: current liabilities - interest-bearing current "
     "liabilities", 1, _q(lambda c: c.b("total_cur_liab") - c.interest_carry_current())),
    ("market_cap", "basics", "总市值 (yuan)", 1, lambda c: c.cap),
    ("net_debt", "basics", "净债务: interest-bearing liabilities - cash", 1, _q(lambda c: c.net_debt())),
    ("net_finance_cash_flow_ttm", "basics", "筹资活动现金流量净额TTM", 1, _ttm("n_cash_flows_fnc_act")),
    ("net_interest_expense", "basics", "净利息费用: interest expense - interest income within finance expense, TTM "
     "(reported from 2018)", 1, _q(lambda c: c.ttm("fin_exp_int_exp") - c.ttm("fin_exp_int_inc"))),
    ("net_invest_cash_flow_ttm", "basics", "投资活动现金流量净额TTM", 1, _ttm("n_cashflow_inv_act")),
    ("net_operate_cash_flow_ttm", "basics", "经营活动现金流量净额TTM", 1, _ttm("n_cashflow_act")),
    ("net_profit_ttm", "basics", "净利润TTM", 1, _ttm("n_income")),
    ("net_working_capital", "basics", "净运营资本: current assets - current liabilities", 1,
     _q(lambda c: c.b("total_cur_assets") - c.b("total_cur_liab"))),
    ("non_operating_net_profit_ttm", "basics", "营业外收支净额TTM", 1,
     _q(lambda c: c.ttm("non_oper_income") - c.ttm("non_oper_exp"))),
    ("np_parent_company_owners_ttm", "basics", "归属于母公司股东的净利润TTM", 1, _ttm("n_income_attr_p")),
    ("OperateNetIncome", "basics", "经营活动净收益: operating profit - value-change profit, TTM", 1,
     _q(lambda c: c.operate_net_income())),
    ("operating_assets", "basics", "经营性资产: total assets - financial assets", 1,
     _q(lambda c: c.operating_assets())),
    ("operating_cost_ttm", "basics", "营业成本TTM", 1, _ttm("oper_cost")),
    ("operating_liability", "basics", "经营性负债: total liabilities - financial liabilities", 1,
     _q(lambda c: c.operating_liability())),
    ("operating_profit_ttm", "basics", "营业利润TTM", 1, _ttm("operate_profit")),
    ("operating_revenue_ttm", "basics", "营业收入TTM", 1, _ttm("revenue")),
    ("retained_earnings", "basics", "留存收益: surplus reserve + undistributed profit", 1,
     _q(lambda c: c.retained_earnings())),
    ("sales_to_price_ratio", "basics", "营收市值比: revenue TTM / market value", 1,
     _over_cap(lambda c: c.ttm("revenue"))),
    ("sale_expense_ttm", "basics", "销售费用TTM", 1, _ttm("sell_exp")),
    ("total_operating_cost_ttm", "basics", "营业总成本TTM", 1, _ttm("total_cogs")),
    ("total_operating_revenue_ttm", "basics", "营业总收入TTM", 1, _ttm("total_revenue")),
    ("total_profit_ttm", "basics", "利润总额TTM", 1, _ttm("total_profit")),
    ("value_change_profit_ttm", "basics", "价值变动净收益TTM: fair-value change + investment income", 1,
     _q(lambda c: c.value_change_profit())),
    # ------------------------------------------------------------------ emotion
    ("AR", "emotion", "人气指标: 100 * sum(high - open, 26) / sum(open - low, 26)", 26,
     lambda c: c.out(100.0 * c.ratio_p(c.sum(c.h - c.o, 26), c.sum(c.o - c.l, 26)))),
    ("ARBR", "emotion", "ARBR: AR - BR", 27, lambda c: c.calc("AR") - c.calc("BR")),
    ("ATR14", "emotion", "14日均幅指标: mean true range over 14 days / close", 15, lambda c: c.out(c.atr(14))),
    ("ATR6", "emotion", "6日均幅指标: mean true range over 6 days / close", 7, lambda c: c.out(c.atr(6))),
    ("BR", "emotion", "意愿指标: 100 * sum(max(0, high - previous close), 26) / sum(max(0, previous close - low), 26)",
     27, lambda c: c.out(100.0 * c.ratio_p(c.sum((c.h - c.c.shift(1)).clip(lower=0.0), 26),
                                          c.sum((c.c.shift(1) - c.l).clip(lower=0.0), 26)))),
    ("DAVOL10", "emotion", "10日平均换手率与120日平均换手率之比", 120, lambda c: c.out(c.davol(10))),
    ("DAVOL20", "emotion", "20日平均换手率与120日平均换手率之比", 120, lambda c: c.out(c.davol(20))),
    ("DAVOL5", "emotion", "5日平均换手率与120日平均换手率之比", 120, lambda c: c.out(c.davol(5))),
    ("MAWVAD", "emotion", "因子WVAD的6日均值", 29, lambda c: c.out(c.mean(c.wvad(), 6))),
    ("money_flow_20", "emotion", "20日资金流量: mean of (high + low + close) / 3 * volume over 20 days", 20,
     lambda c: c.out(c.mean((c.h + c.l + c.c) / 3.0 * c.v, 20))),
    ("PSY", "emotion", "心理线指标: share of up days over 12 days * 100", 13,
     lambda c: c.out(100.0 * c.mean((c.c > c.c.shift(1)).astype(float).where(c.c.shift(1).notna()), 12))),
    ("turnover_volatility", "emotion", "换手率相对波动率: std / mean of daily turnover over 20 days", 20,
     lambda c: c.out(c.ratio_p(c.std(c.turn, 20), c.mean(c.turn, 20)))),
    ("TVMA20", "emotion", "20日成交金额的移动平均值", 20, lambda c: c.out(c.mean(c.amt, 20))),
    ("TVMA6", "emotion", "6日成交金额的移动平均值", 6, lambda c: c.out(c.mean(c.amt, 6))),
    ("TVSTD20", "emotion", "20日成交金额的标准差", 20, lambda c: c.out(c.std(c.amt, 20))),
    ("TVSTD6", "emotion", "6日成交金额的标准差", 6, lambda c: c.out(c.std(c.amt, 6))),
    ("VDEA", "emotion", "计算VMACD因子的中间变量: EMA(VDIFF, 9)", 34, lambda c: c.out(c.vdea())),
    ("VDIFF", "emotion", "计算VMACD因子的中间变量: EMA(volume, 12) - EMA(volume, 26)", 26,
     lambda c: c.out(c.vdiff())),
    ("VEMA10", "emotion", "成交量的10日指数移动平均", 10, lambda c: c.out(c.ema(c.v, 10))),
    ("VEMA12", "emotion", "成交量的12日指数移动平均", 12, lambda c: c.out(c.ema(c.v, 12))),
    ("VEMA26", "emotion", "成交量的26日指数移动平均", 26, lambda c: c.out(c.ema(c.v, 26))),
    ("VEMA5", "emotion", "成交量的5日指数移动平均", 5, lambda c: c.out(c.ema(c.v, 5))),
    ("VMACD", "emotion", "成交量指数平滑异同移动平均线: VDIFF - VDEA", 34,
     lambda c: c.out(c.vdiff() - c.vdea())),
    ("VOL10", "emotion", "10日平均换手率 (%)", 10, lambda c: c.out(c.mean(c.turn, 10))),
    ("VOL120", "emotion", "120日平均换手率 (%)", 120, lambda c: c.out(c.mean(c.turn, 120))),
    ("VOL20", "emotion", "20日平均换手率 (%)", 20, lambda c: c.out(c.mean(c.turn, 20))),
    ("VOL240", "emotion", "240日平均换手率 (%)", 240, lambda c: c.out(c.mean(c.turn, 240))),
    ("VOL5", "emotion", "5日平均换手率 (%)", 5, lambda c: c.out(c.mean(c.turn, 5))),
    ("VOL60", "emotion", "60日平均换手率 (%)", 60, lambda c: c.out(c.mean(c.turn, 60))),
    ("VOSC", "emotion", "成交量震荡: 100 * (MA(volume, 12) - MA(volume, 26)) / MA(volume, 12)", 26,
     lambda c: c.out(100.0 * c.ratio_p(c.mean(c.v, 12) - c.mean(c.v, 26), c.mean(c.v, 12)))),
    ("VR", "emotion", "成交量比率: 100 * (up-day volume + half flat-day volume) / (down-day volume + half flat-day "
     "volume) over 24 days", 25, lambda c: c.out(c.vr(24))),
    ("VROC12", "emotion", "12日量变动速率指标: 100 * (volume / volume 12 days ago - 1)", 12,
     lambda c: c.out(100.0 * (c.ratio_p(c.v, c.v.shift(12)) - 1.0))),
    ("VROC6", "emotion", "6日量变动速率指标: 100 * (volume / volume 6 days ago - 1)", 6,
     lambda c: c.out(100.0 * (c.ratio_p(c.v, c.v.shift(6)) - 1.0))),
    ("VSTD10", "emotion", "10日成交量标准差", 10, lambda c: c.out(c.std(c.v, 10))),
    ("VSTD20", "emotion", "20日成交量标准差", 20, lambda c: c.out(c.std(c.v, 20))),
    ("WVAD", "emotion", "威廉变异离散量: sum over 24 days of (close - open) / (high - low) * volume", 24,
     lambda c: c.out(c.wvad())),
    # ------------------------------------------------------------------ growth
    ("financing_cash_growth_rate", "growth", "筹资活动现金流量净额增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("n_cash_flows_fnc_act")))),
    ("net_asset_growth_rate", "growth", "净资产增长率: equity to parent year on year", 1,
     _q(lambda c: c.yoy(c.b("total_hldr_eqy_exc_min_int")))),
    ("net_operate_cashflow_growth_rate", "growth", "经营活动产生的现金流量净额增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("n_cashflow_act")))),
    ("net_profit_growth_rate", "growth", "净利润增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("n_income")))),
    ("np_parent_company_owners_growth_rate", "growth", "归属母公司股东的净利润增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("n_income_attr_p")))),
    ("operating_revenue_growth_rate", "growth", "营业收入增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("revenue")))),
    ("PEG", "growth", "市盈率相对盈利增长比率: PE TTM / (100 * net income to parent TTM growth)", 1,
     lambda c: replace_inf(c.inputs["basic_pe_ttm"] / (100.0 * c.p(c.yoy(c.ttm("n_income_attr_p")))))),
    ("total_asset_growth_rate", "growth", "总资产增长率: total assets year on year", 1,
     _q(lambda c: c.yoy(c.b("total_assets")))),
    ("total_profit_growth_rate", "growth", "利润总额增长率: TTM year on year", 1,
     _q(lambda c: c.yoy(c.ttm("total_profit")))),
    # ------------------------------------------------------------------ risk
    ("Kurtosis120", "risk", "个股收益的120日峰度", 121, lambda c: c.out(c.ret.rolling(120, min_periods=120).kurt())),
    ("Kurtosis20", "risk", "个股收益的20日峰度", 21, lambda c: c.out(c.ret.rolling(20, min_periods=20).kurt())),
    ("Kurtosis60", "risk", "个股收益的60日峰度", 61, lambda c: c.out(c.ret.rolling(60, min_periods=60).kurt())),
    ("sharpe_ratio_120", "risk", "120日夏普比率: (annualised mean return - 4%) / annualised volatility", 121,
     lambda c: c.out(c.sharpe(120))),
    ("sharpe_ratio_20", "risk", "20日夏普比率", 21, lambda c: c.out(c.sharpe(20))),
    ("sharpe_ratio_60", "risk", "60日夏普比率", 61, lambda c: c.out(c.sharpe(60))),
    ("Skewness120", "risk", "个股收益的120日偏度", 121, lambda c: c.out(c.ret.rolling(120, min_periods=120).skew())),
    ("Skewness20", "risk", "个股收益的20日偏度", 21, lambda c: c.out(c.ret.rolling(20, min_periods=20).skew())),
    ("Skewness60", "risk", "个股收益的60日偏度", 61, lambda c: c.out(c.ret.rolling(60, min_periods=60).skew())),
    ("Variance120", "risk", "120日年化收益方差", 121, lambda c: c.out(c.std(c.ret, 120) ** 2 * 250.0)),
    ("Variance20", "risk", "20日年化收益方差", 21, lambda c: c.out(c.std(c.ret, 20) ** 2 * 250.0)),
    ("Variance60", "risk", "60日年化收益方差", 61, lambda c: c.out(c.std(c.ret, 60) ** 2 * 250.0)),
    # ------------------------------------------------------------------ per share
    ("capital_reserve_fund_per_share", "pershare", "每股资本公积金", 1, _per_share(lambda c: c.b("cap_rese"))),
    ("cash_and_equivalents_per_share", "pershare", "每股现金及现金等价物余额", 1,
     _per_share(lambda c: c.b("c_cash_equ_end_period"))),
    ("cashflow_per_share_ttm", "pershare", "每股现金流量净额: net increase in cash TTM / shares", 1,
     _per_share(lambda c: c.ttm("n_incr_cash_cash_equ"))),
    ("eps_ttm", "pershare", "每股收益TTM: net income to parent TTM / shares", 1,
     _per_share(lambda c: c.ttm("n_income_attr_p"))),
    ("net_asset_per_share", "pershare", "每股净资产: equity to parent / shares", 1,
     _per_share(lambda c: c.b("total_hldr_eqy_exc_min_int"))),
    ("net_operate_cash_flow_per_share", "pershare", "每股经营活动产生的现金流量净额: TTM", 1,
     _per_share(lambda c: c.ttm("n_cashflow_act"))),
    ("operating_profit_per_share", "pershare", "每股营业利润: latest quarter", 1,
     _per_share(lambda c: c.g.single("operate_profit"))),
    ("operating_profit_per_share_ttm", "pershare", "每股营业利润TTM", 1,
     _per_share(lambda c: c.ttm("operate_profit"))),
    ("operating_revenue_per_share", "pershare", "每股营业收入: latest quarter", 1,
     _per_share(lambda c: c.g.single("revenue"))),
    ("operating_revenue_per_share_ttm", "pershare", "每股营业收入TTM", 1, _per_share(lambda c: c.ttm("revenue"))),
    ("retained_earnings_per_share", "pershare", "每股留存收益: (surplus reserve + undistributed profit) / shares", 1,
     _per_share(lambda c: c.retained_earnings())),
    ("retained_profit_per_share", "pershare", "每股未分配利润", 1, _per_share(lambda c: c.b("undistr_porfit"))),
    ("surplus_reserve_fund_per_share", "pershare", "每股盈余公积金", 1, _per_share(lambda c: c.b("surplus_rese"))),
    ("total_operating_revenue_per_share", "pershare", "每股营业总收入: latest quarter", 1,
     _per_share(lambda c: c.g.single("total_revenue"))),
    ("total_operating_revenue_per_share_ttm", "pershare", "每股营业总收入TTM", 1,
     _per_share(lambda c: c.ttm("total_revenue"))),
    # ------------------------------------------------------------------ technical
    ("boll_down", "technical", "下轨线（布林线）: (MA(close, 20) - 2 * std(close, 20)) / close", 20,
     lambda c: c.out((c.mean(c.c, 20) - 2.0 * c.std(c.c, 20)) / c.c)),
    ("boll_up", "technical", "上轨线（布林线）: (MA(close, 20) + 2 * std(close, 20)) / close", 20,
     lambda c: c.out((c.mean(c.c, 20) + 2.0 * c.std(c.c, 20)) / c.c)),
    ("EMAC10", "technical", "10日指数移动均线 / close", 10, lambda c: c.out(c.ema(c.c, 10) / c.c)),
    ("EMAC12", "technical", "12日指数移动均线 / close", 12, lambda c: c.out(c.ema(c.c, 12) / c.c)),
    ("EMAC120", "technical", "120日指数移动均线 / close", 120, lambda c: c.out(c.ema(c.c, 120) / c.c)),
    ("EMAC20", "technical", "20日指数移动均线 / close", 20, lambda c: c.out(c.ema(c.c, 20) / c.c)),
    ("EMAC26", "technical", "26日指数移动均线 / close", 26, lambda c: c.out(c.ema(c.c, 26) / c.c)),
    ("MAC10", "technical", "10日移动均线 / close", 10, lambda c: c.out(c.mean(c.c, 10) / c.c)),
    ("MAC120", "technical", "120日移动均线 / close", 120, lambda c: c.out(c.mean(c.c, 120) / c.c)),
    ("MAC20", "technical", "20日移动均线 / close", 20, lambda c: c.out(c.mean(c.c, 20) / c.c)),
    ("MAC5", "technical", "5日移动均线 / close", 5, lambda c: c.out(c.mean(c.c, 5) / c.c)),
    ("MAC60", "technical", "60日移动均线 / close", 60, lambda c: c.out(c.mean(c.c, 60) / c.c)),
    ("MACDC", "technical", "平滑异同移动平均线: 2 * (DIF - DEA) of MACD(12, 26, 9) / close", 34,
     lambda c: c.out(c.macd() / c.c)),
    ("MFI14", "technical", "资金流量指标: 14-day money flow index", 15, lambda c: c.out(c.mfi(14))),
    ("price_no_fq", "technical", "不复权价格: unadjusted close", 1, lambda c: c.inputs["basic_close"]),
    # ------------------------------------------------------------------ momentum
    ("arron_down_25", "momentum", "Aroon指标下轨: (25 - days since the 25-day low) / 25", 25,
     lambda c: c.out(c.aroon(c.l, 25, low=True))),
    ("arron_up_25", "momentum", "Aroon指标上轨: (25 - days since the 25-day high) / 25", 25,
     lambda c: c.out(c.aroon(c.h, 25, low=False))),
    ("BBIC", "momentum", "BBI 动量: (MA3 + MA6 + MA12 + MA24) / 4 / close", 24,
     lambda c: c.out((c.mean(c.c, 3) + c.mean(c.c, 6) + c.mean(c.c, 12) + c.mean(c.c, 24)) / 4.0 / c.c)),
    ("bear_power", "momentum", "空头力道: (low - EMA(close, 13)) / close", 13,
     lambda c: c.out((c.l - c.ema(c.c, 13)) / c.c)),
    ("BIAS10", "momentum", "10日乖离率: 100 * (close / MA(close, 10) - 1)", 10, lambda c: c.out(c.bias(10))),
    ("BIAS20", "momentum", "20日乖离率", 20, lambda c: c.out(c.bias(20))),
    ("BIAS5", "momentum", "5日乖离率", 5, lambda c: c.out(c.bias(5))),
    ("BIAS60", "momentum", "60日乖离率", 60, lambda c: c.out(c.bias(60))),
    ("bull_power", "momentum", "多头力道: (high - EMA(close, 13)) / close", 13,
     lambda c: c.out((c.h - c.ema(c.c, 13)) / c.c)),
    ("CCI10", "momentum", "10日顺势指标: (TP - MA(TP)) / (0.015 * mean absolute deviation of TP)", 10,
     lambda c: c.out(c.cci(10))),
    ("CCI15", "momentum", "15日顺势指标", 15, lambda c: c.out(c.cci(15))),
    ("CCI20", "momentum", "20日顺势指标", 20, lambda c: c.out(c.cci(20))),
    ("CCI88", "momentum", "88日顺势指标", 88, lambda c: c.out(c.cci(88))),
    ("CR20", "momentum", "CR指标: 100 * sum(max(0, high - previous TP), 20) / sum(max(0, previous TP - low), 20)",
     21, lambda c: c.out(c.cr(20))),
    ("fifty_two_week_close_rank", "momentum", "当前价格处于过去1年股价的位置: (close - 250-day low close) / "
     "(250-day high - low close)", 250, lambda c: c.out(c.ratio_p(c.c - c.c.rolling(250, min_periods=250).min(),
                                                                c.c.rolling(250, min_periods=250).max()
                                                                - c.c.rolling(250, min_periods=250).min()))),
    ("MASS", "momentum", "梅斯线: sum over 25 days of EMA(high - low, 9) / EMA(EMA(high - low, 9), 9)", 41,
     lambda c: c.out(c.sum(c.ema(c.h - c.l, 9) / c.ema(c.ema(c.h - c.l, 9), 9), 25))),
    ("PLRC12", "momentum", "12日收盘价格线性回归系数: slope of close / mean close on day number", 12,
     lambda c: c.out(c.plrc(12))),
    ("PLRC24", "momentum", "24日收盘价格线性回归系数", 24, lambda c: c.out(c.plrc(24))),
    ("PLRC6", "momentum", "6日收盘价格线性回归系数", 6, lambda c: c.out(c.plrc(6))),
    ("Price1M", "momentum", "当前股价除以过去一个月股价均值再减1: close / MA(close, 20) - 1", 20,
     lambda c: c.out(c.c / c.mean(c.c, 20) - 1.0)),
    ("Price1Y", "momentum", "当前股价除以过去一年股价均值再减1: close / MA(close, 250) - 1", 250,
     lambda c: c.out(c.c / c.mean(c.c, 250) - 1.0)),
    ("Price3M", "momentum", "当前股价除以过去三个月股价均值再减1: close / MA(close, 60) - 1", 60,
     lambda c: c.out(c.c / c.mean(c.c, 60) - 1.0)),
    ("Rank1M", "momentum", "1减去过去一个月收益率排名与股票总数的比值: 1 - CSI 300 cross-sectional percentile of "
     "the 20-day return", 20, lambda c: 1.0 - c.cs_rank(c.out(c.ratio_p(c.c, c.c.shift(20)) - 1.0))),
    ("ROC12", "momentum", "12日变动速率: 100 * (close / close 12 days ago - 1)", 12, lambda c: c.out(c.roc(12))),
    ("ROC120", "momentum", "120日变动速率", 120, lambda c: c.out(c.roc(120))),
    ("ROC20", "momentum", "20日变动速率", 20, lambda c: c.out(c.roc(20))),
    ("ROC6", "momentum", "6日变动速率", 6, lambda c: c.out(c.roc(6))),
    ("ROC60", "momentum", "60日变动速率", 60, lambda c: c.out(c.roc(60))),
    ("single_day_VPT", "momentum", "单日价量趋势: return * volume", 1, lambda c: c.out(c.ret * c.v)),
    ("single_day_VPT_12", "momentum", "单日价量趋势12日均值", 13, lambda c: c.out(c.mean(c.ret * c.v, 12))),
    ("single_day_VPT_6", "momentum", "单日价量趋势6日均值", 7, lambda c: c.out(c.mean(c.ret * c.v, 6))),
    ("TRIX10", "momentum", "10日终极指标TRIX: 100 * one-day change of the triple EMA(close, 10)", 31,
     lambda c: c.out(c.trix(10))),
    ("TRIX5", "momentum", "5日终极指标TRIX", 16, lambda c: c.out(c.trix(5))),
    ("Volume1M", "momentum", "当前交易量相比过去1个月日均交易量与过去20日日均收益率乘积: volume / MA(volume, 20) * "
     "mean(return, 20)", 21, lambda c: c.out(c.ratio_p(c.v, c.mean(c.v, 20)) * c.mean(c.ret, 20))),
    # ------------------------------------------------------------------ Barra CNE5 style
    ("size", "style", "市值: standardised log total market value", 1, lambda c: c.style_size()),
    ("beta", "style", "贝塔: standardised 252-day market beta, half-life 63", YEAR, lambda c: c.style_beta()),
    ("momentum", "style", "动量: standardised RSTR, weighted log return from day -525 to -21, half-life 126",
     525, lambda c: c.style_momentum()),
    ("residual_volatility", "style", "残差波动率: 0.74 DASTD + 0.16 CMRA + 0.10 HSIGMA, orthogonal to beta and "
     "size", YEAR, lambda c: c.style_residual_volatility()),
    ("non_linear_size", "style", "非线性市值: cube of size orthogonal to size", 1,
     lambda c: c.style_non_linear_size()),
    ("book_to_price_ratio", "style", "账面市值比: standardised equity to parent / market value", 1,
     lambda c: c.cs_z(c.p(c.b("total_hldr_eqy_exc_min_int")) / c.cap)),
    ("liquidity", "style", "流动性: 0.35 STOM + 0.35 STOQ + 0.30 STOA, orthogonal to size", YEAR,
     lambda c: c.style_liquidity()),
    ("earnings_yield", "style", "盈利预期: (0.21 ETOP + 0.11 CETOP) / 0.32 without analyst forecasts", 1,
     lambda c: c.style_earnings_yield()),
    ("growth", "style", "成长: (0.24 EGRO + 0.47 SGRO) / 0.71 without analyst forecasts", 1,
     lambda c: c.style_growth()),
    ("leverage", "style", "杠杆: 0.38 MLEV + 0.35 DTOA + 0.27 BLEV", 1, lambda c: c.style_leverage()),
)
JQ240_NAMES = tuple(entry[0] for entry in JQ240)
JQ240_FORMULAS = {entry[0]: entry[4] for entry in JQ240}
JQ240_CATEGORIES = tuple(dict.fromkeys(entry[1] for entry in JQ240))


class JQ240Library(PanelFactorLibrary):
    """JoinQuant factor-library definitions re-implemented on Tushare data."""

    family = "jq240"
    required_inputs = ("open", "high", "low", "close", "volume", "amount", "cap")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=f"jq_{ordinal:03d}",
                source_name=name,
                family=self.family,
                ordinal=ordinal,
                formula=f"[{category}] {description}",
                max_lookback=lookback,
                implementation=f"rquant.factors.libraries.jq240.JQ240.{name}",
                catalog_version=1,
            )
            for ordinal, (name, category, description, lookback, _) in enumerate(JQ240, 1)
        )

    @property
    def implementation_fingerprint(self) -> str:
        """Also pin the shared statement grid and the statement fields, which live in other modules."""
        shared = importlib.import_module("rquant.factors.libraries.china_chars")
        return stable_hash(
            {
                "module": super().implementation_fingerprint,
                "quarter_grid": inspect.getsource(shared),
                "statement_fields": {name: list(fields) for name, fields in FUNDAMENTAL_ENDPOINT_FIELDS.items()},
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
                "jq240 needs financial statements; run `rquant data sync-fundamentals` and "
                "`rquant data build-reference` first"
            )
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        missing = sorted({field for fields in ZERO_FILL.values() for field in fields} - set(fundamentals.columns))
        if missing:
            raise DataContractError(
                f"Fundamental reference lacks jq240 statement items {missing[:5]}; run `rquant data "
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
        columns = ["instrument", "datetime", *JQ240_DAILY_BASIC]
        # Reading only the panel's instruments keeps the all-A daily_basic partitions out of memory.
        wanted = [("instrument", "in", list(instruments))]
        daily = pd.concat(
            [pd.read_parquet(root / name, columns=columns, filters=wanted) for name in daily_files], ignore_index=True
        )
        inputs: dict[str, Any] = {}
        for field, scale in JQ240_DAILY_BASIC.items():
            panel = daily.pivot(index="datetime", columns="instrument", values=field)
            inputs[f"basic_{field}"] = panel.reindex(index=dates, columns=instruments) * scale

        factors = pd.read_parquet(root / "market_factors.parquet").set_index("datetime").sort_index()
        inputs["mkt"] = factors["mkt"].reindex(dates)
        inputs["fundamentals"] = fundamentals[fundamentals["instrument"].isin(set(instruments))]

        hashes = {
            "jq240_reference": manifest["fingerprint"],
            "jq240_daily_basic": stable_hash({name: files[name] for name in daily_files}),
            "jq240_market_factors": files["market_factors.parquet"],
            "jq240_fundamentals": files["fundamentals.parquet"],
        }
        return inputs, hashes

    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        calculator = JQFactors(inputs)
        return {name: calculator.calculate(name) for name in JQ240_NAMES}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        calculator = JQFactors(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = calculator.calculate(spec.source_name).to_numpy(dtype=float)


def prepare_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill lines a published statement leaves blank and combine the other-receivables lines."""
    frame = fundamentals.copy()
    for anchor, fields in ZERO_FILL.items():
        published = frame[anchor].notna() if anchor in frame.columns else pd.Series(False, index=frame.index)
        for field in fields:
            if field not in frame.columns:
                frame[field] = np.nan
            frame[field] = frame[field].where(frame[field].notna() | ~published, 0.0)
    combined = frame["oth_receiv"] + frame["int_receiv"] + frame["div_receiv"]
    total = frame["oth_rcv_total"] if "oth_rcv_total" in frame.columns else combined * np.nan
    frame["jq_other_receivables"] = total.where(total.notna(), combined)
    return frame


class Packer:
    """Move each stock's trading days to consecutive rows and back.

    Packed row ``k`` of a column is that stock's ``k``-th trading day, so rolling windows and EMAs count quotes, not
    calendar days; rows after a stock's last quote are missing.
    """

    def __init__(self, trading: np.ndarray, index: pd.Index, columns: pd.Index) -> None:
        self.index = index
        self.columns = columns
        self.rows, self.cols = np.nonzero(trading)
        self.position = (np.cumsum(trading, axis=0) - 1)[self.rows, self.cols]
        self.length = int(trading.sum(axis=0).max()) if trading.size else 0

    def pack(self, panel: Panel) -> Panel:
        values = np.full((self.length, len(self.columns)), np.nan)
        values[self.position, self.cols] = np.asarray(panel, dtype=float)[self.rows, self.cols]
        return Panel(values, columns=self.columns)

    def unpack(self, packed: Panel | np.ndarray) -> Panel:
        values = np.full((len(self.index), len(self.columns)), np.nan)
        values[self.rows, self.cols] = np.asarray(packed, dtype=float)[self.position, self.cols]
        return Panel(values, index=self.index, columns=self.columns)


class JQFactors:
    """Calculator with shared intermediates; ``calculate`` returns one factor panel on the calendar grid."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self.inputs = inputs
        close = inputs["close"]
        self.dates = pd.DatetimeIndex(close.index)
        self.instruments = pd.Index(close.columns)
        self.trading = close.notna()
        self.packer = Packer(self.trading.to_numpy(), self.dates, self.instruments)
        self._packed: dict[str, Panel] = {}
        self._cache: dict[str, Any] = {}
        self._grid: QuarterGrid | None = None
        self.cap = self._panel("cap").where(self._panel("cap") > 0)
        self.shares = self._panel("basic_total_share").where(self._panel("basic_total_share") > 0)
        self.universe = (
            self._panel("universe").reindex(index=self.dates, columns=self.instruments).eq(True)
        )

    def calculate(self, name: str) -> Panel:
        result = JQ240_FORMULAS[name](self)
        if isinstance(result, np.ndarray):
            result = Panel(result, index=self.dates, columns=self.instruments)
        result = replace_inf(result.reindex(index=self.dates, columns=self.instruments).astype(float))
        return result.where(self.trading)

    def calc(self, name: str) -> Panel:
        """A published factor, cached for factors defined from it."""
        key = f"factor:{name}"
        if key not in self._cache:
            self._cache[key] = self.calculate(name)
        return self._cache[key]

    def _panel(self, key: str) -> Any:
        value = self.inputs.get(key)
        if value is None:
            raise DataContractError(f"jq240 input is missing: {key}")
        return value

    # ------------------------------------------------------------------ packed price and volume inputs
    def _pk(self, key: str, build: Callable[[], Panel]) -> Panel:
        if key not in self._packed:
            self._packed[key] = self.packer.pack(build())
        return self._packed[key]

    @property
    def o(self) -> Panel:
        return self._pk("open", lambda: self._panel("open"))

    @property
    def h(self) -> Panel:
        return self._pk("high", lambda: self._panel("high"))

    @property
    def l(self) -> Panel:  # noqa: E743 - the conventional name of the low price
        return self._pk("low", lambda: self._panel("low"))

    @property
    def c(self) -> Panel:
        return self._pk("close", lambda: self._panel("close"))

    @property
    def v(self) -> Panel:
        return self._pk("volume", lambda: self._panel("volume"))

    @property
    def amt(self) -> Panel:
        return self._pk("amount", lambda: self._panel("amount"))

    @property
    def turn(self) -> Panel:
        return self._pk("turnover", lambda: self._panel("basic_turnover_rate"))

    @property
    def ret(self) -> Panel:
        """Return between consecutive quotes of a stock."""
        if "ret" not in self._packed:
            self._packed["ret"] = self.c / self.c.shift(1) - 1.0
        return self._packed["ret"]

    def out(self, packed: Panel) -> Panel:
        return self.packer.unpack(packed)

    @staticmethod
    def ratio_p(numerator: Panel, denominator: Panel) -> Panel:
        return replace_inf(numerator / denominator.mask(denominator.abs() < 1e-12))

    @staticmethod
    def mean(panel: Panel, window: int) -> Panel:
        return panel.rolling(window, min_periods=window).mean()

    @staticmethod
    def sum(panel: Panel, window: int) -> Panel:
        return panel.rolling(window, min_periods=window).sum()

    @staticmethod
    def std(panel: Panel, window: int) -> Panel:
        return panel.rolling(window, min_periods=window).std()

    @staticmethod
    def ema(panel: Panel, span: int) -> Panel:
        """Chinese-software EMA: weight 2 / (span + 1), seeded with the first quote, published after ``span``."""
        return panel.ewm(span=span, adjust=False, min_periods=span).mean()

    def _window(self, panel: Panel, window: int, reduce: Callable[[np.ndarray], np.ndarray]) -> Panel:
        """Apply ``reduce`` to every full trailing window (rows x columns x window), in column blocks."""
        values = panel.to_numpy(dtype=float)
        result = np.full(values.shape, np.nan)
        if len(values) < window:
            return Panel(result, columns=panel.columns)
        for first in range(0, values.shape[1], COLUMN_BLOCK):
            block = values[:, first : first + COLUMN_BLOCK]
            windows = np.lib.stride_tricks.sliding_window_view(block, window, axis=0)
            complete = np.isfinite(windows).all(axis=2)
            with np.errstate(invalid="ignore", divide="ignore"):
                reduced = reduce(np.where(np.isfinite(windows), windows, 0.0))
            result[window - 1 :, first : first + COLUMN_BLOCK] = np.where(complete, reduced, np.nan)
        return Panel(result, columns=panel.columns)

    def atr(self, window: int) -> Panel:
        previous = self.c.shift(1)
        true_range = np.maximum(self.h - self.l, np.maximum((self.h - previous).abs(), (self.l - previous).abs()))
        return self.mean(true_range.where(previous.notna()), window) / self.c

    def davol(self, window: int) -> Panel:
        return self.ratio_p(self.mean(self.turn, window), self.mean(self.turn, 120))

    def wvad(self) -> Panel:
        return self.sum(self.ratio_p(self.c - self.o, self.h - self.l).fillna(0.0).where(self.c.notna()) * self.v, 24)

    def vdiff(self) -> Panel:
        return self.ema(self.v, 12) - self.ema(self.v, 26)

    def vdea(self) -> Panel:
        return self.ema(self.vdiff(), 9)

    def vr(self, window: int) -> Panel:
        change = self.c - self.c.shift(1)
        up = self.sum(self.v.where(change > 0, 0.0).where(change.notna()), window)
        down = self.sum(self.v.where(change < 0, 0.0).where(change.notna()), window)
        flat = self.sum(self.v.where(change == 0, 0.0).where(change.notna()), window)
        return 100.0 * self.ratio_p(up + 0.5 * flat, down + 0.5 * flat)

    def sharpe(self, window: int) -> Panel:
        return self.ratio_p(self.mean(self.ret, window) * 250.0 - RISK_FREE, self.std(self.ret, window) * np.sqrt(250))

    def macd(self) -> Panel:
        dif = self.ema(self.c, 12) - self.ema(self.c, 26)
        return 2.0 * (dif - self.ema(dif, 9))

    def mfi(self, window: int) -> Panel:
        typical = (self.h + self.l + self.c) / 3.0
        flow = typical * self.v
        change = typical - typical.shift(1)
        positive = self.sum(flow.where(change > 0, 0.0).where(change.notna()), window)
        negative = self.sum(flow.where(change < 0, 0.0).where(change.notna()), window)
        return 100.0 * self.ratio_p(positive, positive + negative)

    def aroon(self, panel: Panel, window: int, *, low: bool) -> Panel:
        def since_extreme(windows: np.ndarray) -> np.ndarray:
            # Searching the reversed window finds the most recent extreme and counts the days since it.
            flipped = windows[..., ::-1]
            days = np.argmin(flipped, axis=2) if low else np.argmax(flipped, axis=2)
            return (window - days) / window

        return self._window(panel, window, since_extreme)

    def bias(self, window: int) -> Panel:
        return 100.0 * (self.c / self.mean(self.c, window) - 1.0)

    def cci(self, window: int) -> Panel:
        typical = (self.h + self.l + self.c) / 3.0

        def statistic(windows: np.ndarray) -> np.ndarray:
            mean = windows.mean(axis=2)
            deviation = np.abs(windows - mean[..., None]).mean(axis=2)
            return (windows[..., -1] - mean) / np.where(deviation > 0, 0.015 * deviation, np.nan)

        return self._window(typical, window, statistic)

    def cr(self, window: int) -> Panel:
        previous = ((self.h + self.l + self.c) / 3.0).shift(1)
        strong = self.sum((self.h - previous).clip(lower=0.0), window)
        weak = self.sum((previous - self.l).clip(lower=0.0), window)
        return 100.0 * self.ratio_p(strong, weak)

    def plrc(self, window: int) -> Panel:
        days = np.arange(1, window + 1, dtype=float)
        weights = (days - days.mean()) / ((days - days.mean()) ** 2).sum()

        def slope(windows: np.ndarray) -> np.ndarray:
            mean = windows.mean(axis=2)
            return (windows @ weights) / np.where(mean != 0, mean, np.nan)

        return self._window(self.c, window, slope)

    def roc(self, window: int) -> Panel:
        return 100.0 * (self.ratio_p(self.c, self.c.shift(window)) - 1.0)

    def trix(self, window: int) -> Panel:
        triple = self.ema(self.ema(self.ema(self.c, window), window), window)
        return 100.0 * (self.ratio_p(triple, triple.shift(1)) - 1.0)

    # ------------------------------------------------------------------ statements
    @property
    def g(self) -> QuarterGrid:
        if self._grid is None:
            self._grid = QuarterGrid(prepare_fundamentals(self._panel("fundamentals")), self.dates, self.instruments)
        return self._grid

    def p(self, values: np.ndarray) -> Panel:
        return self.g.panel(values)

    def b(self, name: str) -> np.ndarray:
        """Latest quarter-end balance (or period-end) value."""
        return self.g.value(name)

    def ttm(self, name: str) -> np.ndarray:
        key = f"ttm:{name}"
        if key not in self._cache:
            self._cache[key] = self.g.ttm(name)
        return self._cache[key]

    @staticmethod
    def average(values: np.ndarray) -> np.ndarray:
        """Mean of the latest quarter and the same quarter a year earlier, or the latest alone."""
        return nan_mean(np.stack([values, lag(values, 4)], axis=2), 2)[0]

    @staticmethod
    def yoy(values: np.ndarray) -> np.ndarray:
        previous = lag(values, 4)
        return ratio(values - previous, np.abs(previous))

    @staticmethod
    def index_yoy(values: np.ndarray) -> np.ndarray:
        return ratio(values, lag(values, 4))

    @staticmethod
    def yearly(values: np.ndarray, statistic: str, years: int = 8, minimum: int = 4) -> np.ndarray:
        """Statistic over the latest quarter and the same quarter in each of the previous ``years - 1`` years."""
        stacked = np.stack([lag(values, 4 * k) for k in range(years)], axis=2)
        finite = np.isfinite(stacked)
        count = finite.sum(axis=2)
        enough = count >= minimum
        with np.errstate(invalid="ignore", divide="ignore"):
            if statistic == "max":
                result = np.where(finite, stacked, -np.inf).max(axis=2)
            else:
                mean, _ = nan_mean(stacked, 2)
                if statistic == "mean":
                    result = mean
                else:
                    squares = np.where(finite, (stacked - mean[..., None]) ** 2, 0.0).sum(axis=2)
                    result = ratio(mean, np.sqrt(squares / np.maximum(count - 1, 1)))
        return np.where(enough, result, np.nan)

    def gross_margin(self) -> np.ndarray:
        if "gross_margin" not in self._cache:
            self._cache["gross_margin"] = ratio(self.ttm("revenue") - self.ttm("oper_cost"), self.ttm("revenue"))
        return self._cache["gross_margin"]

    def roa(self) -> np.ndarray:
        return ratio(self.ttm("n_income"), self.average(self.b("total_assets")))

    def roe(self) -> np.ndarray:
        return ratio(self.ttm("n_income_attr_p"), self.average(self.b("total_hldr_eqy_exc_min_int")))

    def ebit(self) -> np.ndarray:
        return self.ttm("total_profit") + self.ttm("fin_exp")

    def value_change_profit(self) -> np.ndarray:
        return self.ttm("fv_value_chg_gain") + self.ttm("invest_income")

    def operate_net_income(self) -> np.ndarray:
        return self.ttm("operate_profit") - self.value_change_profit()

    def fixed(self) -> np.ndarray:
        return self.b("fix_assets") + self.b("const_materials") + self.b("cip")

    def other_receivables(self) -> np.ndarray:
        return self.b("jq_other_receivables")

    def retained_earnings(self) -> np.ndarray:
        return self.b("surplus_rese") + self.b("undistr_porfit")

    def interest_carry_current(self) -> np.ndarray:
        return self.b("st_borr") + self.b("trading_fl") + self.b("int_payable") + self.b("non_cur_liab_due_1y")

    def net_debt(self) -> np.ndarray:
        return self.interest_carry_current() + self.b("lt_borr") + self.b("bond_payable") - self.b("money_cap")

    def financial_assets(self) -> np.ndarray:
        return (self.b("money_cap") + self.b("trad_asset") + self.b("notes_receiv") + self.b("int_receiv")
                + self.b("div_receiv") + self.b("fa_avail_for_sale") + self.b("htm_invest"))

    def financial_liability(self) -> np.ndarray:
        return (self.b("trading_fl") + self.b("st_borr") + self.b("notes_payable") + self.b("int_payable")
                + self.b("div_payable") + self.b("non_cur_liab_due_1y") + self.b("lt_borr") + self.b("bond_payable"))

    def operating_assets(self) -> np.ndarray:
        return self.b("total_assets") - self.financial_assets()

    def operating_liability(self) -> np.ndarray:
        return self.b("total_liab") - self.financial_liability()

    def receivable_turnover(self) -> np.ndarray:
        return ratio(self.ttm("revenue"), self.average(self.b("accounts_receiv") + self.b("notes_receiv")))

    def payable_turnover(self) -> np.ndarray:
        return ratio(self.ttm("oper_cost"), self.average(self.b("acct_payable") + self.b("notes_payable")))

    def inventory_turnover(self) -> np.ndarray:
        return ratio(self.ttm("oper_cost"), self.average(self.b("inventories")))

    def blev(self) -> np.ndarray:
        equity = self.b("total_hldr_eqy_exc_min_int")
        return ratio(equity + self.b("total_ncl"), equity)

    # ------------------------------------------------------------------ cross sections (index members only)
    def cs_z(self, panel: Panel, *, clip: float = 3.0) -> Panel:
        """Standardise every date over that day's index members, truncated at ``clip`` standard deviations."""
        values = panel.where(self.trading)
        members = values.where(self.universe)
        mean = members.mean(axis=1)
        spread = members.std(axis=1)
        z = values.sub(mean, axis=0).div(spread.where(spread > 0), axis=0)
        return z.clip(-clip, clip)

    def cs_rank(self, panel: Panel) -> Panel:
        return panel.where(self.universe & self.trading).rank(axis=1, method="average", pct=True)

    def cs_residual(self, panel: Panel, *regressors: Panel) -> Panel:
        """Residual of a daily cross-sectional OLS on a constant and ``regressors``, fitted on index members."""
        y = panel.to_numpy(dtype=float)
        xs = np.stack([regressor.to_numpy(dtype=float) for regressor in regressors], axis=2)
        member = self.universe.to_numpy()
        result = np.full(y.shape, np.nan)
        for row in range(len(y)):
            usable = np.isfinite(y[row]) & np.isfinite(xs[row]).all(axis=1)
            fit = usable & member[row]
            if fit.sum() <= xs.shape[2] + 2:
                continue
            design = np.column_stack([np.ones(fit.sum()), xs[row, fit]])
            coefficients, *_ = np.linalg.lstsq(design, y[row, fit], rcond=None)
            full = np.column_stack([np.ones(usable.sum()), xs[row, usable]])
            result[row, usable] = y[row, usable] - full @ coefficients
        return Panel(result, index=self.dates, columns=self.instruments)

    def combine(self, *terms: tuple[float, Panel]) -> Panel:
        """Weighted sum of standardised descriptors, re-weighted over the descriptors a stock has (Barra)."""
        total = Panel(0.0, index=self.dates, columns=self.instruments)
        weight = Panel(0.0, index=self.dates, columns=self.instruments)
        for share, descriptor in terms:
            z = self.cs_z(descriptor)
            total = total + share * z.fillna(0.0)
            weight = weight + share * z.notna()
        return total / weight.where(weight > 0)

    # ------------------------------------------------------------------ Barra CNE5 style factors
    def _calendar_returns(self) -> np.ndarray:
        close = self._panel("close")
        return (close / close.shift(1) - 1.0).to_numpy(dtype=float)

    def _ew_regression(self) -> dict[str, np.ndarray]:
        """252-day exponentially weighted market regression (half-life 63) of every stock's daily return."""
        if "ew_regression" in self._cache:
            return self._cache["ew_regression"]
        y = self._calendar_returns()
        market = np.asarray(pd.Series(self._panel("mkt")).reindex(self.dates), dtype=float)
        x = np.repeat(market[:, None], y.shape[1], axis=1)
        mask = np.isfinite(y) & np.isfinite(x)
        y0 = np.where(mask, y, 0.0)
        x0 = np.where(mask, x, 0.0)
        w = mask.astype(float)
        sums = ew_window_sums({"w": w, "x": x0, "y": y0, "xx": x0 * x0, "xy": x0 * y0, "yy": y0 * y0}, YEAR, 63)
        n = ew_window_sums({"n": w}, YEAR, None)["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my = sums["x"] / sums["w"], sums["y"] / sums["w"]
            vx = sums["xx"] / sums["w"] - mx**2
            cov = sums["xy"] / sums["w"] - mx * my
            beta = cov / vx
            alpha = my - beta * mx
            resid = (sums["yy"] - 2 * alpha * sums["y"] - 2 * beta * sums["xy"] + alpha**2 * sums["w"]
                     + 2 * alpha * beta * sums["x"] + beta**2 * sums["xx"]) / sums["w"]
        valid = (n >= 63) & (vx > 0)
        result = {
            "beta": np.where(valid, beta, np.nan),
            "hsigma": np.where(valid & (resid > 0), np.sqrt(np.maximum(resid, 0.0)), np.nan),
        }
        self._cache["ew_regression"] = result
        return result

    def _size(self) -> Panel:
        return self.cs_z(np.log(self.cap))

    def style_size(self) -> Panel:
        return self._size()

    def style_beta(self) -> Panel:
        return self.cs_z(Panel(self._ew_regression()["beta"], index=self.dates, columns=self.instruments))

    def style_momentum(self) -> Panel:
        log_returns = np.log1p(self._calendar_returns())
        lagged = np.full(log_returns.shape, np.nan)
        lagged[21:] = log_returns[:-21]
        mask = np.isfinite(lagged)
        sums = ew_window_sums({"x": np.where(mask, lagged, 0.0)}, 504, 126)
        count = ew_window_sums({"n": mask.astype(float)}, 504, None)["n"]
        rstr = np.where(count >= 252, sums["x"], np.nan)
        return self.cs_z(Panel(rstr, index=self.dates, columns=self.instruments))

    def style_residual_volatility(self) -> Panel:
        returns = self._calendar_returns()
        mask = np.isfinite(returns)
        r0 = np.where(mask, returns, 0.0)
        sums = ew_window_sums({"w": mask.astype(float), "r": r0, "rr": r0 * r0}, YEAR, 42)
        count = ew_window_sums({"n": mask.astype(float)}, YEAR, None)["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums["r"] / sums["w"]
            dastd = np.sqrt(np.maximum(sums["rr"] / sums["w"] - mean**2, 0.0))
        dastd = np.where(count >= 63, dastd, np.nan)
        log_returns = Panel(np.log1p(returns), index=self.dates, columns=self.instruments)
        cumulative = np.stack(
            [log_returns.rolling(21 * months, min_periods=int(21 * months * 0.8)).sum().to_numpy()
             for months in range(1, 13)],
            axis=2,
        )
        with np.errstate(invalid="ignore"):
            cmra = np.log1p(cumulative.max(axis=2)) - np.log1p(cumulative.min(axis=2))
        combined = self.combine(
            (0.74, Panel(dastd, index=self.dates, columns=self.instruments)),
            (0.16, Panel(cmra, index=self.dates, columns=self.instruments)),
            (0.10, Panel(self._ew_regression()["hsigma"], index=self.dates, columns=self.instruments)),
        )
        return self.cs_z(self.cs_residual(combined, self.style_beta(), self._size()))

    def style_non_linear_size(self) -> Panel:
        size = self._size()
        return self.cs_z(self.cs_residual(size**3, size))

    def style_liquidity(self) -> Panel:
        turnover = self._panel("basic_turnover_rate") / 100.0
        turnover = turnover.where(self.trading)

        def share_turnover(window: int, months: int) -> Panel:
            total = turnover.rolling(window, min_periods=window // 2).sum() / months
            return np.log(total.where(total > 0))

        combined = self.combine(
            (0.35, share_turnover(21, 1)), (0.35, share_turnover(63, 3)), (0.30, share_turnover(YEAR, 12))
        )
        return self.cs_z(self.cs_residual(combined, self._size()))

    def style_earnings_yield(self) -> Panel:
        etop = self.p(self.ttm("n_income_attr_p")) / self.cap
        cetop = self.p(self.ttm("n_cashflow_act")) / self.cap
        return self.cs_z(self.combine((0.21, etop), (0.11, cetop)))

    def style_growth(self) -> Panel:
        shares = self.b("total_share")
        egro = annual_trend(ratio(self.g.annual("n_income_attr_p"), shares))
        sgro = annual_trend(ratio(self.g.annual("revenue"), shares))
        egro_panel = self.g.panel(egro, annual=True)
        sgro_panel = self.g.panel(sgro, annual=True)
        return self.cs_z(self.combine((0.24, egro_panel), (0.47, sgro_panel)))

    def style_leverage(self) -> Panel:
        mlev = (self.cap + self.p(self.b("total_ncl"))) / self.cap
        dtoa = self.p(ratio(self.b("total_liab"), self.b("total_assets")))
        blev = self.p(self.blev())
        return self.cs_z(self.combine((0.38, mlev), (0.35, dtoa), (0.27, blev)))


def ew_window_sums(series: Mapping[str, np.ndarray], window: int, half_life: float | None) -> dict[str, np.ndarray]:
    """Exponentially weighted sums over the trailing ``window`` rows; the newest row has weight 1.

    ``half_life=None`` gives plain window sums. Inputs must already have missing values set to zero.
    """
    decay = 1.0 if half_life is None else 0.5 ** (1.0 / half_life)
    tail = decay**window
    result = {}
    for name, values in series.items():
        out = np.empty(values.shape)
        running = np.zeros(values.shape[1:])
        for row in range(len(values)):
            running = decay * running + values[row]
            if row >= window:
                running = running - tail * values[row - window]
            out[row] = running
        result[name] = out
    return result


def annual_trend(values: np.ndarray, years: int = 5) -> np.ndarray:
    """Slope of the latest ``years`` fiscal-year values on time, over their mean absolute value (Barra EGRO/SGRO)."""
    stacked = np.stack([lag(values, 4 * k) for k in range(years - 1, -1, -1)], axis=2)
    complete = np.isfinite(stacked).all(axis=2)
    t = np.arange(years, dtype=float)
    weights = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    filled = np.where(np.isfinite(stacked), stacked, 0.0)
    slope = filled @ weights
    scale = np.abs(filled).mean(axis=2)
    return np.where(complete, ratio(slope, scale), np.nan)
