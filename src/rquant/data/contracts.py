from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Suspensions need no endpoint of their own: Tushare `daily` has no row for a suspended day, and the Qlib provider
# turns that gap into NaN quotes, which Qlib treats as untradable.
RAW_DAILY_ENDPOINTS = ("daily", "adj_factor", "daily_basic", "stk_limit")

# Financial statements are fetched per stock by announcement-date window (the batch *_vip endpoints need more Tushare
# points). Explicit field lists keep the schema stable; every statement carries the point-in-time keys first.
STATEMENT_KEY_FIELDS = ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "update_flag")
FUNDAMENTAL_ENDPOINT_FIELDS = {
    "income": (
        *STATEMENT_KEY_FIELDS,
        "total_revenue",
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
        # jq240 (JoinQuant factor library) items.
        "total_cogs",
        "biz_tax_surchg",
        "assets_impair_loss",
        "fv_value_chg_gain",
        "invest_income",
        "ass_invest_income",
        "non_oper_income",
        "non_oper_exp",
        "fin_exp_int_exp",
        "fin_exp_int_inc",
    ),
    "balancesheet": (
        *STATEMENT_KEY_FIELDS,
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
        # jq240 (JoinQuant factor library) items.
        "notes_receiv",
        "oth_receiv",
        "oth_rcv_total",
        "prepayment",
        "int_receiv",
        "div_receiv",
        "fa_avail_for_sale",
        "htm_invest",
        "cip",
        "const_materials",
        "r_and_d",
        "lt_amor_exp",
        "total_nca",
        "trading_fl",
        "notes_payable",
        "acct_payable",
        "adv_receipts",
        "int_payable",
        "div_payable",
        "total_ncl",
        "cap_rese",
        "surplus_rese",
        "undistr_porfit",
    ),
    "cashflow": (
        *STATEMENT_KEY_FIELDS,
        "n_cashflow_act",
        "c_pay_acq_const_fiolta",
        "n_incr_cash_cash_equ",
        "depr_fa_coga_dpba",
        "amort_intang_assets",
        "lt_amort_deferred_exp",
        # jq240 (JoinQuant factor library) items.
        "c_fr_sale_sg",
        "n_cashflow_inv_act",
        "n_cash_flows_fnc_act",
        "c_cash_equ_end_period",
    ),
}
FUNDAMENTAL_ENDPOINTS = tuple(FUNDAMENTAL_ENDPOINT_FIELDS)

CANONICAL_BAR_COLUMNS = (
    "instrument",
    "datetime",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_pre_close",
    "raw_volume",
    "raw_amount",
    "raw_vwap",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "factor",
    "change",
    "cap",
    "sector",
    "industry",
    "subindustry",
    "limit_buy",
    "limit_sell",
    "in_csi300",
    # Point-in-time ST/*ST/delisting name (Tushare namechange); null when the raw sync has no namechange history.
    "is_st",
    # Trading days since listing, counting the listing day as 1.
    "listed_days",
)


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
