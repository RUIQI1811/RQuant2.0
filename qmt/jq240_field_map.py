#encoding:gbk
# Tushare statement fields used by RQuant (rquant.data.contracts.FUNDAMENTAL_ENDPOINT_FIELDS) mapped to the QMT
# built-in financial tables. Table and field names come from the QMT client's own definitions
# (config/shareincome_new_1.ini, sharebalance_new_1.ini, cashflow_new_1.ini, captial_structure_1.ini).
#
# Read them inside QMT with ContextInfo.get_financial_data([...], codes, start, end, "announce_time"), which is
# point-in-time: a value appears on the first trading day after its announcement ("report_time" leaks future reports).
# A mapping of None means QMT has no matching line; the jq240 port treats it as zero, as RQuant's ZERO_FILL does for
# lines a published statement leaves blank.

INCOME = "ASHAREINCOME"
BALANCE = "ASHAREBALANCESHEET"
CASHFLOW = "ASHARECASHFLOW"
CAPITAL = "CAPITALSTRUCTURE"

STATEMENT_FIELDS = {
    "income": {
        "total_revenue": INCOME + ".revenue",
        "revenue": INCOME + ".revenue_inc",
        "oper_cost": INCOME + ".total_expense",
        "sell_exp": INCOME + ".sale_expense",
        "admin_exp": INCOME + ".less_gerl_admin_exp",
        "fin_exp": INCOME + ".financial_expense",
        "rd_exp": INCOME + ".research_expenses",
        "operate_profit": INCOME + ".oper_profit",
        "total_profit": INCOME + ".tot_profit",
        "income_tax": INCOME + ".inc_tax",
        "n_income": INCOME + ".net_profit_incl_min_int_inc",
        "n_income_attr_p": INCOME + ".net_profit_excl_min_int_inc",
        "total_cogs": INCOME + ".total_operating_cost",
        "biz_tax_surchg": INCOME + ".less_taxes_surcharges_ops",
        "assets_impair_loss": INCOME + ".less_impair_loss_assets",
        "fv_value_chg_gain": INCOME + ".change_income_fair_value",
        "invest_income": INCOME + ".plus_net_invest_inc",
        "ass_invest_income": INCOME + ".incl_inc_invest_assoc_jv_entp",
        "non_oper_income": INCOME + ".plus_non_oper_rev",
        "non_oper_exp": INCOME + ".less_non_oper_exp",
        # QMT has no interest split of the finance expense (less_int_exp / int_inc are bank lines), so
        # net_interest_expense uses the whole finance expense.
        "fin_exp_int_exp": INCOME + ".financial_expense",
        "fin_exp_int_inc": None,
    },
    "balancesheet": {
        "total_share": CAPITAL + ".total_capital",
        "money_cap": BALANCE + ".cash_equivalents",
        "trad_asset": BALANCE + ".tradable_fin_assets",
        "accounts_receiv": BALANCE + ".account_receivable",
        "inventories": BALANCE + ".inventories",
        "total_cur_assets": BALANCE + ".total_current_assets",
        "lt_eqt_invest": BALANCE + ".long_term_eqy_invest",
        "fix_assets": BALANCE + ".fix_assets",
        "intan_assets": BALANCE + ".intang_assets",
        "goodwill": BALANCE + ".goodwill",
        "total_assets": BALANCE + ".tot_assets",
        "st_borr": BALANCE + ".shortterm_loan",
        "non_cur_liab_due_1y": BALANCE + ".non_current_liability_in_one_year",
        "total_cur_liab": BALANCE + ".total_current_liability",
        "lt_borr": BALANCE + ".long_term_loans",
        "bond_payable": BALANCE + ".bonds_payable",
        "total_liab": BALANCE + ".tot_liab",
        "minority_int": BALANCE + ".minority_int",
        "oth_eqt_tools_p_shr": None,
        "total_hldr_eqy_exc_min_int": BALANCE + ".tot_shrhldr_eqy_excl_min_int",
        "total_hldr_eqy_inc_min_int": BALANCE + ".total_equity",
        "notes_receiv": BALANCE + ".bill_receivable",
        "oth_receiv": BALANCE + ".other_receivable",
        "oth_rcv_total": BALANCE + ".other_receivable",
        "prepayment": BALANCE + ".advance_payment",
        "int_receiv": BALANCE + ".int_rcv",
        # QMT names the dividends-receivable line "dividends_payable" (Chinese label: dividends receivable).
        "div_receiv": BALANCE + ".dividends_payable",
        "fa_avail_for_sale": BALANCE + ".fin_assets_avail_for_sale",
        "htm_invest": BALANCE + ".held_to_mty_invest",
        "cip": BALANCE + ".constru_in_process",
        "const_materials": BALANCE + ".construction_materials",
        "r_and_d": BALANCE + ".development_expenditure",
        "lt_amor_exp": BALANCE + ".long_deferred_expense",
        "total_nca": BALANCE + ".total_non_current_assets",
        "trading_fl": BALANCE + ".tradable_fin_liab",
        "notes_payable": BALANCE + ".notes_payable",
        "acct_payable": BALANCE + ".accounts_payable",
        "adv_receipts": BALANCE + ".advance_peceipts",
        "int_payable": BALANCE + ".int_payable",
        "div_payable": BALANCE + ".dividend_payable",
        "total_ncl": BALANCE + ".non_current_liabilities",
        "cap_rese": BALANCE + ".cap_rsrv",
        "surplus_rese": BALANCE + ".surplus_rsrv",
        "undistr_porfit": BALANCE + ".undistributed_profit",
        "taxes_payable": BALANCE + ".taxes_surcharges_payable",
        "defer_tax_liab": BALANCE + ".deferred_tax_liab",
    },
    "cashflow": {
        "n_cashflow_act": CASHFLOW + ".net_cash_flows_oper_act",
        "c_pay_acq_const_fiolta": CASHFLOW + ".cash_pay_acq_const_fiolta",
        "n_incr_cash_cash_equ": CASHFLOW + ".net_incr_cash_cash_equ",
        "depr_fa_coga_dpba": CASHFLOW + ".depr_fa_coga_dpba",
        "amort_intang_assets": CASHFLOW + ".amort_intang_assets",
        "lt_amort_deferred_exp": CASHFLOW + ".amort_lt_deferred_exp",
        "c_fr_sale_sg": CASHFLOW + ".goods_sale_and_service_render_cash",
        "n_cashflow_inv_act": CASHFLOW + ".net_cash_flows_inv_act",
        "n_cash_flows_fnc_act": CASHFLOW + ".net_cash_flows_fnc_act",
        "c_cash_equ_end_period": CASHFLOW + ".cash_cash_equ_end_period",
        "c_recp_cap_contrib": CASHFLOW + ".cash_recp_cap_contrib",
        "c_recp_borrow": CASHFLOW + ".cash_recp_borrow",
        "proc_issue_bonds": CASHFLOW + ".proc_issue_bonds",
        "c_prepay_amt_borr": CASHFLOW + ".cash_prepay_amt_borr",
    },
}

# jq240's daily_basic inputs and where QMT has them.
DAILY_INPUTS = {
    "close": "get_market_data_ex close, dividend_type none (unadjusted)",
    "turnover_rate": "get_turnover_rate, a fraction; x100 to match Tushare percent",
    "total_share": CAPITAL + ".total_capital (announce_time)",
    "circ_mv": "unadjusted close x " + CAPITAL + ".circulating_capital (announce_time)",
    "pe_ttm": "total market value / trailing twelve-month n_income_attr_p",
    "mkt": "value-weighted daily return of all A shares (sector u'\\u6caa\\u6df1A\\u80a1')",
}
