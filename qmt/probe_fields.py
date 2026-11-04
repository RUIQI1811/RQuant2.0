#encoding:gbk
# RQuant2.0 QMT field-coverage probe. READ-ONLY: no orders, no files, no network.
# Checks that every QMT financial field jq240 needs (qmt/jq240_field_map.py) has data for the Shanghai main board.
# Generated from qmt/jq240_field_map.py; regenerate when the map changes.
import time

TAG = "[RQ]"
SECTOR_SH_A = u"\u4e0a\u8bc1A\u80a1"
FIELDS = [
    "ASHAREBALANCESHEET.account_receivable",
    "ASHAREBALANCESHEET.accounts_payable",
    "ASHAREBALANCESHEET.advance_payment",
    "ASHAREBALANCESHEET.advance_peceipts",
    "ASHAREBALANCESHEET.bill_receivable",
    "ASHAREBALANCESHEET.bonds_payable",
    "ASHAREBALANCESHEET.cap_rsrv",
    "ASHAREBALANCESHEET.cash_equivalents",
    "ASHAREBALANCESHEET.constru_in_process",
    "ASHAREBALANCESHEET.construction_materials",
    "ASHAREBALANCESHEET.deferred_tax_liab",
    "ASHAREBALANCESHEET.development_expenditure",
    "ASHAREBALANCESHEET.dividend_payable",
    "ASHAREBALANCESHEET.dividends_payable",
    "ASHAREBALANCESHEET.fin_assets_avail_for_sale",
    "ASHAREBALANCESHEET.fix_assets",
    "ASHAREBALANCESHEET.goodwill",
    "ASHAREBALANCESHEET.held_to_mty_invest",
    "ASHAREBALANCESHEET.int_payable",
    "ASHAREBALANCESHEET.int_rcv",
    "ASHAREBALANCESHEET.intang_assets",
    "ASHAREBALANCESHEET.inventories",
    "ASHAREBALANCESHEET.long_deferred_expense",
    "ASHAREBALANCESHEET.long_term_eqy_invest",
    "ASHAREBALANCESHEET.long_term_loans",
    "ASHAREBALANCESHEET.minority_int",
    "ASHAREBALANCESHEET.non_current_liabilities",
    "ASHAREBALANCESHEET.non_current_liability_in_one_year",
    "ASHAREBALANCESHEET.notes_payable",
    "ASHAREBALANCESHEET.other_receivable",
    "ASHAREBALANCESHEET.shortterm_loan",
    "ASHAREBALANCESHEET.surplus_rsrv",
    "ASHAREBALANCESHEET.taxes_surcharges_payable",
    "ASHAREBALANCESHEET.tot_assets",
    "ASHAREBALANCESHEET.tot_liab",
    "ASHAREBALANCESHEET.tot_shrhldr_eqy_excl_min_int",
    "ASHAREBALANCESHEET.total_current_assets",
    "ASHAREBALANCESHEET.total_current_liability",
    "ASHAREBALANCESHEET.total_equity",
    "ASHAREBALANCESHEET.total_non_current_assets",
    "ASHAREBALANCESHEET.tradable_fin_assets",
    "ASHAREBALANCESHEET.tradable_fin_liab",
    "ASHAREBALANCESHEET.undistributed_profit",
    "ASHARECASHFLOW.amort_intang_assets",
    "ASHARECASHFLOW.amort_lt_deferred_exp",
    "ASHARECASHFLOW.cash_cash_equ_end_period",
    "ASHARECASHFLOW.cash_pay_acq_const_fiolta",
    "ASHARECASHFLOW.cash_prepay_amt_borr",
    "ASHARECASHFLOW.cash_recp_borrow",
    "ASHARECASHFLOW.cash_recp_cap_contrib",
    "ASHARECASHFLOW.depr_fa_coga_dpba",
    "ASHARECASHFLOW.goods_sale_and_service_render_cash",
    "ASHARECASHFLOW.net_cash_flows_fnc_act",
    "ASHARECASHFLOW.net_cash_flows_inv_act",
    "ASHARECASHFLOW.net_cash_flows_oper_act",
    "ASHARECASHFLOW.net_incr_cash_cash_equ",
    "ASHARECASHFLOW.proc_issue_bonds",
    "ASHAREINCOME.change_income_fair_value",
    "ASHAREINCOME.financial_expense",
    "ASHAREINCOME.inc_tax",
    "ASHAREINCOME.incl_inc_invest_assoc_jv_entp",
    "ASHAREINCOME.less_gerl_admin_exp",
    "ASHAREINCOME.less_impair_loss_assets",
    "ASHAREINCOME.less_non_oper_exp",
    "ASHAREINCOME.less_taxes_surcharges_ops",
    "ASHAREINCOME.net_profit_excl_min_int_inc",
    "ASHAREINCOME.net_profit_incl_min_int_inc",
    "ASHAREINCOME.oper_profit",
    "ASHAREINCOME.plus_net_invest_inc",
    "ASHAREINCOME.plus_non_oper_rev",
    "ASHAREINCOME.research_expenses",
    "ASHAREINCOME.revenue",
    "ASHAREINCOME.revenue_inc",
    "ASHAREINCOME.sale_expense",
    "ASHAREINCOME.tot_profit",
    "ASHAREINCOME.total_expense",
    "ASHAREINCOME.total_operating_cost",
    "CAPITALSTRUCTURE.total_capital",
]
_state = {"done": False}


def log(key, value=""):
    text = str(value).replace("\n", " | ")
    if len(text) > 700:
        text = text[:700] + "...(%d chars)" % len(text)
    print("%s %s=%s" % (TAG, key, text))


def init(C):
    log("fieldprobe", "start")
    C.set_universe([c for c in C.get_stock_list_in_sector(SECTOR_SH_A) if c.startswith("60")])


def handlebar(C):
    if _state["done"] or not C.is_last_bar():
        return
    _state["done"] = True
    universe = [c for c in C.get_stock_list_in_sector(SECTOR_SH_A) if c.startswith("60")]
    log("universe", len(universe))
    missing = []
    for start in range(0, len(FIELDS), 10):
        chunk = FIELDS[start:start + 10]
        began = time.time()
        try:
            raw = C.get_raw_financial_data(chunk, universe, "20230101", "20261103")
        except Exception as exc:
            log("chunk%d" % start, "ERROR %s: %r" % (type(exc).__name__, exc))
            continue
        log("chunk%d.seconds" % start, round(time.time() - began, 2))
        for field in chunk:
            filled = sum(1 for code in raw if raw[code].get(field))
            log("cover." + field, filled)
            if filled < len(universe) // 2:
                missing.append(field)
    log("weak_fields", missing)
    sample = C.get_raw_financial_data(["CAPITALSTRUCTURE.total_capital"], ["600519.SH"], "20150101", "20261103")
    log("capital.600519", sample)
    began = time.time()
    turnover = C.get_turnover_rate(universe, "20260901", "20261103")
    log("turnover.universe.seconds", round(time.time() - began, 2))
    log("turnover.universe.shape", getattr(turnover, "shape", None))
    for index in ("000985.CSI", "000002.SH", "000300.SH"):
        try:
            bars = C.get_market_data_ex(["close"], [index], period="1d", count=5, dividend_type="none", subscribe=False)
            log("index." + index, bars[index]["close"].tolist() if index in bars else bars)
        except Exception as exc:
            log("index." + index, "ERROR %s: %r" % (type(exc).__name__, exc))
    log("fieldprobe", "end")
