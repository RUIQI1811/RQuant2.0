# jq240 parity run: computes every jq240 factor for sh_main as of AS_OF inside QMT and prints the last day's values,
# to compare with the factor store RQuant built from Tushare on Windows. READ-ONLY: no orders, no files, no network.
import time
import traceback

AS_OF = "20261021"
DAYS = 1100
PER_LINE = 200
TAG = "[RQ]"
_state = {"done": False}


def log(key, value=""):
    print("%s %s=%s" % (TAG, key, value))


def _number(value):
    return "" if value != value else "%.7g" % value


DIAGNOSE = ["600004.SH", "600006.SH", "600009.SH", "600010.SH"]


def _dated(values):
    return dict((ms_to_date(k).strftime("%Y%m%d"), v) for k, v in sorted((values or {}).items()))


def diagnose(C, inputs):
    """Print what the parity comparison needs to explain differences: mkt, turnover and statement versions."""
    log("diag.mkt_tail", inputs["mkt"].tail(5).round(6).to_dict())
    probe = ["000001.SZ", "300750.SZ", "002594.SZ", "688981.SH", "600000.SH"]
    bars = C.get_market_data_ex(["close", "volume"], probe, period="1d", count=3, dividend_type="none",
                                fill_data=False, subscribe=False)
    log("diag.market_bars", dict((code, len(bars.get(code, []))) for code in probe))
    log("diag.market_floats", dict((code, instrument_info(C, code)["float"]) for code in probe))
    every = [code for code in C.get_stock_list_in_sector(SECTOR_ALL_A) if not code.endswith(".BJ")]
    log("diag.market_codes", "%d all-A, %d SZ, %d 688" % (len(every), sum(c.endswith(".SZ") for c in every),
                                                         sum(c.startswith("688") for c in every)))
    tail = inputs["close"].index[-3:]
    qmt = qmt_turnover_panel(C, DIAGNOSE, tail)
    log("diag.turnover_qmt", qmt.round(4).to_dict())
    log("diag.turnover_used", inputs["basic_turnover_rate"].reindex(index=tail, columns=DIAGNOSE).round(4).to_dict())
    fields = []
    for table in (INCOME, BALANCE, CASHFLOW):
        fields += [table + ".m_anntime", table + ".m_timetag"]
    fields += [INCOME + ".revenue_inc", BALANCE + ".tot_shrhldr_eqy_excl_min_int", CASHFLOW + ".net_cash_flows_oper_act"]
    for mode in ("report_time", "announce_time"):
        raw = C.get_raw_financial_data(fields, DIAGNOSE, "20240101", AS_OF, mode) or {}
        for code in DIAGNOSE:
            for field in fields:
                log("diag.%s.%s.%s" % (mode, code, field), _dated((raw.get(code) or {}).get(field)))


def run_parity(C):
    began = time.time()
    inputs = build_inputs(C, AS_OF, DAYS, log)
    try:
        diagnose(C, inputs)
    except Exception:
        log("diag.error", traceback.format_exc())
    calculator = JQFactors(inputs)
    member = inputs["universe"].iloc[-1]
    codes = [code for code, inside in zip(member.index, member.values) if inside]
    log("parity.date", inputs["close"].index[-1].strftime("%Y%m%d"))
    for first in range(0, len(codes), PER_LINE):
        log("parity.codes.%d" % (first // PER_LINE), ",".join(codes[first:first + PER_LINE]))
    for ordinal, name in enumerate(JQ240_NAMES, 1):
        key = "parity.jq_%03d" % ordinal
        try:
            values = calculator.calculate(name).iloc[-1].reindex(codes).values
        except Exception as exc:
            log(key, "ERROR %s: %r" % (type(exc).__name__, exc))
            continue
        for first in range(0, len(codes), PER_LINE):
            log("%s.%d" % (key, first // PER_LINE), ",".join(_number(v) for v in values[first:first + PER_LINE]))
    log("parity.seconds", round(time.time() - began, 2))


def init(C):
    log("parity", "start")


def handlebar(C):
    if _state["done"] or not C.is_last_bar():
        return
    _state["done"] = True
    try:
        run_parity(C)
    except Exception:
        log("parity.error", traceback.format_exc())
    log("parity", "end")
