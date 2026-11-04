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


def run_parity(C):
    began = time.time()
    inputs = build_inputs(C, AS_OF, DAYS, log)
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
