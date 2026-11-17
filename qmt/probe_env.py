#encoding:gbk
# RQuant2.0 QMT sandbox probe. READ-ONLY: this strategy never places, cancels or queries orders and never touches
# files or the network. It only prints what the built-in Python environment offers, so we can decide how much of the
# research pipeline (jq240 factors + LightGBM) can run inside QMT.
#
# How to run (see qmt/README.md):
#   1. In QMT, download daily history for all Shanghai/Shenzhen A shares (data management, supplement data).
#   2. Create a new Python strategy, paste this whole file, main chart 000300.SH, period 1d.
#   3. Press run (not live trading). Copy every log line that starts with [RQ] and send it back.
#
# The file is ASCII-only and declares GBK, the encoding QMT saves strategy files in (a UTF-8 declaration fails to compile
# there). Chinese sector names are written as unicode escapes.

import sys
import time

TAG = "[RQ]"
SECTOR_SH_A = u"\u4e0a\u8bc1A\u80a1"  # Shanghai A shares
SECTOR_HS_A = u"\u6caa\u6df1A\u80a1"  # Shanghai+Shenzhen A shares
SECTOR_CSI300 = u"\u6caa\u6df1300"  # CSI 300
BAR_FIELDS = ["open", "high", "low", "close", "volume", "amount", "preClose", "suspendFlag"]
MODULES = [
    "numpy", "pandas", "scipy", "sklearn", "statsmodels", "lightgbm", "xgboost", "talib", "numba",
    "pickle", "json", "zlib", "base64", "gzip", "bz2", "lzma", "sqlite3", "psutil", "ctypes",
]
# Candidate financial field spellings; QMT versions differ, so each is tried on its own.
FINANCIAL_FIELDS = [
    "ASHAREBALANCESHEET.tot_assets",
    "ASHAREBALANCESHEET.tot_shrhldr_eqy_excl_min_int",
    "ASHAREINCOME.oper_rev",
    "ASHAREINCOME.net_profit_excl_min_int_inc",
    "ASHARECASHFLOWSTATEMENT.net_cash_flows_oper_act",
    "Balance.tot_assets",
    "Income.revenue_inc",
    "Income.net_profit_excl_min_int_inc",
    "CashFlow.net_cash_flows_oper_act",
    "PershareIndex.s_fa_eps_basic",
    "Capital.total_capital",
    "Capital.circulating_capital",
]

_state = {"done": False}


def log(key, value=""):
    try:
        text = str(value)
    except Exception as exc:  # pragma: no cover - defensive, QMT objects can be odd
        text = "<unprintable %s>" % type(exc).__name__
    if len(text) > 600:
        text = text[:600] + "...(%d chars)" % len(text)
    print("%s %s=%s" % (TAG, key, text))


def init(ContextInfo):
    log("probe", "start")
    log("python", sys.version.replace("\n", " "))
    log("platform", sys.platform)
    log("maxsize_64bit", sys.maxsize > 2 ** 32)
    for name in MODULES:
        try:
            module = __import__(name)
            log("module." + name, getattr(module, "__version__", "ok"))
        except Exception as exc:
            log("module." + name, "MISSING %s: %s" % (type(exc).__name__, exc))
    try:
        codes = [code for code in ContextInfo.get_stock_list_in_sector(SECTOR_SH_A) if code.startswith("60")]
        ContextInfo.set_universe(codes)
        log("universe.set", len(codes))
    except Exception as exc:
        log("universe.set", "ERROR %s: %s" % (type(exc).__name__, exc))


def handlebar(ContextInfo):
    if _state["done"]:
        return
    try:
        if not ContextInfo.is_last_bar():
            return
    except Exception as exc:
        log("is_last_bar", "ERROR %s" % exc)
    _state["done"] = True
    _run(ContextInfo)
    log("probe", "end")


def _step(name, func):
    started = time.time()
    try:
        result = func()
        log(name + ".seconds", round(time.time() - started, 2))
        return result
    except Exception as exc:
        log(name, "ERROR %s: %s" % (type(exc).__name__, exc))
        return None


def _run(C):
    sh = _step("sector.sh_a", lambda: list(C.get_stock_list_in_sector(SECTOR_SH_A)))
    hs = _step("sector.hs_a", lambda: list(C.get_stock_list_in_sector(SECTOR_HS_A)))
    csi = _step("sector.csi300", lambda: list(C.get_stock_list_in_sector(SECTOR_CSI300)))
    for name, codes in (("sh_a", sh), ("hs_a", hs), ("csi300", csi)):
        if codes is not None:
            log("sector.%s.count" % name, len(codes))
            log("sector.%s.sample" % name, codes[:5])
    sh_main = [code for code in (sh or []) if code.startswith("60")]
    log("sh_main.count", len(sh_main))
    sample = sh_main[:5] or ["600000.SH", "600519.SH"]

    _probe_history_data(C)
    _probe_bars(C, sample, sh_main)
    _probe_instrument(C, sample)
    _probe_financials(C, sample, sh_main)
    _probe_history_on_date(C, sample)
    _probe_scoring_speed(len(sh_main) or 1700)
    _probe_memory()


def _probe_history_data(C):
    # get_history_data returns plain dicts of lists, so it may work when pandas is not installed.
    for field in ("close", "open", "high", "low", "volume", "amount"):
        for dividend in (0, 1, 2):  # none, front, back (QMT numeric codes)
            label = "hist.%s.div%d" % (field, dividend)
            data = _step(label, lambda f=field, d=dividend: C.get_history_data(320, "1d", f, d))
            if not data:
                continue
            lengths = sorted(len(values) for values in data.values())
            log(label + ".codes", len(data))
            log(label + ".len_min_median_max", (lengths[0], lengths[len(lengths) // 2], lengths[-1]))
            first = sorted(data)[0]
            log(label + "." + first + ".tail", list(data[first])[-3:])
            if field != "close":
                break


def _probe_bars(C, sample, universe):
    for dividend in ("front", "back", "front_ratio", "none"):
        data = _step(
            "bars.%s" % dividend,
            lambda d=dividend: C.get_market_data_ex(
                BAR_FIELDS, sample[:2], period="1d", count=5, dividend_type=d, subscribe=False
            ),
        )
        if data:
            first = sorted(data)[0]
            frame = data[first]
            log("bars.%s.columns" % dividend, list(frame.columns))
            log("bars.%s.%s.tail" % (dividend, first), frame.tail(2).to_dict("records"))
    if not universe:
        return
    full = _step(
        "bars.universe_320d",
        lambda: C.get_market_data_ex(
            BAR_FIELDS, universe, period="1d", count=320, dividend_type="back", subscribe=False
        ),
    )
    if full:
        lengths = sorted(len(frame) for frame in full.values())
        log("bars.universe_320d.codes_returned", len(full))
        log("bars.universe_320d.rows_min_median_max", (lengths[0], lengths[len(lengths) // 2], lengths[-1]))
        last_index = [frame.index[-1] for frame in full.values() if len(frame)]
        if last_index:
            log("bars.universe_320d.latest_bar", max(last_index))
    long_hist = _step(
        "bars.one_stock_since_2010",
        lambda: C.get_market_data_ex(
            ["close"], [universe[0]], period="1d", start_time="20100101", end_time="", dividend_type="back",
            subscribe=False,
        ),
    )
    if long_hist:
        frame = list(long_hist.values())[0]
        log("bars.one_stock_since_2010.rows", len(frame))
        if len(frame):
            log("bars.one_stock_since_2010.first", frame.index[0])


def _probe_instrument(C, sample):
    for code in sample[:2]:
        detail = _step("instrument." + code, lambda c=code: C.get_instrumentdetail(c))
        if detail:
            keys = sorted(detail) if hasattr(detail, "keys") else dir(detail)
            log("instrument.%s.keys" % code, keys)
            for key in ("InstrumentName", "OpenDate", "FloatVolume", "TotalVolume", "UpStopPrice", "DownStopPrice",
                        "PreClose", "InstrumentStatus"):
                if hasattr(detail, "get") and key in detail:
                    log("instrument.%s.%s" % (code, key), detail[key])
    for name in ("get_turnover_rate", "get_total_share", "get_float_caps", "get_divid_factors",
                 "get_his_st_data", "get_sector", "get_weight_in_index", "get_raw_financial_data",
                 "get_financial_data", "get_factor_data", "get_his_index_data", "get_full_tick"):
        log("api." + name, hasattr(C, name))
    if hasattr(C, "get_turnover_rate"):
        _step("turnover", lambda: log("turnover.sample", C.get_turnover_rate(sample[:2], "20260901", "20261031")))
    if hasattr(C, "get_his_st_data"):
        _step("st_history", lambda: log("st_history.sample", C.get_his_st_data(sample[0])))
    if hasattr(C, "get_sector"):
        _step("csi300_members", lambda: log("csi300_members.count", len(C.get_sector("000300.SH"))))


def _describe(label, data):
    log(label + ".type", type(data).__name__)
    shape = getattr(data, "shape", None)
    if shape is not None:
        log(label + ".shape", shape)
    try:
        text = data.tail(3) if hasattr(data, "tail") else data
        log(label + ".value", str(text).replace("\n", " | "))
    except Exception as exc:
        log(label + ".value", "unprintable %s: %r" % (type(exc).__name__, exc))


def _probe_financials(C, sample, universe):
    codes = sample[:2]
    for field in FINANCIAL_FIELDS:
        for report_type in ("announce_time", "report_time"):
            label = "fin.%s.%s" % (field, report_type)
            data = _step(
                label,
                lambda f=field, r=report_type: C.get_financial_data([f], codes, "20240101", "20261031", r),
            )
            if data is not None:
                _describe(label, data)
                break
    single = _step(
        "fin.single",
        lambda: C.get_financial_data(["ASHAREINCOME.net_profit_excl_min_int_inc"], [codes[0]], "20200101", "20261031",
                                     "announce_time"),
    )
    if single is not None:
        _describe("fin.single", single)
    if hasattr(C, "get_raw_financial_data"):
        fields = ["ASHAREINCOME.net_profit_excl_min_int_inc", "ASHAREBALANCESHEET.tot_assets"]
        raw = _step("fin.raw", lambda: C.get_raw_financial_data(fields, codes, "20200101", "20261031"))
        if raw is not None:
            log("fin.raw.value", raw)
        if universe:
            full = _step(
                "fin.raw_universe",
                lambda: C.get_raw_financial_data(fields, universe, "20200101", "20261031"),
            )
            if full is not None:
                filled = [code for code in full if any(full[code].get(f) for f in fields)]
                log("fin.raw_universe.codes_with_data", (len(filled), len(full)))


def _probe_history_on_date(C, sample):
    # Does the data carry a usable trading calendar back to 2010 for warm-up windows?
    data = _step(
        "calendar.index",
        lambda: C.get_market_data_ex(["close"], ["000300.SH"], period="1d", start_time="20100101", end_time="",
                                     dividend_type="none", subscribe=False),
    )
    if data:
        frame = list(data.values())[0]
        log("calendar.rows", len(frame))


def _probe_scoring_speed(n_stocks):
    # Pure-numpy evaluation of a synthetic gradient-boosted forest the size a jq240 LightGBM model usually has. It
    # tells us whether an embedded model can score the whole universe inside handlebar in time.
    try:
        import numpy as np
    except Exception as exc:
        log("score_speed", "numpy missing: %s" % exc)
        return
    rng = np.random.RandomState(0)
    n_trees, depth, n_features = 500, 6, 240
    nodes = 2 ** depth - 1
    feature = rng.randint(0, n_features, size=(n_trees, nodes))
    threshold = rng.randn(n_trees, nodes)
    leaves = rng.randn(n_trees, 2 ** depth)
    x = rng.randn(n_stocks, n_features)
    started = time.time()
    score = np.zeros(n_stocks)
    rows = np.arange(n_stocks)
    for tree in range(n_trees):
        node = np.zeros(n_stocks, dtype=np.int64)
        for _ in range(depth):
            go_right = x[rows, feature[tree, node]] > threshold[tree, node]
            node = 2 * node + 1 + go_right
        score += leaves[tree, node - nodes]
    log("score_speed.trees_x_stocks", (n_trees, n_stocks))
    log("score_speed.seconds", round(time.time() - started, 3))


def _probe_memory():
    try:
        import psutil

        info = psutil.Process().memory_info()
        log("memory.rss_mb", round(info.rss / 2 ** 20))
        log("memory.system_available_mb", round(psutil.virtual_memory().available / 2 ** 20))
    except Exception as exc:
        log("memory", "unavailable %s" % exc)
