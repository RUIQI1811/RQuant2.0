#encoding:gbk
# RQuant2.0 QMT data-format probe. READ-ONLY: no orders, no files, no network.
# Shows the exact shapes the jq240 strategy adapter must parse and times a full-size data load.
import time

TAG = "[RQ]"
SECTOR_SH_A = u"\u4e0a\u8bc1A\u80a1"
SECTOR_HS_A = u"\u6caa\u6df1A\u80a1"
SECTOR_CSI300 = u"\u6caa\u6df1300"
_state = {"done": False}


def log(key, value=""):
    text = str(value).replace("\n", " | ")
    if len(text) > 900:
        text = text[:900] + "...(%d chars)" % len(text)
    print("%s %s=%s" % (TAG, key, text))


def step(name, func):
    began = time.time()
    try:
        result = func()
        log(name + ".seconds", round(time.time() - began, 2))
        return result
    except Exception as exc:
        log(name, "ERROR %s: %r" % (type(exc).__name__, exc))
        return None


def init(C):
    log("formatprobe", "start")
    C.set_universe([c for c in C.get_stock_list_in_sector(SECTOR_SH_A) if c.startswith("60")])


def handlebar(C):
    if _state["done"] or not C.is_last_bar():
        return
    _state["done"] = True
    two = ["600519.SH", "600000.SH"]
    for table in ("ASHAREINCOME", "ASHAREBALANCESHEET", "ASHARECASHFLOW", "CAPITALSTRUCTURE"):
        fields = [table + ".m_anntime", table + ".m_timetag"]
        fields.append(table + (".total_capital" if table == "CAPITALSTRUCTURE" else ".m_anntime"))
        raw = step("raw.%s" % table, lambda f=fields: C.get_raw_financial_data(f, two, "20230101", "20261103"))
        log("raw.%s.value" % table, raw)
    for mode in ("report_time", "announce_time"):
        raw = step("mode.%s" % mode, lambda m=mode: C.get_raw_financial_data(
            ["ASHAREINCOME.revenue", "CAPITALSTRUCTURE.total_capital"], ["600519.SH"], "20240101", "20261103", m))
        log("mode.%s.value" % mode, raw)
    t = step("turnover", lambda: C.get_turnover_rate(two, "20261027", "20261103"))
    log("turnover.value", t)
    log("turnover.index", list(t.index) if t is not None else None)
    first = step("back.first", lambda: C.get_market_data_ex(
        ["close", "volume", "suspendFlag"], two, period="1d", start_time="20080101", end_time="20100131",
        dividend_type="back", subscribe=False))
    if first:
        for code in two:
            frame = first[code]
            log("back.first.%s" % code, (len(frame), list(frame.index[:2]), frame.head(2).values.tolist()))
    log("divid.600519", step("divid", lambda: C.get_divid_factors("600519.SH")))
    log("csi300.count", len(C.get_stock_list_in_sector(SECTOR_CSI300)))
    hs = C.get_stock_list_in_sector(SECTOR_HS_A)
    sz = [c for c in hs if c.endswith(".SZ")][:3]
    bars = step("sz.bars", lambda: C.get_market_data_ex(["close"], sz, period="1d", count=3, dividend_type="none",
                                                     subscribe=False))
    log("sz.bars.value", {k: v["close"].tolist() for k, v in bars.items()} if bars else bars)
    log("sz.instrument", C.get_instrumentdetail(sz[0]) if sz else None)
    universe = [c for c in C.get_stock_list_in_sector(SECTOR_SH_A) if c.startswith("60")]
    fields = ["open", "high", "low", "close", "volume", "amount", "suspendFlag"]
    full = step("load.back", lambda: C.get_market_data_ex(fields, universe, period="1d", count=600,
                                                       dividend_type="back", subscribe=False))
    raw_full = step("load.none", lambda: C.get_market_data_ex(["close", "volume"], universe, period="1d", count=600,
                                                           dividend_type="none", subscribe=False))
    if full:
        code = universe[0]
        log("load.back.%s.tail" % code, full[code].tail(2).values.tolist())
        suspended = sum(int((frame["suspendFlag"] > 0).sum()) for frame in full.values())
        log("load.back.suspended_rows", suspended)
    fin = step("load.fin", lambda: C.get_raw_financial_data(
        ["ASHAREINCOME.revenue", "ASHAREINCOME.m_anntime", "ASHAREBALANCESHEET.tot_assets",
         "ASHARECASHFLOW.net_cash_flows_oper_act", "CAPITALSTRUCTURE.total_capital"],
        universe, "20150101", "20261103"))
    if fin:
        log("load.fin.600519", fin.get("600519.SH"))
    log("formatprobe", "end")
