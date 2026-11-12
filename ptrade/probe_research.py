# RQuant2.0 PTrade probe, run in PTrade's research (Jupyter) module. READ-ONLY: it never places or queries orders.
# It prints what the PTrade Python environment and data API offer, so the jq240 + LightGBM strategy can be ported
# from QMT (qmt/) to PTrade.
#
# How to run: PTrade -> research -> new notebook -> paste this whole file into one cell -> run. It takes a few
# minutes (PTrade research forbids two standard modules, so this file never imports them). Copy everything it
# prints (or the file probe_result.txt it writes next to the notebook) and send it back.

import builtins
import platform
import time
import traceback

OUT = []


def say(key, value=""):
    try:
        text = str(value)
    except Exception as exc:
        text = "<unprintable %s>" % type(exc).__name__
    text = text.replace("\n", " | ")
    if len(text) > 900:
        text = text[:900] + "...(%d chars)" % len(text)
    line = "[RQ] %s=%s" % (key, text)
    OUT.append(line)
    print(line)


def attempt(key, fn):
    started = time.time()
    try:
        value = fn()
        say(key + ".seconds", "%.2f" % (time.time() - started))
        return value
    except Exception as exc:
        say(key + ".error", "%s: %s" % (type(exc).__name__, exc))
        say(key + ".trace", traceback.format_exc()[-600:])
        return None


def shape(value):
    if value is None:
        return "None"
    if hasattr(value, "shape"):
        return "%s %s" % (type(value).__name__, value.shape)
    if isinstance(value, dict):
        return "dict len %d" % len(value)
    return "%s len %s" % (type(value).__name__, len(value) if hasattr(value, "__len__") else "?")


# 1. Python and libraries ---------------------------------------------------------------------------------------
say("probe", "start " + time.strftime("%Y-%m-%d %H:%M:%S"))
say("python", platform.python_version() + " " + platform.machine())
for name in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "lightgbm", "xgboost", "talib", "numba",
             "bottleneck", "numexpr", "pickle", "json", "zlib", "gzip", "lzma", "sqlite3", "tables", "h5py"]:
    try:
        module = __import__(name)
        say("lib." + name, getattr(module, "__version__", "ok"))
    except Exception as exc:
        say("lib." + name, "MISSING %s" % type(exc).__name__)
for name in ["get_research_path", "get_price", "get_history", "get_fundamentals", "get_Ashares", "get_stock_status",
             "get_stock_info", "get_trade_days", "get_all_trades_days", "get_index_stocks", "get_snapshot",
             "get_stock_blocks", "get_industry_stocks", "get_trading_day"]:
    say("api." + name, name in globals() or hasattr(builtins, name))


# 2. Calendar and universe ---------------------------------------------------------------------------------------
days = attempt("trade_days", lambda: get_trade_days("20080101", time.strftime("%Y%m%d")))  # noqa: F821
if days is not None:
    say("trade_days.shape", shape(days))
    say("trade_days.ends", "%s .. %s" % (list(days[:2]), list(days[-3:])))
ashares = attempt("ashares", lambda: get_Ashares())  # noqa: F821
if ashares:
    sh_main = sorted(code for code in ashares if code.startswith("60") and code.endswith(".SS"))
    say("ashares.count", len(ashares))
    say("ashares.sh_main", len(sh_main))
    say("ashares.sample", ashares[:3] + ashares[-3:])
else:
    sh_main = ["600000.SS", "600570.SS", "601318.SS"]
old = attempt("ashares_2012", lambda: get_Ashares("20120104"))  # noqa: F821
say("ashares_2012.count", len(old) if old else old)
idx = attempt("index_300", lambda: get_index_stocks("000300.SS"))  # noqa: F821
say("index_300.count", len(idx) if idx else idx)

# 3. Daily bars ---------------------------------------------------------------------------------------------------
FIELDS = ["open", "high", "low", "close", "volume", "money", "preclose", "high_limit", "low_limit", "is_open"]
one = attempt("price_one", lambda: get_price("600570.SS", start_date="20080101", end_date=time.strftime("%Y%m%d"),  # noqa: F821
                                             frequency="1d", fields=FIELDS, fq=None))
if one is not None:
    say("price_one.shape", shape(one))
    say("price_one.head", one.head(2))
    say("price_one.tail", one.tail(3))
post = attempt("price_one_post", lambda: get_price("600570.SS", start_date="20080101", end_date=time.strftime("%Y%m%d"),  # noqa: F821
                                                   frequency="1d", fields=["close"], fq="post"))
if post is not None:
    say("price_one_post.ends", "%s / %s" % (post.head(1).to_dict(), post.tail(1).to_dict()))
many = attempt("price_sh_main_1100d", lambda: get_price(sh_main, end_date=time.strftime("%Y%m%d"), frequency="1d",  # noqa: F821
                                                        fields=FIELDS, fq=None, count=1100))
if many is not None:
    say("price_sh_main_1100d.type", shape(many))
    try:
        say("price_sh_main_1100d.columns", list(many.columns)[:12])
        say("price_sh_main_1100d.index", "%s .. %s" % (many.index[:2].tolist(), many.index[-2:].tolist()))
    except Exception as exc:
        say("price_sh_main_1100d.layout_error", exc)
fast = attempt("price_dict_sh_main_1100d", lambda: get_price(sh_main, end_date=time.strftime("%Y%m%d"),  # noqa: F821
                                                             frequency="1d", fields=FIELDS, fq=None, count=1100,
                                                             is_dict=True))
if fast is not None:
    first = next(iter(fast))
    say("price_dict.first", "%s -> %s" % (first, shape(fast[first])))
    try:
        say("price_dict.dtype", fast[first].dtype)
        say("price_dict.sample", fast[first][-2:])
    except Exception as exc:
        say("price_dict.sample_error", exc)
full = attempt("price_all_a_1100d_close", lambda: get_price(ashares, end_date=time.strftime("%Y%m%d"), frequency="1d",  # noqa: F821
                                                            fields=["close", "preclose", "volume"], fq=None,
                                                            count=1100, is_dict=True))
say("price_all_a.size", shape(full))

# 4. Status and listing --------------------------------------------------------------------------------------------
st = attempt("status_st_today", lambda: get_stock_status(sh_main, "ST"))  # noqa: F821
if st:
    say("status_st_today.true", sum(1 for value in st.values() if value is True))
st_old = attempt("status_st_2015", lambda: get_stock_status(sh_main[:300], "ST", "20150105"))  # noqa: F821
if st_old:
    say("status_st_2015.true", sum(1 for value in st_old.values() if value is True))
halt = attempt("status_halt_today", lambda: get_stock_status(sh_main, "HALT"))  # noqa: F821
if halt:
    say("status_halt_today.true", sum(1 for value in halt.values() if value is True))
info = attempt("stock_info", lambda: get_stock_info(sh_main, ["stock_name", "listed_date", "de_listed_date"]))  # noqa: F821
if info:
    say("stock_info.sample", {code: info[code] for code in list(info)[:2]})

# 5. Fundamentals --------------------------------------------------------------------------------------------------
TABLES = ["balance_statement", "income_statement", "cashflow_statement", "valuation"]
for table in TABLES:
    frame = attempt("fund_all_fields." + table, lambda: get_fundamentals("600570.SS", table, date="20241031"))  # noqa: F821
    if frame is not None:
        say("fund_all_fields.%s.columns" % table, list(getattr(frame, "columns", [])))
        say("fund_all_fields.%s.row" % table, frame.iloc[:1].to_dict("records") if hasattr(frame, "iloc") else frame)
# Point in time: what a 2024-04-15 query sees (the 2023 annual report is published around end of April).
for merge in (None, 1):
    pit = attempt("fund_pit_merge_%s" % merge, lambda: get_fundamentals(  # noqa: F821
        ["600570.SS", "600000.SS"], "income_statement", ["np_parent_company_owners", "operating_revenue"],
        date="20240415", merge_type=merge))
    if pit is not None:
        say("fund_pit_merge_%s.value" % merge, pit)
years = attempt("fund_years", lambda: get_fundamentals(  # noqa: F821
    ["600570.SS", "600000.SS"], "income_statement", ["np_parent_company_owners", "operating_revenue"],
    start_year="2021", end_year="2024"))
say("fund_years.type", shape(years))
try:
    say("fund_years.sample", years["600570.SS"] if not hasattr(years, "iloc") else years.head(10))
except Exception as exc:
    say("fund_years.sample_error", exc)
# Speed: 100 stocks x 20 fields x 4 years in year mode.
cols = ["np_parent_company_owners", "operating_revenue", "total_operating_revenue", "operating_cost",
        "operating_profit", "total_profit", "net_profit", "income_tax_cost", "financial_expense",
        "administration_expense", "operating_expense", "total_operating_cost", "invest_income",
        "fair_value_change_income", "asset_impairment_loss", "operating_tax_surcharges", "non_operating_income",
        "non_operating_expense", "invest_income_associates", "interest_expense"]
bulk = attempt("fund_bulk_100x20x4y", lambda: get_fundamentals(sh_main[:100], "income_statement", cols,  # noqa: F821
                                                                start_year="2021", end_year="2024"))
say("fund_bulk.type", shape(bulk))
val = attempt("valuation_day_sh_main", lambda: get_fundamentals(  # noqa: F821
    sh_main, "valuation", ["total_shares", "a_floats", "turnover_rate", "pe_ttm", "total_value", "float_value"],
    date=str(days[-2]).replace("-", "")[:8] if days is not None else None))
if val is not None:
    say("valuation_day.shape", shape(val))
    say("valuation_day.head", val.head(3))

say("probe", "done " + time.strftime("%Y-%m-%d %H:%M:%S"))
try:
    with open("probe_result.txt", "w") as handle:
        handle.write("\n".join(OUT) + "\n")
    say("written", "probe_result.txt next to this notebook")
except Exception as exc:
    say("write_error", exc)
