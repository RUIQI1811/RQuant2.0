# Build the jq240 calculator's input dict from PTrade's built-in data, matching RQuant's Tushare-based inputs.
# Python 3.11 / numpy 1.23 / pandas 1.5 (Guojin PTrade). Runs after jq240_core.py in the bundled file, in PTrade's
# research notebook or a strategy, where get_price, get_fundamentals, get_trade_days, get_Ashares, get_stock_status
# and get_stock_info are globals. PTrade research rejects code that touches the system modules, so none are used.
#
# RQuant conventions reproduced here (same as qmt/runtime/qmt_inputs.py):
# * canonical prices are the raw price times factor = adj / (adj x close at the first quote), the first quote being
#   the first traded bar on or after the data start (2008-05-12); adj is chained from the raw close and preclose,
#   which is how Tushare's adj_factor moves; volume is raw shares / factor and amount is raw yuan;
# * cap = raw close x total shares; basic_* are daily_basic in yuan, shares and percent;
# * mkt is the circulating-cap-weighted return of all A shares listed 30+ days, weighted by the previous day's cap;
# * statements are one row per stock and quarter end, dated by the earliest of the three statements' announcements;
# * sh_main: 60xxxx Shanghai main board, no ST or delisting status that day, more than 60 trading days listed.
# PTrade codes end in .SS; RQuant and QMT write .SH. Known gap: stocks delisted before ``end`` are absent.
import time

import numpy as np
import pandas as pd

DATA_START = "20080512"
CALENDAR_START = "20050101"
FUNDAMENTAL_YEARS_BACK = 6
MARKET_SEASONING_DAYS = 30
MARKET_MIN_STOCKS = 100
NEW_LISTING_EXCLUDED_DAYS = 60
BAR_FIELDS = ["open", "high", "low", "close", "volume", "money", "preclose"]
BAR_CHUNK = 300
VALUATION_CHUNK = 1000
STATEMENT_CODES = 200
STATEMENT_FIELDS_PER_CALL = 30
CACHE_RECENT_DAYS = 3  # with a cache, the last days are always asked again

# Tushare statement fields used by jq240 -> PTrade get_fundamentals table and field. None: PTrade has no such line,
# treated as zero on a published statement (as RQuant's ZERO_FILL does for blank lines). A tuple: the first field
# with a value on that row.
PT_INCOME, PT_BALANCE, PT_CASHFLOW = "income_statement", "balance_statement", "cashflow_statement"
PTRADE_FIELDS = {
    "income": {
        "total_revenue": "total_operating_revenue",
        "revenue": "operating_revenue",
        "oper_cost": "operating_cost",
        "sell_exp": "operating_expense",
        "admin_exp": "administration_expense",
        "fin_exp": "financial_expense",
        "rd_exp": "r_and_d",
        "operate_profit": "operating_profit",
        "total_profit": "total_profit",
        "income_tax": "income_tax_cost",
        "n_income": "net_profit",
        "n_income_attr_p": "np_parent_company_owners",
        "total_cogs": "total_operating_cost",
        "biz_tax_surchg": "operating_tax_surcharges",
        "assets_impair_loss": "asset_impairment_loss",
        "fv_value_chg_gain": "fair_value_change_income",
        "invest_income": "invest_income",
        "ass_invest_income": "invest_income_associates",
        "non_oper_income": "non_operating_income",
        "non_oper_exp": "non_operating_expense",
        # No interest split of the finance expense for industrial companies (interest_expense is a bank line), so
        # net_interest_expense uses the whole finance expense, as the QMT port does.
        "fin_exp_int_exp": "financial_expense",
        "fin_exp_int_inc": None,
    },
    "balancesheet": {
        "total_share": "paidin_capital",
        "money_cap": "cash_equivalents",
        "trad_asset": "trading_assets",
        "accounts_receiv": "account_receivable",
        "inventories": "inventories",
        "total_cur_assets": "total_current_assets",
        "lt_eqt_invest": "longterm_equity_invest",
        # Third-quarter reports leave fixed_assets blank; total_fixed_asset (with assets in liquidation) is filled.
        "fix_assets": ("fixed_assets", "total_fixed_asset"),
        "intan_assets": "intangible_assets",
        "goodwill": "good_will",
        "total_assets": "total_assets",
        "st_borr": "shortterm_loan",
        "non_cur_liab_due_1y": "non_current_liability_in_one_year",
        "total_cur_liab": "total_current_liability",
        "lt_borr": "longterm_loan",
        "bond_payable": "bonds_payable",
        "total_liab": "total_liability",
        "minority_int": "minority_interests",
        "oth_eqt_tools_p_shr": None,
        "total_hldr_eqy_exc_min_int": "se_without_mi",
        "total_hldr_eqy_inc_min_int": "total_shareholder_equity",
        "notes_receiv": "bill_receivable",
        "oth_receiv": "other_receivable",
        "oth_rcv_total": "other_receivable",
        "prepayment": "advance_payment",
        "int_receiv": "interest_receivable",
        "div_receiv": "dividend_receivable",
        "fa_avail_for_sale": "hold_for_sale_assets",
        "htm_invest": "hold_to_maturity_investments",
        "cip": "constru_in_process",
        "const_materials": "construction_materials",
        "r_and_d": "development_expenditure",
        "lt_amor_exp": "long_deferred_expense",
        "total_nca": "total_non_current_assets",
        "trading_fl": "trading_liability",
        "notes_payable": "notes_payable",
        "acct_payable": "accounts_payable",
        "adv_receipts": "advance_receipts",
        "int_payable": "interest_payable",
        "div_payable": "dividend_payable",
        "total_ncl": "total_non_current_liability",
        "cap_rese": "capital_reserve_fund",
        "surplus_rese": "surplus_reserve_fund",
        "undistr_porfit": "retained_profit",
        "taxes_payable": "taxs_payable",
        "defer_tax_liab": "deferred_tax_liability",
    },
    "cashflow": {
        "n_cashflow_act": "net_operate_cash_flow",
        "c_pay_acq_const_fiolta": "fix_intan_other_asset_acqui_cash",
        "n_incr_cash_cash_equ": "cash_equivalent_increase",
        "depr_fa_coga_dpba": "fixed_asset_depreciation",
        "amort_intang_assets": "intangible_asset_amortization",
        "lt_amort_deferred_exp": "deferred_expense_amort",
        "c_fr_sale_sg": "goods_sale_service_render_cash",
        "n_cashflow_inv_act": "net_invest_cash_flow",
        "n_cash_flows_fnc_act": "net_finance_cash_flow",
        "c_cash_equ_end_period": "end_period_cash_equivalent",
        "c_recp_cap_contrib": "cash_from_invest",
        "c_recp_borrow": "cash_from_borrowing",
        "proc_issue_bonds": "cash_from_bonds_issue",
        "c_prepay_amt_borr": "borrowing_repayment",
    },
}
ENDPOINT_TABLES = {"income": PT_INCOME, "balancesheet": PT_BALANCE, "cashflow": PT_CASHFLOW}


def to_rquant(code):
    return code[:-3] + ".SH" if code.endswith(".SS") else code


def _chunks(codes, size):
    for first in range(0, len(codes), size):
        yield codes[first:first + size]


def _day(value):
    return pd.Timestamp(str(value).replace("-", "")[:8])


def trading_calendar(end):
    """Exchange trading days from 2005 through ``end`` (YYYYMMDD)."""
    days = pd.DatetimeIndex([_day(day) for day in get_trade_days(CALENDAR_START, end)])  # noqa: F821
    return days[days <= _day(end)]


def sh_main_codes(end):
    return sorted(code for code in get_Ashares(end) if code.startswith("60") and code.endswith(".SS"))  # noqa: F821


def all_a_codes(end):
    return sorted(code for code in get_Ashares(end) if code.endswith(".SS") or code.endswith(".SZ"))  # noqa: F821


def _bars(codes, fields, start, end, calendar):
    """Yield (code, dates, structured array) of unadjusted daily bars on exchange trading days, oldest first."""
    for chunk in _chunks(codes, BAR_CHUNK):
        data = get_price(chunk, start_date=start, end_date=end, frequency="1d", fields=fields,  # noqa: F821
                         fq=None, is_dict=True) or {}
        for code in chunk:
            bars = data.get(code)
            if bars is None or len(bars) == 0:
                continue
            dates = pd.DatetimeIndex([_day(value) for value in bars["datetime"]])
            keep = dates.isin(calendar) & ~dates.duplicated(keep="last")
            yield code, dates[keep], bars[keep]


def _traded(bars):
    return (np.asarray(bars["volume"], dtype=float) > 0) & (np.asarray(bars["close"], dtype=float) > 0)


def canonical_factor(close, pre_close, traded):
    """RQuant's canonical factor per bar, or None when the stock never traded (see qmt_inputs.canonical_factor)."""
    close = np.asarray(close, dtype=float)
    pre_close = np.asarray(pre_close, dtype=float)
    previous = np.r_[np.nan, close[:-1]]
    step = np.ones(len(close))
    usable = (previous > 0) & (pre_close > 0)
    step[usable] = previous[usable] / pre_close[usable]
    adj = np.cumprod(step)
    first = np.flatnonzero(traded)
    if not len(first):
        return None
    first = first[0]
    return adj / (adj[first] * close[first])


def price_panels(codes, dates, end, calendar):
    """Canonical and raw daily panels on ``dates``; days without a traded bar are missing."""
    names = ("open", "high", "low", "close", "volume", "amount", "raw_close", "raw_volume", "raw_open",
             "raw_pre_close", "factor")
    columns = dict((name, {}) for name in names)
    for code, index, bars in _bars(codes, BAR_FIELDS, DATA_START, end, calendar):
        traded = _traded(bars)
        factor = canonical_factor(bars["close"], bars["preclose"], traded)
        if factor is None:
            continue
        keep = traded & index.isin(dates)
        at = index[keep]
        factor = factor[keep]
        raw = dict((field, np.asarray(bars[field], dtype=float)[keep]) for field in BAR_FIELDS)
        for field in ("open", "high", "low", "close"):
            columns[field][code] = pd.Series(raw[field] * factor, index=at)
        columns["volume"][code] = pd.Series(raw["volume"] / factor, index=at)
        columns["amount"][code] = pd.Series(raw["money"], index=at)
        columns["raw_close"][code] = pd.Series(raw["close"], index=at)
        columns["raw_volume"][code] = pd.Series(raw["volume"], index=at)
        columns["raw_open"][code] = pd.Series(raw["open"], index=at)
        columns["raw_pre_close"][code] = pd.Series(raw["preclose"], index=at)
        columns["factor"][code] = pd.Series(factor, index=at)
    return dict((name, pd.DataFrame(values).reindex(index=dates, columns=codes)) for name, values in columns.items())


def _cacheable(dates):
    """Days whose answers may be kept for later runs: all but the last CACHE_RECENT_DAYS (they may still change)."""
    return set(dates[:-CACHE_RECENT_DAYS]) if len(dates) > CACHE_RECENT_DAYS else set()


def valuation_panels(codes, dates, fields, log=print, cache=None):
    """{field: dates x codes} from PTrade's daily valuation table; gaps take the stock's nearest known value.

    ``cache`` (a dict kept between runs, as a live strategy left running does) holds the answers of earlier days, so a
    later run only asks for the days it has not seen.
    """
    values = dict((field, {}) for field in fields)
    store = cache.setdefault(("valuation",) + tuple(fields), {}) if cache is not None else {}
    keep = _cacheable(dates)
    began, asked = time.time(), 0
    for number, day in enumerate(dates):
        if day in store:
            for field in fields:
                if field in store[day]:
                    values[field][day] = store[day][field]
            continue
        asked += 1
        stamp = day.strftime("%Y%m%d")
        for chunk in _chunks(codes, VALUATION_CHUNK):
            frame = None
            for attempt in range(3):
                try:
                    frame = get_fundamentals(chunk, "valuation", fields, date=stamp)  # noqa: F821
                except Exception:
                    frame = None
                if frame is not None and len(frame):
                    break
                time.sleep(0.5)
            if frame is None or not len(frame):
                continue
            if "secu_code" in frame.columns:
                frame = frame.set_index("secu_code")
            for field in fields:
                if field in frame.columns:
                    values[field][day] = pd.to_numeric(frame[field], errors="coerce").combine_first(
                        values[field].get(day, pd.Series(dtype=float)))
        if day in keep and any(day in values[field] for field in fields):
            store[day] = dict((field, values[field][day]) for field in fields if day in values[field])
        if number and number % 200 == 0:
            log("inputs.valuation_progress", "%d/%d days %.0fs" % (number, len(dates), time.time() - began))
    if cache is not None:
        log("inputs.valuation_asked_days", "%d of %d" % (asked, len(dates)))
    panels = {}
    for field in fields:
        panel = pd.DataFrame(values[field]).T.reindex(index=dates, columns=codes)
        panels[field] = panel.where(panel > 0).ffill().bfill()
    return panels


def _statement_table(codes, table, fields, start_year, end_year, log=print):
    """One table's rows as a frame with columns secu_code, end_date, publ_date and ``fields``."""
    parts, missing = [], set(fields)
    for chunk in _chunks(codes, STATEMENT_CODES):
        for first in range(0, len(fields), STATEMENT_FIELDS_PER_CALL):
            wanted = fields[first:first + STATEMENT_FIELDS_PER_CALL]
            frame = None
            for attempt in range(3):
                try:
                    frame = get_fundamentals(chunk, table, wanted, start_year=str(start_year),  # noqa: F821
                                             end_year=str(end_year))
                except Exception as exc:
                    log("inputs.statement_retry", "%s %s" % (table, exc))
                    frame = None
                if frame is not None:
                    break
                time.sleep(1.0)
            if frame is None or not len(frame):
                continue
            frame = frame.reset_index()
            missing -= set(frame.columns)
            keys = [name for name in ("secu_code", "end_date", "publ_date") if name in frame.columns]
            parts.append(frame[keys + [name for name in wanted if name in frame.columns]])
    if missing:
        log("inputs.statement_missing_fields", "%s %s" % (table, sorted(missing)))
    if not parts:
        return pd.DataFrame(columns=["secu_code", "end_date", "publ_date"] + list(fields))
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.groupby(["secu_code", "end_date"], as_index=False).first()
    return frame


def _candidates(value):
    """The PTrade fields of one PTRADE_FIELDS entry, in order of preference."""
    if value is None:
        return ()
    return tuple(value) if isinstance(value, tuple) else (value,)


def fundamentals_frame(codes, start_year, end_year, log=print):
    """One row per stock and quarter end in Tushare column names, as rquant's fundamentals.parquet."""
    names = [name for endpoint in ("income", "balancesheet", "cashflow") for name in PTRADE_FIELDS[endpoint]]
    merged = None
    absent = []
    for endpoint, mapping in PTRADE_FIELDS.items():
        table = ENDPOINT_TABLES[endpoint]
        fields = sorted(set(field for value in mapping.values() for field in _candidates(value)))
        absent += [name for name, field in mapping.items() if field is None]
        frame = _statement_table(codes, table, fields, start_year, end_year, log)
        frame = frame.rename(columns={"publ_date": "publ_" + endpoint})
        out = frame[["secu_code", "end_date", "publ_" + endpoint]].copy()
        for name, value in mapping.items():
            if value is None:
                continue
            column = pd.Series(np.nan, index=frame.index)
            for field in _candidates(value):
                if field in frame.columns:
                    column = column.combine_first(pd.to_numeric(frame[field], errors="coerce"))
            out[name] = column
        merged = out if merged is None else merged.merge(out, on=["secu_code", "end_date"], how="outer")
    published = merged[["publ_" + endpoint for endpoint in PTRADE_FIELDS]].apply(
        lambda column: pd.to_datetime(column, errors="coerce"))
    merged["ann_date"] = published.min(axis=1)
    merged["end_date"] = pd.to_datetime(merged["end_date"], errors="coerce")
    merged = merged[merged["ann_date"].notna() & merged["end_date"].notna()]
    for name in absent:
        merged[name] = 0.0
    merged = merged.rename(columns={"secu_code": "instrument"})
    merged = merged.sort_values(["instrument", "end_date"]).reset_index(drop=True)
    return merged[["instrument", "end_date", "ann_date"] + names]


def listing_dates(codes):
    info = get_stock_info(codes, ["listed_date"]) or {}  # noqa: F821
    opened = {}
    for code in codes:
        value = str((info.get(code) or {}).get("listed_date") or "")
        opened[code] = _day(value) if len(value) >= 8 else pd.NaT
    return opened


def market_return(codes, dates, end, calendar, opened, floats):
    """Circulating-cap-weighted all-A return, as rquant.data.reference's ``mkt``."""
    start = calendar[max(calendar.searchsorted(dates[0]) - 1, 0)]
    window = calendar[(calendar >= start) & (calendar <= dates[-1])]
    closes, returns = {}, {}
    for code, index, bars in _bars(codes, ["close", "preclose", "volume"], start.strftime("%Y%m%d"), end, calendar):
        traded = _traded(bars)
        close = np.asarray(bars["close"], dtype=float)[traded]
        pre_close = np.asarray(bars["preclose"], dtype=float)[traded]
        safe = np.where(pre_close > 0, pre_close, np.nan)
        closes[code] = pd.Series(close, index=index[traded])
        returns[code] = pd.Series(close / safe - 1.0, index=index[traded])
    codes = sorted(closes)
    close = pd.DataFrame(closes).reindex(index=window, columns=codes)
    ret = pd.DataFrame(returns).reindex(index=window, columns=codes)
    shares = floats.reindex(index=window, columns=codes).ffill().bfill()
    # The previous trading day's circulating cap, as reference.py weights by the previous daily_basic.
    weights = (close * shares).shift(1)
    listed = pd.DatetimeIndex([opened.get(code, pd.NaT) for code in codes])
    age = (window.values[:, None] - listed.values[None, :]).astype("timedelta64[D]").astype(float)
    seasoned = pd.DataFrame((age >= MARKET_SEASONING_DAYS) & ~np.isnan(age), index=window, columns=codes)
    valid = ret.notna() & (weights > 0) & seasoned
    count = valid.sum(axis=1)
    weighted = (ret.where(valid) * weights.where(valid)).sum(axis=1) / weights.where(valid).sum(axis=1)
    return weighted.where(count >= MARKET_MIN_STOCKS).reindex(dates), count.reindex(dates)


def status_panel(codes, dates, kind, log=print, cache=None):
    """dates x codes booleans from get_stock_status(codes, kind, date) (kind 'ST' or 'DELISTING'); ``cache`` as in
    valuation_panels."""
    rows = {}
    store = cache.setdefault(("status", kind), {}) if cache is not None else {}
    keep = _cacheable(dates)
    for day in dates:
        known = store.get(day, {})
        wanted = [code for code in codes if code not in known]
        flags = {}
        if wanted:
            try:
                flags = get_stock_status(wanted, kind, day.strftime("%Y%m%d")) or {}  # noqa: F821
            except Exception as exc:
                log("inputs.status_error", "%s %s %s" % (kind, day.date(), exc))
                flags = None
        row = dict(known)
        if flags is not None:
            row.update((code, flags.get(code) is True) for code in wanted)
            if day in keep:
                store[day] = row
        rows[day] = dict((code, row.get(code, False)) for code in codes)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=dates, columns=codes).fillna(False).astype(bool)


def universe_panel(codes, dates, calendar, traded, opened, excluded):
    """sh_main membership: traded, not ST or delisting that day, more than 60 open days since listing (inclusive)."""
    position = calendar.searchsorted(dates)
    columns = {}
    for code in codes:
        listed = opened.get(code, pd.NaT)
        start = calendar.searchsorted(listed) if listed == listed else 0
        columns[code] = (position - start + 1) > NEW_LISTING_EXCLUDED_DAYS
    frame = pd.DataFrame(columns, index=dates).reindex(columns=codes).fillna(False).astype(bool)
    return frame & traded & ~excluded


def build_inputs(end, days, log=print, codes=None, status_days=None, cache=None):
    """jq240 inputs for sh_main (or ``codes``) over the ``days`` trading days ending at ``end`` (YYYYMMDD).

    ``status_days`` limits the per-day ST lookups to the last that many days (earlier days use the first looked-up
    day's flags); None looks up every day. ``cache``: a dict kept between calls; the per-day valuation and status
    answers of earlier days are taken from it (the slow part of a cold run) and new ones are added.
    """
    began = time.time()
    calendar = trading_calendar(end)
    dates = calendar[-days:]
    codes = list(codes) if codes is not None else sh_main_codes(end)
    every = all_a_codes(end)
    log("inputs.codes", "%d sh_main, %d all-A" % (len(codes), len(every)))
    log("inputs.dates", "%s..%s (%d)" % (dates[0].date(), dates[-1].date(), len(dates)))
    panels = price_panels(codes, dates, end, calendar)
    log("inputs.bars_seconds", round(time.time() - began, 2))
    opened = listing_dates(every)
    start = calendar[max(calendar.searchsorted(dates[0]) - 1, 0)]
    window = calendar[(calendar >= start) & (calendar <= dates[-1])]
    valuation = valuation_panels(every, window, ["total_shares", "a_floats"], log, cache)
    log("inputs.valuation_seconds", round(time.time() - began, 2))
    total = valuation["total_shares"].reindex(index=dates, columns=codes)
    floats = valuation["a_floats"].reindex(index=dates, columns=codes)
    raw_close = panels.pop("raw_close")
    raw_volume = panels.pop("raw_volume")
    quotes = dict((name, panels.pop(name)) for name in ("raw_open", "raw_pre_close", "factor"))
    inputs = dict(panels)
    inputs["quotes"] = quotes
    inputs["cap"] = raw_close * total
    inputs["basic_close"] = raw_close
    inputs["basic_total_share"] = total.where(raw_close.notna())
    inputs["basic_circ_mv"] = raw_close * floats
    # Tushare's turnover_rate: traded shares / circulating shares, in percent.
    inputs["basic_turnover_rate"] = (raw_volume / floats.where(floats > 0) * 100.0).where(raw_close.notna())
    first_year = max(2005, dates[0].year - FUNDAMENTAL_YEARS_BACK)
    fundamentals = fundamentals_frame(codes, first_year, _day(end).year, log)
    inputs["fundamentals"] = fundamentals
    log("inputs.fundamental_rows", len(fundamentals))
    log("inputs.fundamentals_seconds", round(time.time() - began, 2))
    grid = QuarterGrid(prepare_fundamentals(fundamentals), dates, pd.Index(codes))  # noqa: F821
    profit = grid.panel(grid.ttm("n_income_attr_p"))
    inputs["basic_pe_ttm"] = (inputs["cap"] / profit.where(profit > 0)).where(raw_close.notna())
    mkt, count = market_return(every, dates, end, calendar, opened, valuation["a_floats"])
    inputs["mkt"] = mkt
    last = count.iloc[-1] if len(count) else 0
    log("inputs.mkt_stocks_last", int(last) if last == last else 0)
    looked = dates if status_days is None else dates[-status_days:]
    excluded = status_panel(codes, looked, "ST", log, cache) | status_panel(codes, looked, "DELISTING", log, cache)
    excluded = excluded.reindex(index=dates).bfill().fillna(False).astype(bool)
    quotes["st"] = set(code for code in codes if excluded[code].iloc[-1])
    inputs["universe"] = universe_panel(codes, dates, calendar, raw_close.notna(), opened, excluded)
    log("inputs.universe_last", int(inputs["universe"].iloc[-1].sum()))
    log("inputs.seconds", round(time.time() - began, 2))
    return inputs
