# Build the jq240 calculator's input dict from QMT's built-in data, matching RQuant's Tushare-based inputs.
# Python 3.6 / numpy 1.19 / pandas 0.22. Runs after jq240_core.py and jq240_field_map.py in the bundled strategy.
#
# RQuant conventions reproduced here (rquant.data.canonical, rquant.data.reference, rquant.universe):
# * canonical prices are the raw price times factor = adj / (adj x close at the first quote), the first quote being
#   the first traded bar on or after the data start (2008-05-12); adj is chained from QMT's raw close and preClose,
#   which is how Tushare's adj_factor moves; volume is raw shares / factor and amount is raw yuan;
# * cap = raw close x total shares; basic_* are daily_basic in yuan, shares and percent;
# * mkt is the circulating-cap-weighted return of all A shares listed 30+ days, weighted by the previous day's cap;
# * statements are one row per stock and quarter end, dated by the latest of the three statements' announcements;
# * the cross-sectional universe is sh_main: 60xxxx.SH, no ST or delisting name, more than 60 trading days listed.
# Known gaps: ST status is today's name for every day of the window, and stocks delisted before today are absent.
import time

import numpy as np
import pandas as pd

DATA_START = "20080512"
CALENDAR_START = "20050101"
FUNDAMENTAL_START = "20050101"
BEIJING_MS = 8 * 3600 * 1000
SECTOR_ALL_A = u"沪深A股"
CALENDAR_CODE = "000300.SH"
MARKET_SEASONING_DAYS = 30
MARKET_MIN_STOCKS = 100
NEW_LISTING_EXCLUDED_DAYS = 60
BAR_FIELDS = ["open", "high", "low", "close", "volume", "amount", "preClose", "suspendFlag"]
CHUNK = 300
TOTAL_SHARES = "CAPITALSTRUCTURE.total_capital"
FLOAT_SHARES = "CAPITALSTRUCTURE.circulating_capital"


def ms_to_date(ms):
    return pd.Timestamp(int(ms) + BEIJING_MS, unit="ms").normalize()


def _date_index(labels):
    return pd.DatetimeIndex([pd.Timestamp(str(label)[:8]) for label in labels])


def _chunks(codes, size=CHUNK):
    for first in range(0, len(codes), size):
        yield codes[first:first + size]


def trading_calendar(C, end):
    """Every exchange trading day from 2005 through ``end``."""
    bars = C.get_market_data_ex(["close"], [CALENDAR_CODE], period="1d", start_time=CALENDAR_START, end_time=end,
                                dividend_type="none", fill_data=False, subscribe=False)
    frame = bars[CALENDAR_CODE]
    return _date_index(frame.index[np.asarray(frame["close"], dtype=float) > 0])


def sh_main_codes(C):
    return sorted(code for code in C.get_stock_list_in_sector(SECTOR_ALL_A)
                  if code.endswith(".SH") and code.startswith("60"))


def _bar_frames(C, fields, codes, start, end, calendar):
    """Yield (code, frame) of unadjusted daily bars on exchange trading days, oldest first."""
    for chunk in _chunks(codes):
        data = C.get_market_data_ex(fields, chunk, period="1d", start_time=start, end_time=end,
                                    dividend_type="none", fill_data=False, subscribe=False)
        for code in chunk:
            frame = data.get(code)
            if frame is None or len(frame) == 0:
                continue
            frame = frame.copy()
            frame.index = _date_index(frame.index)
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            yield code, frame[frame.index.isin(calendar)]


def _traded(frame):
    traded = (np.asarray(frame["volume"], dtype=float) > 0) & (np.asarray(frame["close"], dtype=float) > 0)
    if "suspendFlag" in frame.columns:
        traded &= ~(np.asarray(frame["suspendFlag"], dtype=float) > 0)
    return traded


def canonical_factor(close, pre_close, traded):
    """RQuant's canonical factor per bar, or None when the stock never traded.

    adj steps by close[t-1] / preClose[t] on every bar (1 when either is missing), and is scaled so the first traded
    bar's canonical close is 1.
    """
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


def price_panels(C, codes, dates, end, calendar):
    """Canonical and raw daily panels on ``dates``; days without a traded bar are missing."""
    columns = {name: {} for name in ("open", "high", "low", "close", "volume", "amount", "raw_close")}
    for code, frame in _bar_frames(C, BAR_FIELDS, codes, DATA_START, end, calendar):
        traded = _traded(frame)
        factor = canonical_factor(frame["close"], frame["preClose"], traded)
        if factor is None:
            continue
        keep = traded & frame.index.isin(dates)
        index = frame.index[keep]
        factor = factor[keep]
        raw = {field: np.asarray(frame[field], dtype=float)[keep] for field in BAR_FIELDS[:6]}
        for field in ("open", "high", "low", "close"):
            columns[field][code] = pd.Series(raw[field] * factor, index=index)
        columns["volume"][code] = pd.Series(raw["volume"] * 100.0 / factor, index=index)
        columns["amount"][code] = pd.Series(raw["amount"], index=index)
        columns["raw_close"][code] = pd.Series(raw["close"], index=index)
    return {name: pd.DataFrame(values).reindex(index=dates, columns=codes) for name, values in columns.items()}


def _step_panel(points, dates, codes):
    """Daily panel from {code: {change_date_ms: value}}, each value holding from its date on.

    Days before a stock's first change take its first value (QMT's share history starts around 2015).
    """
    columns = {}
    for code in codes:
        changes = points.get(code) or {}
        series = pd.Series({ms_to_date(ms): float(value) for ms, value in changes.items()
                            if value is not None and value == value and value > 0})
        if not len(series):
            continue
        series = series[~series.index.duplicated(keep="last")].sort_index()
        columns[code] = series.reindex(series.index.union(dates)).ffill().bfill().reindex(dates)
    return pd.DataFrame(columns).reindex(index=dates, columns=codes)


def share_history(C, codes, field, end):
    points = {}
    for chunk in _chunks(codes, 500):
        raw = C.get_raw_financial_data([field], chunk, FUNDAMENTAL_START, end) or {}
        for code in chunk:
            points[code] = (raw.get(code) or {}).get(field) or {}
    return points


def turnover_panel(C, codes, dates):
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    parts = []
    for chunk in _chunks(codes):
        frame = C.get_turnover_rate(chunk, start, end)
        if frame is not None and len(frame):
            frame = frame.copy()
            frame.index = _date_index(frame.index)
            parts.append(frame)
    if not parts:
        return pd.DataFrame(np.nan, index=dates, columns=codes)
    # QMT gives a fraction; Tushare's daily_basic turnover_rate is in percent.
    return pd.concat(parts, axis=1).reindex(index=dates, columns=codes).astype(float) * 100.0


def fundamentals_frame(C, codes, end, total_points):
    """One row per stock and quarter end in Tushare column names, as rquant's fundamentals.parquet."""
    names = [name for endpoint in ("income", "balancesheet", "cashflow") for name in STATEMENT_FIELDS[endpoint]]
    targets = {}
    for endpoint in ("income", "balancesheet", "cashflow"):
        for name, target in STATEMENT_FIELDS[endpoint].items():
            if target and not target.startswith(CAPITAL):
                targets[name] = target
    wanted = sorted(set(targets.values()))
    announcements = [table + ".m_anntime" for table in (INCOME, BALANCE, CASHFLOW)]
    rows = []
    for chunk in _chunks(codes):
        raw = {}
        for first in range(0, len(wanted), 20):
            fields = wanted[first:first + 20] + announcements
            part = C.get_raw_financial_data(fields, chunk, FUNDAMENTAL_START, end) or {}
            for code, values in part.items():
                raw.setdefault(code, {}).update(values)
        for code in chunk:
            values = raw.get(code) or {}
            periods = set()
            for name in announcements:
                periods.update((values.get(name) or {}).keys())
            shares = sorted((ms_to_date(ms), float(value)) for ms, value in (total_points.get(code) or {}).items()
                            if value is not None and value == value and value > 0)
            for period in sorted(periods):
                announced = [(values.get(name) or {}).get(period) for name in announcements]
                announced = [value for value in announced if value is not None and value == value and value > 0]
                if not announced:
                    continue
                end_date = ms_to_date(period)
                row = {"instrument": code, "end_date": end_date, "ann_date": ms_to_date(max(announced))}
                for name, target in targets.items():
                    value = (values.get(target) or {}).get(period)
                    row[name] = np.nan if value is None else float(value)
                known = [value for when, value in shares if when <= end_date]
                row["total_share"] = known[-1] if known else (shares[0][1] if shares else np.nan)
                rows.append(row)
    return pd.DataFrame(rows, columns=["instrument", "end_date", "ann_date"] + names)


def instrument_info(C, code):
    """(listing date or NaT, name) from QMT's instrument detail."""
    detail = C.get_instrumentdetail(code) or {}
    opened = str(detail.get("OpenDate") or "")
    opened = pd.Timestamp(opened[:8]) if len(opened) >= 8 and opened[:8] not in ("00000000", "99999999") else pd.NaT
    return opened, u"%s" % (detail.get("InstrumentName") or u"")


def market_return(C, dates, end, calendar, opened):
    """Circulating-cap-weighted all-A return (no Beijing exchange), as rquant.data.reference's ``mkt``."""
    codes = sorted(code for code in C.get_stock_list_in_sector(SECTOR_ALL_A) if not code.endswith(".BJ"))
    start = calendar[max(calendar.searchsorted(dates[0]) - 1, 0)]
    window = calendar[(calendar >= start) & (calendar <= dates[-1])]
    closes, returns = {}, {}
    for code, frame in _bar_frames(C, ["close", "preClose", "volume", "suspendFlag"], codes,
                                   start.strftime("%Y%m%d"), end, calendar):
        traded = _traded(frame)
        close = np.asarray(frame["close"], dtype=float)[traded]
        pre_close = np.asarray(frame["preClose"], dtype=float)[traded]
        safe = np.where(pre_close > 0, pre_close, np.nan)
        closes[code] = pd.Series(close, index=frame.index[traded])
        returns[code] = pd.Series(close / safe - 1.0, index=frame.index[traded])
    codes = sorted(closes)
    close = pd.DataFrame(closes).reindex(index=window, columns=codes)
    ret = pd.DataFrame(returns).reindex(index=window, columns=codes)
    floats = _step_panel(share_history(C, codes, FLOAT_SHARES, end), window, codes)
    # The previous trading day's circulating cap, as reference.py weights by the previous daily_basic.
    weights = (close * floats).shift(1)
    for code in codes:
        if code not in opened:
            opened[code] = instrument_info(C, code)[0]
    listed = pd.DatetimeIndex([opened[code] for code in codes])
    age = (window.values[:, None] - listed.values[None, :]).astype("timedelta64[D]").astype(float)
    seasoned = pd.DataFrame((age >= MARKET_SEASONING_DAYS) & ~np.isnan(age), index=window, columns=codes)
    valid = ret.notna() & (weights > 0) & seasoned
    count = valid.sum(axis=1)
    weighted = (ret.where(valid) * weights.where(valid)).sum(axis=1) / weights.where(valid).sum(axis=1)
    return weighted.where(count >= MARKET_MIN_STOCKS).reindex(dates), count.reindex(dates)


def universe_panel(codes, dates, calendar, traded, opened, names):
    """sh_main membership: traded, no ST or delisting name, more than 60 open days since listing (inclusive)."""
    position = calendar.searchsorted(dates)
    columns = {}
    for code in codes:
        name = names.get(code, u"").upper()
        if u"ST" in name or u"退" in name:
            columns[code] = np.zeros(len(dates), dtype=bool)
            continue
        listed = opened.get(code)
        start = calendar.searchsorted(listed) if listed is not None and listed == listed else 0
        columns[code] = (position - start + 1) > NEW_LISTING_EXCLUDED_DAYS
    frame = pd.DataFrame(columns, index=dates).reindex(columns=codes).fillna(False).astype(bool)
    return frame & traded


def build_inputs(C, end, days, log=print, codes=None):
    """jq240 inputs for sh_main (or ``codes``) over the ``days`` trading days ending at ``end`` (YYYYMMDD)."""
    began = time.time()
    calendar = trading_calendar(C, end)
    dates = calendar[-days:]
    codes = list(codes) if codes is not None else sh_main_codes(C)
    log("inputs.codes", len(codes))
    log("inputs.dates", "%s..%s (%d)" % (dates[0].date(), dates[-1].date(), len(dates)))
    panels = price_panels(C, codes, dates, end, calendar)
    log("inputs.bars_seconds", round(time.time() - began, 2))
    total_points = share_history(C, codes, TOTAL_SHARES, end)
    total = _step_panel(total_points, dates, codes)
    floats = _step_panel(share_history(C, codes, FLOAT_SHARES, end), dates, codes)
    raw_close = panels.pop("raw_close")
    inputs = dict(panels)
    inputs["cap"] = raw_close * total
    inputs["basic_close"] = raw_close
    inputs["basic_total_share"] = total.where(raw_close.notna())
    inputs["basic_circ_mv"] = raw_close * floats
    inputs["basic_turnover_rate"] = turnover_panel(C, codes, dates).where(raw_close.notna())
    log("inputs.shares_seconds", round(time.time() - began, 2))
    fundamentals = fundamentals_frame(C, codes, end, total_points)
    inputs["fundamentals"] = fundamentals
    log("inputs.fundamental_rows", len(fundamentals))
    log("inputs.fundamentals_seconds", round(time.time() - began, 2))
    grid = QuarterGrid(prepare_fundamentals(fundamentals), dates, pd.Index(codes))
    profit = grid.panel(grid.ttm("n_income_attr_p"))
    inputs["basic_pe_ttm"] = (inputs["cap"] / profit.where(profit > 0)).where(raw_close.notna())
    opened, names = {}, {}
    for code in codes:
        opened[code], names[code] = instrument_info(C, code)
    mkt, count = market_return(C, dates, end, calendar, opened)
    inputs["mkt"] = mkt
    last = count.iloc[-1] if len(count) else 0
    log("inputs.mkt_stocks_last", int(last) if last == last else 0)
    inputs["universe"] = universe_panel(codes, dates, calendar, raw_close.notna(), opened, names)
    log("inputs.universe_last", int(inputs["universe"].iloc[-1].sum()))
    log("inputs.seconds", round(time.time() - began, 2))
    return inputs
