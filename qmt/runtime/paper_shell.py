# jq240 + LightGBM paper trading inside QMT: run once after the close. It scores sh_main for every day since
# PAPER_START, replays the paper book (top PAPER_TOPK, drop PAPER_N_DROP a day, next-open fills, A-share costs)
# from PAPER_CAPITAL in cash, and prints the book and the orders for the next open.
# READ-ONLY: no orders, no files, no network.
import time
import traceback

PAPER_START = "20261021"  # first scored day; the book is cash until the open after it
PAPER_CAPITAL = 150000.0
PAPER_TOPK = 30
PAPER_N_DROP = 3
PAPER_BACKUPS = 5
PAPER_DAYS = 1100
DOWNLOAD_DAYS = 10  # calendar days of daily bars to refresh through QMT's own data service on start; 0 = off
DOWNLOAD_SECONDS = 600
TAG = "[RQ]"
_state = {"done": False}


def log(key, value=""):
    print("%s %s=%s" % (TAG, key, value))


def beijing_now():
    """Beijing wall-clock time as YYYYMMDDHHMM."""
    return time.strftime("%Y%m%d%H%M", time.gmtime(time.time() + 8 * 3600))


def download_recent_bars(C):
    """Ask QMT to fetch the last DOWNLOAD_DAYS of daily bars (the same as its data manager's download) for every
    A share and the calendar index, so that the last session's bars are in before scoring."""
    if DOWNLOAD_DAYS <= 0:
        return
    many, one = globals().get("download_history_data2"), globals().get("download_history_data")
    if many is None and one is None:
        log("download.skip", "this QMT has no download_history_data")
        return
    began = time.time()
    start = time.strftime("%Y%m%d", time.gmtime(time.time() + 8 * 3600 - DOWNLOAD_DAYS * 86400))
    # The calendar index and the sh_main stocks the model scores first; the other A shares (for the market return)
    # only if time allows.
    everything = set(C.get_stock_list_in_sector(SECTOR_ALL_A))
    first = [CALENDAR_CODE] + sh_main_codes(C)
    codes = first + sorted(everything - set(first))
    failed = 0
    if many is not None:
        try:
            many(codes, "1d", start, "")
            codes = []
        except Exception:
            log("download.batch_error", traceback.format_exc().splitlines()[-1])
            if one is None:
                return
    for position, code in enumerate(codes):
        if time.time() - began > DOWNLOAD_SECONDS:
            log("download.stopped", "after %d of %d codes" % (position, len(codes)))
            break
        try:
            one(code, "1d", start, "")
        except Exception:
            failed += 1
    log("download.bars", "from %s, %d failed, %.1f s" % (start, failed, time.time() - began))


def complete_days(inputs, now):
    """Trading days with finished daily bars: a day with no close for any stock (QMT lists today in the calendar
    before its daily bars arrive) is dropped, and so is today while its session is still open (before 15:00)."""
    close = inputs["close"]
    dates = close.index[close.notna().any(axis=1).values]
    if len(dates) and dates[-1] == pd.Timestamp(now[:8]) and now[8:] < "1500":
        dates = dates[:-1]
    return dates


def limit_price(pre_close, pct):
    """Exchange limit price: pre-close times (1 +/- pct), rounded half up to the fen."""
    return np.floor(pre_close * pct * 100.0 + 0.5 + 1e-6) / 100.0


def quote_day(inputs, date):
    """paper.Day for ``date``: adjusted prices, factor and Tushare-style open-limit flags (10%, 5% for ST names)."""
    quotes = inputs["quotes"]
    close = inputs["close"].loc[date].dropna()
    codes = close.index
    raw_open = quotes["raw_open"].loc[date].reindex(codes)
    pre = quotes["raw_pre_close"].loc[date].reindex(codes)
    pct = pd.Series([0.05 if code in quotes["st"] else 0.10 for code in codes], index=codes)
    up, down = limit_price(pre, 1 + pct), limit_price(pre, 1 - pct)
    limit_buy = raw_open.round(2) >= up.round(2)
    limit_sell = raw_open.round(2) <= down.round(2)
    return Day(date, inputs["open"].loc[date].reindex(codes).to_dict(), close.to_dict(),
               quotes["factor"].loc[date].reindex(codes).to_dict(), limit_buy.to_dict(), limit_sell.to_dict())


def plan_day(inputs, date):
    """A stand-in for the next session, whose quotes are unknown: every stock that traded today is tradable at
    today's close."""
    close = inputs["close"].loc[date].dropna()
    none = dict((code, False) for code in close.index)
    factor = inputs["quotes"]["factor"].loc[date].reindex(close.index)
    return Day(date, close.to_dict(), close.to_dict(), factor.to_dict(), none, none)


def daily_scores(inputs, window):
    """Model scores of every universe member on each day of ``window``, keyed by day."""
    calculator = JQFactors(inputs)
    prepared = prepare_model(MODEL)
    panels = {}
    for ordinal, name in enumerate(JQ240_NAMES, 1):
        key = "jq_%03d" % ordinal
        if key in prepared["features"]:
            panels[key] = calculator.calculate(name).reindex(window)
    missing = [name for name in prepared["features"] if name not in panels]
    if missing:
        raise ValueError("model needs factors the calculator lacks: %s" % ",".join(missing))
    scores = {}
    for date in window:
        member = inputs["universe"].loc[date]
        codes = sorted(code for code, inside in zip(member.index, member.values) if inside)
        if not codes:
            continue
        matrix = np.column_stack([panels[name].loc[date].reindex(codes).values for name in prepared["features"]])
        scores[date] = pd.Series(predict(prepared, normalise(matrix, prepared)), index=codes)
    return scores


def _shares(book, code, factor):
    return int(round(book.amount[code] * factor))


def run_paper(C):
    began = time.time()
    download_recent_bars(C)
    now = beijing_now()
    inputs = build_inputs(C, now[:8], PAPER_DAYS, log)
    listed = inputs["close"].index
    dates = complete_days(inputs, now)
    if not len(dates):
        log("paper.error", "no trading day with daily bars")
        return
    skipped = listed[listed > dates[-1]]
    if len(skipped):
        log("paper.skipped_days", "%s (session open or daily bars not in yet)" % ",".join(
            day.strftime("%Y%m%d") for day in skipped))
    last = dates[-1]
    log("paper.data_through", last.strftime("%Y%m%d"))
    log("paper.model", "%s year=%s fit=%s" % (MODEL["run"], MODEL["prediction_year"], "..".join(MODEL["fit_period"])))
    window = dates[dates >= pd.Timestamp(PAPER_START)]
    if not len(window):
        log("paper.error", "no trading day on or after PAPER_START %s" % PAPER_START)
        return
    scores = daily_scores(inputs, window)
    log("paper.scored_days", len(scores))
    log("paper.score_seconds", round(time.time() - began, 2))
    book = Book(PAPER_CAPITAL)
    for previous, date in zip(window[:-1], window[1:]):
        if previous not in scores:
            continue
        fills = trade_day(scores[previous], book, quote_day(inputs, date), PAPER_TOPK, PAPER_N_DROP)
        for fill in fills:
            log("paper.fill.%s" % date.strftime("%Y%m%d"), "%s %s %d @ %.3f value %.2f cost %.2f" % (
                fill["side"], fill["code"], int(round(fill["shares"])), fill["value"] / fill["shares"],
                fill["value"], fill["cost"]))
        log("paper.nav.%s" % date.strftime("%Y%m%d"), "%.2f cash %.2f holdings %d" % (
            book.value(), book.cash, len(book.amount)))
    # Suspended holdings keep their last factor and close.
    factor = inputs["quotes"]["factor"].ffill().loc[last]
    raw_close = inputs["basic_close"].ffill().loc[last]
    for code in sorted(book.amount):
        log("paper.hold", "%s %d shares close %.2f value %.2f held_days %d" % (
            code, _shares(book, code, factor.get(code, np.nan)), raw_close.get(code, np.nan),
            book.amount[code] * book.price[code], book.count.get(code, 0)))
    gain = (book.value() / PAPER_CAPITAL - 1) * 100
    log("paper.value", "%.2f cash %.2f return %.2f%%" % (book.value(), book.cash, gain))
    if last not in scores:
        log("paper.error", "no scores for %s" % last.strftime("%Y%m%d"))
        return
    today = scores[last]
    plan = plan_day(inputs, last)
    sells, buys = decide(today, book, plan, PAPER_TOPK, PAPER_N_DROP)
    for code, amount in sells:
        shares = _shares(book, code, factor[code])
        log("paper.next.sell", "%s %d shares score %s" % (code, shares, _rank_text(today, code)))
    for code, amount in buys:
        shares, value = int(round(amount * factor[code])), amount * plan.close[code]
        log("paper.next.buy", "%s about %d shares at %.2f (about %.0f yuan) score %s" % (
            code, shares, raw_close[code], value, _rank_text(today, code)))
    chosen = set(code for code, _ in buys) | set(book.amount)
    spare = [code for code in ranked(today) if code not in chosen and code in plan.close][:PAPER_BACKUPS]
    log("paper.next.backups", ",".join("%s(%s)" % (code, _rank_text(today, code)) for code in spare))
    log("paper.top%d" % PAPER_TOPK, ",".join(ranked(today)[:PAPER_TOPK]))
    log("paper.seconds", round(time.time() - began, 2))


def _rank_text(scores, code):
    if code not in scores.index:
        return "n/a (left universe)"
    order = list(ranked(scores))
    return "%.5f rank %d/%d" % (scores[code], order.index(code) + 1, len(order))


def init(C):
    log("paper", "start")


def handlebar(C):
    if _state["done"] or not C.is_last_bar():
        return
    _state["done"] = True
    try:
        run_paper(C)
    except Exception:
        log("paper.error", traceback.format_exc())
    log("paper", "end")
