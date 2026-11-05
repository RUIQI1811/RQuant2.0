# jq240 + LightGBM live trading inside QMT (run in QMT's live-trading mode with the account selected).
#
# Start it before 09:15 Beijing time and leave it running; each Beijing day it plans afresh. On the day's first bar
# it refreshes recent daily bars and scores sh_main with the last finished session's data (several minutes, so start
# by 09:00). At 09:30:05 it reads the account, decides like the backtest's TopkDropoutStrategy
# (top LIVE_TOPK, drop at most LIVE_N_DROP a day) against the real holdings and the real open, and sends the sells; at
# 09:30:25 it sends the buys with the cash then available. Orders are marketable limit orders. It trades at most once
# a day and never after LIVE_STOP_AT.
#
# LIVE_DRY_RUN = True (the default) sends nothing: it prints the orders it would send, at once if the market is shut.
# Set it to False in the QMT editor only after a dry run looks right. The account number stays in the QMT editor.
LIVE_DRY_RUN = True
LIVE_ACCOUNT = ""  # empty: the account selected for the strategy in QMT
LIVE_CAPITAL = 150000.0  # the most the strategy may hold, holdings at market plus cash it spends
LIVE_TOPK = 30
LIVE_N_DROP = 3
LIVE_KEEP = set()  # sh_main codes the strategy must leave alone, e.g. set(["600000.SH"])
LIVE_SELL_AT = "093005"
LIVE_BUY_AT = "093025"
LIVE_STOP_AT = "100000"
LIVE_SLIP = 0.01  # limit price offset from the last price; capped at the daily limit
LIVE_MAX_ORDER = 2.5  # no single buy above this many equal slots (LIVE_CAPITAL / LIVE_TOPK)
LIVE_TAG = "RQ"
BUY_OP, SELL_OP, BY_SHARES, LIMIT_PRICE, RIGHT_AWAY = 23, 24, 1101, 11, 2
_live = {"day": None, "planned": False, "plan": None, "sold": None, "done": False}


def beijing_clock():
    """Beijing wall-clock time as YYYYMMDDHHMMSS, whatever the machine's time zone."""
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(time.time() + 8 * 3600))


def account_id():
    return LIVE_ACCOUNT or str(globals().get("account", "") or "")


def masked(account):
    return ("*" * max(len(account) - 4, 0)) + account[-4:]


def managed(code):
    return code.startswith("60") and code.endswith(".SH") and code not in LIVE_KEEP


def read_account(account):
    """(available cash, {code: (shares, sellable shares)}) for the strategy's sh_main codes."""
    rows = get_trade_detail_data(account, "stock", "account")
    if not rows:
        raise ValueError("no account data for %s: is the account logged in and selected?" % masked(account))
    cash = float(rows[0].m_dAvailable)
    positions = {}
    for row in get_trade_detail_data(account, "stock", "position"):
        exchange = {"SSE": "SH", "SZSE": "SZ"}.get(str(row.m_strExchangeID), str(row.m_strExchangeID))
        code = "%s.%s" % (row.m_strInstrumentID, exchange)
        shares = int(row.m_nVolume)
        if shares > 0 and managed(code):
            positions[code] = (shares, int(row.m_nCanUseVolume))
    return cash, positions


def sent_today(account, tag):
    rows = get_trade_detail_data(account, "stock", "order") or []
    return [row for row in rows if str(getattr(row, "m_strRemark", "")).startswith(tag)]


def live_plan(C):
    """Scores of the last finished session, or None (logged) when the data is not current."""
    began = time.time()
    download_recent_bars(C)
    clock = beijing_clock()
    inputs = build_inputs(C, clock[:8], PAPER_DAYS, log)
    listed = inputs["close"].index
    dates = complete_days(inputs, clock[:12])
    today = pd.Timestamp(clock[:8])
    if not len(dates):
        log("live.error", "no trading day with daily bars")
        return None
    last = dates[-1]
    log("live.data_through", last.strftime("%Y%m%d"))
    missing = listed[(listed > last) & (listed < today)]
    if len(missing):
        log("live.error", "daily bars of %s are missing: download the daily bars in QMT's data manager, then run again" % (
            ",".join(day.strftime("%Y%m%d") for day in missing)))
        return None
    scores = daily_scores(inputs, pd.DatetimeIndex([last]))[last]
    log("live.model", "%s year=%s" % (MODEL["run"], MODEL["prediction_year"]))
    log("live.top%d" % LIVE_TOPK, ",".join(ranked(scores)[:LIVE_TOPK]))
    log("live.plan_seconds", round(time.time() - began, 2))
    return {"date": last, "scores": scores, "st": inputs["quotes"]["st"]}


def tick_day(C, codes, st):
    """paper.Day from QMT's live ticks: open and last price in real yuan, factor 1, open-limit flags."""
    ticks = C.get_full_tick(list(codes)) or {}
    open_, last, factor, limit_buy, limit_sell, limits = {}, {}, {}, {}, {}, {}
    for code in codes:
        tick = ticks.get(code) or {}
        price, opened, pre = (float(tick.get(key) or 0.0) for key in ("lastPrice", "open", "lastClose"))
        factor[code] = 1.0
        if price <= 0 or opened <= 0 or pre <= 0:
            last[code] = np.nan
            continue
        pct = 0.05 if code in st else 0.10
        up, down = float(limit_price(pre, 1 + pct)), float(limit_price(pre, 1 - pct))
        open_[code], last[code], limits[code] = opened, price, (up, down)
        limit_buy[code] = round(opened, 2) >= round(up, 2)
        limit_sell[code] = round(opened, 2) <= round(down, 2)
    return Day(None, open_, last, factor, limit_buy, limit_sell), limits


def send(C, account, side, code, shares, price, tag):
    log("live.%s" % ("dry_" + side if LIVE_DRY_RUN else side), "%s %d shares limit %.2f (about %.0f yuan)" % (
        code, shares, price, shares * price))
    if not LIVE_DRY_RUN:
        passorder(SELL_OP if side == "sell" else BUY_OP, BY_SHARES, account, code, LIMIT_PRICE, float(price),
                  int(shares), "RQuant", RIGHT_AWAY, tag, C)


def live_sell(C, plan, account, tag):
    cash, positions = read_account(account)
    held = sorted(positions)
    codes = sorted(set(plan["scores"].index) | set(held))
    day, limits = tick_day(C, codes, plan["st"])
    if sum(1 for code in plan["scores"].index if code in day.open) < 0.5 * len(plan["scores"]):
        return None  # the market has not opened
    value = sum(positions[code][0] * day.close[code] for code in held if day.close[code] == day.close[code])
    budget = max(0.0, min(cash, LIVE_CAPITAL - value))
    log("live.account", "%s cash %.2f holdings %d value %.2f budget %.2f" % (
        masked(account), cash, len(held), value, budget))
    book = Book(budget, dict((code, float(positions[code][0])) for code in held),
                dict((code, day.close[code]) for code in held),
                dict((code, 1 if positions[code][1] > 0 else 0) for code in held))
    sells, buys = decide(plan["scores"], book, day, LIVE_TOPK, LIVE_N_DROP)
    sold = 0.0
    for code, shares in sells:
        sellable = positions[code][1]
        price = max(limits[code][1], round(day.close[code] * (1 - LIVE_SLIP), 2))
        if sellable > 0:
            send(C, account, "sell", code, sellable, price, tag)
            sold += sellable * price * (1 - CLOSE_COST)
    return {"day": day, "limits": limits, "buys": buys, "sold": sold, "value_after": value - sum(
        positions[code][0] * day.close[code] for code, _ in sells)}


def live_buy(C, sold, account, tag):
    cash, _ = read_account(account)
    if LIVE_DRY_RUN:
        cash += sold["sold"]  # nothing was really sold
    day, limits, buys = sold["day"], sold["limits"], sold["buys"]
    budget = max(0.0, min(cash, LIVE_CAPITAL - sold["value_after"]))
    orders = []
    for code, amount in buys:
        price = min(limits[code][0], round(day.close[code] * (1 + LIVE_SLIP), 2))
        orders.append((code, amount, price))
    planned = sum(amount * price for _, amount, price in orders) * (1 + OPEN_COST)
    scale = min(1.0, budget / planned) if planned > 0 else 0.0
    log("live.buy_budget", "%.2f planned %.2f scale %.3f" % (budget, planned, scale))
    ceiling = LIVE_MAX_ORDER * LIVE_CAPITAL / LIVE_TOPK
    for code, amount, price in orders:
        shares = int(amount * scale // TRADE_UNIT * TRADE_UNIT)
        if shares <= 0:
            log("live.skip_buy", "%s too dear for its share of the cash" % code)
        elif shares * price > ceiling:
            log("live.skip_buy", "%s order %.0f yuan above the %.0f ceiling" % (code, shares * price, ceiling))
        else:
            send(C, account, "buy", code, shares, price, tag)


def live_step(C, plan):
    clock = beijing_clock()
    moment = clock[8:]
    in_session = LIVE_SELL_AT <= moment < LIVE_STOP_AT
    if not LIVE_DRY_RUN and not in_session:
        if moment >= LIVE_STOP_AT:
            log("live.skip", "past %s Beijing time: no orders today" % LIVE_STOP_AT)
            _live["done"] = True
        return
    account = account_id()
    if not account:
        raise ValueError("no account: select one for the strategy or set LIVE_ACCOUNT")
    tag = "%s%s" % (LIVE_TAG, clock[:8])
    if _live["sold"] is None:
        if not LIVE_DRY_RUN and sent_today(account, tag):
            log("live.skip", "orders tagged %s already sent today" % tag)
            _live["done"] = True
            return
        _live["sold"] = live_sell(C, plan, account, tag)
        if _live["sold"] is None:
            if LIVE_DRY_RUN and not in_session:
                log("live.error", "no live quotes: the dry run needs today's ticks")
                _live["done"] = True
            return
    if LIVE_DRY_RUN or moment >= LIVE_BUY_AT:
        live_buy(C, _live["sold"], account, tag)
        log("live.done", "dry run, nothing sent" if LIVE_DRY_RUN else "orders sent")
        _live["done"] = True


def init(C):
    log("live", "start dry_run=%s" % LIVE_DRY_RUN)


def handlebar(C):
    if not C.is_last_bar():
        return
    day = beijing_clock()[:8]
    if _live["day"] != day:
        # A new Beijing day: plan afresh, so a strategy left running trades each day on that day's plan.
        _live.update(day=day, planned=False, plan=None, sold=None, done=False)
    if _live["done"]:
        return
    if not _live["planned"]:
        _live["planned"] = True
        try:
            _live["plan"] = live_plan(C)
        except Exception:
            log("live.error", traceback.format_exc())
        if _live["plan"] is None:
            _live["done"] = True
            return
    try:
        live_step(C, _live["plan"])
    except Exception:
        log("live.error", traceback.format_exc())
        _live["done"] = True
