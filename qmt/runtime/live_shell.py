# jq240 + LightGBM live trading inside QMT (run in QMT's live-trading mode with the account selected).
#
# Start it by 08:45 Beijing time and leave it running; each Beijing day it plans afresh. On the day's first bar (or
# its timer) it refreshes recent daily bars and scores sh_main with the last finished session's data (about 15
# minutes). With LIVE_AUCTION (the default) it trades in the opening call auction, so it deals at the open as the
# backtest does: between LIVE_AUCTION_AT and LIVE_AUCTION_END it reads the account, decides like the backtest's
# TopkDropoutStrategy (top LIVE_TOPK, drop at most LIVE_N_DROP a day) against the real holdings and the previous
# close, and sends the sells at the limit-down price and the buys at most LIVE_AUCTION_BUY above the previous close;
# the auction fills both at the open. The buys can only use the cash on hand before the auction (the sales pay out at
# 09:25), so the account keeps about one day's sales in cash. If the plan is not ready by LIVE_AUCTION_END it trades
# after the open instead: sells at LIVE_SELL_AT, buys at LIVE_BUY_AT and LIVE_BUY_GAP seconds after the sells, as
# marketable limit orders. It trades at most once a day and never after LIVE_STOP_AT.
#
# LIVE_DRY_RUN = True (the default) sends nothing: it prints the orders it would send, at once if the market is shut.
# Set it to False in the QMT editor only after a dry run looks right. The account number stays in the QMT editor.
LIVE_DRY_RUN = True
LIVE_ACCOUNT = ""  # empty: the account selected for the strategy in QMT
LIVE_CAPITAL = 0  # the most the strategy may hold (holdings at market plus cash); 0 = all its holdings and cash
LIVE_TOPK = 30
LIVE_N_DROP = 3
LIVE_KEEP = set()  # sh_main codes the strategy must leave alone, e.g. set(["600000.SH"])
LIVE_AUCTION = True  # trade in the opening call auction (at the open); False: after the open, as below
LIVE_AUCTION_AT = "092100"  # orders sent from 09:20 on cannot be cancelled, so the auction keeps them
LIVE_AUCTION_END = "092450"
LIVE_AUCTION_BUY = 0.03  # auction buy limit above the previous close (the cash it ties up); above it, no fill at the open
LIVE_TIMER = True  # also wake every 30 s on a QMT timer, so the plan is ready before the auction without ticks
LIVE_SELL_AT = "093005"
LIVE_BUY_AT = "093025"
LIVE_BUY_GAP = 20  # seconds between sending the sells and the buys, whatever the clock says, so the sales can fill
LIVE_STOP_AT = "100000"
LIVE_SLIP = 0.01  # limit price offset from the last price; capped at the daily limit
LIVE_MAX_ORDER = 2.5  # no single buy above this many equal slots (strategy money / LIVE_TOPK)
LIVE_FILL_MAX = 1.3  # idle cash tops names up lot by lot, to at most this many slots each
LIVE_TOPUP = False  # False: only today's new buys get the idle cash; True: held names are topped up too
LIVE_FIRST_LOT_MAX = 1.5  # a new name whose one lot is above a slot is still bought if the lot is under this
LIVE_TOPUP_MIN = 0.2  # top-ups of a held name under this many slots are not sent (5 yuan minimum commission)
LIVE_RETRY = 600  # seconds before planning again after the data was not ready
LIVE_TAG = "RQ"
BUY_OP, SELL_OP, BY_SHARES, LIMIT_PRICE, RIGHT_AWAY = 23, 24, 1101, 11, 2
_live = {"day": None, "planned": False, "plan": None, "sold": None, "sold_at": None, "retry_at": 0, "done": False}


def beijing_clock():
    """Beijing wall-clock time as YYYYMMDDHHMMSS, whatever the machine's time zone."""
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(time.time() + 8 * 3600))


def seconds():
    return time.time()


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
    counts = inputs["close"].notna().sum(axis=1)
    log("live.bars", ",".join("%s:%d" % (day.strftime("%Y%m%d"), counts[day]) for day in listed[-3:]))
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


def tick_day(C, codes, st, auction=False):
    """paper.Day from QMT's live ticks: open and last price in real yuan, factor 1, open-limit flags. In the auction
    there is no open yet: the previous close stands for both, and no code counts as opening at a limit."""
    ticks = C.get_full_tick(list(codes)) or {}
    open_, last, factor, limit_buy, limit_sell, limits = {}, {}, {}, {}, {}, {}
    for code in codes:
        tick = ticks.get(code) or {}
        price, opened, pre = (float(tick.get(key) or 0.0) for key in ("lastPrice", "open", "lastClose"))
        if auction and pre > 0:
            price = opened = pre
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


def live_sell(C, plan, account, tag, auction=False):
    cash, positions = read_account(account)
    held = sorted(positions)
    codes = sorted(set(plan["scores"].index) | set(held))
    day, limits = tick_day(C, codes, plan["st"], auction)
    if sum(1 for code in plan["scores"].index if code in day.open) < 0.5 * len(plan["scores"]):
        return None  # the market has not opened
    value = sum(positions[code][0] * day.close[code] for code in held if day.close[code] == day.close[code])
    capital = LIVE_CAPITAL if LIVE_CAPITAL > 0 else cash + value
    budget = max(0.0, min(cash, capital - value))
    log("live.account", "%s cash %.2f holdings %d value %.2f budget %.2f at %s" % (
        masked(account), cash, len(held), value, budget, beijing_clock()[8:]))
    book = Book(budget, dict((code, float(positions[code][0])) for code in held),
                dict((code, day.close[code]) for code in held),
                dict((code, 1 if positions[code][1] > 0 else 0) for code in held))
    sells, buys = decide(plan["scores"], book, day, LIVE_TOPK, LIVE_N_DROP)
    sold, sent, proceeds = 0.0, 0, 0.0
    kept = dict((code, positions[code][0]) for code in held)
    for code, shares in sells:
        sellable = positions[code][1]
        # In the auction a sell at the limit-down price fills at the open, whatever the open is.
        price = limits[code][1] if auction else max(limits[code][1], round(day.close[code] * (1 - LIVE_SLIP), 2))
        if sellable > 0:
            send(C, account, "sell", code, sellable, price, tag)
            sent += 1
            if day.close[code] == day.close[code]:
                proceeds += sellable * day.close[code] * (1 - CLOSE_COST)
            sold += sellable * price * (1 - CLOSE_COST)
            kept[code] -= sellable
            if kept[code] <= 0:
                kept.pop(code)
    value_after = sum(kept[code] * day.close[code] for code in kept if day.close[code] == day.close[code])
    return {"day": day, "limits": limits, "buys": buys, "sold": sold, "kept": kept, "value_after": value_after,
            "capital": capital, "sells": sent, "auction": auction, "proceeds": proceeds}


def _lots(shares):
    return int(shares // TRADE_UNIT * TRADE_UNIT)


def live_buy(C, sold, account, tag):
    """New names as the backtest sizes them, at most one slot (the strategy's money / LIVE_TOPK) each; then the idle
    cash goes out one lot at a time, always to the name furthest below its share, up to LIVE_FILL_MAX slots a name.
    With ~6,000 yuan slots and 100-share lots of up to 5,000 yuan, flooring each name to a slot left a quarter of the
    money idle."""
    cash, _ = read_account(account)
    if LIVE_DRY_RUN:
        cash += sold["sold"]  # nothing was really sold
    day, limits, kept = sold["day"], sold["limits"], sold["kept"]
    budget = max(0.0, min(cash, sold["capital"] - sold["value_after"]))
    slot = (sold["value_after"] + budget) / LIVE_TOPK
    ceiling = LIVE_MAX_ORDER * sold["capital"] / LIVE_TOPK
    slip = LIVE_AUCTION_BUY if sold.get("auction") else LIVE_SLIP

    def price_of(code):
        # Sized at the limit price: the broker holds back limit x shares of cash until the order fills.
        return min(limits[code][0], round(day.close[code] * (1 + slip), 2))

    orders = [(code, min(amount, slot / price_of(code)), price_of(code)) for code, amount in sold["buys"]]
    planned = sum(amount * price for _, amount, price in orders) * (1 + OPEN_COST)
    scale = min(1.0, budget / planned) if planned > 0 else 0.0
    log("live.buy_budget", "%.2f planned %.2f scale %.3f slot %.0f" % (budget, planned, scale, slot))
    left = budget
    shares = {}  # code -> shares to buy today
    for code, amount, price in orders:
        lot = TRADE_UNIT * price * (1 + OPEN_COST)
        count = _lots(amount * scale)
        if count <= 0 and TRADE_UNIT * price <= LIVE_FIRST_LOT_MAX * slot and lot <= left:
            count = TRADE_UNIT  # one lot is a little above a slot: still worth holding
        if count <= 0:
            log("live.skip_buy", "%s one lot (%.0f yuan) is too dear for its share of the cash" % (
                code, TRADE_UNIT * price))
            continue
        shares[code] = count
        left -= count * price * (1 + OPEN_COST)
    # Auction buys can only use the cash on hand before 09:25, so keep enough back that tomorrow starts with
    # LIVE_N_DROP slots of cash. Sales in today's auction pay out after it and count towards that.
    coming = sold.get("proceeds", 0.0) if sold.get("auction") else 0.0
    reserve = max(0.0, LIVE_N_DROP * slot - coming) if LIVE_AUCTION else 0.0
    log("live.reserve", "%.0f kept for tomorrow's auction buys" % reserve)
    # The rest of the cash, lot by lot, to whichever of today's buys (and held names, with LIVE_TOPUP) holds least.
    held = dict((code, kept[code] * day.close[code]) for code in kept
                if LIVE_TOPUP and day.close[code] == day.close[code] and day.tradable(code, BUY))
    names = set(held) | set(shares)
    added = dict((code, 0) for code in names)

    def value(code):
        return held.get(code, 0.0) + (shares.get(code, 0) + added[code]) * price_of(code)

    while True:
        best = None
        for code in names:
            price = price_of(code)
            lot = TRADE_UNIT * price * (1 + OPEN_COST)
            if lot > left - reserve or value(code) + TRADE_UNIT * price > LIVE_FILL_MAX * slot:
                continue
            if best is None or (value(code), code) < (value(best), best):
                best = code
        if best is None:
            break
        added[best] += TRADE_UNIT
        left -= TRADE_UNIT * price_of(best) * (1 + OPEN_COST)
    for code, amount, price in orders:
        count = shares.get(code, 0) + added.get(code, 0)
        if count <= 0:
            continue
        if count * price > ceiling:
            log("live.skip_buy", "%s order %.0f yuan above the %.0f ceiling" % (code, count * price, ceiling))
            continue
        send(C, account, "buy", code, count, price, tag)
    for code in sorted(held, key=lambda name: (held[name], name)):
        count = added.get(code, 0)
        if count and count * price_of(code) >= LIVE_TOPUP_MIN * slot:  # not worth a 5 yuan commission below
            send(C, account, "topup", code, count, price_of(code), tag)


def live_step(C, plan):
    clock = beijing_clock()
    moment = clock[8:]
    auction = LIVE_AUCTION and _live["sold"] is None and LIVE_AUCTION_AT <= moment < LIVE_AUCTION_END
    in_session = auction or LIVE_SELL_AT <= moment < LIVE_STOP_AT
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
        _live["sold"] = live_sell(C, plan, account, tag, auction and not LIVE_DRY_RUN)
        _live["sold_at"] = seconds()
        if _live["sold"] is None:
            if LIVE_DRY_RUN and not in_session:
                log("live.error", "no live quotes: the dry run needs today's ticks")
                _live["done"] = True
            return
    # Buy at LIVE_BUY_AT, and never sooner than LIVE_BUY_GAP seconds after sells were sent: a start after the open (or a
    # machine clock ahead of the exchange) must not send the buys with the sells, before the sales add to the cash.
    # In the auction the buys go with the sells: they spend only the cash on hand before it.
    waited = seconds() - _live["sold_at"] >= (LIVE_BUY_GAP if _live["sold"]["sells"] else 0)
    if LIVE_DRY_RUN or _live["sold"]["auction"] or (moment >= LIVE_BUY_AT and waited):
        live_buy(C, _live["sold"], account, tag)
        log("live.done", "dry run, nothing sent" if LIVE_DRY_RUN else "orders sent")
        _live["done"] = True


def init(C):
    log("live", "start dry_run=%s auction=%s" % (LIVE_DRY_RUN, LIVE_AUCTION))
    if LIVE_TIMER:
        try:
            C.run_time("live_timer", "30nSecond", "2020-01-01 00:00:00", "SH")
            log("live.timer", "every 30 s")
        except Exception:
            log("live.timer_error", traceback.format_exc().splitlines()[-1])


def live_timer(C):
    live_tick(C)


def handlebar(C):
    if not C.is_last_bar():
        return
    live_tick(C)


def live_tick(C):
    day = beijing_clock()[:8]
    if _live["day"] != day:
        # A new Beijing day: plan afresh, so a strategy left running trades each day on that day's plan.
        _live.update(day=day, planned=False, plan=None, sold=None, sold_at=None, retry_at=0, done=False)
    if _live["done"]:
        return
    if not _live["planned"]:
        if seconds() < _live["retry_at"]:
            return
        _live["planned"] = True
        try:
            _live["plan"] = live_plan(C)
        except Exception:
            log("live.error", traceback.format_exc())
        if _live["plan"] is None:
            # Data not ready (say the timer planned before the last session's bars were in): try again later today,
            # until the trading window has passed.
            if LIVE_DRY_RUN or beijing_clock()[8:] >= LIVE_STOP_AT:
                _live["done"] = True
            else:
                _live.update(planned=False, retry_at=seconds() + LIVE_RETRY)
                log("live.retry", "planning again in %d s" % LIVE_RETRY)
            return
    try:
        live_step(C, _live["plan"])
    except Exception:
        log("live.error", traceback.format_exc())
        _live["done"] = True
