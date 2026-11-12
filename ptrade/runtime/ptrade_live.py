# jq240 + LightGBM live trading in Guojin PTrade (the trading module: 量化 -> 交易). Paste the whole generated file
# (ptrade/generated/jq240_ptrade_live.py) into a new PTrade strategy, daily frequency, and start a trade with it.
#
# PTrade runs the strategy on the broker's servers and leaves it running from day to day. Each trading day:
# * LIVE_PLAN_AT (08:00 Beijing) builds the jq240 inputs from PTrade's data through the last finished session and
#   scores sh_main with the exported LightGBM window. The first run after a start takes about 20 minutes; the per-day
#   lookups stay in memory, so later days take a few minutes. It also plans at once when the trade is started.
# * LIVE_AUCTION_AT (09:21) decides like the backtest's TopkDropoutStrategy (top LIVE_TOPK, drop at most LIVE_N_DROP)
#   against the real holdings and the previous close, and sends the sells at the limit-down price and the buys at
#   most LIVE_AUCTION_BUY above the previous close, so the opening auction fills both at the open. Buys only use the
#   cash on hand before the auction, so the account keeps about LIVE_N_DROP slots in cash for the next day.
# * If the plan was not ready for the auction: sells at LIVE_SELL_AT and buys at LIVE_BUY_AT after the open, as
#   marketable limit orders.
# It trades at most once a day, and not at all on a day the account already has orders for sh_main stocks.
#
# LIVE_DRY_RUN = True (the default) sends nothing: it logs the orders it would send, once at start and then at each
# auction. Set it to False only after a dry run looks right. The account is the one the PTrade login trades; nothing
# about it is written here.
LIVE_DRY_RUN = True
LIVE_CAPITAL = 0  # the most the strategy may hold (holdings at market plus cash); 0 = all its holdings and cash
LIVE_TOPK = 30
LIVE_N_DROP = 3
LIVE_KEEP = set()  # sh_main codes the strategy must leave alone, e.g. set(["600000.SS"])
LIVE_AUCTION = True
LIVE_PLAN_AT = "08:00"
LIVE_AUCTION_AT = "09:21"  # orders sent from 09:20 on cannot be cancelled, so the auction keeps them
LIVE_SELL_AT = "09:31"
LIVE_BUY_AT = "09:32"
LIVE_AUCTION_BUY = 0.03  # auction buy limit above the previous close (the cash it ties up); above it, no fill at the open
LIVE_SLIP = 0.01  # after the open: limit price offset from the last price; capped at the daily limit
LIVE_MAX_ORDER = 2.5  # no single buy above this many equal slots (strategy money / LIVE_TOPK)
LIVE_FILL_MAX = 1.3  # idle cash tops today's buys up lot by lot, to at most this many slots each
LIVE_TOPUP = False  # False: only today's new buys get the idle cash; True: held names are topped up too
LIVE_FIRST_LOT_MAX = 1.5  # a new name whose one lot is above a slot is still bought if the lot is under this
LIVE_TOPUP_MIN = 0.2  # top-ups of a held name under this many slots are not sent (5 yuan minimum commission)
LIVE_DAYS = 1100  # trading days of history the factors are computed on (as the QMT strategy)
LIVE_TAG = "[RQ]"
SNAPSHOT_CHUNK = 300
HALTED = ("HALT", "SUSP", "STOPT", "DELISTED")
_pt = {"context": None, "plan": None, "planning": False, "sold": None, "done": None, "dry": set(), "cache": {}}


def rq_log(key, value=""):
    log.info("%s %s=%s" % (LIVE_TAG, key, value))  # noqa: F821 - PTrade's logger


def beijing_clock():
    """Beijing wall-clock time as YYYYMMDDHHMMSS, whatever the server's time zone."""
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(time.time() + 8 * 3600))


def _where(exc):
    """Where an exception came from, innermost last, without the standard traceback helpers (PTrade may refuse them)."""
    spots, frame = [], exc.__traceback__
    while frame is not None:
        spots.append("%s:%d" % (frame.tb_frame.f_code.co_name, frame.tb_lineno))
        frame = frame.tb_next
    return "%s: %s at %s" % (type(exc).__name__, exc, " > ".join(spots[-5:]))


def managed(code):
    return code.startswith("60") and code.endswith(".SS") and code not in LIVE_KEEP


def account_label():
    try:
        return str(get_user_name() or "ptrade")  # noqa: F821
    except Exception:
        return "ptrade"


def read_account(account):
    """(available cash, {code: (shares, sellable shares)}) for the strategy's sh_main codes."""
    portfolio = _pt["context"].portfolio
    positions = {}
    for code, position in (portfolio.positions or {}).items():
        shares = int(getattr(position, "amount", 0) or 0)
        if shares > 0 and managed(code):
            positions[code] = (shares, int(getattr(position, "enable_amount", 0) or 0))
    return float(portfolio.cash), positions


def sent_today():
    """Today's orders in the account for sh_main stocks (any source), as get_all_orders lists them."""
    rows = get_all_orders() or []  # noqa: F821
    return [row for row in rows if managed(str(row.get("symbol", "")))]


def tick_day(C, codes, st, auction=False):
    """paper.Day from PTrade's snapshots: open and last price in yuan, factor 1, open-limit flags, and the exchange's
    limit prices. In the auction the previous close stands for both prices and no code opens at a limit; outside
    trading hours (a dry run) a stock without a last price uses its previous close."""
    snaps = {}
    for chunk in _chunks(list(codes), SNAPSHOT_CHUNK):
        snaps.update(get_snapshot(chunk) or {})  # noqa: F821
    open_, last, factor, limit_buy, limit_sell, limits = {}, {}, {}, {}, {}, {}
    for code in codes:
        snap = snaps.get(code) or {}
        pre, price, opened = (float(snap.get(key) or 0.0) for key in ("preclose_px", "last_px", "open_px"))
        if auction or price <= 0:
            price = pre
        if auction or opened <= 0:
            opened = price
        factor[code] = 1.0
        if str(snap.get("trade_status", "")) in HALTED or price <= 0 or pre <= 0:
            last[code] = np.nan
            continue
        pct = 0.05 if code in st else 0.10
        up = float(snap.get("up_px") or 0.0) or float(limit_price(pre, 1 + pct))
        down = float(snap.get("down_px") or 0.0) or float(limit_price(pre, 1 - pct))
        open_[code], last[code], limits[code] = opened, price, (up, down)
        limit_buy[code] = (not auction) and round(opened, 2) >= round(up, 2)
        limit_sell[code] = (not auction) and round(opened, 2) <= round(down, 2)
    return Day(None, open_, last, factor, limit_buy, limit_sell), limits


def send(C, account, side, code, shares, price, tag):
    rq_log("live.%s" % ("dry_" + side if LIVE_DRY_RUN else side), "%s %d shares limit %.2f (about %.0f yuan)" % (
        code, shares, price, shares * price))
    if LIVE_DRY_RUN:
        return
    amount = -int(shares) if side == "sell" else int(shares)
    if order(code, amount, limit_price=round(float(price), 2)) is None:  # noqa: F821
        rq_log("live.order_failed", "%s %d" % (code, amount))


def previous_session(today):
    """The last trading day before ``today`` (YYYYMMDD), and whether ``today`` is a trading day."""
    days = trading_calendar(today)
    stamp = pd.Timestamp(today)
    before = days[days < stamp]
    return (before[-1] if len(before) else None), bool(len(days) and days[-1] == stamp)


def live_plan():
    """Scores of the last finished session, or None (logged) when the data is not current."""
    began = time.time()
    clock = beijing_clock()
    inputs = build_inputs(clock[:8], LIVE_DAYS, rq_log, cache=_pt["cache"])
    listed = inputs["close"].index
    counts = inputs["close"].notna().sum(axis=1)
    rq_log("live.bars", ",".join("%s:%d" % (day.strftime("%Y%m%d"), counts[day]) for day in listed[-3:]))
    dates = complete_days(inputs, clock[:12])
    today = pd.Timestamp(clock[:8])
    if not len(dates):
        rq_log("live.error", "no trading day with daily bars")
        return None
    last = dates[-1]
    rq_log("live.data_through", last.strftime("%Y%m%d"))
    missing = listed[(listed > last) & (listed < today)]
    if len(missing):
        rq_log("live.error", "daily bars of %s are missing" % ",".join(day.strftime("%Y%m%d") for day in missing))
        return None
    scores = daily_scores(inputs, pd.DatetimeIndex([last]))[last]
    rq_log("live.model", "%s year=%s" % (MODEL["run"], MODEL["prediction_year"]))
    rq_log("live.top%d" % LIVE_TOPK, ",".join(to_rquant(code) for code in ranked(scores)[:LIVE_TOPK]))
    rq_log("live.plan_seconds", round(time.time() - began, 2))
    return {"date": last, "scores": scores, "st": inputs["quotes"]["st"], "made": clock[:8]}


def rq_prepare(context):
    """Plan for the next session unless today's plan is in hand (or another job is making it)."""
    _pt["context"] = context
    today = beijing_clock()[:8]
    plan = _pt["plan"]
    if (plan is not None and plan["made"] == today) or _pt["planning"]:
        return
    _pt["planning"] = True
    try:
        _pt["plan"] = live_plan()
    except Exception as exc:
        rq_log("live.error", _where(exc))
    finally:
        _pt["planning"] = False


def _ready_plan(today):
    """Today's plan when it scores the session before today, else None (logged)."""
    plan = _pt["plan"]
    if plan is None:
        rq_log("live.skip", "no plan yet (still planning: %s)" % _pt["planning"])
        return None
    previous, trading = previous_session(today)
    if not trading:
        rq_log("live.skip", "%s is not a trading day" % today)
        return None
    if previous is None or plan["date"] != previous:
        rq_log("live.skip", "the plan scores %s, not the last session %s" % (
            plan["date"].strftime("%Y%m%d"), previous.strftime("%Y%m%d") if previous is not None else "?"))
        return None
    return plan


def rq_trade(context, mode):
    """mode 'auction' (sells and buys at once), 'sell' (after the open) or 'dry' (log only, at start)."""
    _pt["context"] = context
    clock = beijing_clock()
    today = clock[:8]
    if _pt["done"] == today or getattr(g, "rq_done", None) == today:  # noqa: F821
        return
    if LIVE_DRY_RUN and ((today, mode) in _pt["dry"] or (mode == "sell" and (today, "auction") in _pt["dry"])):
        return
    if mode != "dry":
        plan = _ready_plan(today)
    else:
        plan = _pt["plan"]
    if plan is None:
        return
    if not LIVE_DRY_RUN:
        orders = sent_today()
        if orders:
            rq_log("live.skip", "the account already has %d sh_main orders today: no orders" % len(orders))
            _pt["done"] = g.rq_done = today  # noqa: F821
            return
    account = account_label()
    sold = live_sell(context, plan, account, today, mode == "auction")
    if sold is None:
        rq_log("live.error", "no quotes for the planned stocks")
        return
    if LIVE_DRY_RUN:
        _pt["dry"].add((today, mode))
    if mode == "sell" and not LIVE_DRY_RUN:
        _pt["sold"] = (today, sold)
        return
    live_buy(context, sold, account, today)
    rq_log("live.done", "dry run, nothing sent" if LIVE_DRY_RUN else "orders sent at %s" % clock[8:])
    if not LIVE_DRY_RUN:
        _pt["done"] = g.rq_done = today  # noqa: F821


def rq_auction(context):
    if LIVE_AUCTION:
        try:
            rq_trade(context, "auction")
        except Exception as exc:
            rq_log("live.error", _where(exc))


def rq_after_open_sell(context):
    try:
        rq_trade(context, "sell")
    except Exception as exc:
        rq_log("live.error", _where(exc))


def rq_after_open_buy(context):
    _pt["context"] = context
    today = beijing_clock()[:8]
    held = _pt["sold"]
    if held is None or held[0] != today or _pt["done"] == today:
        return
    try:
        live_buy(context, held[1], account_label(), today)
        rq_log("live.done", "orders sent after the open")
    except Exception as exc:
        rq_log("live.error", _where(exc))
    _pt["done"] = g.rq_done = today  # noqa: F821


def initialize(context):
    _pt["context"] = context
    rq_log("live", "start dry_run=%s auction=%s topk=%d n_drop=%d" % (LIVE_DRY_RUN, LIVE_AUCTION, LIVE_TOPK,
                                                                      LIVE_N_DROP))
    run_daily(context, rq_prepare, time=LIVE_PLAN_AT)  # noqa: F821
    run_daily(context, rq_auction, time=LIVE_AUCTION_AT)  # noqa: F821
    run_daily(context, rq_after_open_sell, time=LIVE_SELL_AT)  # noqa: F821
    run_daily(context, rq_after_open_buy, time=LIVE_BUY_AT)  # noqa: F821


def before_trading_start(context, data):
    """Runs when the trade starts and at 09:10 each day: plans if LIVE_PLAN_AT did not; a dry run logs its orders."""
    rq_prepare(context)
    if LIVE_DRY_RUN:
        moment = beijing_clock()[8:12]
        if not ("0915" <= moment < "0935"):  # the auction and after-open jobs log their own dry run
            try:
                rq_trade(context, "dry")
            except Exception as exc:
                rq_log("live.error", _where(exc))


def handle_data(context, data):
    pass
