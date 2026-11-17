# Paper book for the live strategy: Qlib's TopkDropoutStrategy (topk, n_drop, method top/bottom, hold_thresh 1,
# only_tradable, risk_degree 1) traded at the next open by Qlib's SimulatorExecutor with RQuant's A-share costs.
# Amounts are Qlib's adjusted amounts (real shares / adjustment factor) and prices are adjusted prices, so a
# replay from the same book gives the same orders as RQuant's backtest. Pure pandas/numpy; no orders are placed.
import numpy as np
import pandas as pd

TRADE_UNIT = 100
COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005  # sells from 2023-08-28
# Qlib clips orders against cash with its own cost ratios before RQuant's cost model prices them.
OPEN_COST = COMMISSION
CLOSE_COST = COMMISSION + 0.001
SELL, BUY = 0, 1


def trade_cost(value, sell):
    if value <= 1e-5:
        return 0.0
    cost = max(MIN_COMMISSION, value * COMMISSION)
    return cost + (value * STAMP_TAX if sell else 0.0)


def round_amount(amount, factor):
    """Qlib's round_amount_by_trade_unit: whole lots of real shares, back in adjusted amount."""
    return (amount * factor + 0.1) // TRADE_UNIT * TRADE_UNIT / factor


class Day(object):
    """One trading day's quotes: adjusted open/close and adjustment factor per code, and the open-limit flags.

    A code without a finite close is suspended, as in Qlib.
    """

    def __init__(self, date, open_, close, factor, limit_buy, limit_sell):
        self.date = date
        self.open = open_
        self.close = close
        self.factor = factor
        self.limit_buy = limit_buy
        self.limit_sell = limit_sell

    def suspended(self, code):
        value = self.close.get(code, np.nan)
        return not (value == value)

    def tradable(self, code, direction=None):
        if self.suspended(code):
            return False
        buy, sell = bool(self.limit_buy.get(code, False)), bool(self.limit_sell.get(code, False))
        if direction is None:
            return not (buy or sell)
        return not (buy if direction == BUY else sell)

    def deal_price(self, code):
        value = self.open.get(code, np.nan)
        if not (value == value) or value <= 1e-08:
            value = self.close.get(code, np.nan)
        return value


class Book(object):
    def __init__(self, cash, amount=None, price=None, count=None):
        self.cash = float(cash)
        self.amount = dict(amount or {})
        self.price = dict(price or {})
        self.count = dict(count or {})

    def copy(self):
        return Book(self.cash, self.amount, self.price, self.count)

    def value(self):
        return self.cash + sum(self.amount[code] * self.price[code] for code in self.amount)

    def deal(self, code, direction, amount, day):
        """Qlib Exchange.deal_order against this book; returns (amount, price, value, cost), amount 0 if not dealt."""
        price = day.deal_price(code)
        factor = day.factor[code]
        if direction == SELL:
            held = self.amount.get(code, 0.0)
            if not np.isclose(amount, held):
                amount = round_amount(min(held, amount), factor)
            if self.cash + amount * price < max(amount * price * CLOSE_COST, MIN_COMMISSION):
                amount = 0.0
        else:
            value = amount * price
            if self.cash < max(value * OPEN_COST, MIN_COMMISSION):
                amount = 0.0
            elif self.cash < value + max(value * OPEN_COST, MIN_COMMISSION):
                amount = round_amount(min(_cash_limited_amount(price, self.cash), amount), factor)
            else:
                amount = round_amount(amount, factor)
        value = amount * price
        cost = trade_cost(value, direction == SELL)
        if value <= 1e-5:
            return 0.0, price, 0.0, 0.0
        if direction == SELL:
            left = self.amount[code] - amount
            if abs(left) < 1e-5:
                for table in (self.amount, self.price, self.count):
                    table.pop(code, None)
            else:
                self.amount[code] = left
            self.cash += value - cost
        else:
            if code not in self.amount:
                self.amount[code], self.price[code], self.count[code] = amount, price, 0
            else:
                self.amount[code] += amount
            self.cash -= value + cost
        return amount, price, value, cost

    def close_day(self, day):
        """Mark holdings to the close (suspended ones keep their price) and count one more held day."""
        for code in self.amount:
            if not day.suspended(code):
                self.price[code] = day.close[code]
            self.count[code] = self.count.get(code, 0) + 1


def _cash_limited_amount(price, cash):
    if cash < MIN_COMMISSION:
        return 0.0
    if cash >= MIN_COMMISSION / OPEN_COST + MIN_COMMISSION:
        return cash / (1 + OPEN_COST) / price
    return (cash - MIN_COMMISSION) / price


def ranked(scores):
    """Codes by score, highest first; ties by code, missing scores last.

    Qlib sorts with an unstable quicksort, so the order of equal scores depends on the numpy build; a short model
    (few trees) gives many equal scores, so the paper book fixes the order instead.
    """
    scores = scores.sort_index()
    return scores.index[np.argsort(-scores.values, kind="mergesort")]


def decide(scores, book, day, topk, n_drop):
    """Qlib TopkDropoutStrategy.generate_trade_decision for ``day``.

    ``scores`` is the previous trading day's prediction (a Series indexed by code); ``day`` holds the quotes the
    tradability checks and the sell simulation use. Ties in score are ordered by ``ranked``. Returns (sells, buys):
    sells and buys as (code, adjusted amount) in the order Qlib places them.
    """
    scores = scores.dropna()

    def first_n(codes, n, reverse=False):
        picked = []
        for code in (reversed(list(codes)) if reverse else codes):
            if day.tradable(code):
                picked.append(code)
                if len(picked) >= n:
                    break
        return picked[::-1] if reverse else picked

    temp = book.copy()
    current = list(temp.amount)
    last = ranked(scores.reindex(current))
    others = ranked(scores[~scores.index.isin(last)])
    today = first_n(others, n_drop + topk - len(last))
    comb = ranked(scores.reindex(last.union(pd.Index(today))))
    sell = last[last.isin(first_n(comb, n_drop, reverse=True))]
    buy = today[: len(sell) + topk - len(last)]
    sells = []
    for code in current:
        if not day.tradable(code, SELL):
            continue
        if code in sell:
            if temp.count.get(code, 0) < 1:
                continue
            amount = temp.amount[code]
            sells.append((code, amount))
            temp.deal(code, SELL, amount, day)
    cash = temp.cash
    value = cash / len(buy) if len(buy) > 0 else 0
    buys = []
    for code in buy:
        if not day.tradable(code, BUY):
            continue
        amount = round_amount(value / day.deal_price(code), day.factor[code])
        buys.append((code, amount))
    return sells, buys


def trade_day(scores, book, day, topk, n_drop):
    """Decide with the previous day's ``scores``, execute at ``day``'s open, mark at its close. Returns the fills."""
    sells, buys = decide(scores, book, day, topk, n_drop)
    fills = []
    for direction, orders in ((SELL, sells), (BUY, buys)):
        for code, amount in orders:
            dealt, price, value, cost = book.deal(code, direction, amount, day)
            if dealt > 0:
                fills.append({"date": day.date, "code": code, "side": "sell" if direction == SELL else "buy",
                              "amount": dealt, "shares": dealt * day.factor[code], "price": price,
                              "value": value, "cost": cost})
    book.close_day(day)
    return fills
