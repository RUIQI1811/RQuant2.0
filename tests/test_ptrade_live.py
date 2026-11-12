"""The PTrade live strategy (ptrade/runtime/ptrade_live.py, bundled as ptrade/generated/jq240_ptrade_live.py) against
a fake PTrade trading module: the synthetic market of the QMT tests, a fake account, snapshots and order capture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))
sys.path.insert(0, str(ROOT / "tests"))

import build_jq240  # noqa: E402
import test_ptrade_inputs as pti  # noqa: E402
import test_qmt_inputs as qmt  # noqa: E402

TODAY = pd.Timestamp("2011-07-01")  # the session after the synthetic market's last daily bar


class LiveFake(pti.FakePTrade):
    """FakePTrade whose calendar also lists TODAY, which has no daily bar yet (get_price stops before today)."""

    def get_trade_days(self, start, end):
        days = qmt.CALENDAR.append(pd.DatetimeIndex([TODAY]))
        days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
        return np.array(days.strftime("%Y-%m-%d"))


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, text) -> None:
        self.lines.append(str(text))

    warn = error = info


class Holder:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


class Live:
    """The bundled strategy run against the fakes. ``clock`` is Beijing time YYYYMMDDHHMMSS and can be moved."""

    def __init__(self, positions: dict, cash: float = 50_000.0, dry_run: bool = False, orders_today=()) -> None:
        self.fake = LiveFake()
        self.logger = Logger()
        self.sent: list[tuple] = []
        self.jobs: list[tuple] = []
        self.account_orders = [dict(symbol=code) for code in orders_today]
        self.clock = "20110701080000"
        self.context = Holder(portfolio=Holder(cash=cash, positions={
            code: Holder(amount=shares, enable_amount=free) for code, (shares, free) in positions.items()}))
        self.g = Holder()
        names: dict = {"log": self.logger, "g": self.g}
        source = build_jq240.build_ptrade_live()
        exec(compile(source, str(build_jq240.PTRADE_LIVE_OUTPUT), "exec"), names)  # noqa: S102 - our own file
        for name in pti.API:
            names[name] = getattr(self.fake, name)
        features = [f"jq_{ordinal:03d}" for ordinal in range(1, 234)]
        names.update(
            MARKET_MIN_STOCKS=1, LIVE_DAYS=qmt.DAYS, LIVE_TOPK=2, LIVE_N_DROP=1, LIVE_CAPITAL=100_000.0,
            LIVE_DRY_RUN=dry_run, beijing_clock=lambda: self.clock, get_snapshot=self.get_snapshot,
            order=self.order, get_all_orders=lambda: list(self.account_orders),
            run_daily=lambda context, func, time: self.jobs.append((func.__name__, time)),
            get_user_name=lambda: "1234567890",
            MODEL={"run": "fake", "prediction_year": 2011, "fit_period": ["2010-01-01", "2010-12-31"],
                   "features": features, "mean": [0.0] * 233, "std": [1.0] * 233,
                   "trees": [{"feature": [0], "threshold": [0.0], "left": [~0], "right": [~1], "leaf": [-1.0, 1.0]}]},
        )
        self.names = names

    def close(self, code: str) -> float:
        return float(self.fake.bars[code]["close"].iloc[-1])

    def get_snapshot(self, codes):
        """In the auction (before 09:25) no open or last price yet; after it, open 1% and last 0.5% above the close."""
        opened = self.clock[8:12] >= "0925"
        snaps = {}
        for code in codes:
            if code not in self.fake.bars:
                continue
            pre = self.close(code)
            pct = 0.05 if code == "600002.SS" else 0.10
            snaps[code] = {"preclose_px": pre, "open_px": pre * 1.01 if opened else 0.0,
                           "last_px": pre * 1.005 if opened else 0.0,
                           "up_px": float(self.names["limit_price"](pre, 1 + pct)),
                           "down_px": float(self.names["limit_price"](pre, 1 - pct)),
                           "trade_status": "TRADE" if opened else "OCALL"}
        return snaps

    def order(self, code, amount, limit_price=None):
        self.sent.append((code, amount, limit_price))
        self.account_orders.append({"symbol": code})
        return f"order-{len(self.sent)}"

    def run(self, name: str, clock: str):
        self.clock = clock
        if name == "before_trading_start":
            return self.names[name](self.context, None)
        return self.names[name](self.context)

    def lines(self) -> list[str]:
        return self.logger.lines


def test_jobs_and_auction_trade_sells_and_buys_at_once() -> None:
    # 600002 is an ST name, out of the universe, so it goes; the SZ holding is not the strategy's.
    live = Live({"600002.SS": (300, 300), "000001.SZ": (500, 500)})
    live.run("initialize", "20110701080000")
    assert live.jobs == [("rq_prepare", "08:00"), ("rq_auction", "09:21"), ("rq_after_open_sell", "09:31"),
                         ("rq_after_open_buy", "09:32")]
    live.run("rq_prepare", "20110701080000")
    assert live.sent == []
    live.run("before_trading_start", "20110701091000")  # the plan is in hand: nothing more
    live.run("rq_auction", "20110701092100")
    sells = [order for order in live.sent if order[1] < 0]
    buys = [order for order in live.sent if order[1] > 0]
    assert [(code, amount) for code, amount, _ in sells] == [("600002.SS", -300)]
    assert abs(sells[0][2] - live.names["limit_price"](live.close("600002.SS"), 0.95)) < 1e-9  # limit-down, ST 5%
    assert buys and {code for code, _, _ in buys} <= {"600001.SS", "600003.SS", "600004.SS"}
    assert all(amount % 100 == 0 and price <= round(live.close(code) * 1.03, 2) + 1e-9 for code, amount, price in buys)
    assert sum(amount * price for _, amount, price in buys) <= 50_000.0  # only the cash before the auction
    count = len(live.sent)
    live.run("rq_auction", "20110701092130")
    live.run("rq_after_open_sell", "20110701093100")
    live.run("rq_after_open_buy", "20110701093200")
    assert len(live.sent) == count and live.g.rq_done == "20110701"
    lines = live.lines()
    assert "[RQ] live.data_through=20110630" in lines
    assert not any("error" in line for line in lines), lines[-8:]
    assert any(line.startswith("[RQ] live.top2=") and ".SH" in line for line in lines)
    assert not any("1234567890" in line for line in lines)


def test_dry_run_logs_orders_at_start_and_sends_nothing() -> None:
    live = Live({"600002.SS": (300, 300)}, dry_run=True)
    live.run("initialize", "20110701200000")
    live.run("before_trading_start", "20110701200000")  # started in the evening: plans and logs at once
    lines = live.lines()
    assert live.sent == []
    assert any(line.startswith("[RQ] live.dry_sell=600002.SS 300 shares") for line in lines), lines[-10:]
    assert any(line.startswith("[RQ] live.dry_buy=") for line in lines)
    assert "[RQ] live.done=dry run, nothing sent" in lines and not any("error" in line for line in lines)


def test_after_open_fallback_when_the_plan_misses_the_auction() -> None:
    live = Live({"600002.SS": (300, 300)})
    live.run("rq_auction", "20110701092100")  # no plan yet
    assert live.sent == [] and any("live.skip=no plan yet" in line for line in live.lines())
    live.run("rq_prepare", "20110701092400")
    live.run("rq_after_open_sell", "20110701093100")
    assert [(code, amount) for code, amount, _ in live.sent] == [("600002.SS", -300)]
    last = live.close("600002.SS") * 1.005
    assert abs(live.sent[0][2] - round(last * 0.99, 2)) < 1e-9  # marketable: 1% under the last price
    live.run("rq_after_open_buy", "20110701093200")
    assert any(amount > 0 for _, amount, _ in live.sent[1:])
    assert "[RQ] live.done=orders sent after the open" in live.lines()


def test_orders_already_in_the_account_today_stop_the_strategy() -> None:
    live = Live({"600002.SS": (300, 300)}, orders_today=["600001.SS"])
    live.run("rq_prepare", "20110701080000")
    live.run("rq_auction", "20110701092100")
    assert live.sent == [] and live.g.rq_done == "20110701"
    assert any("already has 1 sh_main orders today" in line for line in live.lines())


def test_no_trading_on_a_day_off_or_with_a_stale_plan() -> None:
    live = Live({"600002.SS": (300, 300)})
    live.run("rq_prepare", "20110701080000")
    live.run("rq_auction", "20110702092100")  # a Saturday
    assert live.sent == [] and any("is not a trading day" in line for line in live.lines())
    live.names["_pt"]["plan"]["date"] = pd.Timestamp("2011-06-29")
    live.run("rq_auction", "20110701092100")
    assert live.sent == [] and any("not the last session 20110630" in line for line in live.lines())


def test_a_new_day_plans_again_and_the_cache_spares_old_lookups() -> None:
    live = Live({})
    live.run("rq_prepare", "20110630080000")
    first = dict(live.fake.calls)
    live.run("rq_prepare", "20110630090000")  # same day: the plan stands
    assert live.fake.calls == first
    live.run("rq_prepare", "20110701080000")
    again = {name: live.fake.calls[name] - first.get(name, 0) for name in live.fake.calls}
    recent = live.names["CACHE_RECENT_DAYS"]
    # Only the new day and the last few (which may still change) are looked up again.
    assert again["get_stock_status.ST"] <= recent + 1
    assert again["get_fundamentals.valuation"] <= recent + 1
    assert live.names["_pt"]["plan"]["made"] == "20110701"


def test_committed_ptrade_live_bundle_is_current() -> None:
    source = build_jq240.PTRADE_LIVE_OUTPUT.read_text(encoding="ascii")
    assert source == build_jq240.build_ptrade_live()
    for word in ("import os", "import sys", "import traceback", "globals(", "__import__", "\ndef log("):
        assert word not in source
    assert "LIVE_DRY_RUN = True" in source
