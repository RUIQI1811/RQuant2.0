"""The QMT paper book (qmt/runtime/paper.py): Qlib TopkDropoutStrategy decisions and SimulatorExecutor fills.

The book was also checked against RQuant's own backtest of sh_main jq240 (top 30, drop 3, 200k, 2026): started each
day from the backtest's book, its fills and account value matched on every day without tied scores among the
stocks involved; the other days differ only in which of two equal-score stocks is taken.
"""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = runpy.run_path(str(ROOT / "qmt" / "runtime" / "paper.py"))
Book, Day = PAPER["Book"], PAPER["Day"]


def day(prices: dict[str, float], *, factor: float = 1.0, limit_buy=(), limit_sell=(), suspended=()) -> object:
    codes = list(prices)
    close = {code: (np.nan if code in suspended else prices[code]) for code in codes}
    return Day(
        pd.Timestamp("2026-01-05"),
        dict(prices),
        close,
        {code: factor for code in codes},
        {code: code in limit_buy for code in codes},
        {code: code in limit_sell for code in codes},
    )


def scores(**values: float) -> pd.Series:
    return pd.Series(values)


def test_costs_and_lots() -> None:
    assert PAPER["trade_cost"](10_000.0, False) == 5.0
    assert PAPER["trade_cost"](100_000.0, False) == pytest.approx(30.0)
    assert PAPER["trade_cost"](100_000.0, True) == pytest.approx(30.0 + 50.0)
    assert PAPER["trade_cost"](0.0, True) == 0.0
    # 1234 adjusted shares at factor 0.5 are 617 real shares: six lots, back in adjusted units.
    assert PAPER["round_amount"](1234.0, 0.5) == pytest.approx(1200.0)


def test_empty_book_buys_the_top_tradable_names_equally() -> None:
    book = Book(30_000.0)
    quotes = day({"A": 10.0, "B": 20.0, "C": 5.0, "D": 8.0}, limit_buy={"B"})
    fills = PAPER["trade_day"](scores(A=0.9, B=0.8, C=0.7, D=0.1), book, quotes, 2, 1)
    # B opens limit-up, so the next name stands in. Each buy gets half the cash; the first one's commission leaves
    # too little for the second's last lot, so Qlib clips it by a lot.
    assert [(fill["code"], fill["side"], fill["shares"]) for fill in fills] == [("A", "buy", 1500.0),
                                                                              ("C", "buy", 2900.0)]
    assert book.cash == pytest.approx(30_000.0 - 15_000.0 - 5.0 - 14_500.0 - 5.0)


def test_drops_the_worst_holding_for_a_better_new_name() -> None:
    book = Book(0.0, {"A": 100.0, "B": 100.0}, {"A": 10.0, "B": 10.0}, {"A": 3, "B": 3})
    quotes = day({"A": 10.0, "B": 10.0, "C": 10.0})
    sells, buys = PAPER["decide"](scores(A=0.5, B=0.1, C=0.9), book, quotes, 2, 1)
    assert sells == [("B", 100.0)]
    # Cash after the simulated sale (1000 less the 5 yuan minimum commission and 0.05% stamp tax) buys C in lots.
    assert buys == [("C", 0.0)]
    rich = Book(5_000.0, book.amount, book.price, book.count)
    assert PAPER["decide"](scores(A=0.5, B=0.1, C=0.9), rich, quotes, 2, 1)[1] == [("C", 500.0)]


def test_untradable_holdings_are_skipped_when_choosing_what_to_drop() -> None:
    held = {"A": 100.0, "B": 100.0}
    quotes = day({"A": 10.0, "B": 10.0, "C": 10.0}, limit_sell={"B"})
    book = Book(10_000.0, held, {"A": 10.0, "B": 10.0}, {"A": 3, "B": 3})
    # B opens limit-down, so Qlib drops the worst tradable holding instead.
    assert PAPER["decide"](scores(A=0.5, B=0.1, C=0.9), book, quotes, 2, 1)[0] == [("A", 100.0)]
    # A holding bought today (held for 0 days) is not sold.
    fresh = Book(10_000.0, held, {"A": 10.0, "B": 10.0}, {"A": 3, "B": 0})
    assert PAPER["decide"](scores(A=0.5, B=0.1, C=0.9), fresh, day({"A": 10.0, "B": 10.0, "C": 10.0}), 2, 1)[0] == []


def test_holdings_outside_the_universe_are_sold_first() -> None:
    book = Book(0.0, {"A": 100.0, "X": 100.0}, {"A": 10.0, "X": 10.0}, {"A": 3, "X": 3})
    quotes = day({"A": 10.0, "X": 10.0, "C": 10.0})
    sells, _ = PAPER["decide"](scores(A=0.5, C=0.1), book, quotes, 2, 1)
    assert sells == [("X", 100.0)]


def test_suspended_holdings_keep_their_price() -> None:
    book = Book(0.0, {"A": 100.0}, {"A": 10.0}, {"A": 1})
    PAPER["trade_day"](scores(A=0.5), book, day({"A": 12.0}, suspended={"A"}), 1, 1)
    assert book.price["A"] == 10.0 and book.count["A"] == 2


def test_equal_scores_are_ranked_by_code() -> None:
    order = PAPER["ranked"](pd.Series({"C": 0.5, "A": 0.5, "B": 0.9, "D": np.nan}))
    assert list(order) == ["B", "A", "C", "D"]


def test_paper_modules_are_python36_ascii() -> None:
    for name in ("paper.py", "paper_shell.py"):
        source = (ROOT / "qmt" / "runtime" / name).read_bytes().decode("ascii")
        tree = ast.parse(source, feature_version=(3, 6))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert imported <= {"numpy", "pandas", "time", "traceback"}
        assert "passorder" not in source and "order_target" not in source
