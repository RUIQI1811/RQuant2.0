from __future__ import annotations

import importlib.util
import unittest


class _Calendar:
    def get_trade_step(self) -> int:
        return 3

    def get_trade_len(self) -> int:
        return 10

    def get_step_time(self, trade_step=None, shift=0):  # type: ignore[no-untyped-def]
        import pandas as pd

        day = pd.Timestamp("2024-01-10") - pd.Timedelta(days=shift)
        return day, day + pd.Timedelta(hours=23)

    def get_freq(self) -> str:
        return "day"


class _Signal:
    def __init__(self, scores) -> None:  # type: ignore[no-untyped-def]
        self.scores = scores

    def get_signal(self, start_time, end_time):  # type: ignore[no-untyped-def]
        return self.scores


class _Position:
    def __init__(self, cash: float, amounts: dict[str, float], counts: dict[str, int]) -> None:
        self.cash, self.amounts, self.counts = cash, dict(amounts), dict(counts)

    def get_cash(self) -> float:
        return self.cash

    def get_stock_list(self) -> list[str]:
        return list(self.amounts)

    def get_stock_amount(self, code: str) -> float:
        return self.amounts[code]

    def get_stock_count(self, code: str, bar: str) -> int:
        return self.counts[code]


class _Exchange:
    """Every stock trades at 10 a share unless it is listed as blocked; selling pays 0.1% cost."""

    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked

    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):  # type: ignore[no-untyped-def]
        return stock_id not in self.blocked

    def check_order(self, order) -> bool:  # type: ignore[no-untyped-def]
        return True

    def deal_order(self, order, position):  # type: ignore[no-untyped-def]
        value = order.amount * 10.0
        position.amounts.pop(order.stock_id)
        return value, value * 0.001, 10.0

    def get_deal_price(self, stock_id, start_time, end_time, direction):  # type: ignore[no-untyped-def]
        return 10.0

    def get_factor(self, stock_id, start_time, end_time) -> float:  # type: ignore[no-untyped-def]
        return 1.0

    def round_amount_by_trade_unit(self, amount: float, factor: float) -> float:
        return float(int(amount // 100) * 100)


@unittest.skipUnless(importlib.util.find_spec("qlib") is not None, "Qlib is required")
class ScoreRulesTests(unittest.TestCase):
    def _strategy(  # type: ignore[no-untyped-def]
        self, scores, holdings, *, topk=10, n_drop=3, blocked=(), min_score_z=None, exit_rank_pct=None
    ):
        from rquant.qlib_ext.strategy import ScoreRuleTopkDropoutStrategy

        strategy = object.__new__(ScoreRuleTopkDropoutStrategy)
        strategy.topk, strategy.n_drop = topk, n_drop
        strategy.method_sell, strategy.method_buy = "bottom", "top"
        strategy.hold_thresh, strategy.only_tradable, strategy.forbid_all_trade_at_limit = 1, True, False
        strategy.risk_degree = 1.0
        strategy.min_score_z, strategy.exit_rank_pct = min_score_z, exit_rank_pct
        strategy.signal = _Signal(scores)
        account = type("Account", (), {})()
        account.current_position = _Position(
            100_000.0, {code: 300.0 for code in holdings}, {code: 2 for code in holdings}
        )
        strategy.level_infra = {"trade_calendar": _Calendar()}
        strategy.common_infra = {"trade_account": account, "trade_exchange": _Exchange(set(blocked))}
        return strategy

    @staticmethod
    def _orders(decision) -> list[tuple[str, int, float]]:  # type: ignore[no-untyped-def]
        return [(order.stock_id, int(order.direction), order.amount) for order in decision.order_list]

    def _scores(self, seed: int, stocks: int = 60):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(size=stocks), index=[f"S{i:03d}" for i in range(stocks)])

    # Break caught: the re-implemented decision drifting from Qlib's TopkDropoutStrategy when neither rule binds
    # (which holdings it drops, how many it buys, cash per buy, untradable and unscored stocks).
    def test_rules_that_never_bind_reproduce_qlibs_topk_dropout_orders(self) -> None:
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

        for seed in range(25):
            with self.subTest(seed=seed):
                scores = self._scores(seed)
                codes = list(scores.index)
                holdings = codes[seed % 7 :: 6][: 4 + seed % 8] + (["GONE"] if seed % 3 == 0 else [])
                # The lowest score ranks at 100%, outside every exit_rank_pct below 1.
                holdings = [code for code in holdings if code != scores.idxmin()]
                blocked = set(codes[seed % 5 :: 9])
                options = {"topk": 10, "n_drop": 1 + seed % 3, "blocked": blocked}
                expected = TopkDropoutStrategy.generate_trade_decision(self._strategy(scores, holdings, **options))
                for rules in ({"min_score_z": -1e9}, {"exit_rank_pct": 0.999999}):
                    actual = self._strategy(scores, holdings, **options, **rules).generate_trade_decision()
                    self.assertEqual(self._orders(expected), self._orders(actual), rules)

    # Break caught: a score line that buys below it, re-spends the skipped cash on the other buys, or touches sells.
    def test_score_line_skips_weak_buys_and_keeps_their_cash(self) -> None:
        import pandas as pd

        scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, **{f"N{i:02d}": -0.0001 * i for i in range(40)}, "H1": -1.0})
        holdings = ["H1"]
        plain = self._strategy(scores, holdings, topk=4, n_drop=1, min_score_z=-1e9).generate_trade_decision()
        lined = self._strategy(scores, holdings, topk=4, n_drop=1, min_score_z=3.0).generate_trade_decision()
        z = (scores - scores.mean()) / scores.std()
        self.assertGreater(z["B"], 3.0)
        self.assertLess(z["C"], 3.0)
        self.assertEqual(
            [("H1", 0), ("A", 1), ("B", 1), ("C", 1), ("N00", 1)], [order[:2] for order in self._orders(plain)]
        )
        self.assertEqual([("H1", 0), ("A", 1), ("B", 1)], [order[:2] for order in self._orders(lined)])
        per_buy = {order[2] for order in self._orders(plain) if order[1] == 1}
        self.assertEqual(per_buy, {order[2] for order in self._orders(lined) if order[1] == 1})

    # Break caught: a rank exit that leaves weak holdings in the book, sells strong ones, or does not replace them.
    def test_rank_exit_sells_every_holding_outside_the_top_fraction_and_replaces_it(self) -> None:
        import pandas as pd

        scores = pd.Series({f"S{i:02d}": float(100 - i) for i in range(100)})
        holdings = ["S01", "S05", "S15", "S30", "S60", "S90"]
        plain = self._strategy(scores, holdings, topk=6, n_drop=1).generate_trade_decision()
        ruled = self._strategy(scores, holdings, topk=6, n_drop=1, exit_rank_pct=0.2).generate_trade_decision()
        sold = [order[0] for order in self._orders(ruled) if order[1] == 0]
        bought = [order[0] for order in self._orders(ruled) if order[1] == 1]
        self.assertEqual([("S90", 0), ("S00", 1)], [order[:2] for order in self._orders(plain)])
        self.assertEqual(["S30", "S60", "S90"], sold)  # ranks 31%, 61%, 91%; S15 is at 16%
        self.assertEqual(["S00", "S02", "S03"], bought)
        blocked = self._strategy(scores, holdings, topk=6, n_drop=1, blocked={"S60"}, exit_rank_pct=0.2)
        self.assertNotIn("S60", [order[0] for order in self._orders(blocked.generate_trade_decision())])

    def test_config_rejects_rules_where_they_cannot_apply(self) -> None:
        from rquant.workflow.backtest import PortfolioConfig

        for options in (
            {"exit_rank_pct": 0.0},
            {"exit_rank_pct": 1.0},
            {"min_score_z": float("nan")},
            {"strategy": "enhanced-indexing", "min_score_z": 2.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PortfolioConfig(**options)
        PortfolioConfig(min_score_z=2.5, exit_rank_pct=0.2)


if __name__ == "__main__":
    unittest.main()
