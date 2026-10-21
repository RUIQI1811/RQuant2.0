from __future__ import annotations

import unittest
from datetime import date

from rquant.qlib_ext.exchange import AshareCostModel, is_open_limit_blocked, round_buy_shares


class AshareExchangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.costs = AshareCostModel()

    def test_commission_minimum(self) -> None:
        self.assertEqual(5.0, self.costs.commission(1_000.0))
        self.assertAlmostEqual(30.0, self.costs.commission(100_000.0))

    def test_stamp_tax_switch(self) -> None:
        value = 100_000.0
        self.assertEqual(100.0, self.costs.stamp_tax(value, date(2023, 8, 27), is_sell=True))
        self.assertEqual(50.0, self.costs.stamp_tax(value, date(2023, 8, 28), is_sell=True))
        self.assertEqual(0.0, self.costs.stamp_tax(value, date(2023, 8, 28), is_sell=False))

    def test_round_lot(self) -> None:
        self.assertEqual(0, round_buy_shares(99.9))
        self.assertEqual(100, round_buy_shares(199.9))
        self.assertEqual(0, round_buy_shares(-1))

    def test_open_limit_direction(self) -> None:
        self.assertTrue(is_open_limit_blocked(11.0, 11.0, buy=True))
        self.assertFalse(is_open_limit_blocked(11.0, 10.0, buy=False))
        self.assertTrue(is_open_limit_blocked(9.0, 9.0, buy=False))
        self.assertFalse(is_open_limit_blocked(9.0, 10.0, buy=True))


class AshareExchangeImpactTests(unittest.TestCase):
    def _exchange(self, impact_cost: float, volume: float | None) -> object:
        from rquant.qlib_ext.exchange import AshareExchange

        exchange = object.__new__(AshareExchange)
        exchange.impact_cost = impact_cost
        exchange.get_volume = lambda *args, **kwargs: volume
        return exchange

    def test_impact_is_quadratic_in_participation(self) -> None:
        order = type("Order", (), {"stock_id": "SH600000", "start_time": None, "end_time": None})()
        exchange = self._exchange(0.1, 10_000.0)
        # 100,000 traded against 10,000 shares * 100 = 1,000,000 daily value: 0.1 * 0.1^2.
        self.assertAlmostEqual(0.001, exchange._impact_ratio(order, 100.0, 100_000.0))

    def test_impact_disabled_or_without_volume(self) -> None:
        order = type("Order", (), {"stock_id": "SH600000", "start_time": None, "end_time": None})()
        self.assertEqual(0.0, self._exchange(0.0, 10_000.0)._impact_ratio(order, 100.0, 100_000.0))
        self.assertEqual(0.1, self._exchange(0.1, None)._impact_ratio(order, 100.0, 100_000.0))


if __name__ == "__main__":
    unittest.main()
