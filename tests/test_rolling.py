from __future__ import annotations

import unittest
from datetime import date, timedelta

from rquant.workflow.rolling import generate_rolling_windows, is_five_day_anchor, previous_trading_day


class RollingWindowTests(unittest.TestCase):
    def test_first_year_uses_prior_three_calendar_years(self) -> None:
        window = generate_rolling_windows(
            first_prediction_year=2013,
            last_prediction_year=2013,
            horizon="1d",
            through=date(2013, 12, 31),
        )[0]
        self.assertEqual(date(2010, 1, 1), window.train_start)
        self.assertEqual(date(2012, 7, 1), window.validation_start)
        self.assertEqual(date(2013, 1, 1), window.test_start)
        self.assertEqual(1, window.purge_trading_days)

    def test_current_year_is_clipped(self) -> None:
        through = date(2026, 8, 3)
        window = generate_rolling_windows(
            first_prediction_year=2026,
            last_prediction_year=2026,
            horizon="5d",
            through=through,
        )[0]
        self.assertEqual(through, window.test_end)
        self.assertEqual(5, window.purge_trading_days)

    def test_trading_day_purge(self) -> None:
        start = date(2024, 1, 1)
        calendar = [start + timedelta(days=value) for value in range(10)]
        self.assertEqual(start + timedelta(days=6), previous_trading_day(calendar, start + timedelta(days=9), 2))

    def test_five_day_anchor_is_global_position(self) -> None:
        self.assertTrue(is_five_day_anchor(0))
        self.assertTrue(is_five_day_anchor(10))
        self.assertFalse(is_five_day_anchor(11))


if __name__ == "__main__":
    unittest.main()
