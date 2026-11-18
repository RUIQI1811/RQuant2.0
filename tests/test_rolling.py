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

    def test_window_lengths_follow_configuration(self) -> None:
        window = generate_rolling_windows(
            first_prediction_year=2015,
            last_prediction_year=2015,
            horizon="1d",
            through=date(2015, 12, 31),
            train_years=5,
            validation_months=3,
        )[0]
        self.assertEqual(date(2010, 1, 1), window.train_start)
        self.assertEqual(date(2014, 10, 1), window.validation_start)
        self.assertEqual(date(2014, 9, 30), window.inner_train_end)

    def test_first_year_respects_formal_start_and_train_years(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot predict before 2015"):
            generate_rolling_windows(
                first_prediction_year=2014,
                last_prediction_year=2014,
                horizon="1d",
                through=date(2014, 12, 31),
                train_years=5,
            )
        with self.assertRaisesRegex(ValueError, "validation_months"):
            generate_rolling_windows(
                first_prediction_year=2013,
                last_prediction_year=2013,
                horizon="1d",
                through=date(2013, 12, 31),
                validation_months=36,
            )

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

    # Break caught: sub-year windows that overlap, leave gaps, train past their first day or lose their label.
    def test_quarterly_windows_tile_the_year_and_train_on_the_three_years_before_each(self) -> None:
        windows = generate_rolling_windows(
            first_prediction_year=2025,
            last_prediction_year=2026,
            horizon="1d",
            through=date(2026, 10, 21),
            retrain_months=3,
        )
        self.assertEqual(
            ["2025-01", "2025-04", "2025-07", "2025-10", "2026-01", "2026-04", "2026-07", "2026-10"],
            [window.label for window in windows],
        )
        for previous, window in zip(windows, windows[1:], strict=False):
            self.assertEqual(previous.test_end + timedelta(days=1), window.test_start)
        april = windows[5]
        self.assertEqual(
            (date(2023, 4, 1), date(2025, 10, 1), date(2025, 9, 30), date(2026, 3, 31), date(2026, 3, 31)),
            (april.train_start, april.validation_start, april.inner_train_end, april.validation_end, april.refit_end),
        )
        self.assertEqual((date(2026, 4, 1), date(2026, 6, 30), 2026), (april.test_start, april.test_end, 2026))
        self.assertEqual((date(2026, 10, 1), date(2026, 10, 21)), (windows[-1].test_start, windows[-1].test_end))
        self.assertEqual(date(2025, 12, 31), windows[3].test_end)
        yearly = generate_rolling_windows(
            first_prediction_year=2026, last_prediction_year=2026, horizon="1d", through=date(2026, 10, 21)
        )
        self.assertEqual(["2026"], [window.label for window in yearly])
        self.assertEqual(date(2025, 12, 31), yearly[0].refit_end)
        clipped = generate_rolling_windows(
            first_prediction_year=2026, last_prediction_year=2026, horizon="1d", through=date(2026, 5, 6),
            retrain_months=3,
        )
        self.assertEqual(["2026-01", "2026-04"], [window.label for window in clipped])
        with self.assertRaisesRegex(ValueError, "retrain_months"):
            generate_rolling_windows(
                first_prediction_year=2026, last_prediction_year=2026, horizon="1d", through=date(2026, 5, 6),
                retrain_months=5,
            )

    # Break caught: older training years that move the validation or test months, reach before the formal start, or
    # change which years can be predicted.
    def test_older_train_years_start_training_earlier_but_not_before_the_formal_start(self) -> None:
        def windows(older: int) -> list:
            return generate_rolling_windows(
                first_prediction_year=2013, last_prediction_year=2026, horizon="1d", through=date(2026, 10, 21),
                older_train_years=older,
            )

        plain, longer = windows(0), windows(3)
        self.assertEqual([date(year - 3, 1, 1) for year in range(2013, 2027)], [w.train_start for w in plain])
        self.assertEqual(
            [date(2010, 1, 1)] * 4 + [date(year - 6, 1, 1) for year in range(2017, 2027)],
            [w.train_start for w in longer],
        )
        for before, after in zip(plain, longer, strict=True):
            self.assertEqual(
                (before.validation_start, before.refit_end, before.test_start, before.test_end, before.label),
                (after.validation_start, after.refit_end, after.test_start, after.test_end, after.label),
            )
        with self.assertRaisesRegex(ValueError, "2013"):
            generate_rolling_windows(
                first_prediction_year=2012, last_prediction_year=2013, horizon="1d", through=date(2013, 5, 6),
                older_train_years=3,
            )
        with self.assertRaisesRegex(ValueError, "older_train_years"):
            windows(-1)

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
