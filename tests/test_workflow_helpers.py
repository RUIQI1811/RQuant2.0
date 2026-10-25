from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.qlib_ext.loader import _partitions_in_range
from rquant.workflow.backtest import PortfolioConfig, _metrics, _rebalance_dates, _rebalance_description


class LoaderPartitionTests(unittest.TestCase):
    def test_only_years_inside_the_requested_range_are_read(self) -> None:
        paths = [Path(f"factors/combined/year={year}/factors.parquet") for year in range(2010, 2016)]
        selected = _partitions_in_range(paths, pd, "2012-07-01", "2014-12-31")
        self.assertEqual(["year=2012", "year=2013", "year=2014"], [path.parent.name for path in selected])
        self.assertEqual(paths, _partitions_in_range(paths, pd, None, None))


class BacktestMetricTests(unittest.TestCase):
    def test_sharpe_uses_arithmetic_mean_daily_return(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
        metrics = _metrics(np, returns, pd.Series([0.0] * 5))
        expected = returns.mean() * 252 / (returns.std(ddof=1) * math.sqrt(252))
        self.assertAlmostEqual(expected, metrics["sharpe_zero_rf"])


class RebalanceScheduleTests(unittest.TestCase):
    def test_rebalance_dates_take_every_nth_trading_day_from_the_first(self) -> None:
        calendar = pd.bdate_range("2024-01-01", periods=12)
        self.assertEqual(list(calendar[[0, 5, 10]]), list(_rebalance_dates(pd, calendar, 5)))
        self.assertEqual(list(calendar), list(_rebalance_dates(pd, calendar, 1)))

    def test_portfolio_config_rejects_invalid_rebalance_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "rebalance_days"):
            PortfolioConfig(rebalance_days=0)
        with self.assertRaisesRegex(ValueError, "n_drop"):
            PortfolioConfig(topk=50, n_drop=51)

    def test_rebalance_description_names_full_and_dropout_modes(self) -> None:
        self.assertEqual(
            "every 5 trading days, full rebalance into the latest top 50",
            _rebalance_description(PortfolioConfig(topk=50, n_drop=50, rebalance_days=5)),
        )
        self.assertEqual(
            "daily, drop at most 5 of the top 50",
            _rebalance_description(PortfolioConfig(topk=50, n_drop=5, rebalance_days=1)),
        )


if __name__ == "__main__":
    unittest.main()
