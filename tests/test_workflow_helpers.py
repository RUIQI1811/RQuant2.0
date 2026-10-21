from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.qlib_ext.loader import _partitions_in_range
from rquant.workflow.backtest import _metrics


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


if __name__ == "__main__":
    unittest.main()
