from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.qlib_ext.loader import _partitions_in_range, _single_precision
from rquant.qlib_ext.processors import ProcessInf
from rquant.workflow.backtest import PortfolioConfig, _metrics, _rebalance_dates, _rebalance_description


class LoaderPartitionTests(unittest.TestCase):
    def test_only_years_inside_the_requested_range_are_read(self) -> None:
        paths = [Path(f"factors/combined/year={year}/factors.parquet") for year in range(2010, 2016)]
        selected = _partitions_in_range(paths, pd, "2012-07-01", "2014-12-31")
        self.assertEqual(["year=2012", "year=2013", "year=2014"], [path.parent.name for path in selected])
        self.assertEqual(paths, _partitions_in_range(paths, pd, None, None))

    def test_factor_columns_are_read_in_single_precision(self) -> None:
        frame = pd.DataFrame(
            {"datetime": pd.to_datetime(["2024-01-02"]), "instrument": ["SH600000"], "f1": [1.5], "f2": [np.nan]}
        )
        converted = _single_precision(frame)
        self.assertEqual(["float32", "float32"], [str(converted[name].dtype) for name in ("f1", "f2")])
        self.assertEqual(frame["instrument"].dtype, converted["instrument"].dtype)
        self.assertEqual(frame["datetime"].dtype, converted["datetime"].dtype)


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


class ProcessInfTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-29", periods=6)
        instruments = ["SH600000", "SH600004", "SH600006", "SH600007"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        rng = np.random.default_rng(7)
        values = rng.normal(size=(len(index), 3)).astype("float32")
        values[[0, 5, 9], 0] = np.inf
        values[[2, 13], 1] = -np.inf
        values[[3, 6], 0] = np.nan
        values[20:24, 2] = np.inf  # every value of one date is infinite
        columns = pd.MultiIndex.from_tuples([("feature", "a"), ("feature", "b"), ("label", "LABEL0")])
        return pd.DataFrame(values, index=index, columns=columns)

    def test_infinities_take_the_same_date_mean_of_finite_values(self) -> None:
        frame = self._panel()
        result = ProcessInf()(frame.copy())
        self.assertFalse(np.isinf(result.to_numpy()).any())
        day = frame.index.get_level_values("datetime")[0]
        column = ("feature", "a")
        finite = frame.loc[day, column].replace(np.inf, np.nan).mean()
        self.assertAlmostEqual(float(finite), float(result.loc[(day, "SH600000"), column]), places=6)
        self.assertTrue(np.isnan(result[("feature", "a")].iloc[3]))
        self.assertTrue(result[("label", "LABEL0")].iloc[20:24].isna().all())
        untouched = ~np.isinf(frame.to_numpy()) & ~np.isnan(frame.to_numpy())
        np.testing.assert_array_equal(frame.to_numpy()[untouched], result.to_numpy()[untouched])

    @unittest.skipUnless(importlib.util.find_spec("qlib") is not None, "pyqlib is required")
    def test_matches_qlib_process_inf(self) -> None:
        from qlib.data.dataset.processor import ProcessInf as QlibProcessInf

        frame = self._panel()
        expected = QlibProcessInf()(frame.copy())
        pd.testing.assert_frame_equal(expected, ProcessInf()(frame.copy()), check_dtype=False)


if __name__ == "__main__":
    unittest.main()
