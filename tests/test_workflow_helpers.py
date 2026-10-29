from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.qlib_ext.loader import _load_feature_block, _partitions_in_range
from rquant.qlib_ext.processors import DropnaLabel, Fillna, ProcessInf, ZScoreNorm
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


class ProcessorTests(unittest.TestCase):
    def test_zscore_large_finite_values_keep_signal_and_use_only_fit_period(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        index = pd.MultiIndex.from_product([dates, range(100)], names=["datetime", "instrument"])
        values = np.full((300, 2), 7.0, dtype=np.float32)
        values[:200, 0] = np.tile(np.array([-3.6e18, 3.6e18], dtype=np.float32), 100)
        values[0, 0] = np.nan
        values[200:, 0] = 1e30  # Must not enter fitted statistics.
        frame = pd.DataFrame(np.asfortranarray(values), index=index, columns=["ratio", "constant"], copy=False)
        fit = values[:200, 0].astype(np.float64)
        expected = ((values[:, 0].astype(np.float64) - np.nanmean(fit)) / np.nanstd(fit)).astype(np.float32)
        processor = ZScoreNorm(fit_start_time=dates[0], fit_end_time=dates[1])
        with np.errstate(over="raise"):
            processor.fit(frame)
            result = processor(frame)
        self.assertTrue(np.isfinite(processor.std_train).all())
        np.testing.assert_allclose(result["ratio"], expected, rtol=1e-6, equal_nan=True)
        np.testing.assert_array_equal(result["constant"], values[:, 1])
        self.assertEqual([np.dtype("float32")] * 2, list(result.dtypes))
        self.assertEqual(1, len(result._mgr.blocks))

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
        return pd.DataFrame(np.asfortranarray(values), index=index, columns=columns, copy=False)

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

    @unittest.skipUnless(importlib.util.find_spec("qlib") is not None, "pyqlib is required")
    def test_processor_chain_matches_qlib_and_keeps_one_block(self) -> None:
        from qlib.data.dataset import processor as qlib_processor

        def run(processors, frame):
            for processor in processors:
                processor.fit(frame)
                frame = processor(frame)
            return frame

        fit = {"fit_start_time": "2024-01-29", "fit_end_time": "2024-02-02"}
        frame = self._panel()
        frame.iloc[7, 2] = np.nan
        ours = run([ProcessInf(), ZScoreNorm(**fit), Fillna(), DropnaLabel()], frame.copy())
        expected = run(
            [
                qlib_processor.ProcessInf(),
                qlib_processor.ZScoreNorm(**fit),
                qlib_processor.Fillna(),
                qlib_processor.DropnaLabel(),
            ],
            frame.copy(),
        )
        pd.testing.assert_frame_equal(expected, ours, check_dtype=False, rtol=1e-6)
        self.assertEqual(1, len(ours._mgr.blocks))

    def test_dropna_label_copies_only_when_a_label_is_missing(self) -> None:
        frame = self._panel().fillna(0)
        self.assertIs(frame, DropnaLabel()(frame))
        frame.iloc[1, 2] = np.nan
        self.assertEqual(len(frame) - 1, len(DropnaLabel()(frame)))


class PartitionReadTests(unittest.TestCase):
    # Break caught: a block whose rows, order or single-precision rounding differ from reading, filtering, concatenating
    # and sorting the partitions with pandas.
    def test_feature_block_matches_the_pandas_read_of_the_partitions(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        rng = np.random.default_rng(3)
        names = [f"f{position}" for position in range(5)]
        paths = []
        for year, dates in ((2023, ["2023-12-28", "2023-12-29"]), (2024, ["2024-01-02", "2024-01-03"])):
            # Instruments out of order inside each date, so the block has to be sorted.
            keys = [(date, code) for date in dates for code in ("SH600004", "SH600000", "SH600006")]
            values = rng.normal(size=(len(keys), len(names))) * 1e3
            values[0, 1], values[4, 2], values[5, 0] = np.nan, np.inf, 1.0000000001
            frame = pd.DataFrame(values, columns=names)
            frame.insert(0, "instrument", [code for _, code in keys])
            frame.insert(0, "datetime", pd.to_datetime([date for date, _ in keys]))
            path = Path(directory.name) / f"year={year}" / "factors.parquet"
            path.parent.mkdir()
            frame.to_parquet(path, index=False)
            paths.append(path)

        seen = []

        def labels(index: pd.MultiIndex) -> np.ndarray:
            seen.append(index)
            return np.arange(len(index), dtype=np.float32)

        index, block = _load_feature_block(
            pd,
            paths,
            names,
            "2023-12-29",
            "2024-01-02",
            ["SH600000", "SH600004"],
            labels=labels,
            columns_per_read=2,
        )

        expected = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        expected = expected[
            expected["datetime"].between(pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-02"))
            & expected["instrument"].isin(["SH600000", "SH600004"])
        ]
        expected = expected.set_index(["datetime", "instrument"]).sort_index().astype("float32")
        pd.testing.assert_index_equal(expected.index, index)
        self.assertEqual((len(expected), len(names) + 1), block.shape)
        self.assertTrue(block.flags["F_CONTIGUOUS"])
        self.assertEqual(np.float32, block.dtype)
        np.testing.assert_array_equal(expected.to_numpy(), block[:, :-1])
        # The label is evaluated on the sorted index and lands in the trailing column.
        self.assertEqual(1, len(seen))
        pd.testing.assert_index_equal(index, seen[0])
        np.testing.assert_array_equal(np.arange(len(index), dtype=np.float32), block[:, -1])

    def test_store_with_other_columns_is_refused(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "factors.parquet"
        pd.DataFrame({"datetime": pd.to_datetime(["2024-01-02"]), "instrument": ["SH600000"], "f2": [1.0]}).to_parquet(
            path, index=False
        )
        with self.assertRaisesRegex(DataContractError, "locked catalog"):
            _load_feature_block(pd, [path], ["f1"], None, None, None)


if __name__ == "__main__":
    unittest.main()
