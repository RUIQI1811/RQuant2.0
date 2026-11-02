from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("qlib", "cvxpy", "ecos", "sklearn"))

FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor", "change", "limit_buy", "limit_sell")


def write_provider(root: Path, rng, stocks: int = 12):  # type: ignore[no-untyped-def]
    """A Qlib day provider of ``stocks`` stocks and the benchmark over 70 trading days, plus reference float caps."""
    import numpy as np
    import pandas as pd

    dates = pd.bdate_range("2021-01-04", periods=70)
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("".join(f"{day:%Y-%m-%d}\n" for day in dates), encoding="utf-8")
    instruments = [f"SH6000{value:02d}" for value in range(stocks)]
    (root / "instruments").mkdir()
    (root / "instruments" / "all.txt").write_text(
        "".join(f"{code}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}\n" for code in [*instruments, "SH000300"]),
        encoding="utf-8",
    )
    market = rng.normal(0.0, 0.01, size=len(dates))
    closes = {}
    for position, code in enumerate([*instruments, "SH000300"]):
        beta = 0.5 + position / 12
        returns = beta * market + rng.normal(0.0, 0.015, size=len(dates))
        close = 10.0 * np.cumprod(1.0 + returns)
        volume = np.full(len(dates), 1e6)
        if code == "SH600003":
            close[30:33] = np.nan  # suspended for three days
            volume[30:33] = np.nan
        values = {
            "open": close * (1.0 + rng.normal(0.0, 0.002, size=len(dates))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "vwap": close,
            "factor": np.ones(len(dates)),
            "change": np.concatenate(([np.nan], close[1:] / close[:-1] - 1.0)),
            "limit_buy": np.zeros(len(dates)),  # flags: 1 when the open is limit-up
            "limit_sell": np.zeros(len(dates)),
        }
        directory = root / "features" / code.lower()
        directory.mkdir(parents=True)
        for field in FIELDS:
            np.concatenate(([0.0], values[field])).astype(np.float32).tofile(directory / f"{field}.day.bin")
        closes[code] = close
    reference = root / "reference" / "daily_basic" / "year=2021"
    reference.mkdir(parents=True)
    caps = pd.DataFrame(
        [
            {"datetime": day, "instrument": code, "circ_mv": 1e5 * (1 + position)}
            for day in dates
            for position, code in enumerate(instruments)
        ]
    )
    caps.to_parquet(reference / "data.parquet", index=False)
    return dates, instruments, pd.DataFrame(closes, index=dates)


@unittest.skipUnless(HAS_STACK, "pyqlib, cvxpy, ecos and scikit-learn are required")
class EnhancedIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dates, self.instruments, self.closes = write_provider(self.root / "qlib", np.random.default_rng(4))

    # Break caught: a risk window that is not the trailing lookback ending on the signal date, or a risk model that
    # departs from Qlib's prepare_riskdata example (clipped daily returns into StructuredCovEstimator).
    def test_risk_model_follows_qlib_prepare_riskdata(self) -> None:
        import numpy as np
        from qlib.model.riskmodel import StructuredCovEstimator

        from rquant.errors import DataContractError
        from rquant.qlib_ext.strategy import RiskModelData

        codes = self.instruments[:8]
        close = self.closes[self.instruments]
        data = RiskModelData(
            close=close, volume=close * 0 + 1, float_cap=close * 0 + 1, lookback=20, num_factors=3
        )
        exposures, factor_cov, specific_risk = data.risk(self.dates[40], codes)

        # prepare_riskdata calls pct_change() under the pandas default of its time, which pads missing prices.
        price = close.loc[self.dates[21] : self.dates[40], codes]
        returns = price.ffill().pct_change(fill_method=None)
        returns = returns.clip(returns.quantile(0.025), returns.quantile(0.975), axis=1)
        expected = StructuredCovEstimator(num_factors=3).predict(
            returns, is_price=False, return_decomposed_components=True
        )
        np.testing.assert_allclose(expected[0], exposures)
        np.testing.assert_allclose(expected[1], factor_cov)
        np.testing.assert_allclose(np.sqrt(expected[2]), specific_risk)
        self.assertEqual((8, 3), exposures.shape)

        # Across a suspension the price is carried, so the stock earns zero and the move lands on resumption.
        suspended = close.loc[self.dates[21] : self.dates[40], "SH600003"]
        moved = suspended.ffill().pct_change(fill_method=None)
        self.assertEqual(0.0, moved.loc[self.dates[31]])
        self.assertTrue(np.isfinite(moved.loc[self.dates[33]]))
        weights = data.benchmark_weight(self.dates[40], ["SH600000", "SH600001", "SH699999"])
        np.testing.assert_allclose([1 / 2, 1 / 2, 0.0], weights)
        with self.assertRaises(DataContractError):
            data.risk(self.dates[10], codes)

    # Break caught: the enhanced-indexing path not reaching Qlib's optimiser, holding more than the per-stock
    # deviation allows, or leaving positions open at the end.
    def test_backtest_optimises_weights_within_the_benchmark_deviation(self) -> None:
        import numpy as np
        import pandas as pd

        from rquant.workflow.backtest import PortfolioBacktestRunner, PortfolioConfig

        rng = np.random.default_rng(9)
        run_directory = self.root / "run"
        run_directory.mkdir()
        signal_dates = self.dates[30:60]
        pd.DataFrame(
            [
                {"datetime": day, "instrument": code, "score": rng.normal()}
                for day in signal_dates
                for code in self.instruments
            ]
        ).to_parquet(run_directory / "predictions.parquet", index=False)
        config = PortfolioConfig(
            initial_capital=10_000_000,
            strategy="enhanced-indexing",
            risk_lookback_days=20,
            risk_factors=3,
            benchmark_deviation=0.05,
            turnover_limit=2.0,
            rebalance_days=5,
        )
        result = PortfolioBacktestRunner(
            qlib_root=self.root / "qlib",
            run_directory=run_directory,
            config=config,
            reference_root=self.root / "qlib" / "reference",
        ).run()

        self.assertEqual("complete", result["status"])
        self.assertEqual("LiquidatingEnhancedIndexingStrategy", result["strategy"])
        self.assertFalse(result["constraints"]["equal_weight"])
        self.assertTrue(result["terminal_liquidation"]["complete"])
        optimizations = pd.read_parquet(run_directory / "optimizations.parquet")
        self.assertEqual(6, len(optimizations))
        self.assertTrue((optimizations["holdings"] > 0).all())
        self.assertTrue(np.isfinite(optimizations["tracking_error"]).all())
        # 12 stocks with float caps 1..12: no weight may exceed its cap weight by more than 5 points.
        positions = pd.read_parquet(run_directory / "positions.parquet")
        first = positions[positions["datetime"] == positions["datetime"].min()].set_index("instrument")
        prices = self.closes.loc[positions["datetime"].min()]
        value = (first["adjusted_amount"] * prices.reindex(first.index)).dropna()
        weights = value / value.sum()
        caps = pd.Series(np.arange(1, 13), index=self.instruments, dtype=float)
        benchmark = caps / caps.sum()
        self.assertTrue((weights <= benchmark.reindex(weights.index) + 0.05 + 0.01).all())


if __name__ == "__main__":
    unittest.main()
