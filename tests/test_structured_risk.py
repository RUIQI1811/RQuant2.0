from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("qlib", "cvxpy", "ecos", "pandas"))


def synthetic_market(rng, days: int = 120, stocks: int = 150):  # type: ignore[no-untyped-def]
    """Descriptors, sectors and caps of ``stocks`` over ``days``, and prices whose percent returns are the previous
    day's model exposures times known factor returns plus small noise."""
    import numpy as np
    import pandas as pd

    from rquant.qlib_ext.riskmodel import CNE6_DESCRIPTOR_NAMES, CNE6_STYLES, style_exposures

    dates = pd.bdate_range("2021-01-04", periods=days)
    instruments = [f"SH6{value:05d}" for value in range(stocks)]
    sector = np.tile(np.array([1.0, 2.0, 3.0] * (stocks // 3)), (days, 1))
    sector[:, :5] = np.nan  # no Shenwan membership
    sector = pd.DataFrame(sector, index=dates, columns=instruments)
    cap = pd.DataFrame(rng.lognormal(10, 1, size=(days, stocks)), index=dates, columns=instruments)
    base = rng.normal(size=(stocks, len(CNE6_DESCRIPTOR_NAMES)))
    frames = []
    exposures = {}
    for position, day in enumerate(dates):
        values = base + 0.05 * rng.normal(size=base.shape)
        values[position % stocks, 3] = np.nan
        frame = pd.DataFrame(values, index=pd.Index(instruments, name="instrument"), columns=CNE6_DESCRIPTOR_NAMES)
        frames.append(frame.assign(datetime=day).set_index("datetime", append=True).swaplevel())
        styles = style_exposures(frame, cap.loc[day].to_numpy())
        industry = np.column_stack(
            [sector.loc[day].to_numpy() == code for code in (1.0, 2.0, 3.0)] + [~np.isfinite(sector.loc[day])]
        ).astype(float)
        exposures[day] = np.hstack([industry, styles])
    factors = 3 + 1 + len(CNE6_STYLES)
    true_returns = rng.normal(0.0, 0.5, size=(days, factors))
    returns = np.zeros((days, stocks))
    for position in range(1, days):
        returns[position] = exposures[dates[position - 1]] @ true_returns[position]
        returns[position] += rng.normal(0.0, 0.01, size=stocks)
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + returns / 100.0, axis=0), index=dates, columns=instruments)
    descriptors = pd.concat(frames).sort_index()
    return dates, instruments, close, cap, sector, descriptors, true_returns, exposures


@unittest.skipUnless(HAS_STACK, "pyqlib, cvxpy, ecos and pandas are required")
class StructuredRiskModelTests(unittest.TestCase):
    # Break caught: a size factor centred on the equal-weighted mean (so the cap-weighted benchmark carries a size
    # exposure of its own), or outliers left to dominate the cross-section.
    def test_standardise_centres_on_cap_weights_and_winsorises(self) -> None:
        import numpy as np

        from rquant.qlib_ext.riskmodel import standardise

        rng = np.random.default_rng(1)
        values = rng.normal(size=500)
        values[0] = 1e6
        values[1] = np.nan
        cap = rng.lognormal(0, 1, size=500)
        result = standardise(values, cap)[:, 0]
        finite = np.isfinite(result)
        self.assertTrue(np.isnan(result[1]))
        self.assertAlmostEqual(0.0, float(np.average(result[finite], weights=cap[finite])), places=10)
        self.assertAlmostEqual(1.0, float(np.std(result[finite], ddof=1)), places=10)
        self.assertLess(result[0], 4.0)

    # Break caught: factor returns regressed on same-day exposures (look-ahead), the wrong regression weights, a
    # covariance that ignores the half-life, or specific risk taken from total instead of residual returns.
    def test_factor_returns_covariance_and_specific_risk(self) -> None:
        import numpy as np
        import pandas as pd

        from rquant.errors import DataContractError
        from rquant.qlib_ext.riskmodel import StructuredRiskModelData

        rng = np.random.default_rng(2)
        dates, instruments, close, cap, sector, descriptors, true_returns, exposures = synthetic_market(rng)
        signal = dates[-1]
        model = StructuredRiskModelData(
            close=close,
            volume=close * 0 + 1,
            float_cap=cap,
            sector=sector,
            descriptors=[descriptors.loc[: dates[59]], descriptors.loc[dates[60] :]],
            signal_dates=[signal],
            lookback=100,
            half_life=30,
        )
        self.assertEqual(["industry_1", "industry_2", "industry_3", "industry_unknown"], model.factor_names[:4])
        # Noise of 0.01% a stock; the 5-stock unknown industry is estimated least precisely.
        np.testing.assert_allclose(true_returns[1:], model.factor_returns.to_numpy(), atol=0.03)

        codes = instruments[10:40]
        loadings, factor_cov, specific = model.risk(signal, codes)
        np.testing.assert_allclose(exposures[signal][10:40], loadings)
        history = model.factor_returns.tail(100).to_numpy()
        weights = 0.5 ** (np.arange(100)[::-1] / 30)
        weights /= weights.sum()
        centred = history - weights @ history
        np.testing.assert_allclose((centred * weights[:, None]).T @ centred, factor_cov)
        np.testing.assert_allclose(0.01, specific, rtol=0.3)
        with self.assertRaises(DataContractError):
            model.risk(dates[-2], codes)
        self.assertIsInstance(model.factor_returns.index, pd.DatetimeIndex)

    # Break caught: the industry and size limits never reaching Qlib's optimiser.
    def test_optimiser_keeps_industry_and_size_exposure_within_limits(self) -> None:
        import numpy as np
        from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer

        from rquant.qlib_ext.riskmodel import StructuredRiskModelData

        rng = np.random.default_rng(3)
        dates, instruments, close, cap, sector, descriptors, _, _ = synthetic_market(rng)
        signal = dates[-1]
        model = StructuredRiskModelData(
            close=close,
            volume=close * 0 + 1,
            float_cap=cap,
            sector=sector,
            descriptors=[descriptors],
            signal_dates=[signal],
            lookback=60,
            industry_deviation=0.02,
            size_deviation=0.1,
        )
        loadings, factor_cov, specific = model.risk(signal, instruments)
        size = model.factor_names.index("size")
        expected = -loadings[:, size] + 0.1 * rng.normal(size=len(instruments))  # a small-cap signal
        benchmark = cap.loc[signal].to_numpy() / cap.loc[signal].sum()

        def active(f_dev):  # type: ignore[no-untyped-def]
            weight = EnhancedIndexingOptimizer(lamb=0.1, delta=2.0, b_dev=0.05, f_dev=f_dev)(
                r=expected, F=loadings, cov_b=factor_cov, var_u=specific**2, w0=benchmark, wb=benchmark
            )
            return (weight - benchmark) @ loadings

        free = active(None)
        limited = active(model.factor_deviation())
        self.assertLess(free[size], -0.1)
        self.assertGreaterEqual(limited[size], -0.1 - 1e-4)  # ECOS tolerance
        self.assertLessEqual(float(np.abs(limited[:4]).max()), 0.02 + 1e-4)
        self.assertEqual(100.0, model.factor_deviation()[model.factor_names.index("momentum")])


@unittest.skipUnless(
    HAS_STACK and importlib.util.find_spec("sklearn") is not None, "pyqlib, cvxpy, ecos and scikit-learn are required"
)
class StructuredBacktestTests(unittest.TestCase):
    # Break caught: the report path not loading the barra_cne6 store and the canonical sectors for the structured
    # model, or not passing its limits to the optimiser.
    def test_enhanced_indexing_backtest_with_the_structured_model(self) -> None:
        import numpy as np
        import pandas as pd
        from test_enhanced_indexing import write_provider

        from rquant.factors.catalog import get_catalog
        from rquant.qlib_ext.riskmodel import CNE6_DESCRIPTOR_NAMES
        from rquant.workflow.backtest import PortfolioBacktestRunner, PortfolioConfig

        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # More stocks than the model's 23 factors, so every day's regression is determined.
            dates, instruments, _ = write_provider(root / "qlib", rng, stocks=40)
            rows = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"]).to_frame(
                index=False
            )
            bars = rows.assign(sector=[1.0 + position % 3 for position in range(len(rows))])
            (root / "bars" / "year=2021").mkdir(parents=True)
            bars.to_parquet(root / "bars" / "year=2021" / "bars.parquet", index=False)
            canonical = get_catalog().source_to_canonical("barra_cne6")
            store = rows.assign(
                **{canonical[name]: rng.normal(size=len(rows)).astype("float32") for name in CNE6_DESCRIPTOR_NAMES}
            )
            (root / "factors" / "barra_cne6" / "year=2021").mkdir(parents=True)
            store.to_parquet(root / "factors" / "barra_cne6" / "year=2021" / "factors.parquet", index=False)
            # The provider's reference float caps live under qlib/reference; the runner reads bars next to it.
            (root / "qlib" / "reference").rename(root / "reference")
            run_directory = root / "run"
            run_directory.mkdir()
            signal_rows = [(day, code) for day in dates[30:60] for code in instruments]
            pd.DataFrame(signal_rows, columns=["datetime", "instrument"]).assign(
                score=rng.normal(size=len(signal_rows))
            ).to_parquet(run_directory / "predictions.parquet", index=False)
            config = PortfolioConfig(
                initial_capital=10_000_000,
                strategy="enhanced-indexing",
                risk_model="structured",
                risk_lookback_days=20,
                benchmark_deviation=0.05,
                turnover_limit=2.0,
                industry_deviation=0.02,
                rebalance_days=5,
            )
            result = PortfolioBacktestRunner(
                qlib_root=root / "qlib",
                run_directory=run_directory,
                config=config,
                reference_root=root / "reference",
                factor_root=root / "factors",
            ).run()

            self.assertEqual("complete", result["status"])
            self.assertIn("CNE6 style risk model", result["constraints"]["weights"])
            optimizations = pd.read_parquet(run_directory / "optimizations.parquet")
            self.assertEqual(6, len(optimizations))
            self.assertTrue((optimizations["holdings"] > 0).all())
            self.assertTrue(optimizations["solved"].all())
            self.assertLessEqual(float(optimizations["max_industry_active"].max()), 0.02 + 1e-4)
            self.assertLessEqual(float(optimizations["size_active"].abs().max()), 0.2 + 1e-4)
            positions = pd.read_parquet(run_directory / "positions.parquet")
            self.assertFalse(positions.empty)


if __name__ == "__main__":
    unittest.main()
