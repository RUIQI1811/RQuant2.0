from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from talib import abstract

from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, PanelFactorEngine, create_factor_engine
from rquant.factors.libraries.talib_indicators import TALIB_INDICATORS, TALIB_NAMES, TALibLibrary
from rquant.io import atomic_write_json


def synthetic_bars(days: int = 400, stocks: int = 5, seed: int = 5) -> dict[str, pd.DataFrame]:
    """OHLCV panels with a suspension, a late listing and a limit-locked day, shaped like the engine's inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=days, name="datetime")
    instruments = pd.Index([f"SZ{1 + index:06d}" for index in range(stocks)], name="instrument")
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (days, stocks)), axis=0))
    open_ = close * np.exp(rng.normal(0.0, 0.008, (days, stocks)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    volume = rng.lognormal(13.0, 0.4, (days, stocks))
    open_[150, 0] = high[150, 0] = low[150, 0] = close[150, 0]
    panels = {
        field: pd.DataFrame(values, index=dates, columns=instruments)
        for field, values in (("open", open_), ("high", high), ("low", low), ("close", close), ("volume", volume))
    }
    for panel in panels.values():
        panel.iloc[120:140, 1] = np.nan
        panel.iloc[:90, 2] = np.nan
    panels["volume"].iloc[180, 2] = np.nan
    return panels


class TALibCatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        catalog = get_catalog()
        names = catalog.canonical_names("talib")
        self.assertEqual(125, len(names))
        self.assertEqual(("ta_001", "ta_125"), (names[0], names[-1]))
        self.assertEqual(len(set(TALIB_NAMES)), len(TALIB_NAMES))
        self.assertEqual(1101, len(catalog.specs))

    def test_declared_lookbacks_and_parameters_match_the_installed_library(self) -> None:
        for name, function, _, parameters, position, _, lookback in TALIB_INDICATORS:
            indicator = abstract.Function(function)
            self.assertLessEqual(set(parameters), set(indicator.parameters), name)
            if parameters:
                indicator.set_parameters(parameters)
            self.assertEqual(lookback, indicator.lookback, name)
            self.assertLess(position, len(indicator.output_names), name)

    def test_fingerprint_pins_the_talib_version(self) -> None:
        self.assertEqual(TALibLibrary().implementation_fingerprint, TALibLibrary().implementation_fingerprint)
        self.assertEqual("0.8.2", talib.__version__)


class TALibCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_bars()
        cls.outputs = TALibLibrary().calculate(cls.inputs)

    def test_all_outputs_are_aligned_and_missing_without_a_quote(self) -> None:
        close = self.inputs["close"]
        quoted = pd.concat(self.inputs.values(), axis=1).notna().T.groupby(level=0).all().T
        for name in TALIB_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            self.assertTrue(panel.where(~quoted).isna().all().all(), name)
        for name in TALIB_NAMES[:64]:
            self.assertTrue(np.isfinite(self.outputs[name].to_numpy()).mean() > 0.6, name)
        patterns = np.column_stack([self.outputs[name].to_numpy().ravel() for name in TALIB_NAMES[64:]])
        self.assertTrue(np.isin(patterns[np.isfinite(patterns)], (-200.0, -100.0, 0.0, 100.0, 200.0)).all())
        self.assertTrue((np.abs(np.nan_to_num(patterns)) > 0).any(axis=0).sum() > 20)

    # Break caught: an indicator that reads a later quote than the row it is published on.
    def test_no_output_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (130, 310):
            last = dates[cut]
            past = TALibLibrary().calculate({field: panel.loc[:last] for field, panel in self.inputs.items()})
            for name in TALIB_NAMES:
                np.testing.assert_allclose(past[name].to_numpy(), self.outputs[name].loc[:last].to_numpy(),
                                           rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)

    # Break caught: a suspension feeding NaN into TA-Lib, which poisons recursive indicators for the rest of history.
    def test_indicators_run_over_the_stocks_own_quotes_across_a_suspension(self) -> None:
        column = 1
        close = self.inputs["close"].iloc[:, column]
        quoted = close.dropna()
        expected = talib.RSI(quoted.to_numpy(), timeperiod=14)
        np.testing.assert_allclose(self.outputs["rsi"].iloc[:, column].dropna().to_numpy(), expected[14:])
        ema = talib.EMA(quoted.to_numpy(), timeperiod=30)
        self.assertAlmostEqual(ema[-1] / quoted.iloc[-1] - 1.0, self.outputs["ema"].iloc[-1, column])
        self.assertTrue(np.isfinite(self.outputs["trix"].iloc[-1, column]))
        # The late listing starts its lookback at its own first quote.
        first = self.inputs["close"].iloc[:, 2].first_valid_index()
        start = self.inputs["close"].index.get_loc(first)
        self.assertTrue(self.outputs["rsi"].iloc[: start + 14, 2].isna().all())
        self.assertTrue(np.isfinite(self.outputs["rsi"].iloc[start + 14, 2]))

    def test_volume_lines_are_published_as_flows(self) -> None:
        row, column = 380, 0
        close = self.inputs["close"].iloc[:, column].to_numpy()
        volume = self.inputs["volume"].iloc[:, column].to_numpy()
        obv = talib.OBV(close, volume)
        self.assertAlmostEqual((obv[row] - obv[row - 20]) / volume[row - 19 : row + 1].sum(),
                               self.outputs["obv"].iloc[row, column])
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        self.assertAlmostEqual((close[row] - lower[row]) / (upper[row] - lower[row]),
                               self.outputs["bb_pctb"].iloc[row, column])
        self.assertAlmostEqual((upper[row] - lower[row]) / middle[row], self.outputs["bb_width"].iloc[row, column])

    # Break caught: a price-level indicator published raw, which ranks stocks by their arbitrary adjusted price level.
    def test_outputs_do_not_depend_on_the_adjusted_price_level(self) -> None:
        scaled = {field: panel.copy() for field, panel in self.inputs.items()}
        for field in ("open", "high", "low", "close"):
            scaled[field] = scaled[field] * 7.0
        scaled["volume"] = scaled["volume"] / 7.0
        outputs = TALibLibrary().calculate(scaled)
        for name in TALIB_NAMES:
            np.testing.assert_allclose(outputs[name].to_numpy(), self.outputs[name].to_numpy(), rtol=1e-6, atol=1e-8,
                                       equal_nan=True, err_msg=name)


class TALibEngineTests(unittest.TestCase):
    def test_engine_publishes_the_library_outputs(self) -> None:
        inputs = synthetic_bars(days=200, stocks=3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            bars = pd.concat({field: panel.stack(future_stack=True) for field, panel in inputs.items()}, axis=1)
            bars = bars.reset_index()
            bars["amount"] = bars["volume"] * bars["close"]
            bars["vwap"] = bars["close"]
            bars["in_csi300"] = True
            path = canonical / "bars" / "year=2016" / "bars.parquet"
            path.parent.mkdir(parents=True)
            bars.to_parquet(path, index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-fixture"})

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="talib"))
            self.assertIsInstance(engine, PanelFactorEngine)
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(125, result["columns"])
            store = root / "factors" / "talib"
            stored = pd.concat([pd.read_parquet(path) for path in sorted(store.glob("year=*/*.parquet"))])
            stored = stored.set_index(["datetime", "instrument"])
            direct = TALibLibrary().calculate(inputs)
            for canonical_name, name in (("ta_019", "rsi"), ("ta_045", "ad"), ("ta_083", "engulfing")):
                expected = direct[name].stack(future_stack=True).reindex(stored.index)
                np.testing.assert_allclose(stored[canonical_name].to_numpy(), expected.to_numpy(), rtol=1e-6,
                                           equal_nan=True, err_msg=name)


if __name__ == "__main__":
    unittest.main()
