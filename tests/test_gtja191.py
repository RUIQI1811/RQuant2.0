from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, PanelFactorEngine, create_factor_engine
from rquant.factors.libraries.gtja191 import (
    GTJA191,
    GTJA191_BUILD_EXCLUSIONS,
    GTJA191_FORMULA_NAMES,
    GTJA191_FORMULA_NOTES,
    GTJA191_NAMES,
    GTJA191DataError,
    GTJA191ExternalData,
    GTJA191Library,
    GTJA191Panels,
    count,
    highday,
    lowday,
    mean,
    regbeta,
    regresi,
    sma_cn,
    sumif,
    wma,
)
from rquant.factors.panel_operators import correlation, delay, element_max, element_min, safe_div, ts_sum
from rquant.io import atomic_write_json, stable_hash


def complete_panels(days: int = 540, symbols: int = 5) -> GTJA191Panels:
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    columns = [f"S{number:03d}" for number in range(symbols)]
    rng = np.random.default_rng(20260807)
    close = pd.DataFrame(
        30.0 + rng.normal(0.02, 0.3, (days, symbols)).cumsum(axis=0),
        index=dates,
        columns=columns,
    )
    open_ = close * (1.0 + rng.normal(0.0, 0.005, close.shape))
    high = pd.DataFrame(np.maximum(open_, close) + 0.5 + rng.random(close.shape), index=dates, columns=columns)
    low = pd.DataFrame(np.minimum(open_, close) - 0.5 - rng.random(close.shape), index=dates, columns=columns)
    volume = pd.DataFrame(rng.lognormal(12.0, 0.3, close.shape), index=dates, columns=columns)
    benchmark_close = pd.Series(4000.0 + rng.normal(0.0, 8.0, days).cumsum(), index=dates)
    external = GTJA191ExternalData(
        benchmark_open=benchmark_close * (1.0 + rng.normal(0.0, 0.002, days)),
        benchmark_close=benchmark_close,
        mkt=pd.Series(rng.normal(0.0, 0.01, days), index=dates),
        smb=pd.Series(rng.normal(0.0, 0.01, days), index=dates),
        hml=pd.Series(rng.normal(0.0, 0.01, days), index=dates),
    )
    return GTJA191Panels(
        open=open_,
        close=close,
        high=high,
        low=low,
        volume=volume,
        amount=close * volume,
        vwap=(high + low + close) / 3.0,
        returns=close.pct_change(fill_method=None),
        external=external,
    )


def panel_inputs(panels: GTJA191Panels) -> dict[str, object]:
    return {
        "open": panels.open,
        "high": panels.high,
        "low": panels.low,
        "close": panels.close,
        "volume": panels.volume,
        "amount": panels.amount,
        "vwap": panels.vwap,
        "benchmark_open": panels.external.benchmark_open,
        "benchmark_close": panels.external.benchmark_close,
        "mkt": panels.external.mkt,
        "smb": panels.external.smb,
        "hml": panels.external.hml,
    }


class GTJA191OperatorTests(unittest.TestCase):
    def test_report_specific_operators_have_locked_examples(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 4.0, 7.0, 5.0]})
        expected_sma = pd.DataFrame({"a": [1.0, 2.0, 11.0 / 3.0, 37.0 / 9.0]})
        pd.testing.assert_frame_equal(sma_cn(frame, 3, 1), expected_sma)
        expected_wma = (7.0 + 0.9 * 4.0 + 0.9**2) / (1.0 + 0.9 + 0.9**2)
        self.assertAlmostEqual(wma(frame, 3).iloc[2, 0], expected_wma)
        self.assertEqual(highday(frame, 4).iloc[-1, 0], 1.0)
        self.assertEqual(lowday(frame, 4).iloc[-1, 0], 3.0)

    def test_regression_and_conditional_windows(self) -> None:
        independent = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        dependent = independent * 2.0 + 3.0
        self.assertAlmostEqual(regbeta(dependent, independent, 4).iloc[-1, 0], 2.0)
        self.assertAlmostEqual(regresi(dependent, independent, 4).iloc[-1, 0], 0.0)
        condition = independent > 2.0
        self.assertEqual(count(condition, 3).iloc[-1, 0], 2.0)
        self.assertEqual(sumif(independent, 3, condition).iloc[-1, 0], 7.0)


class GTJA191LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panels = complete_panels()
        cls.calculator = GTJA191(cls.panels)

    def test_catalog_temporarily_excludes_alpha030(self) -> None:
        names = get_catalog().canonical_names("gtja191")
        self.assertEqual(GTJA191_NAMES, names)
        self.assertEqual("gtja_001", names[0])
        self.assertEqual("gtja_191", names[-1])
        self.assertEqual(frozenset({"gtja_030"}), GTJA191_BUILD_EXCLUSIONS)
        self.assertNotIn("gtja_030", names)
        self.assertEqual(190, len(GTJA191Library().specs))
        self.assertEqual(191, len(GTJA191_FORMULA_NAMES))
        specs = GTJA191Library().specs
        self.assertEqual(("alpha029", "alpha031"), (specs[28].source_name, specs[29].source_name))
        self.assertEqual((29, 31), (specs[28].ordinal, specs[29].ordinal))

    def test_representative_formulas_match_direct_calculation(self) -> None:
        expected_015 = self.panels.open / delay(self.panels.close, 1) - 1.0
        pd.testing.assert_frame_equal(self.calculator.calculate(15), expected_015)
        expected_126 = (self.panels.close + self.panels.high + self.panels.low) / 3.0
        pd.testing.assert_frame_equal(self.calculator.calculate(126), expected_126)
        expected_191 = correlation(self.panels.volume.rolling(20).mean(), self.panels.low, 5)
        expected_191 += (self.panels.high + self.panels.low) / 2.0 - self.panels.close
        pd.testing.assert_frame_equal(self.calculator.calculate(191), expected_191)

    def test_alpha159_preserves_report_literal_expression(self) -> None:
        previous = delay(self.panels.close, 1)
        floor = element_min(self.panels.low, previous)
        spread = element_max(self.panels.high, previous) - floor

        def term(window: int, weight: int) -> pd.DataFrame:
            return safe_div(self.panels.close - ts_sum(floor, window), ts_sum(spread, window)) * weight

        expected = (term(6, 12 * 24) + term(12, 6 * 24) + term(24, 6 * 24)) * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        pd.testing.assert_frame_equal(self.calculator.calculate(159), expected)
        self.assertIn("literal", GTJA191_FORMULA_NOTES["gtja_159"].lower())

    def test_alpha181_uses_documented_twenty_day_denominator(self) -> None:
        benchmark = pd.DataFrame(
            np.repeat(self.panels.external.benchmark_close.to_numpy()[:, None], self.panels.close.shape[1], axis=1),
            index=self.panels.close.index,
            columns=self.panels.close.columns,
        )
        centered_return = self.panels.returns - mean(self.panels.returns, 20)
        centered_benchmark = benchmark - mean(benchmark, 20)
        expected = safe_div(
            ts_sum(centered_return - centered_benchmark.pow(2), 20),
            ts_sum(centered_benchmark.pow(3), 20),
        )
        pd.testing.assert_frame_equal(self.calculator.calculate(181), expected)
        self.assertIn("20-day", GTJA191_FORMULA_NOTES["gtja_181"])

    def test_external_dependencies_fail_explicitly(self) -> None:
        panels = complete_panels(days=80)
        panels = GTJA191Panels(
            open=panels.open,
            close=panels.close,
            high=panels.high,
            low=panels.low,
            volume=panels.volume,
            amount=panels.amount,
            vwap=panels.vwap,
            returns=panels.returns,
        )
        with self.assertRaisesRegex(GTJA191DataError, "mkt, smb, hml"):
            GTJA191(panels).calculate(30)
        with self.assertRaisesRegex(GTJA191DataError, "benchmark_close"):
            GTJA191(panels).calculate(181)

    def test_all_191_formulas_are_callable_and_aligned(self) -> None:
        for name in GTJA191_FORMULA_NAMES:
            with self.subTest(name=name):
                result = self.calculator.calculate(name)
                self.assertEqual(self.panels.close.shape, result.shape)
                self.assertEqual(self.panels.close.index.tolist(), result.index.tolist())
                self.assertEqual(self.panels.close.columns.tolist(), result.columns.tolist())

    def test_generic_engine_selects_panel_backend_and_runs_full_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = create_factor_engine(temporary, FactorBuildConfig(factor_set="gtja191"))
            self.assertIsInstance(engine, PanelFactorEngine)
            output = engine.run(panel_inputs(complete_panels(days=80, symbols=3)))
        self.assertEqual(GTJA191_NAMES, tuple(output))


class GTJA191FactorStoreTests(unittest.TestCase):
    def test_panel_engine_writes_standard_factor_store_with_audited_references(self) -> None:
        panels = complete_panels(days=80, symbols=3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            bars_root = canonical / "bars"
            reference_root = canonical / "reference"
            bars_root.mkdir(parents=True)
            reference_root.mkdir(parents=True)
            records = []
            for instrument in panels.close.columns:
                frame = pd.DataFrame(
                    {
                        "datetime": panels.close.index,
                        "instrument": instrument,
                        "open": panels.open[instrument],
                        "high": panels.high[instrument],
                        "low": panels.low[instrument],
                        "close": panels.close[instrument],
                        "volume": panels.volume[instrument],
                        "amount": panels.amount[instrument],
                        "vwap": panels.vwap[instrument],
                        "in_csi300": True,
                    }
                )
                records.append(frame)
            pd.concat(records, ignore_index=True).to_parquet(bars_root / "part.parquet", index=False)
            canonical_payload = {"status": "complete", "fingerprint": "canonical-test"}
            atomic_write_json(canonical / "manifest.json", canonical_payload)
            pd.DataFrame(
                {
                    "ts_code": "399300.SZ",
                    "datetime": panels.close.index,
                    "open": panels.external.benchmark_open,
                    "close": panels.external.benchmark_close,
                }
            ).to_parquet(reference_root / "index_daily.parquet", index=False)
            style_path = root / "style.parquet"
            pd.DataFrame(
                {
                    "datetime": panels.close.index,
                    "mkt": panels.external.mkt,
                    "smb": panels.external.smb,
                    "hml": panels.external.hml,
                }
            ).to_parquet(style_path, index=False)
            engine = create_factor_engine(
                root / "cache",
                FactorBuildConfig(
                    factor_set="gtja191",
                    external_inputs=(("style_factors", str(style_path)),),
                ),
            )
            manifest = engine.build_from_canonical(canonical, root / "factors")
            stored = json.loads((root / "factors" / "gtja191" / "manifest.json").read_text())
            partition = next((root / "factors" / "gtja191").glob("year=*/factors.parquet"))
            frame = pd.read_parquet(partition)

        self.assertEqual("complete", manifest["status"])
        self.assertEqual(190, manifest["columns"])
        self.assertEqual("pandas_panel_v1", manifest["execution"]["backend"])
        expected_fingerprint = stable_hash({key: value for key, value in stored.items() if key != "fingerprint"})
        self.assertEqual(expected_fingerprint, stored["fingerprint"])
        self.assertEqual(["datetime", "instrument", *GTJA191_NAMES], frame.columns.tolist())

    def test_factor_store_builds_without_style_factor_source(self) -> None:
        panels = complete_panels(days=5, symbols=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            bars_root = canonical / "bars"
            reference_root = canonical / "reference"
            bars_root.mkdir(parents=True)
            reference_root.mkdir(parents=True)
            instrument = panels.close.columns[0]
            pd.DataFrame(
                {
                    "datetime": panels.close.index,
                    "instrument": instrument,
                    "open": panels.open[instrument],
                    "high": panels.high[instrument],
                    "low": panels.low[instrument],
                    "close": panels.close[instrument],
                    "volume": panels.volume[instrument],
                    "amount": panels.amount[instrument],
                    "vwap": panels.vwap[instrument],
                    "in_csi300": True,
                }
            ).to_parquet(bars_root / "part.parquet", index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "test"})
            pd.DataFrame(
                {
                    "ts_code": "000300.SH",
                    "datetime": panels.close.index,
                    "open": panels.external.benchmark_open,
                    "close": panels.external.benchmark_close,
                }
            ).to_parquet(reference_root / "index_daily.parquet", index=False)
            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="gtja191"))
            manifest = engine.build_from_canonical(canonical, root / "factors")

        self.assertEqual("complete", manifest["status"])
        self.assertEqual(190, manifest["columns"])
        self.assertEqual({"benchmark"}, set(manifest["reference_inputs"]))


if __name__ == "__main__":
    unittest.main()
