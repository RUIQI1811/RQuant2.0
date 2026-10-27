from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine, cpp_compiler_available, create_factor_engine
from rquant.factors.libraries.gtja191 import (
    GTJA191_BUILD_EXCLUSIONS,
    GTJA191_FORMULA_NAMES,
    GTJA191_FORMULA_NOTES,
    GTJA191_NAMES,
    GTJA191DataError,
    GTJA191Library,
)
from rquant.factors.panel_operators import delay, element_max, element_min, safe_div, ts_sum
from rquant.io import atomic_write_json, stable_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_reference.gtja191 import GTJA191, GTJA191ExternalData, GTJA191Panels, mean  # noqa: E402

KUNQUANT_READY = importlib.util.find_spec("KunQuant") is not None and cpp_compiler_available()
# Cells allowed to differ from the pandas reference.  GTJA113 tests 2-day windows of 5- and 20-day sums for being
# constant, and whether two sums of different prices compare equal depends on how the rolling sum was accumulated.
ROUNDING_SENSITIVE_CELLS = {"gtja_113": 4}


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


def market_panels(days: int = 280, symbols: int = 16, seed: int = 7) -> GTJA191Panels:
    """Tick-rounded prices with ties, a limit-locked run, a late listing, a delisting, a suspension and gaps."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    columns = [f"S{number:03d}" for number in range(symbols)]
    close = np.round(20.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (days, symbols)), axis=0)), 2)
    open_ = np.round(close * (1.0 + rng.normal(0.0, 0.006, close.shape)), 2)
    high = np.round(np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.02, close.shape)), 2)
    low = np.round(np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.02, close.shape)), 2)
    volume = np.round(rng.lognormal(12.0, 0.5, close.shape), -2)
    for field in (open_, high, low, close):
        field[100:112, 3] = close[99, 3]
    volume[150:160, 5] = 1000.0
    observed = np.ones(close.shape, dtype=bool)
    observed[:60, 7] = False
    observed[200:, 8] = False
    observed[120:135, 9] = False
    for column in range(10, symbols, 3):
        observed[rng.choice(days, 5, replace=False), column] = False

    def panel(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.where(observed, values, np.nan), index=dates, columns=columns)

    open_frame, high_frame, low_frame, close_frame, volume_frame = map(panel, (open_, high, low, close, volume))
    benchmark_close = pd.Series(4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, days))), index=dates)
    return GTJA191Panels(
        open=open_frame,
        close=close_frame,
        high=high_frame,
        low=low_frame,
        volume=volume_frame,
        amount=close_frame * volume_frame,
        vwap=(high_frame + low_frame + close_frame) / 3.0,
        returns=safe_div(close_frame, close_frame.shift(1)) - 1.0,
        external=GTJA191ExternalData(
            benchmark_open=benchmark_close * (1.0 + rng.normal(0.0, 0.003, days)),
            benchmark_close=benchmark_close,
        ),
    )


def kunquant_inputs(panels: GTJA191Panels) -> dict[str, np.ndarray]:
    fields = {
        "open": panels.open,
        "high": panels.high,
        "low": panels.low,
        "close": panels.close,
        "volume": panels.volume,
        "amount": panels.amount,
        "vwap": panels.vwap,
    }
    arrays = {name: frame.to_numpy(dtype=float) for name, frame in fields.items()}
    for name in ("benchmark_open", "benchmark_close"):
        series = getattr(panels.external, name).to_numpy(dtype=float)
        arrays[name] = np.repeat(series[:, None], panels.close.shape[1], axis=1)
    return arrays


def run_graph(build, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compile and run a small KunQuant graph of ``kunquant_panel`` operators."""

    from KunQuant.Driver import KunCompilerConfig
    from KunQuant.jit import cfake
    from KunQuant.Op import Builder, Input, Output
    from KunQuant.runner import KunRunner
    from KunQuant.Stage import Function

    builder = Builder()
    with builder:
        outputs = build({name: Input(name) for name in inputs})
        for name, value in outputs.items():
            Output(value, name)
    config = KunCompilerConfig(dtype="double", input_layout="TS", output_layout="TS", options={"no_fast_stat": True})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        library = cfake.compileit(
            [("graph", Function(builder.ops), config)],
            "rquant_gtja191_operators",
            cfake.CppCompilerConfig(),
            tempdir=temporary,
        )
        module = library.getModule("graph")
        rows = next(iter(inputs.values())).shape[0]
        executor = KunRunner.createSingleThreadExecutor()
        arrays = {name: np.ascontiguousarray(values, dtype=np.float64) for name, values in inputs.items()}
        return KunRunner.runGraph(executor, module, arrays, 0, rows, {})


class GTJA191CatalogTests(unittest.TestCase):
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

    def test_generic_engine_selects_kunquant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = create_factor_engine(temporary, FactorBuildConfig(factor_set="gtja191"))
        self.assertIsInstance(engine, KunQuantFactorEngine)
        self.assertEqual(("benchmark_open", "benchmark_close"), engine.reference_fields)
        self.assertTrue(engine.strict_float)

    def test_missing_amount_vwap_or_benchmark_fails_explicitly(self) -> None:
        library = GTJA191Library()
        inputs = kunquant_inputs(complete_panels(days=20, symbols=2))
        library.validate_inputs(inputs)
        for field in ("amount", "vwap", "benchmark_close"):
            broken = dict(inputs)
            broken[field] = inputs[field].copy()
            broken[field][5] = np.nan
            with self.subTest(field=field), self.assertRaisesRegex(GTJA191DataError, field):
                library.validate_inputs(broken)


class GTJA191ReferenceTests(unittest.TestCase):
    """The pandas reference keeps the report's literal formulas the KunQuant graph is checked against."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panels = complete_panels()
        cls.calculator = GTJA191(cls.panels)

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


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class GTJA191OperatorTests(unittest.TestCase):
    def test_report_specific_operators_have_locked_examples(self) -> None:
        from rquant.factors.extensions import kunquant_panel as k
        from rquant.factors.libraries import gtja191

        column = np.array([1.0, 4.0, 7.0, 5.0])
        independent = np.array([1.0, 2.0, 3.0, 4.0])
        inputs = {
            "a": np.repeat(column[:, None], 4, axis=1),
            "x": np.repeat(independent[:, None], 4, axis=1),
        }

        def build(nodes: dict) -> dict:
            a, x = nodes["a"], nodes["x"]
            dependent = x * 2.0 + 3.0
            return {
                "sma": k.sma_cn(a, 3, 1),
                "wma": gtja191.wma(a, 3),
                "highday": gtja191.highday(a, 4),
                "lowday": gtja191.lowday(a, 4),
                "count": k.count(k.gt(x, 2.0), 3),
                "sumif": k.sumif(x, 3, k.gt(x, 2.0)),
                "sequence_beta": gtja191.sequence_regbeta(dependent, 4),
                "filtered_beta": gtja191.filtered_regbeta(dependent, x, k.always(), 4),
            }

        output = {name: values[:, 0] for name, values in run_graph(build, inputs).items()}
        np.testing.assert_allclose(output["sma"], [1.0, 2.0, 11.0 / 3.0, 37.0 / 9.0])
        self.assertAlmostEqual(output["wma"][2], (7.0 + 0.9 * 4.0 + 0.9**2) / (1.0 + 0.9 + 0.9**2))
        self.assertEqual(1.0, output["highday"][-1])
        self.assertEqual(3.0, output["lowday"][-1])
        self.assertEqual(2.0, output["count"][-1])
        self.assertEqual(7.0, output["sumif"][-1])
        self.assertAlmostEqual(2.0, output["sequence_beta"][-1])
        self.assertAlmostEqual(2.0, output["filtered_beta"][-1])


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class GTJA191KunQuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Windows cannot delete a JIT-compiled DLL while this process still has it loaded.
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.engine = KunQuantFactorEngine(cls.temporary.name, FactorBuildConfig(factor_set="gtja191", workers=2))
        cls.module = cls.engine.compile()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_all_factors_match_the_pandas_reference(self) -> None:
        panels = market_panels()
        output = self.engine.run(kunquant_inputs(panels), module=self.module)
        self.assertEqual(GTJA191_NAMES, tuple(output))
        calculator = GTJA191(panels)
        excess: dict[str, int] = {}
        for name in GTJA191_NAMES:
            expected = calculator.calculate(name).to_numpy(dtype=float)
            actual = output[name]
            self.assertEqual(expected.shape, actual.shape)
            # GTJA149 regresses on 252 benchmark down days, more history than this panel has.
            self.assertEqual(name != "gtja_149", bool(np.isfinite(expected).any()), name)
            missing = np.isnan(expected)
            agree = np.isnan(actual) == missing
            agree[~missing] &= np.isclose(actual[~missing], expected[~missing], rtol=1e-7, atol=1e-9)
            if int((~agree).sum()) > ROUNDING_SENSITIVE_CELLS.get(name, 0):
                excess[name] = int((~agree).sum())
        self.assertEqual({}, excess)

    def test_engine_writes_standard_factor_store_with_audited_references(self) -> None:
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
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-test"})
            pd.DataFrame(
                {
                    "ts_code": "399300.SZ",
                    "datetime": panels.close.index,
                    "open": panels.external.benchmark_open,
                    "close": panels.external.benchmark_close,
                }
            ).to_parquet(reference_root / "index_daily.parquet", index=False)
            engine = KunQuantFactorEngine(root / "cache", FactorBuildConfig(factor_set="gtja191"))
            engine.compile = lambda: self.module  # type: ignore[method-assign]
            manifest = engine.build_from_canonical(canonical, root / "factors")
            stored = json.loads((root / "factors" / "gtja191" / "manifest.json").read_text())
            partition = next((root / "factors" / "gtja191").glob("year=*/factors.parquet"))
            frame = pd.read_parquet(partition)

        self.assertEqual("complete", manifest["status"])
        self.assertEqual(190, manifest["columns"])
        self.assertEqual({"benchmark"}, set(manifest["reference_inputs"]))
        expected_fingerprint = stable_hash({key: value for key, value in stored.items() if key != "fingerprint"})
        self.assertEqual(expected_fingerprint, stored["fingerprint"])
        self.assertEqual(["datetime", "instrument", *GTJA191_NAMES], frame.columns.tolist())
        stored_015 = frame.pivot(index="datetime", columns="instrument", values="gtja_015")
        expected_015 = GTJA191(panels).calculate(15).reindex(index=stored_015.index, columns=stored_015.columns)
        np.testing.assert_allclose(stored_015.to_numpy(), expected_015.to_numpy(), rtol=1e-12)

if __name__ == "__main__":
    unittest.main()
