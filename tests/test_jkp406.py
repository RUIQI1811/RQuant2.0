from __future__ import annotations

import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_jkp153 import KUNQUANT_READY, synthetic_inputs

from rquant.factors.catalog import get_catalog
from rquant.factors.libraries.registry import get_library_registry
from rquant.io import atomic_write_json, sha256_file, stable_hash


@lru_cache(maxsize=1)
def compiled():
    from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine

    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    engine = KunQuantFactorEngine(temporary.name, FactorBuildConfig(factor_set="jkp406", workers=2))
    return temporary, engine, engine.compile(), engine.compile_statements()


def calculate(inputs):
    from rquant.factors.libraries.jkp406 import JKP406Library

    _, engine, module, statement_modules = compiled()
    close = inputs["close"]
    prepared = JKP406Library().align_reference_inputs(inputs, close.index, close.columns)
    statements = engine.run_statements(prepared, close.index, close.columns, statement_modules)
    arrays = {}
    for name in engine.input_fields:
        value = statements.daily(name) if name in statements else prepared[name]
        if isinstance(value, pd.Series):
            value = np.repeat(value.to_numpy()[:, None], len(close.columns), axis=1)
        arrays[name] = np.ascontiguousarray(value, dtype=float)
    output = engine.run(arrays, module=module, statements=statements)
    return {
        spec.source_name: pd.DataFrame(output[spec.canonical_name], index=close.index, columns=close.columns)
        for spec in JKP406Library().specs
    }


def write_reference_fixture(root, inputs):
    canonical = root / "canonical"
    reference = canonical / "reference"
    reference.mkdir(parents=True)
    dates = inputs["close"].index
    bars = pd.concat(
        [inputs[field].stack(future_stack=True).rename(field) for field in ("high", "low", "close", "amount", "cap")],
        axis=1,
    ).reset_index()
    bars["in_csi300"] = inputs["universe"].stack(future_stack=True).to_numpy()
    for year, frame in bars.groupby(bars["datetime"].dt.year):
        path = canonical / "bars" / f"year={year}" / "bars.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
    daily = pd.concat(
        [
            (inputs["basic_turnover_rate"].stack(future_stack=True) * 100).rename("turnover_rate"),
            inputs["basic_close"].stack(future_stack=True).rename("close"),
            (inputs["basic_total_share"].stack(future_stack=True) / 10_000).rename("total_share"),
            (inputs["basic_dv_ttm"].stack(future_stack=True) * 100).rename("dv_ttm"),
        ],
        axis=1,
    ).reset_index()
    files = {}
    for year, frame in daily.groupby(daily["datetime"].dt.year):
        name = f"daily_basic/year={year}/data.parquet"
        path = reference / name
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        files[name] = sha256_file(path)
    pd.DataFrame(
        {"datetime": dates, **{name: inputs[name].to_numpy() for name in ("mkt", "smb", "hml")}, "stocks": 200}
    ).to_parquet(reference / "market_factors.parquet", index=False)
    inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
    pd.DataFrame(
        {"ts_code": [f"60000{i}.SH" for i in range(len(inputs["close"].columns))], "list_date": "20040601"}
    ).to_parquet(reference / "stock_basic.parquet", index=False)
    for name in ("market_factors.parquet", "fundamentals.parquet", "stock_basic.parquet"):
        files[name] = sha256_file(reference / name)
    atomic_write_json(
        canonical / "manifest.json",
        {"status": "complete", "fingerprint": "canonical-fixture", "raw_fingerprint": "raw-fixture"},
    )
    manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(reference / "manifest.json", manifest)
    return canonical, manifest


class JKP406CatalogTests(unittest.TestCase):
    def test_legacy_reference_manifests_still_load_and_track_the_listing_file_hash(self):
        from rquant.factors.libraries.jkp406 import JKP406Library

        inputs = synthetic_inputs(days=80, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = write_reference_fixture(Path(temporary), inputs)
            del manifest["files"]["stock_basic.parquet"]
            manifest.pop("fingerprint")
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(root / "reference" / "manifest.json", manifest)
            _, hashes = JKP406Library().load_reference_inputs(root, inputs["close"].index, inputs["close"].columns)
            self.assertEqual(sha256_file(root / "reference" / "stock_basic.parquet"), hashes["jkp406_stock_basic"])

    def test_reference_loading_names_its_inputs_and_rejects_changed_listing_dates(self):
        from rquant.errors import DataContractError
        from rquant.factors.libraries.jkp406 import JKP406Library

        inputs = synthetic_inputs(days=80, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = write_reference_fixture(Path(temporary), inputs)
            library = JKP406Library()
            loaded, hashes = library.load_reference_inputs(root, inputs["close"].index, inputs["close"].columns)
            self.assertEqual(manifest["fingerprint"], hashes["jkp406_reference"])
            self.assertTrue(all(name.startswith("jkp406_") for name in hashes))
            pd.testing.assert_frame_equal(inputs["basic_close"], loaded["basic_close"], check_freq=False)
            path = root / "reference" / "stock_basic.parquet"
            pd.read_parquet(path).assign(list_date="20200101").to_parquet(path, index=False)
            with self.assertRaisesRegex(DataContractError, "manifest"):
                library.load_reference_inputs(root, inputs["close"].index, inputs["close"].columns)

    def test_inherited_formula_changes_invalidate_the_extended_cache(self) -> None:
        import inspect

        from rquant.factors.libraries import jkp153
        from rquant.factors.libraries.jkp406 import JKP406Library

        library = JKP406Library()
        previous = library.implementation_fingerprint
        getsource = inspect.getsource
        with patch(
            "inspect.getsource",
            side_effect=lambda obj: getsource(obj) + "\n# changed core" if obj is jkp153 else getsource(obj),
        ):
            self.assertNotEqual(previous, library.implementation_fingerprint)

    def test_extended_library_is_buildable_and_keeps_the_core_contract(self) -> None:
        catalog = get_catalog()
        self.assertIn("jkp406", catalog.factor_sets())
        from rquant.factors.libraries.jkp406_catalog import JKP406_NAMES, JKP406_SKIPPED, JKP406_SOURCE_NAMES

        specs = catalog.select("jkp406")
        self.assertGreater(len(specs), 300)
        self.assertEqual(
            tuple(f"jkpx_{i:03d}" for i in range(1, len(specs) + 1)), tuple(s.canonical_name for s in specs)
        )
        self.assertEqual(JKP406_NAMES, tuple(s.source_name for s in specs))
        self.assertEqual(set(JKP406_SOURCE_NAMES), set(JKP406_NAMES) | set(JKP406_SKIPPED))
        self.assertFalse(set(JKP406_NAMES) & set(JKP406_SKIPPED))
        self.assertEqual(len(JKP406_SOURCE_NAMES), len(set(JKP406_SOURCE_NAMES)))
        self.assertTrue(all(JKP406_SKIPPED.values()))
        self.assertEqual(144, len(catalog.select("jkp153")))
        self.assertEqual("jkp_144", catalog.select("jkp153")[-1].canonical_name)
        self.assertEqual("kunquant", get_library_registry().select("jkp406")[0].backend)

    def test_inventory_reconciles_pinned_source_without_filling_the_count_gap(self) -> None:
        from rquant.factors.libraries.jkp406_catalog import (
            ADVERTISED_COUNT,
            JKP406_NAMES,
            JKP406_SOURCE_NAMES,
            SOURCE_COUNT_DISCREPANCY,
        )

        self.assertEqual(404, len(JKP406_SOURCE_NAMES))
        self.assertEqual(ADVERTISED_COUNT, len(JKP406_SOURCE_NAMES) + SOURCE_COUNT_DISCREPANCY)
        self.assertEqual(362, len(JKP406_NAMES))


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and C++ compiler required")
class JKP406CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs(stocks=5)
        cls.outputs = calculate(cls.inputs)

    def test_every_output_matches_independent_panel_reference(self):
        from panel_reference.jkp406 import JKP406Chars

        from rquant.factors.libraries.jkp406_catalog import JKP406_NAMES

        reference = JKP406Chars(self.inputs)
        for name in JKP406_NAMES:
            with self.subTest(name=name):
                actual, expected = self.outputs[name], reference.calculate(name)
                if name not in {"seas_6_10an", "seas_6_10na"}:
                    self.assertTrue(np.isfinite(actual.to_numpy()).any(), name)
                np.testing.assert_array_equal(actual.isna(), expected.isna())
                np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=2e-8, atol=1e-8)

    def test_every_core_characteristic_keeps_its_values(self):
        from panel_reference.jkp153 import JKPChars

        from rquant.factors.libraries.jkp153 import JKP_NAMES

        reference = JKPChars(self.inputs)
        for name in JKP_NAMES:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    self.outputs[name], reference.calculate(name), equal_nan=True, rtol=2e-8, atol=1e-8
                )

    def test_prefix_recomputation_cannot_see_future_announcements_or_quotes(self):
        truncated = {}
        cutoff = self.inputs["close"].index[1200]
        for key, value in self.inputs.items():
            if key == "fundamentals":
                value = value[pd.to_datetime(value["ann_date"]) <= cutoff]
            elif isinstance(value, (pd.DataFrame, pd.Series)) and isinstance(value.index, pd.DatetimeIndex):
                value = value.loc[:cutoff]
            truncated[key] = value
        short = calculate(truncated)
        for name, values in short.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    values, self.outputs[name].loc[:cutoff], equal_nan=True, rtol=2e-8, atol=1e-8
                )

    def test_new_returns_and_ratios_match_direct_definitions(self):
        close = self.inputs["close"].ffill()
        expected = (close / close.shift(42) - 1).where(self.inputs["close"].notna())
        np.testing.assert_allclose(self.outputs["ret_2_0"], expected, equal_nan=True)
        from panel_reference.jkp153 import JKPChars

        ref = JKPChars(self.inputs)
        assets = ref.grid.panel(ref.x("at")).where(self.inputs["close"].notna())
        np.testing.assert_allclose(self.outputs["assets"], assets, equal_nan=True)
        sales = ref.x("sale")
        ebitda_margin = ref.grid.panel(np.where(sales > 0, ref.x("ebitda") / sales, np.nan))
        ebitda_margin = ebitda_margin.where(self.inputs["close"].notna())
        np.testing.assert_allclose(self.outputs["ebitda_sale"], ebitda_margin, equal_nan=True)

    def test_zero_denominators_stay_missing_and_signed_fixed_capital_is_preserved(self):
        inputs = synthetic_inputs(days=80, stocks=4)
        reports = inputs["fundamentals"].copy()
        reports["revenue"] = 0.0
        reports["total_liab"] = 200.0
        reports["fix_assets"] = -100.0
        inputs["fundamentals"] = reports
        out = calculate(inputs)
        self.assertTrue(out["ebitda_sale"].isna().all().all())
        values = out["lt_ppen"].to_numpy()
        self.assertTrue(np.isfinite(values).any())
        np.testing.assert_allclose(values[np.isfinite(values)], -2.0)

    def test_no_output_fills_a_suspension_with_a_quote(self):
        missing = self.inputs["close"].isna().to_numpy()
        self.assertTrue(missing.any())
        for name, values in self.outputs.items():
            with self.subTest(name=name):
                self.assertTrue(np.isnan(values.to_numpy()[missing]).all())

    def test_quick_ratio_rejects_negative_current_liabilities(self):
        inputs = synthetic_inputs(days=80, stocks=4)
        inputs["fundamentals"] = inputs["fundamentals"].assign(total_cur_liab=-100.0)
        out = calculate(inputs)
        self.assertTrue(out["caliq_cl"].isna().all().all())

    def test_high_low_volatility_adjusts_overnight_price_gaps(self):
        inputs = synthetic_inputs(days=80, stocks=4)
        dates, stocks = inputs["close"].index, inputs["close"].columns
        price = 10 * np.exp(np.arange(80) * 0.001)
        price[45:] *= 1.7
        for key, multiplier in (("close", 1.0), ("high", 1.01), ("low", 0.99)):
            inputs[key] = pd.DataFrame(np.repeat((price * multiplier)[:, None], 4, axis=1), index=dates, columns=stocks)
        out = calculate(inputs)["rvolhl_21d"]
        high, low, previous = price * 1.01, price * 0.99, np.roll(price, 1)
        high_t, low_t = high.copy(), low.copy()
        below, above = previous < low, previous > high
        high_t[below], low_t[below] = high[below] - low[below] + previous[below], previous[below]
        high_t[above], low_t[above] = previous[above], low[above] + previous[above] - high[above]
        beta = np.log(high_t[1:] / low_t[1:]) ** 2 + np.log(high[:-1] / low[:-1]) ** 2
        gamma = np.log(np.maximum(high_t[1:], high[:-1]) / np.minimum(low_t[1:], low[:-1])) ** 2
        k, c = np.sqrt(8 / np.pi), 3 - 2 * np.sqrt(2)
        sigma = np.maximum((np.sqrt(beta / 2) - np.sqrt(beta)) / (k * c) + np.sqrt(gamma / (k * k * c)), 0)
        expected = pd.Series(np.r_[np.nan, sigma]).rolling(21, min_periods=15).mean()
        np.testing.assert_allclose(out.iloc[:, 0], expected, equal_nan=True, rtol=1e-10, atol=1e-12)

    def test_statement_outputs_use_a_distinct_namespace(self):
        from rquant.factors.libraries.jkp153 import JKP153Library
        from rquant.factors.libraries.jkp406 import JKP406Library

        self.assertFalse(set(JKP406Library().statement_outputs) & set(JKP153Library().statement_outputs))

    def test_engine_builds_an_audited_temporary_factor_store(self):
        from rquant.factors.engine import FactorBuildConfig, create_factor_engine

        inputs = synthetic_inputs(days=320, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, manifest = write_reference_fixture(root, inputs)
            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="jkp406"))
            _, _, module, statement_modules = compiled()
            engine.compile = lambda: module
            engine.compile_statements = lambda: statement_modules
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(362, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["jkp406_reference"])
            built = pd.concat(pd.read_parquet(p) for p in (root / "factors" / "jkp406").glob("year=*/*.parquet"))
            eligible = inputs["universe"].any(axis=0)
            self.assertEqual(set(eligible.index[eligible]), set(built["instrument"]))
            columns = get_catalog().source_to_canonical("jkp406")
            expected = calculate(inputs)
            for source in ("assets", "ebitda_sale", "ret_2_0", "cash_conversion", "beta_21d"):
                actual = built.pivot(index="datetime", columns="instrument", values=columns[source])
                want = expected[source].reindex(index=actual.index, columns=actual.columns)
                np.testing.assert_allclose(actual, want, equal_nan=True, rtol=2e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
