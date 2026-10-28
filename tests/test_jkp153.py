from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.factors.catalog import get_catalog
from rquant.factors.engine import (
    FactorBuildConfig,
    KunQuantFactorEngine,
    cpp_compiler_available,
    create_factor_engine,
)
from rquant.factors.libraries.jkp153 import FLOWS, ITEMS, JKP_NAMES, JKP153Library
from rquant.io import atomic_write_json, sha256_file, stable_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_reference.jkp153 import JKP153Library as ReferenceLibrary  # noqa: E402
from panel_reference.jkp153 import (  # noqa: E402
    JKPChars,
    corwin_schultz,
    cross_section_rank,
    step_sums,
)

KUNQUANT_READY = importlib.util.find_spec("KunQuant") is not None and cpp_compiler_available()


@lru_cache(maxsize=1)
def compiled():
    # Windows cannot delete a JIT-compiled DLL while this process still has it loaded.
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    engine = KunQuantFactorEngine(temporary.name, FactorBuildConfig(factor_set="jkp153", workers=2))
    return temporary, engine, engine.compile(), engine.compile_statements()


def calculate(inputs: dict[str, object]) -> dict[str, pd.DataFrame]:
    """KunQuant outputs by source name, as panels shaped like the inputs."""
    _, engine, module, statement_modules = compiled()
    close = inputs["close"]
    prepared = JKP153Library().align_reference_inputs(inputs, close.index, close.columns)
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
        for spec in JKP153Library().specs
    }

FLOW_LINES = sorted({line for lines in FLOWS.values() for line in lines})
STOCK_LINES = sorted({line for lines in ITEMS.values() for line in lines})
# Outputs that need more history than the fixture has, or a statement line the fixture leaves out on purpose.
SPARSE = {"seas_6_10an", "seas_6_10na"}


def synthetic_inputs(days: int = 1700, stocks: int = 6, seed: int = 21) -> dict[str, object]:
    """Daily panels and quarterly statements shaped like the engine's jkp153 inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    factors = {name: rng.normal(0.0, 0.006, days) for name in ("smb", "hml")}
    factors["mkt"] = rng.normal(0.0003, 0.012, days)
    loadings = rng.uniform(0.6, 1.4, stocks)
    returns = factors["mkt"][:, None] * loadings + 0.3 * factors["smb"][:, None] + rng.normal(0.0, 0.015,
                                                                                             (days, stocks))
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=instruments)
    close.iloc[300:312, 1] = np.nan
    close.iloc[:60, 2] = np.nan
    high = close * np.exp(np.abs(rng.normal(0.0, 0.01, close.shape)))
    low = close * np.exp(-np.abs(rng.normal(0.0, 0.01, close.shape)))
    shares = rng.uniform(1e8, 1e9, stocks)
    # Issued shares drift at a different rate per stock: with a constant count, eqnpo_12m (price growth minus market
    # value growth) cancels to rounding noise for every stock, and ranking that noise differs between log builds.
    cap = close * shares * np.linspace(1.0, 1.2, days)[:, None] ** np.linspace(0.5, 1.5, stocks)
    amount = cap * rng.uniform(0.002, 0.03, close.shape)
    turnover = pd.DataFrame(rng.uniform(0.002, 0.04, close.shape), index=dates, columns=instruments)
    dividend = pd.DataFrame(rng.uniform(0.0, 0.03, close.shape), index=dates, columns=instruments)
    dividend.iloc[:, stocks - 2] = np.nan
    universe = pd.DataFrame(True, index=dates, columns=instruments)
    universe.iloc[:400, 0] = False

    rows = []
    for column, instrument in enumerate(instruments):
        scale = shares[column] * 10.0
        for end in pd.date_range("2005-03-31", dates[-1], freq="QE"):
            quarter = (end.month - 1) // 3 + 1
            row = {"instrument": instrument, "end_date": end,
                   "ann_date": end + pd.Timedelta(days=int(rng.integers(20, 110)))}
            for line in FLOW_LINES:
                row[line] = float(rng.uniform(0.01, 0.06) * scale * quarter)
            row["oper_cost"] = row["revenue"] * rng.uniform(0.5, 0.8)
            for line in (*STOCK_LINES, "total_cur_assets", "total_cur_liab"):
                row[line] = float(rng.uniform(0.05, 0.4) * scale)
            row["total_assets"] = 6.0 * scale * rng.uniform(0.9, 1.1)
            row["total_liab"] = 0.5 * row["total_assets"]
            row["total_nca"] = 0.5 * row["total_assets"]
            row["total_ncl"] = 0.2 * row["total_assets"]
            row["total_hldr_eqy_exc_min_int"] = 0.45 * row["total_assets"]
            rows.append(row)
    fundamentals = pd.DataFrame(rows).drop(index=11).reset_index(drop=True)
    fundamentals.loc[fundamentals["instrument"].eq(instruments[3]), "notes_payable"] = np.nan
    inputs: dict[str, object] = {
        "high": high,
        "low": low,
        "close": close,
        "amount": amount,
        "cap": cap,
        "basic_turnover_rate": turnover,
        "basic_close": close / np.linspace(1.0, 1.3, days)[:, None],
        "basic_total_share": pd.DataFrame(np.tile(shares, (days, 1)), index=dates, columns=instruments)
        * np.linspace(1.0, 1.1, days)[:, None],
        "basic_dv_ttm": dividend,
        "fundamentals": fundamentals,
        "list_date": pd.Series(pd.Timestamp("2004-06-01"), index=instruments),
        "universe": universe,
    }
    for name, values in factors.items():
        inputs[name] = pd.Series(values, index=dates)
    return inputs


def truncated(inputs: dict[str, object], last: pd.Timestamp) -> dict[str, object]:
    """Return what was knowable at the close of ``last``."""
    result: dict[str, object] = {}
    for key, value in inputs.items():
        if key == "fundamentals":
            result[key] = value[value["ann_date"] <= last]
        elif key == "list_date":
            result[key] = value
        else:
            result[key] = value.loc[:last]
    return result


class JKP153CatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        names = get_catalog().canonical_names("jkp153")
        self.assertEqual(144, len(names))
        self.assertEqual(("jkp_001", "jkp_144"), (names[0], names[-1]))
        self.assertEqual(len(set(JKP_NAMES)), len(JKP_NAMES))
        for dropped in ("emp_gr1", "sale_emp_gr1", "ivol_hxz4_21d", "iskew_hxz4_21d", "seas_11_15an", "eqpo_me"):
            self.assertNotIn(dropped, JKP_NAMES)

    def test_every_characteristic_has_a_calculator(self) -> None:
        for name in JKP_NAMES:
            self.assertTrue(hasattr(JKPChars, f"_c_{name}"), name)


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class JKP153CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.outputs = calculate(cls.inputs)

    # Break caught: a KunQuant formula, window, regression or ranking step that drifts from the pandas reference.
    def test_every_characteristic_matches_the_pandas_reference(self) -> None:
        expected = ReferenceLibrary().calculate(self.inputs)
        for name in JKP_NAMES:
            np.testing.assert_allclose(self.outputs[name].to_numpy(), expected[name].to_numpy(), rtol=1e-7,
                                       atol=1e-10, equal_nan=True, err_msg=name)

    def test_all_characteristics_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in JKP_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)
            if name not in SPARSE:
                self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)

    # Break caught: any characteristic that reads a later quote, factor return, statement or index membership.
    def test_no_characteristic_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (900, 1500):
            last = dates[cut]
            past = calculate(truncated(self.inputs, last))
            for name in JKP_NAMES:
                np.testing.assert_allclose(past[name].to_numpy(), self.outputs[name].loc[:last].to_numpy(),
                                           rtol=1e-8, atol=1e-11, equal_nan=True, err_msg=name)

    def test_characteristics_match_direct_definitions(self) -> None:
        close = self.inputs["close"]
        row, column = 1500, 0
        price = close.iloc[:, column].to_numpy()
        np.testing.assert_allclose(self.outputs["ret_12_1"].iloc[row, column], price[row - 21] / price[row - 252] - 1)
        np.testing.assert_allclose(self.outputs["ret_60_12"].iloc[row, column],
                                   price[row - 252] / price[row - 1260] - 1)
        months = [price[row - 21 * k] / price[row - 21 * (k + 1)] - 1 for k in range(12)]
        np.testing.assert_allclose(self.outputs["seas_1_1an"].iloc[row, column], months[11])
        np.testing.assert_allclose(self.outputs["seas_1_1na"].iloc[row, column], np.mean(months[:11]))
        returns = (close / close.shift(1) - 1.0).iloc[row - 20 : row + 1, column].to_numpy()
        np.testing.assert_allclose(self.outputs["rvol_21d"].iloc[row, column], returns.std(ddof=1))
        np.testing.assert_allclose(self.outputs["rmax1_21d"].iloc[row, column], returns.max())
        market = self.inputs["mkt"].iloc[row - 20 : row + 1].to_numpy()
        design = np.column_stack([np.ones(21), market])
        coefficients, *_ = np.linalg.lstsq(design, returns, rcond=None)
        residual = returns - design @ coefficients
        np.testing.assert_allclose(self.outputs["ivol_capm_21d"].iloc[row, column],
                                   np.sqrt((residual**2).sum() / 19), rtol=1e-7)
        turnover = self.inputs["basic_turnover_rate"].iloc[row - 125 : row + 1, column]
        np.testing.assert_allclose(self.outputs["turnover_126d"].iloc[row, column], turnover.mean())
        np.testing.assert_allclose(self.outputs["zero_trades_21d"].iloc[row, column],
                                   1.0 / self.inputs["basic_turnover_rate"].iloc[row - 20 : row + 1, column].mean()
                                   / 480_000.0)
        np.testing.assert_allclose(self.outputs["div12m_me"].iloc[row, column],
                                   self.inputs["basic_dv_ttm"].iloc[row, column])

    def test_quarterly_ratios_appear_on_the_announcement_day(self) -> None:
        fundamentals = self.inputs["fundamentals"]
        instrument = "SH600000"
        reports = fundamentals[fundamentals["instrument"].eq(instrument)].sort_values("end_date")
        dates = self.inputs["close"].index
        report, previous = reports.iloc[30], reports.iloc[29]
        position = int(np.searchsorted(dates.to_numpy(), report["ann_date"].to_numpy()))
        if position <= int(np.searchsorted(dates.to_numpy(), previous["ann_date"].to_numpy())):
            self.skipTest("fixture announced the two quarters out of order")
        cash = (report["money_cap"] + report["trad_asset"]) / report["total_assets"]
        self.assertAlmostEqual(cash, self.outputs["cash_at"][instrument].iloc[position])
        self.assertNotAlmostEqual(cash, self.outputs["cash_at"][instrument].iloc[position - 1])

    def test_composites_rank_only_universe_members(self) -> None:
        changed = dict(self.inputs)
        changed["close"] = self.inputs["close"].copy()
        changed["close"].iloc[:400, 0] *= np.linspace(1.0, 3.0, 400)
        moved = calculate(changed)["mispricing_perf"]
        np.testing.assert_allclose(moved.iloc[300:400, 3].to_numpy(),
                                   self.outputs["mispricing_perf"].iloc[300:400, 3].to_numpy(), equal_nan=True)
        quality = self.outputs["qmj"].to_numpy()
        self.assertTrue(np.isfinite(quality).any())


class JKP153HelperTests(unittest.TestCase):
    def test_step_sums_add_every_step_th_row(self) -> None:
        values = np.arange(20.0)[:, None]
        values[7] = np.nan
        total, count = step_sums(values, 3, 4)
        self.assertEqual(19 + 16 + 13 + 10, total[19, 0])
        self.assertEqual(4, count[19, 0])
        self.assertEqual(16 + 13 + 10, total[16, 0])
        self.assertEqual(3, count[16, 0])

    def test_cross_section_rank_ranks_against_members(self) -> None:
        values = np.array([[1.0, 2.0, 3.0, 10.0, np.nan]])
        universe = np.array([[True, True, True, False, True]])
        np.testing.assert_allclose(cross_section_rank(values, universe, z_score=False)[0, :4],
                                   [1 / 3, 2 / 3, 1.0, 3.5 / 3])
        z = cross_section_rank(values, universe, z_score=True)[0]
        np.testing.assert_allclose(z[:3], [-1.0, 0.0, 1.0])

    def test_corwin_schultz_is_zero_for_flat_bars_and_positive_for_wide_ones(self) -> None:
        high = np.array([[10.0], [10.0], [10.5], [10.2]])
        low = np.array([[10.0], [10.0], [9.5], [9.7]])
        spread = corwin_schultz(high, low)
        self.assertTrue(np.isnan(spread[0, 0]))
        self.assertEqual(0.0, spread[1, 0])
        self.assertGreaterEqual(spread[3, 0], 0.0)


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class JKP153EngineTests(unittest.TestCase):
    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=320, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            dates = inputs["close"].index
            fields = ("high", "low", "close", "amount", "cap")
            bars = pd.concat([inputs[field].stack(future_stack=True).rename(field) for field in fields],
                             axis=1).reset_index()
            bars["in_csi300"] = inputs["universe"].stack(future_stack=True).to_numpy()
            for year_value, frame in bars.groupby(bars["datetime"].dt.year):
                path = canonical / "bars" / f"year={year_value}" / "bars.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-fixture",
                                                            "raw_fingerprint": "raw-fixture"})
            files = {}
            daily = pd.concat(
                [(inputs["basic_turnover_rate"].stack() * 100.0).rename("turnover_rate"),
                 inputs["basic_close"].stack(future_stack=True).rename("close"),
                 (inputs["basic_total_share"].stack() / 10_000.0).rename("total_share"),
                 (inputs["basic_dv_ttm"].stack(future_stack=True) * 100.0).rename("dv_ttm")],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                path = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(path)
            pd.DataFrame({"datetime": dates, "mkt": inputs["mkt"].to_numpy(), "smb": inputs["smb"].to_numpy(),
                          "hml": inputs["hml"].to_numpy(), "stocks": 200}).to_parquet(
                reference / "market_factors.parquet", index=False)
            files["market_factors.parquet"] = sha256_file(reference / "market_factors.parquet")
            inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            pd.DataFrame({"ts_code": [f"60000{i}.SH" for i in range(4)], "list_date": "20040601"}).to_parquet(
                reference / "stock_basic.parquet", index=False)
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="jkp153"))
            self.assertIsInstance(engine, KunQuantFactorEngine)
            _, _, module, statement_modules = compiled()
            engine.compile = lambda: module
            engine.compile_statements = lambda: statement_modules
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(144, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["jkp153_reference"])
            store = root / "factors" / "jkp153"
            stored = pd.concat([pd.read_parquet(path) for path in sorted(store.glob("year=*/*.parquet"))])
            stored = stored.set_index(["datetime", "instrument"])
            direct = ReferenceLibrary().calculate(inputs)
            for name in ("be_me", "ret_1_0", "cash_at", "prc"):
                canonical_name = f"jkp_{JKP_NAMES.index(name) + 1:03d}"
                expected = direct[name].stack(future_stack=True).reindex(stored.index)
                np.testing.assert_allclose(stored[canonical_name].to_numpy(), expected.to_numpy(), rtol=1e-9,
                                           equal_nan=True, err_msg=name)


if __name__ == "__main__":
    unittest.main()
