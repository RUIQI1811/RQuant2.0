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
from rquant.factors.libraries.barra_cne6 import CNE6_NAMES, BarraCNE6Library
from rquant.io import atomic_write_json, sha256_file, stable_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_reference.barra_cne6 import BarraCNE6Library as ReferenceLibrary  # noqa: E402
from panel_reference.barra_cne6 import (  # noqa: E402
    CNE6Descriptors,
    ew_market_fit,
    ew_window_sums,
    five_year_slope,
)

KUNQUANT_READY = importlib.util.find_spec("KunQuant") is not None and cpp_compiler_available()


@lru_cache(maxsize=1)
def compiled():
    # Windows cannot delete a JIT-compiled DLL while this process still has it loaded.
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    engine = KunQuantFactorEngine(temporary.name, FactorBuildConfig(factor_set="barra_cne6", workers=2))
    return temporary, engine, engine.compile(), engine.compile_statements()


def calculate(inputs: dict[str, object]) -> dict[str, pd.DataFrame]:
    """KunQuant outputs by source name, as panels shaped like the inputs."""
    _, engine, module, statement_modules = compiled()
    close = inputs["close"]
    statements = engine.run_statements(inputs, close.index, close.columns, statement_modules)
    arrays = {}
    for name in engine.input_fields:
        value = statements.daily(name) if name in statements else inputs[name]
        if isinstance(value, pd.Series):
            value = np.repeat(value.to_numpy()[:, None], len(close.columns), axis=1)
        arrays[name] = np.ascontiguousarray(value, dtype=float)
    output = engine.run(arrays, module=module, statements=statements)
    return {
        spec.source_name: pd.DataFrame(output[spec.canonical_name], index=close.index, columns=close.columns)
        for spec in BarraCNE6Library().specs
    }

FLOWS = ("revenue", "oper_cost", "fin_exp", "total_profit", "n_income_attr_p", "n_cashflow_act", "n_cashflow_inv_act",
         "c_pay_acq_const_fiolta", "n_incr_cash_cash_equ", "depr_fa_coga_dpba", "amort_intang_assets",
         "lt_amort_deferred_exp")
STOCKS = ("total_share", "money_cap", "st_borr", "non_cur_liab_due_1y", "lt_borr", "bond_payable",
          "oth_eqt_tools_p_shr", "total_assets", "total_liab", "total_hldr_eqy_exc_min_int")


def synthetic_inputs(days: int = 1600, stocks: int = 6, seed: int = 5) -> dict[str, object]:
    """Daily panels and quarterly statements shaped like the engine's barra_cne6 inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    market = rng.normal(0.0003, 0.012, days)
    loadings = rng.uniform(0.6, 1.4, stocks)
    returns = market[:, None] * loadings[None, :] + rng.normal(0.0, 0.015, (days, stocks))
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=instruments)
    close.iloc[300:312, 1] = np.nan
    close.iloc[:50, 2] = np.nan
    shares = rng.uniform(1e8, 1e9, stocks)
    cap = close * shares
    codes = [float(index % 2 + 1) for index in range(stocks - 1)] + [np.nan]
    sector = pd.DataFrame(np.tile(codes, (days, 1)), index=dates, columns=instruments)
    turnover = pd.DataFrame(rng.uniform(0.002, 0.04, (days, stocks)), index=dates, columns=instruments)
    dividend = pd.DataFrame(rng.uniform(0.0, 0.03, (days, stocks)), index=dates, columns=instruments)
    dividend.iloc[:, stocks - 2] = np.nan
    universe = pd.DataFrame(True, index=dates, columns=instruments)
    universe.iloc[:400, 0] = False

    rows = []
    for column, instrument in enumerate(instruments):
        scale = shares[column] * 10.0
        for end in pd.date_range("2006-03-31", dates[-1], freq="QE"):
            quarter = (end.month - 1) // 3 + 1
            row = {"instrument": instrument, "end_date": end,
                   "ann_date": end + pd.Timedelta(days=int(rng.integers(20, 110)))}
            for field in FLOWS:
                row[field] = float(rng.uniform(0.02, 0.06) * scale * quarter)
            for field in STOCKS:
                row[field] = float(rng.uniform(0.2, 1.0) * scale)
            row["total_share"] = shares[column] * rng.uniform(0.95, 1.05)
            row["total_assets"] = 4.0 * scale * rng.uniform(0.9, 1.1)
            row["total_liab"] = 0.5 * row["total_assets"]
            row["total_hldr_eqy_exc_min_int"] = 0.45 * row["total_assets"]
            rows.append(row)
    fundamentals = pd.DataFrame(rows).drop(index=9).reset_index(drop=True)
    return {
        "close": close,
        "cap": cap,
        "sector": sector,
        "basic_turnover_rate": turnover,
        "basic_dv_ttm": dividend,
        "mkt": pd.Series(market, index=dates),
        "fundamentals": fundamentals,
        "universe": universe,
    }


def truncated(inputs: dict[str, object], last: pd.Timestamp) -> dict[str, object]:
    """Return what was knowable at the close of ``last``."""
    result: dict[str, object] = {}
    for key, value in inputs.items():
        result[key] = value[value["ann_date"] <= last] if key == "fundamentals" else value.loc[:last]
    return result


def weighted_fit(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    keep = np.isfinite(y) & np.isfinite(x)
    design = np.column_stack([np.ones(keep.sum()), x[keep]])
    root = np.sqrt(weights[keep])
    (alpha, beta), *_ = np.linalg.lstsq(design * root[:, None], y[keep] * root, rcond=None)
    residual = y[keep] - alpha - beta * x[keep]
    return alpha, beta, float(np.sqrt(np.sum(weights[keep] * residual**2) / weights[keep].sum()))


class BarraCNE6CatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        names = get_catalog().canonical_names("barra_cne6")
        self.assertEqual(38, len(names))
        self.assertEqual(("cne6_001", "cne6_038"), (names[0], names[-1]))
        self.assertEqual(len(set(CNE6_NAMES)), len(CNE6_NAMES))
        for dropped in ("epibs", "egrlf", "etopf_std", "dpibs"):
            self.assertNotIn(dropped, CNE6_NAMES)

    def test_every_descriptor_has_a_calculator(self) -> None:
        for name in CNE6_NAMES:
            self.assertTrue(hasattr(CNE6Descriptors, f"_d_{name}"), name)


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class BarraCNE6CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.outputs = calculate(cls.inputs)

    # Break caught: a KunQuant formula, window or cross-sectional step that drifts from the pandas reference.
    def test_every_descriptor_matches_the_pandas_reference(self) -> None:
        expected = ReferenceLibrary().calculate(self.inputs)
        for name in CNE6_NAMES:
            np.testing.assert_allclose(self.outputs[name].to_numpy(), expected[name].to_numpy(), rtol=1e-7,
                                       atol=1e-10, equal_nan=True, err_msg=name)

    def test_all_descriptors_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in CNE6_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)

    # Break caught: any descriptor that reads a later quote, market return, statement or index membership.
    def test_no_descriptor_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (900, 1450):
            last = dates[cut]
            past = calculate(truncated(self.inputs, last))
            for name in CNE6_NAMES:
                expected = self.outputs[name].loc[:last].to_numpy()
                np.testing.assert_allclose(past[name].to_numpy(), expected, rtol=1e-8, atol=1e-11, equal_nan=True,
                                           err_msg=name)

    def test_descriptors_match_direct_definitions(self) -> None:
        close = self.inputs["close"]
        cap = self.inputs["cap"]
        turnover = self.inputs["basic_turnover_rate"]
        row, column = 1500, 0
        returns = (close / close.shift(1) - 1.0).to_numpy()
        market = self.inputs["mkt"].to_numpy()
        np.testing.assert_allclose(self.outputs["lncap"].iloc[row, column], np.log(cap.iloc[row, column]))
        weights = 0.5 ** (np.arange(251, -1, -1) / 63.0)
        window = slice(row - 251, row + 1)
        _, beta, sigma = weighted_fit(returns[window, column], market[window], weights)
        np.testing.assert_allclose(self.outputs["hbeta"].iloc[row, column], beta, rtol=1e-7)
        np.testing.assert_allclose(self.outputs["hsigma"].iloc[row, column], sigma, rtol=1e-6)
        np.testing.assert_allclose(
            self.outputs["stom"].iloc[row, column], np.log(turnover.iloc[row - 20 : row + 1, column].sum())
        )
        np.testing.assert_allclose(
            self.outputs["stoa"].iloc[row, column], np.log(turnover.iloc[row - 251 : row + 1, column].sum() / 12.0)
        )
        log_returns = np.log1p(returns[:, column])
        raw = [np.sum(0.5 ** (np.arange(251, -1, -1) / 126.0) * log_returns[end - 251 : end + 1])
               for end in range(row - 21, row - 10)]
        np.testing.assert_allclose(self.outputs["rstr"].iloc[row, column], np.mean(raw), rtol=1e-8)
        cumulative = [log_returns[row - 21 * months + 1 : row + 1].sum() for months in range(1, 13)]
        np.testing.assert_allclose(
            self.outputs["cmra"].iloc[row, column], np.log1p(max(cumulative)) - np.log1p(min(cumulative))
        )
        price = close.ffill()
        season = np.mean([price.iloc[row - 252 * k + 21, column] / price.iloc[row - 252 * k, column] - 1.0
                          for k in range(1, 6)])
        np.testing.assert_allclose(self.outputs["season"].iloc[row, column], season)

    def test_industry_momentum_uses_only_universe_members(self) -> None:
        indmom = self.outputs["indmom"]
        # Stock 5 has no industry code; stock 0 is outside the universe for the first 400 rows.
        self.assertTrue(indmom.iloc[:, 5].isna().all())
        self.assertTrue(np.isfinite(indmom.iloc[350, 0]))
        changed = self.inputs.copy()
        changed["cap"] = self.inputs["cap"].copy()
        changed["cap"].iloc[:400, 0] *= 50.0
        moved = calculate(changed)["indmom"]
        np.testing.assert_allclose(moved.iloc[200:400, 2].to_numpy(), indmom.iloc[200:400, 2].to_numpy(),
                                   equal_nan=True)

    def test_midcap_is_standardised_over_members(self) -> None:
        midcap = self.outputs["midcap"].to_numpy()
        members = self.inputs["universe"].to_numpy() & np.isfinite(midcap)
        row = 1000
        self.assertAlmostEqual(1.0, midcap[row, members[row]].std(ddof=1), places=8)

    def test_dividend_yield_is_zero_for_non_payers(self) -> None:
        self.assertTrue(self.outputs["dtop"].iloc[:, 4].dropna().eq(0.0).all())
        self.assertGreater(self.outputs["dtop"].iloc[:, 4].notna().sum(), 1000)


class BarraCNE6HelperTests(unittest.TestCase):
    # Break caught: a decaying recursion that drifts from explicitly weighted window sums.
    def test_window_sums_match_explicit_weights(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=(300, 3))
        fast = ew_window_sums({"v": values}, 40, 9.0)["v"]
        weights = 0.5 ** (np.arange(39, -1, -1) / 9.0)
        for end in (39, 150, 299):
            np.testing.assert_allclose(fast[end], weights @ values[end - 39 : end + 1], rtol=1e-10)
        np.testing.assert_allclose(fast[10], (0.5 ** (np.arange(10, -1, -1) / 9.0)) @ values[:11], rtol=1e-10)

    def test_market_fit_matches_weighted_least_squares_with_missing_rows(self) -> None:
        rng = np.random.default_rng(4)
        market = rng.normal(0.0, 0.01, 200)
        y = (0.001 + 1.3 * market)[:, None] + rng.normal(0.0, 0.005, (200, 1))
        y[[50, 120], 0] = np.nan
        market[90] = np.nan
        fit = ew_market_fit(y, market, 100, 25, 60)
        weights = 0.5 ** (np.arange(99, -1, -1) / 25.0)
        alpha, beta, sigma = weighted_fit(y[100:200, 0], market[100:200], weights)
        self.assertAlmostEqual(alpha, fit["alpha"][199, 0])
        self.assertAlmostEqual(beta, fit["beta"][199, 0])
        self.assertAlmostEqual(sigma, fit["resid_std"][199, 0])
        self.assertTrue(np.isnan(fit["beta"][40, 0]))

    def test_five_year_slope_uses_available_years(self) -> None:
        values = np.full((1, 20), np.nan)
        values[0, [3, 7, 11, 15, 19]] = [10.0, 12.0, np.nan, 16.0, 18.0]
        slope = five_year_slope(values)
        time = np.array([1.0, 2.0, 4.0, 5.0])
        expected = np.polyfit(time, [10.0, 12.0, 16.0, 18.0], 1)[0] / 14.0
        self.assertAlmostEqual(expected, slope[0, 19])
        self.assertTrue(np.isnan(slope[0, 11]))


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class BarraCNE6EngineTests(unittest.TestCase):
    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=300, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            dates = inputs["close"].index
            bars = pd.concat(
                [inputs[field].stack(future_stack=True).rename(field) for field in ("close", "cap", "sector")],
                axis=1,
            ).reset_index()
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
                 (inputs["basic_dv_ttm"].stack(future_stack=True) * 100.0).rename("dv_ttm")],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                path = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(path)
            pd.DataFrame({"datetime": dates, "mkt": inputs["mkt"].to_numpy(), "smb": 0.0, "hml": 0.0,
                          "stocks": 200}).to_parquet(reference / "market_factors.parquet", index=False)
            files["market_factors.parquet"] = sha256_file(reference / "market_factors.parquet")
            inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="barra_cne6"))
            self.assertIsInstance(engine, KunQuantFactorEngine)
            _, _, module, statement_modules = compiled()
            engine.compile = lambda: module
            engine.compile_statements = lambda: statement_modules
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(38, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["barra_cne6_reference"])
            store = root / "factors" / "barra_cne6"
            stored = pd.concat([pd.read_parquet(path) for path in sorted(store.glob("year=*/*.parquet"))])
            stored = stored.set_index(["datetime", "instrument"])
            direct = ReferenceLibrary().calculate(inputs)
            for canonical_name, name in (("cne6_001", "lncap"), ("cne6_003", "hbeta"), ("cne6_017", "btop"),
                                         ("cne6_038", "dtop")):
                expected = direct[name].stack(future_stack=True).reindex(stored.index)
                np.testing.assert_allclose(stored[canonical_name].to_numpy(), expected.to_numpy(), rtol=1e-9,
                                           equal_nan=True, err_msg=name)


if __name__ == "__main__":
    unittest.main()
