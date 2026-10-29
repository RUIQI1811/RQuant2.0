from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.data.reference import first_reported
from rquant.factors.catalog import get_catalog
from rquant.factors.engine import (
    FactorBuildConfig,
    KunQuantFactorEngine,
    cpp_compiler_available,
    create_factor_engine,
)
from rquant.factors.libraries.china_chars import CHINA_CHARS_NAMES, ChinaCharsLibrary
from rquant.factors.panel_operators import rolling_regression
from rquant.factors.statements import QuarterGrid
from rquant.io import atomic_write_json, sha256_file, stable_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_reference.china_chars import ChinaChars, calculate  # noqa: E402

KUNQUANT_READY = importlib.util.find_spec("KunQuant") is not None and cpp_compiler_available()

FLOWS = (
    "total_revenue",
    "revenue",
    "oper_cost",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "rd_exp",
    "operate_profit",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "n_cashflow_act",
    "c_pay_acq_const_fiolta",
    "n_incr_cash_cash_equ",
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
)
STOCKS = (
    "total_share",
    "money_cap",
    "trad_asset",
    "accounts_receiv",
    "inventories",
    "total_cur_assets",
    "lt_eqt_invest",
    "fix_assets",
    "intan_assets",
    "goodwill",
    "total_assets",
    "st_borr",
    "non_cur_liab_due_1y",
    "total_cur_liab",
    "lt_borr",
    "bond_payable",
    "total_liab",
    "minority_int",
    "oth_eqt_tools_p_shr",
    "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int",
)


def synthetic_inputs(days: int = 1400, stocks: int = 6, seed: int = 7) -> dict[str, object]:
    """Daily panels and quarterly statements shaped like the engine's china_chars inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    market = rng.normal(0.0003, 0.012, days)
    smb = rng.normal(0.0, 0.006, days)
    hml = rng.normal(0.0, 0.005, days)
    loadings = rng.uniform(0.6, 1.4, stocks)
    returns = market[:, None] * loadings[None, :] + 0.4 * smb[:, None] + rng.normal(0.0, 0.015, (days, stocks))
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=instruments)
    close.iloc[100:110, 1] = np.nan
    close.iloc[:40, 2] = np.nan
    shares = rng.uniform(1e8, 1e9, stocks)
    cap = close * shares
    amount = cap * rng.uniform(0.002, 0.03, (days, stocks))
    codes = [float(index % 3 + 1) for index in range(stocks - 1)] + [np.nan]
    sector = pd.DataFrame(np.tile(codes, (days, 1)), index=dates, columns=instruments)

    rows = []
    for column, instrument in enumerate(instruments):
        scale = shares[column] * 10.0
        for end in pd.date_range("2011-03-31", dates[-1], freq="QE"):
            quarter = (end.month - 1) // 3 + 1
            announce = end + pd.Timedelta(days=int(rng.integers(20, 110)))
            row = {"instrument": instrument, "end_date": end, "ann_date": announce}
            for field in FLOWS:
                row[field] = float(rng.uniform(0.02, 0.06) * scale * quarter)
            for field in STOCKS:
                row[field] = float(rng.uniform(0.2, 1.0) * scale)
            row["total_assets"] = 4.0 * scale * rng.uniform(0.9, 1.1)
            row["total_liab"] = 0.5 * row["total_assets"]
            row["total_hldr_eqy_inc_min_int"] = 0.5 * row["total_assets"]
            row["total_hldr_eqy_exc_min_int"] = 0.45 * row["total_assets"]
            rows.append(row)
    fundamentals = pd.DataFrame(rows)
    # One quarter is never reported and one stock has no R&D line.
    fundamentals = fundamentals.drop(index=7).reset_index(drop=True)
    fundamentals.loc[fundamentals["instrument"].eq(instruments[3]), "rd_exp"] = np.nan

    def market_series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=dates)

    return {
        "close": close,
        "amount": amount,
        "cap": cap,
        "sector": sector,
        "basic_turnover_rate": amount / cap,
        "basic_circ_mv": cap * 0.8,
        "basic_close": close / np.linspace(1.0, 1.2, days)[:, None],
        "mkt": market_series(market),
        "smb": market_series(smb),
        "hml": market_series(hml),
        "fundamentals": fundamentals,
        "list_date": pd.Series(pd.Timestamp("2005-06-01"), index=instruments),
    }


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


class ChinaCharsCatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        catalog = get_catalog()
        names = catalog.canonical_names("china_chars")
        self.assertEqual(109, len(names))
        self.assertEqual("cnc_001", names[0])
        self.assertEqual("cnc_109", names[-1])
        self.assertEqual(len(set(CHINA_CHARS_NAMES)), len(CHINA_CHARS_NAMES))
        self.assertEqual(2064, len(catalog.specs))
        for dropped in ("hn", "lfe", "adm", "gad", "idvq", "idsq", "tail", "mkt_beta"):
            self.assertNotIn(dropped, CHINA_CHARS_NAMES)

    def test_every_characteristic_has_a_reference_calculator(self) -> None:
        for name in CHINA_CHARS_NAMES:
            self.assertTrue(hasattr(ChinaChars, f"_char_{name}"), name)


def kunquant_arrays(engine: KunQuantFactorEngine, inputs: dict[str, object]) -> dict[str, np.ndarray]:
    """The daily graph's input arrays, with market series repeated for every instrument."""
    close = inputs["close"]
    arrays = {}
    for name in engine.input_fields:
        value = inputs[name]
        if isinstance(value, pd.Series):
            value = np.repeat(value.reindex(close.index).to_numpy(dtype=float)[:, None], close.shape[1], axis=1)
        arrays[name] = np.asarray(value, dtype=float)
    return arrays


@unittest.skipUnless(KUNQUANT_READY, "KunQuant and a C++ compiler are required")
class ChinaCharsCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Windows cannot delete a JIT-compiled DLL while this process still has it loaded.
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.engine = KunQuantFactorEngine(cls.temporary.name, FactorBuildConfig(factor_set="china_chars", workers=2))
        cls.module = cls.engine.compile()
        cls.statements = cls.engine.compile_statements()
        cls.inputs = synthetic_inputs()
        cls.outputs = cls.calculate(cls.inputs)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def calculate(cls, inputs: dict[str, object]) -> dict[str, pd.DataFrame]:
        """KunQuant outputs by source name, as panels shaped like the inputs."""
        close = inputs["close"]
        aligned = ChinaCharsLibrary().align_reference_inputs(inputs, close.index, close.columns)
        statements = cls.engine.run_statements(aligned, close.index, close.columns, cls.statements)
        aligned.update({name: statements.daily(name) for name in cls.engine.input_fields if name in statements})
        output = cls.engine.run(kunquant_arrays(cls.engine, aligned), module=cls.module, statements=statements)
        return {
            spec.source_name: pd.DataFrame(output[spec.canonical_name], index=close.index, columns=close.columns)
            for spec in ChinaCharsLibrary().specs
        }

    def test_every_characteristic_matches_the_pandas_reference(self) -> None:
        expected = calculate(self.inputs)
        for name in CHINA_CHARS_NAMES:
            np.testing.assert_allclose(self.outputs[name].to_numpy(), expected[name].to_numpy(), rtol=1e-9,
                                       atol=1e-13, equal_nan=True, err_msg=name)

    def test_all_characteristics_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in CHINA_CHARS_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            # Days without a quote never publish a value.
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)

    # Break caught: any characteristic that reads a later quote, market return or statement than the row it is on.
    def test_no_characteristic_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (700, 1300):
            last = dates[cut]
            past = self.calculate(truncated(self.inputs, last))
            for name in CHINA_CHARS_NAMES:
                expected = self.outputs[name].loc[:last].to_numpy()
                actual = past[name].to_numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)

    def test_selected_characteristics_match_direct_definitions(self) -> None:
        close = self.inputs["close"]
        row, column = 600, 0
        np.testing.assert_allclose(
            self.outputs["m1"].iloc[row, column], close.iloc[row, column] / close.iloc[row - 21, column] - 1.0
        )
        np.testing.assert_allclose(
            self.outputs["size"].iloc[row, column], np.log(self.inputs["basic_circ_mv"].iloc[row, column])
        )
        window = close.iloc[row - 239 : row + 1, column]
        np.testing.assert_allclose(self.outputs["w52"].iloc[row, column], close.iloc[row, column] / window.max())
        returns = (close / close.shift(1) - 1.0).iloc[row - 119 : row + 1, column]
        market = self.inputs["mkt"].iloc[row - 119 : row + 1]
        slope = np.polyfit(market.to_numpy(), returns.to_numpy(), 1)[0]
        np.testing.assert_allclose(self.outputs["beta"].iloc[row, column], slope, rtol=1e-8)
        listed = self.inputs["list_date"].iloc[column]
        self.assertAlmostEqual((close.index[row] - listed).days / 30.4375, self.outputs["age"].iloc[row, column])

    def test_quarterly_values_appear_on_the_announcement_day(self) -> None:
        fundamentals = self.inputs["fundamentals"]
        instrument = "SH600000"
        reports = fundamentals[fundamentals["instrument"].eq(instrument)].sort_values("end_date")
        dates = self.inputs["close"].index
        report = reports.iloc[14]
        previous = reports.iloc[13]
        position = int(np.searchsorted(dates.to_numpy(), report["ann_date"].to_numpy()))
        if position <= int(np.searchsorted(dates.to_numpy(), previous["ann_date"].to_numpy())):
            self.skipTest("fixture announced the two quarters out of order")
        cash_to_assets = self.outputs["cta"][instrument]
        self.assertAlmostEqual(report["money_cap"] / report["total_assets"], cash_to_assets.iloc[position])
        self.assertNotAlmostEqual(report["money_cap"] / report["total_assets"], cash_to_assets.iloc[position - 1])

    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=300, stocks=4)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            dates = inputs["close"].index
            records = []
            for field in ("close", "amount", "cap", "sector"):
                stacked = inputs[field].stack(future_stack=True).rename(field)
                records.append(stacked)
            bars = pd.concat(records, axis=1).reset_index()
            bars["in_csi300"] = True
            year = reference.parent / "bars" / "year=2014" / "bars.parquet"
            year.parent.mkdir(parents=True)
            bars.to_parquet(year, index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-fixture",
                                                            "raw_fingerprint": "raw-fixture"})
            files = {}
            daily = pd.concat(
                [
                    inputs["basic_turnover_rate"].stack().rename("turnover_rate") * 100.0,
                    (inputs["basic_circ_mv"].stack() / 10_000.0).rename("circ_mv"),
                    inputs["basic_close"].stack().rename("close"),
                ],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                path = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(path)
            pd.DataFrame(
                {"datetime": dates, "mkt": inputs["mkt"].to_numpy(), "smb": inputs["smb"].to_numpy(),
                 "hml": inputs["hml"].to_numpy(), "stocks": 200}
            ).to_parquet(reference / "market_factors.parquet", index=False)
            files["market_factors.parquet"] = sha256_file(reference / "market_factors.parquet")
            inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            pd.DataFrame(
                {"ts_code": [f"60000{i}.SH" for i in range(4)], "list_date": "20050601"}
            ).to_parquet(reference / "stock_basic.parquet", index=False)
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="china_chars"))
            self.assertIsInstance(engine, KunQuantFactorEngine)
            engine.compile = lambda: self.module  # type: ignore[method-assign]
            engine.compile_statements = lambda: self.statements  # type: ignore[method-assign]
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(109, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["china_chars_reference"])
            stored = pd.read_parquet(root / "factors" / "china_chars" / "year=2014" / "factors.parquet")
            self.assertEqual(["datetime", "instrument", *get_catalog().canonical_names("china_chars")],
                             list(stored.columns))
            direct = calculate(inputs)
            stored = stored.set_index(["datetime", "instrument"])
            for canonical_name, name in (("cnc_001", "size"), ("cnc_015", "idvc"), ("cnc_041", "roe"),
                                         ("cnc_061", "bm"), ("cnc_095", "age")):
                values = stored[canonical_name].unstack()
                expected = direct[name].loc[values.index, values.columns]
                np.testing.assert_allclose(values.to_numpy(), expected.to_numpy(), rtol=1e-9, equal_nan=True,
                                           err_msg=name)

            with (reference / "manifest.json").open("r", encoding="utf-8") as handle:
                tampered = json.load(handle)
            tampered["files"]["fundamentals.parquet"] = "0" * 64
            tampered.pop("fingerprint")
            tampered["fingerprint"] = stable_hash(tampered)
            atomic_write_json(reference / "manifest.json", tampered)
            with self.assertRaisesRegex(Exception, "differs from its manifest"):
                engine.build_from_canonical(canonical, root / "factors")


class QuarterGridTests(unittest.TestCase):
    def test_year_to_date_flows_become_single_quarters_and_wait_for_earlier_quarters(self) -> None:
        dates = pd.bdate_range("2020-01-01", "2021-06-30")
        instruments = pd.Index(["SZ000001"])
        frame = pd.DataFrame(
            {
                "instrument": "SZ000001",
                "end_date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]),
                # Q2 is announced after Q3, so Q3 must wait for it.
                "ann_date": pd.to_datetime(["2020-04-20", "2020-11-02", "2020-10-20", "2021-03-30"]),
                "revenue": [10.0, 25.0, 45.0, 70.0],
            }
        )
        grid = QuarterGrid(frame, dates, instruments)
        np.testing.assert_allclose(grid.single("revenue")[0], [10.0, 15.0, 20.0, 25.0])
        anchor = pd.Series(grid.anchor[:, 0], index=dates)
        self.assertEqual(0, anchor.loc["2020-10-20"])
        self.assertEqual(2, anchor.loc["2020-11-02"])
        self.assertEqual(-1, grid.annual_anchor[dates.get_loc(pd.Timestamp("2021-03-29")), 0])
        self.assertEqual(3, grid.annual_anchor[dates.get_loc(pd.Timestamp("2021-03-30")), 0])

    def test_first_reported_version_wins(self) -> None:
        raw = pd.DataFrame(
            {
                "ts_code": ["600000.SH"] * 4,
                "ann_date": ["20200425", "20200425", "20200425", "20200425"],
                "f_ann_date": ["20200425", "20200425", "20200801", "20200425"],
                "end_date": ["20200331", "20200331", "20200331", "20200401"],
                "report_type": ["1", "1", "1", "1"],
                "update_flag": ["0", "1", "1", "0"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = first_reported(raw, ["revenue"], pd, np)
        self.assertEqual(["20200331"], result["end_date"].tolist())
        self.assertEqual([2.0], result["revenue"].tolist())
        self.assertEqual(["20200425"], result["available"].tolist())


class RollingRegressionTests(unittest.TestCase):
    # Break caught: prefix-sum moments that drift from explicitly computed window residuals.
    def test_prefix_sums_match_explicit_window_residuals(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(0.0, 0.01, size=(400, 3))
        y = 0.0004 + x @ np.array([1.1, 0.3, -0.2]) + rng.standard_t(4, size=(400, 1)) * 0.01
        y = np.column_stack([y[:, 0], y[:, 0] * 0.5 + rng.normal(0.0, 0.01, 400)])
        y[rng.random(y.shape) < 0.05] = np.nan
        x[[17, 230], 2] = np.nan
        slices = {"long": (60, 140), "short": (100, 140)}
        fast = rolling_regression(y, x, 150, 90, residual_slices=slices)
        slow = direct_regression(y, x, 150, 90, residual_slices=slices)
        for name in ("beta", "slopes", "resid_std", "resid_skew", "coskew", *slices):
            np.testing.assert_allclose(fast[name], slow[name], rtol=1e-7, atol=1e-10, equal_nan=True, err_msg=name)

    def test_matches_least_squares_with_missing_rows(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 2))
        y = (0.5 + x @ np.array([1.2, -0.7]))[:, None] + rng.normal(scale=0.1, size=(60, 1))
        y[[5, 33], 0] = np.nan
        x[40, 1] = np.nan
        fit = rolling_regression(y, x, 30, 20)
        rows = np.arange(30, 60)
        keep = np.isfinite(y[rows, 0]) & np.isfinite(x[rows]).all(axis=1)
        design = np.column_stack([np.ones(keep.sum()), x[rows][keep]])
        coefficients, *_ = np.linalg.lstsq(design, y[rows, 0][keep], rcond=None)
        residuals = y[rows, 0][keep] - design @ coefficients
        self.assertAlmostEqual(coefficients[1], fit["beta"][59, 0])
        self.assertAlmostEqual(coefficients[1:].sum(), fit["slopes"][59, 0])
        self.assertAlmostEqual(np.sqrt((residuals**2).sum() / (keep.sum() - 3)), fit["resid_std"][59, 0])
        self.assertTrue(np.isnan(fit["beta"][28, 0]))


def direct_regression(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
    minimum: int,
    *,
    residual_slices: dict[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Window-by-window reference for ``rolling_regression``: explicit residuals for every window."""
    dates, stocks = y.shape
    regressors = x.shape[1] + 1
    residual_slices = dict(residual_slices or {})
    names = ("beta", "slopes", "resid_std", "resid_skew", "coskew", *residual_slices)
    out = {name: np.full((dates, stocks), np.nan) for name in names}
    if dates < window:
        return out
    design = np.column_stack([np.ones(dates), x])
    design_ok = np.isfinite(design).all(axis=1)
    design = np.where(design_ok[:, None], design, 0.0)
    windows = np.lib.stride_tricks.sliding_window_view(design, window, axis=0)  # (n, k, w)
    design_mask = np.lib.stride_tricks.sliding_window_view(design_ok, window)
    market = windows[:, 1, :]
    identity = np.eye(regressors)
    for column in range(stocks):
        series = y[:, column]
        observed = np.isfinite(series)
        if observed.sum() < minimum:
            continue
        values = np.lib.stride_tricks.sliding_window_view(np.where(observed, series, 0.0), window)
        mask = np.lib.stride_tricks.sliding_window_view(observed, window) & design_mask
        count = mask.sum(axis=1)
        valid = count >= minimum
        if not valid.any():
            continue
        weighted = windows * mask[:, None, :]
        gram = weighted @ windows.transpose(0, 2, 1)
        gram[~valid] = identity
        moment = (weighted * values[:, None, :]).sum(axis=2)
        try:
            coefficients = np.linalg.solve(gram, moment[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = (np.linalg.pinv(gram) @ moment[..., None])[..., 0]
        residuals = (values - (windows * coefficients[:, :, None]).sum(axis=1)) * mask
        with np.errstate(invalid="ignore", divide="ignore"):
            squared = (residuals**2).sum(axis=1)
            variance = squared / count
            dof = np.where(count > regressors, count - regressors, np.nan)
            resid_std = np.sqrt(squared / dof)
            resid_skew = ((residuals**3).sum(axis=1) / count) / variance**1.5
            centered = (market - (market * mask).sum(axis=1, keepdims=True) / count[:, None]) * mask
            coskew = ((residuals * centered**2).sum(axis=1) / count) / (
                np.sqrt(variance) * ((centered**2).sum(axis=1) / count)
            )
            slopes = coefficients[:, 1:].sum(axis=1)
        rows = slice(window - 1, dates)
        keep = valid & (variance > 0)
        out["beta"][rows, column] = np.where(valid, coefficients[:, 1], np.nan)
        out["slopes"][rows, column] = np.where(valid, slopes, np.nan)
        out["resid_std"][rows, column] = np.where(keep, resid_std, np.nan)
        out["resid_skew"][rows, column] = np.where(keep, resid_skew, np.nan)
        out["coskew"][rows, column] = np.where(keep, coskew, np.nan)
        for name, (first, last) in residual_slices.items():
            part = residuals[:, first:last]
            part_mask = mask[:, first:last]
            part_count = part_mask.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = part.sum(axis=1) / part_count
                spread = np.sqrt((((part - mean[:, None]) * part_mask) ** 2).sum(axis=1) / (part_count - 1))
                momentum = mean / spread
            enough = valid & (part_count >= (last - first) // 2) & (spread > 0)
            out[name][rows, column] = np.where(enough, momentum, np.nan)
    return out


if __name__ == "__main__":
    unittest.main()
