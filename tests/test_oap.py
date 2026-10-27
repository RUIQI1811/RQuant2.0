from __future__ import annotations

import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS, STATEMENT_KEY_FIELDS
from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, KunQuantFactorEngine, create_factor_engine
from rquant.factors.libraries.oap import (
    OAP,
    OAP_NAMES,
    OAP_SKIPPED,
    REQUIRED_LINES,
    OAPLibrary,
)
from rquant.io import atomic_write_json, sha256_file, stable_hash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_reference.oap import (  # noqa: E402
    OAPLibrary as ReferenceLibrary,
)
from panel_reference.oap import (
    corwin_schultz,
    cross_residual,
    group_stat,
    monthly,
    prepare_fundamentals,
    price_delay,
    rolling_coskew,
)


@lru_cache(maxsize=1)
def compiled():
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    engine = KunQuantFactorEngine(temporary.name, FactorBuildConfig(factor_set="oap", workers=2))
    return temporary, engine, engine.compile(), engine.compile_statements()


def calculate(inputs):
    _, engine, module, statement_modules = compiled()
    close = inputs["close"]
    prepared = OAPLibrary().align_reference_inputs(inputs, close.index, close.columns)
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
        for spec in OAPLibrary().specs
    }


BALANCE = tuple(f for f in FUNDAMENTAL_ENDPOINT_FIELDS["balancesheet"] if f not in STATEMENT_KEY_FIELDS)
FLOWS = tuple(
    f
    for endpoint in ("income", "cashflow")
    for f in FUNDAMENTAL_ENDPOINT_FIELDS[endpoint]
    if f not in STATEMENT_KEY_FIELDS and f != "c_cash_equ_end_period"
)


def synthetic_inputs(days: int = 2200, stocks: int = 40, seed: int = 5) -> dict[str, object]:
    """Daily panels and quarterly statements shaped like the engine's oap inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    market = rng.normal(0.0003, 0.012, days)
    smb = rng.normal(0.0, 0.006, days)
    hml = rng.normal(0.0, 0.005, days)
    loadings = rng.uniform(0.6, 1.4, stocks)
    returns = market[:, None] * loadings[None, :] + rng.normal(0.0, 0.015, (days, stocks))
    close = 10.0 * np.cumprod(1.0 + returns, axis=0)
    open_ = close * np.exp(rng.normal(0.0, 0.008, (days, stocks)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    volume = rng.lognormal(15.0, 0.4, (days, stocks))
    panels = {
        field: pd.DataFrame(values, index=dates, columns=instruments)
        for field, values in (("high", high), ("low", low), ("close", close), ("volume", volume))
    }
    for panel in panels.values():
        panel.iloc[300:315, 1] = np.nan
        panel.iloc[:60, 2] = np.nan
    shares = rng.uniform(1e8, 1e9, stocks)
    cap = panels["close"] * shares
    universe = pd.DataFrame(True, index=dates, columns=instruments)
    universe.iloc[:500, stocks - 1] = False
    sector = pd.DataFrame(np.tile(np.arange(stocks) % 4 + 1.0, (days, 1)), index=dates, columns=instruments)
    # Trailing-twelve-month dividend per share: every other stock pays each June 15 and stock 0 stops after 2016.
    dps = pd.DataFrame(np.nan, index=dates, columns=instruments)
    for column in range(0, stocks, 2):
        first, last = (2015 if column == 4 else 2012), (2016 if column == 0 else 2019)
        for year in range(first, last + 1):
            paid = (dates >= f"{year}-06-15") & (dates < f"{year + 1}-06-15")
            dps.iloc[paid, column] = 0.2 + 0.05 * (year - 2012)
            # Tushare double-counts the new payment for a few days after its ex-date.
            spike = (dates >= f"{year}-06-15") & (dates < f"{year}-06-22")
            dps.iloc[spike, column] = 2.0 * (0.2 + 0.05 * (year - 2012))
    raw_close = panels["close"] / np.linspace(1.0, 1.3, days)[:, None]

    rows = []
    for column, instrument in enumerate(instruments):
        scale = shares[column] * 10.0
        for end in pd.date_range("2005-03-31", dates[-1], freq="QE"):
            quarter = (end.month - 1) // 3 + 1
            announce = end + pd.Timedelta(days=int(rng.integers(20, 110)))
            row = {"instrument": instrument, "end_date": end, "ann_date": announce}
            for field in FLOWS:
                row[field] = float(rng.uniform(0.02, 0.06) * scale * quarter)
            for field in BALANCE:
                row[field] = float(rng.uniform(0.02, 0.2) * scale)
            row["total_assets"] = 4.0 * scale * rng.uniform(0.9, 1.1)
            row["total_cur_assets"] = 0.4 * row["total_assets"]
            row["total_nca"] = 0.6 * row["total_assets"]
            row["total_liab"] = 0.5 * row["total_assets"]
            row["total_cur_liab"] = 0.3 * row["total_assets"]
            row["total_ncl"] = 0.2 * row["total_assets"]
            row["total_hldr_eqy_inc_min_int"] = 0.5 * row["total_assets"]
            row["total_hldr_eqy_exc_min_int"] = 0.45 * row["total_assets"]
            row["revenue"] = float(rng.uniform(0.3, 0.5) * scale * quarter)
            row["n_income_attr_p"] = float(rng.uniform(-0.01, 0.06) * scale * quarter)
            if end.year < 2018 or column == 5:
                row["rd_exp"] = np.nan
            rows.append(row)
    fundamentals = pd.DataFrame(rows)
    fundamentals = fundamentals.drop(index=9).reset_index(drop=True)

    list_date = pd.Series(dates[0] - pd.to_timedelta(rng.integers(100, 4000, stocks), unit="D"), index=instruments)
    list_date.iloc[3] = dates[min(400, days - 1)]
    list_date.iloc[5] = dates[min(1800, days - 1)]
    return {
        **panels,
        "amount": panels["close"] * panels["volume"],
        "cap": cap,
        "sector": sector,
        "basic_close": raw_close,
        "basic_turnover_rate": panels["volume"]
        / pd.DataFrame(np.tile(shares * 0.8, (days, 1)), index=dates, columns=instruments),
        "basic_total_share": pd.DataFrame(np.tile(shares, (days, 1)), index=dates, columns=instruments),
        "basic_dv_ttm": dps / raw_close,
        "mkt": pd.Series(market, index=dates),
        "smb": pd.Series(smb, index=dates),
        "hml": pd.Series(hml, index=dates),
        "fundamentals": fundamentals,
        "list_date": list_date,
        "universe": universe,
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


class OAPCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_oap_predictor(self) -> None:
        names = get_catalog().canonical_names("oap")
        self.assertEqual(len(OAP), len(names))
        self.assertEqual(("oap_001", f"oap_{len(OAP):03d}"), (names[0], names[-1]))
        self.assertEqual(len(set(OAP_NAMES)), len(OAP_NAMES))
        # Implemented and skipped signals partition the 212 OAP predictors.
        self.assertFalse(set(OAP_NAMES) & set(OAP_SKIPPED))
        self.assertEqual(212, len(OAP_NAMES) + len(OAP_SKIPPED))
        self.assertEqual(142, len(OAP_NAMES))

    def test_statement_lines_are_synced(self) -> None:
        synced = {field for fields in FUNDAMENTAL_ENDPOINT_FIELDS.values() for field in fields}
        self.assertFalse(set(REQUIRED_LINES) - synced)


class OAPCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.outputs = calculate(cls.inputs)

    def test_every_signal_matches_independent_reference(self) -> None:
        reference = ReferenceLibrary().calculate(self.inputs)
        for name in OAP_NAMES:
            np.testing.assert_allclose(
                self.outputs[name].to_numpy(),
                reference[name].to_numpy(),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
                err_msg=name,
            )

    def test_all_signals_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in OAP_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            # Days without a quote never publish a value.
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)

    # Break caught: any signal that reads a later quote, factor return, statement or index membership.
    def test_no_signal_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (900, 1700):
            last = dates[cut]
            past = calculate(truncated(self.inputs, last))
            for name in OAP_NAMES:
                expected = self.outputs[name].loc[:last].to_numpy()
                actual = past[name].to_numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=name)

    def test_price_signals_match_direct_definitions(self) -> None:
        close = self.inputs["close"]
        row, column = 1200, 5
        price = close.iloc[:, column]
        mom6 = price.iloc[row - 21] / price.iloc[row - 126] - 1.0
        self.assertAlmostEqual(mom6, self.outputs["Mom6m"].iloc[row, column])
        self.assertAlmostEqual(
            price.iloc[row - 252] / price.iloc[row - 378] - 1.0, self.outputs["MRreversal"].iloc[row, column]
        )
        returns = price.pct_change()
        self.assertAlmostEqual(returns.iloc[row - 20 : row + 1].max(), self.outputs["MaxRet"].iloc[row, column])
        monthly_returns = price / price.shift(21) - 1.0
        season = np.mean([monthly_returns.iloc[row - 21 * (j - 1)] for j in (24, 36, 48)])
        self.assertAlmostEqual(season, self.outputs["MomSeason"].iloc[row, column])
        amount = self.inputs["amount"].iloc[:, column]
        self.assertAlmostEqual(np.log(amount.iloc[row - 41 : row - 20].sum()), self.outputs["DolVol"].iloc[row, column])

    def test_beta_uses_non_overlapping_monthly_returns(self) -> None:
        row, column = 1600, 7
        price = self.inputs["close"].iloc[:, column].to_numpy()
        market = pd.Series(self.inputs["mkt"]).rolling(21, min_periods=15).sum().to_numpy()
        months = np.arange(row, row - 21 * 60, -21)[::-1]
        months = months[months >= 21]
        stock = price[months] / price[months - 21] - 1.0
        slope = np.polyfit(market[months], stock, 1)[0]
        self.assertAlmostEqual(slope, self.outputs["Beta"].iloc[row, column], places=8)

    def test_statement_signals_appear_on_the_announcement_day(self) -> None:
        fundamentals = self.inputs["fundamentals"]
        instrument = "SH600004"
        reports = fundamentals[fundamentals["instrument"].eq(instrument)].sort_values("end_date")
        dates = self.inputs["close"].index
        for position in range(40, 50):
            report, previous = reports.iloc[position], reports.iloc[position - 1]
            row = int(np.searchsorted(dates.to_numpy(), report["ann_date"].to_numpy()))
            if 0 < row < len(dates) and row > int(np.searchsorted(dates.to_numpy(), previous["ann_date"].to_numpy())):
                break
        else:
            self.skipTest("fixture announced every quarter out of order")
        cash = (report["money_cap"] + report["trad_asset"]) / report["total_assets"]
        self.assertAlmostEqual(cash, self.outputs["Cash"][instrument].iloc[row])
        self.assertNotAlmostEqual(cash, self.outputs["Cash"][instrument].iloc[row - 1])
        leverage = report["total_liab"] / self.inputs["cap"][instrument].iloc[row]
        self.assertAlmostEqual(leverage, self.outputs["Leverage"][instrument].iloc[row])

    def test_dividend_events(self) -> None:
        dates = self.inputs["close"].index

        def row(day: str) -> int:
            return int(np.searchsorted(dates, pd.Timestamp(day)))

        # Payers have no year of history before mid-2013, so a 2012 initiation is unknown; stock 4 starts in 2015.
        self.assertTrue(np.isnan(self.outputs["DivInit"].iloc[row("2012-06-15") + 5, 2]))
        self.assertEqual(1.0, self.outputs["DivInit"].iloc[row("2015-06-15") + 20, 4])
        self.assertEqual(0.0, self.outputs["DivInit"].iloc[row("2015-06-15") + 20, 2])
        self.assertEqual(0.0, self.outputs["DivInit"].iloc[row("2016-06-15") + 20, 4])
        self.assertEqual(1.0, self.outputs["DivOmit"].iloc[row("2017-06-15") + 5, 0])
        self.assertEqual(0.0, self.outputs["DivOmit"].iloc[row("2017-06-15") + 5, 2])
        # The trailing amount ignores the ex-date double count and moves once, 15 rows after the payment.
        paid = row("2015-06-15") + 14
        self.assertEqual(1.0, self.outputs["DivSeason"].iloc[paid + 240, 2])
        self.assertEqual(0.0, self.outputs["DivSeason"].iloc[paid + 100, 2])
        self.assertTrue(np.isnan(self.outputs["DivSeason"].iloc[paid + 240, 1]))
        self.assertEqual(3.0, self.outputs["DivYieldST"].iloc[paid + 240, 2])
        self.assertEqual(0.0, self.outputs["DivYieldST"].iloc[paid + 100, 2])

    def test_cross_sectional_flags_use_index_members_only(self) -> None:
        for name in ("MomRev", "AccrualsBM", "ShareVol", "IndRetBig", "AbnormalAccruals"):
            self.assertTrue(self.outputs[name].iloc[:500, -1].isna().all(), name)
        values = self.outputs["MomRev"].to_numpy()
        self.assertTrue(set(np.unique(values[np.isfinite(values)])) <= {0.0, 1.0})
        self.assertEqual(1.0, self.outputs["IndIPO"].iloc[600, 3])
        self.assertEqual(0.0, self.outputs["IndIPO"].iloc[2000, 3])
        # A blank R&D line means no R&D only once R&D is its own line (2018); stock 5 lists in late 2018 without R&D.
        self.assertTrue(np.isnan(self.outputs["RDIPO"].iloc[600, 3]))
        self.assertEqual(1.0, self.outputs["RDIPO"].iloc[2100, 5])
        self.assertEqual(0.0, self.outputs["RDIPO"].iloc[2100, 4])


class OAPHelperTests(unittest.TestCase):
    def test_monthly_applies_to_each_residue_class(self) -> None:
        values = np.arange(60, dtype=float)[:, None]
        result = monthly(values, lambda block: block - np.concatenate([[np.nan], block[:-1, 0]])[:, None])
        np.testing.assert_array_equal(np.full(39, 21.0), result[21:, 0])
        self.assertTrue(np.isnan(result[:21, 0]).all())

    def test_rolling_coskew_matches_its_definition(self) -> None:
        rng = np.random.default_rng(2)
        market = rng.normal(size=80)
        stock = (0.5 * market + rng.normal(size=80))[:, None]
        result = rolling_coskew(stock, market, 50, 40)
        y, m = stock[30:80, 0], market[30:80]
        expected = np.mean((y - y.mean()) * (m - m.mean()) ** 2) / (y.std() * m.var())
        self.assertAlmostEqual(expected, result[79, 0])

    def test_price_delay_matches_ols(self) -> None:
        rng = np.random.default_rng(3)
        market = rng.normal(size=300)
        stock = (0.8 * market + 0.3 * np.concatenate([[0.0], market[:-1]]) + rng.normal(0, 0.5, 300))[:, None]
        result = price_delay(stock, market, 252, 150)
        rows = slice(48, 300)
        design = np.column_stack([np.ones(252)] + [market[48 - j : 300 - j] for j in range(5)])
        beta, *_ = np.linalg.lstsq(design, stock[rows, 0], rcond=None)
        slopes = beta[1:]
        self.assertAlmostEqual((np.arange(5) * slopes).sum() / slopes.sum(), result["slope"][299, 0])
        y = stock[rows, 0]
        r2_full = 1 - np.sum((y - design @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        r2_one = np.corrcoef(market[48:300], stock[rows, 0])[0, 1] ** 2
        self.assertAlmostEqual(1 - r2_one / r2_full, result["rsq"][299, 0])

    def test_corwin_schultz_is_zero_without_range(self) -> None:
        flat = np.ones((3, 1))
        np.testing.assert_array_equal([0.0, 0.0], corwin_schultz(flat, flat)[1:, 0])

    def test_group_stat_and_cross_residual(self) -> None:
        values = np.array([[1.0, 2.0, 3.0, 10.0]])
        groups = np.array([[1.0, 1.0, 2.0, 2.0]])
        universe = np.array([[True, True, True, False]])
        np.testing.assert_array_equal([[1.5, 1.5, 3.0, 3.0]], group_stat(values, groups, universe, "mean"))
        x = np.arange(40, dtype=float)[None, :]
        y = 2.0 + 3.0 * x
        residual = cross_residual(y, [x], np.ones((1, 40), dtype=bool))
        np.testing.assert_allclose(0.0, residual, atol=1e-9)

    def test_published_statements_zero_fill_blank_lines(self) -> None:
        frame = pd.DataFrame(
            {
                "total_assets": [10.0, np.nan],
                "goodwill": [np.nan, np.nan],
                "revenue": [1.0, 1.0],
                "n_cashflow_act": [1.0, 1.0],
            }
        )
        prepared = prepare_fundamentals(frame)
        self.assertEqual(0.0, prepared["goodwill"].iloc[0])
        self.assertTrue(np.isnan(prepared["goodwill"].iloc[1]))


class OAPEngineTests(unittest.TestCase):
    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=300, stocks=6)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            records = [
                inputs[field].stack(future_stack=True).rename(field)
                for field in ("high", "low", "close", "volume", "amount", "cap", "sector")
            ]
            bars = pd.concat(records, axis=1).reset_index()
            bars["in_csi300"] = True
            bars_path = canonical / "bars" / "year=2012" / "bars.parquet"
            bars_path.parent.mkdir(parents=True)
            bars.to_parquet(bars_path, index=False)
            atomic_write_json(
                canonical / "manifest.json",
                {"status": "complete", "fingerprint": "canonical-fixture", "raw_fingerprint": "raw-fixture"},
            )
            files = {}
            daily = pd.concat(
                [
                    inputs["basic_close"].stack().rename("close"),
                    (inputs["basic_turnover_rate"].stack() * 100.0).rename("turnover_rate"),
                    (inputs["basic_total_share"].stack() / 10_000.0).rename("total_share"),
                    (inputs["basic_dv_ttm"].stack(future_stack=True) * 100.0).rename("dv_ttm"),
                ],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                path = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(path)
            pd.DataFrame(
                {
                    "datetime": inputs["close"].index,
                    "mkt": inputs["mkt"].to_numpy(),
                    "smb": inputs["smb"].to_numpy(),
                    "hml": inputs["hml"].to_numpy(),
                    "stocks": 200,
                }
            ).to_parquet(reference / "market_factors.parquet", index=False)
            files["market_factors.parquet"] = sha256_file(reference / "market_factors.parquet")
            inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            listed = inputs["list_date"]
            pd.DataFrame(
                {
                    "ts_code": [f"{code[2:]}.SH" for code in listed.index],
                    "list_date": listed.dt.strftime("%Y%m%d").to_numpy(),
                }
            ).to_parquet(reference / "stock_basic.parquet", index=False)
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="oap"))
            self.assertIsInstance(engine, KunQuantFactorEngine)
            _, _, module, statement_modules = compiled()
            engine.compile = lambda: module
            engine.compile_statements = lambda: statement_modules
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(len(OAP), result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["oap_reference"])
            stored = pd.read_parquet(root / "factors" / "oap" / "year=2012" / "factors.parquet")
            self.assertEqual(["datetime", "instrument", *get_catalog().canonical_names("oap")], list(stored.columns))
            direct = ReferenceLibrary().calculate(inputs)
            for position in (0, 19, 70, 138):
                canonical_name, name = f"oap_{position + 1:03d}", OAP_NAMES[position]
                values = stored.set_index(["datetime", "instrument"])[canonical_name].unstack()
                expected = direct[name].loc[values.index, values.columns]
                np.testing.assert_allclose(
                    values.to_numpy(), expected.to_numpy(), rtol=1e-9, equal_nan=True, err_msg=name
                )

            stale = inputs["fundamentals"].drop(columns=["c_recp_borrow"])
            stale.to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)
            with self.assertRaisesRegex(Exception, "lacks oap statement items"):
                engine.build_from_canonical(canonical, root / "factors")


if __name__ == "__main__":
    unittest.main()


def test_oap_uses_compiled_backend():
    assert OAPLibrary.backend == "kunquant"
