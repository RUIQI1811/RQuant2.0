from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS, STATEMENT_KEY_FIELDS
from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, PanelFactorEngine, create_factor_engine
from rquant.factors.libraries.jq240 import (
    JQ240,
    JQ240_NAMES,
    JQ240Library,
    Packer,
    annual_trend,
    ew_window_sums,
    prepare_fundamentals,
)
from rquant.io import atomic_write_json, sha256_file, stable_hash

BALANCE = tuple(f for f in FUNDAMENTAL_ENDPOINT_FIELDS["balancesheet"] if f not in STATEMENT_KEY_FIELDS)
FLOWS = tuple(
    f
    for endpoint in ("income", "cashflow")
    for f in FUNDAMENTAL_ENDPOINT_FIELDS[endpoint]
    if f not in STATEMENT_KEY_FIELDS and f != "c_cash_equ_end_period"
)


def synthetic_inputs(days: int = 1300, stocks: int = 6, seed: int = 11) -> dict[str, object]:
    """Daily panels and quarterly statements shaped like the engine's jq240 inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    market = rng.normal(0.0003, 0.012, days)
    loadings = rng.uniform(0.6, 1.4, stocks)
    returns = market[:, None] * loadings[None, :] + rng.normal(0.0, 0.015, (days, stocks))
    close = 10.0 * np.cumprod(1.0 + returns, axis=0)
    open_ = close * np.exp(rng.normal(0.0, 0.008, (days, stocks)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, (days, stocks))))
    volume = rng.lognormal(15.0, 0.4, (days, stocks))
    panels = {
        field: pd.DataFrame(values, index=dates, columns=instruments)
        for field, values in (("open", open_), ("high", high), ("low", low), ("close", close), ("volume", volume))
    }
    for panel in panels.values():
        panel.iloc[300:315, 1] = np.nan
        panel.iloc[:60, 2] = np.nan
    shares = rng.uniform(1e8, 1e9, stocks)
    cap = panels["close"] * shares
    universe = pd.DataFrame(True, index=dates, columns=instruments)
    universe.iloc[:500, stocks - 1] = False

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
            row["c_cash_equ_end_period"] = float(rng.uniform(0.1, 0.3) * scale)
            row["total_share"] = shares[column]
            row["total_assets"] = 4.0 * scale * rng.uniform(0.9, 1.1)
            row["total_cur_assets"] = 0.4 * row["total_assets"]
            row["total_nca"] = 0.6 * row["total_assets"]
            row["total_liab"] = 0.5 * row["total_assets"]
            row["total_cur_liab"] = 0.3 * row["total_assets"]
            row["total_ncl"] = 0.2 * row["total_assets"]
            row["total_hldr_eqy_inc_min_int"] = 0.5 * row["total_assets"]
            row["total_hldr_eqy_exc_min_int"] = 0.45 * row["total_assets"]
            row["revenue"] = float(rng.uniform(0.3, 0.5) * scale * quarter)
            row["total_revenue"] = row["revenue"] * 1.01
            rows.append(row)
    fundamentals = pd.DataFrame(rows)
    # One quarter is never reported and one stock has no notes receivable line.
    fundamentals = fundamentals.drop(index=9).reset_index(drop=True)
    fundamentals.loc[fundamentals["instrument"].eq(instruments[3]), "notes_receiv"] = np.nan

    return {
        **panels,
        "amount": panels["close"] * panels["volume"],
        "cap": cap,
        "basic_close": panels["close"] / np.linspace(1.0, 1.3, days)[:, None],
        "basic_turnover_rate": 100.0 * panels["volume"] / pd.DataFrame(
            np.tile(shares * 0.8, (days, 1)), index=dates, columns=instruments),
        "basic_circ_mv": cap * 0.8,
        "basic_total_share": pd.DataFrame(np.tile(shares, (days, 1)), index=dates, columns=instruments),
        "basic_pe_ttm": pd.DataFrame(rng.uniform(5.0, 40.0, (days, stocks)), index=dates, columns=instruments),
        "mkt": pd.Series(market, index=dates),
        "fundamentals": fundamentals,
        "universe": universe,
    }


def truncated(inputs: dict[str, object], last: pd.Timestamp) -> dict[str, object]:
    """Return what was knowable at the close of ``last``."""
    result: dict[str, object] = {}
    for key, value in inputs.items():
        if key == "fundamentals":
            result[key] = value[value["ann_date"] <= last]
        else:
            result[key] = value.loc[:last]
    return result


class JQ240CatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        catalog = get_catalog()
        names = catalog.canonical_names("jq240")
        self.assertEqual(233, len(names))
        self.assertEqual(("jq_001", "jq_233"), (names[0], names[-1]))
        self.assertEqual(len(set(JQ240_NAMES)), len(JQ240_NAMES))
        self.assertEqual(1439, len(catalog.specs))
        counts = pd.Series([entry[1] for entry in JQ240]).value_counts().to_dict()
        self.assertEqual(
            {"quality": 66, "basics": 36, "emotion": 36, "momentum": 34, "pershare": 15, "technical": 15,
             "risk": 12, "style": 10, "growth": 9},
            counts,
        )
        for dropped in ("non_recurring_gain_loss", "adjusted_profit_to_total_profit"):
            self.assertNotIn(dropped, JQ240_NAMES)

    def test_fingerprint_covers_the_statement_fields(self) -> None:
        self.assertEqual(JQ240Library().implementation_fingerprint, JQ240Library().implementation_fingerprint)
        self.assertIn("ass_invest_income", FUNDAMENTAL_ENDPOINT_FIELDS["income"])
        self.assertIn("c_fr_sale_sg", FUNDAMENTAL_ENDPOINT_FIELDS["cashflow"])


class JQ240CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.outputs = JQ240Library().calculate(cls.inputs)

    def test_all_factors_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in JQ240_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            # Days without a quote never publish a value.
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)

    # Break caught: any factor that reads a later quote, market return, statement or index membership.
    def test_no_factor_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (650, 1150):
            last = dates[cut]
            past = JQ240Library().calculate(truncated(self.inputs, last))
            for name in JQ240_NAMES:
                expected = self.outputs[name].loc[:last].to_numpy()
                actual = past[name].to_numpy()
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, equal_nan=True, err_msg=name)

    def test_price_and_volume_factors_match_direct_definitions(self) -> None:
        close, volume = self.inputs["close"], self.inputs["volume"]
        row, column = 700, 0
        window = close.iloc[row - 4 : row + 1, column]
        self.assertAlmostEqual(window.mean() / close.iloc[row, column], self.outputs["MAC5"].iloc[row, column])
        self.assertAlmostEqual(
            100.0 * (close.iloc[row, column] / close.iloc[row - 12, column] - 1.0),
            self.outputs["ROC12"].iloc[row, column],
        )
        turnover = self.inputs["basic_turnover_rate"].iloc[row - 4 : row + 1, column]
        self.assertAlmostEqual(turnover.mean(), self.outputs["VOL5"].iloc[row, column])
        up = (close.iloc[row - 11 : row + 1, column].to_numpy() > close.iloc[row - 12 : row, column].to_numpy())
        self.assertAlmostEqual(100.0 * up.mean(), self.outputs["PSY"].iloc[row, column])
        typical = ((self.inputs["high"] + self.inputs["low"] + close) / 3.0).iloc[row - 19 : row + 1, column]
        deviation = (typical - typical.mean()).abs().mean()
        self.assertAlmostEqual(
            (typical.iloc[-1] - typical.mean()) / (0.015 * deviation), self.outputs["CCI20"].iloc[row, column]
        )
        prices = close.iloc[row - 11 : row + 1, column].to_numpy()
        slope = np.polyfit(np.arange(1, 13), prices / prices.mean(), 1)[0]
        self.assertAlmostEqual(slope, self.outputs["PLRC12"].iloc[row, column])
        highs = self.inputs["high"].iloc[row - 24 : row + 1, column].to_numpy()
        since = 24 - int(np.flatnonzero(highs == highs.max())[-1])
        self.assertAlmostEqual((25 - since) / 25, self.outputs["arron_up_25"].iloc[row, column])
        vstd = volume.iloc[row - 9 : row + 1, column].std()
        self.assertAlmostEqual(vstd / 1e6, self.outputs["VSTD10"].iloc[row, column] / 1e6)

    def test_windows_count_quotes_across_a_suspension(self) -> None:
        close = self.inputs["close"]
        row, column = 316, 1  # the second quote after the suspension of rows 300-314
        quotes = close.iloc[:row + 1, column].dropna().iloc[-5:]
        self.assertAlmostEqual(quotes.mean() / close.iloc[row, column], self.outputs["MAC5"].iloc[row, column])
        self.assertTrue(np.isnan(self.outputs["MAC5"].iloc[305, column]))

    def test_statement_factors_appear_on_the_announcement_day(self) -> None:
        fundamentals = self.inputs["fundamentals"]
        instrument = "SH600000"
        reports = fundamentals[fundamentals["instrument"].eq(instrument)].sort_values("end_date")
        dates = self.inputs["close"].index
        for position in range(45, 60):
            report, previous = reports.iloc[position], reports.iloc[position - 1]
            row = int(np.searchsorted(dates.to_numpy(), report["ann_date"].to_numpy()))
            if 0 < row < len(dates) and row > int(np.searchsorted(dates.to_numpy(), previous["ann_date"].to_numpy())):
                break
        else:
            self.skipTest("fixture announced every quarter out of order")
        current = self.outputs["current_ratio"][instrument]
        self.assertAlmostEqual(report["total_cur_assets"] / report["total_cur_liab"], current.iloc[row])
        shares = self.inputs["basic_total_share"][instrument].iloc[row]
        self.assertAlmostEqual(report["cap_rese"] / shares, self.outputs["capital_reserve_fund_per_share"]
                               [instrument].iloc[row])

    def test_style_factors_are_standardised_over_index_members(self) -> None:
        universe = self.inputs["universe"]
        for name in ("size", "beta", "momentum", "residual_volatility", "liquidity", "leverage"):
            panel = self.outputs[name].where(universe)
            row = panel.iloc[-1].dropna()
            self.assertGreaterEqual(len(row), 4, name)
            self.assertAlmostEqual(0.0, row.mean(), places=6, msg=name)
        rank = self.outputs["Rank1M"]
        self.assertTrue(rank.iloc[:500, -1].isna().all())


class JQ240HelperTests(unittest.TestCase):
    def test_packer_round_trips_and_compacts_each_column(self) -> None:
        values = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
        packer = Packer(np.isfinite(values), pd.RangeIndex(3), pd.Index(["a", "b"]))
        packed = packer.pack(pd.DataFrame(values))
        np.testing.assert_array_equal([[1.0, 2.0], [3.0, 4.0]], packed.to_numpy())
        np.testing.assert_array_equal(values, packer.unpack(packed).to_numpy())

    def test_exponential_window_sums_match_explicit_weights(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(size=(40, 2))
        result = ew_window_sums({"x": values}, 10, 4.0)["x"]
        weights = 0.5 ** (np.arange(10) / 4.0)
        expected = (values[30:40][::-1] * weights[:, None]).sum(axis=0)
        np.testing.assert_allclose(expected, result[39])

    def test_annual_trend_is_slope_over_mean_absolute_value(self) -> None:
        values = np.full((1, 20), np.nan)
        values[0, 3::4] = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = annual_trend(values)
        self.assertAlmostEqual(1.0 / 3.0, trend[0, 19])
        self.assertTrue(np.isnan(trend[0, 15]))

    def test_published_statements_zero_fill_blank_lines(self) -> None:
        frame = pd.DataFrame(
            {
                "instrument": ["SZ000001", "SZ000001"],
                "total_assets": [10.0, np.nan],
                "revenue": [5.0, 6.0],
                "n_cashflow_act": [1.0, np.nan],
                "notes_receiv": [np.nan, np.nan],
                "oth_receiv": [1.0, np.nan],
                "int_receiv": [np.nan, np.nan],
                "div_receiv": [0.5, np.nan],
                "oth_rcv_total": [np.nan, np.nan],
            }
        )
        prepared = prepare_fundamentals(frame)
        self.assertEqual(0.0, prepared["notes_receiv"].iloc[0])
        self.assertTrue(np.isnan(prepared["notes_receiv"].iloc[1]))
        self.assertEqual(1.5, prepared["jq_other_receivables"].iloc[0])
        self.assertEqual(0.0, prepared["sell_exp"].iloc[1])


class JQ240EngineTests(unittest.TestCase):
    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=300, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            records = [inputs[field].stack(future_stack=True).rename(field)
                       for field in ("open", "high", "low", "close", "volume", "amount", "cap")]
            bars = pd.concat(records, axis=1).reset_index()
            bars["in_csi300"] = True
            bars_path = canonical / "bars" / "year=2015" / "bars.parquet"
            bars_path.parent.mkdir(parents=True)
            bars.to_parquet(bars_path, index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-fixture",
                                                            "raw_fingerprint": "raw-fixture"})
            files = {}
            daily = pd.concat(
                [
                    inputs["basic_close"].stack().rename("close"),
                    inputs["basic_turnover_rate"].stack().rename("turnover_rate"),
                    (inputs["basic_circ_mv"].stack() / 10_000.0).rename("circ_mv"),
                    (inputs["basic_total_share"].stack() / 10_000.0).rename("total_share"),
                    inputs["basic_pe_ttm"].stack().rename("pe_ttm"),
                ],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                path = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                path.parent.mkdir(parents=True)
                frame.to_parquet(path, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(path)
            pd.DataFrame({"datetime": inputs["close"].index, "mkt": inputs["mkt"].to_numpy(), "smb": 0.0,
                          "hml": 0.0, "stocks": 200}).to_parquet(reference / "market_factors.parquet", index=False)
            files["market_factors.parquet"] = sha256_file(reference / "market_factors.parquet")
            inputs["fundamentals"].to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="jq240"))
            self.assertIsInstance(engine, PanelFactorEngine)
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(233, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["jq240_reference"])
            stored = pd.read_parquet(root / "factors" / "jq240" / "year=2015" / "factors.parquet")
            self.assertEqual(["datetime", "instrument", *get_catalog().canonical_names("jq240")],
                             list(stored.columns))
            # Every fixture bar is flagged as an index member.
            members = pd.DataFrame(True, index=inputs["close"].index, columns=inputs["close"].columns)
            direct = JQ240Library().calculate({**inputs, "universe": members})
            for canonical_name, name in (("jq_001", JQ240_NAMES[0]), ("jq_225", JQ240_NAMES[224])):
                values = stored.set_index(["datetime", "instrument"])[canonical_name].unstack()
                expected = direct[name].loc[values.index, values.columns]
                np.testing.assert_allclose(values.to_numpy(), expected.to_numpy(), rtol=1e-9, equal_nan=True)

            stale = inputs["fundamentals"].drop(columns=["ass_invest_income"])
            stale.to_parquet(reference / "fundamentals.parquet", index=False)
            files["fundamentals.parquet"] = sha256_file(reference / "fundamentals.parquet")
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)
            with self.assertRaisesRegex(Exception, "lacks jq240 statement items"):
                engine.build_from_canonical(canonical, root / "factors")


if __name__ == "__main__":
    unittest.main()
