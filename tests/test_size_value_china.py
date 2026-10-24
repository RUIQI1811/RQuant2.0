from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.data.ch4_factors import CH4_FACTORS, CH4FactorBuilder
from rquant.factors.catalog import get_catalog
from rquant.factors.engine import FactorBuildConfig, PanelFactorEngine, create_factor_engine
from rquant.factors.libraries.size_value_china import (
    SIZE_VALUE_CHINA_NAMES,
    SizeValueChina,
    SizeValueChinaLibrary,
    rolling_ols,
)
from rquant.io import atomic_write_json, sha256_file, stable_hash


def synthetic_inputs(days: int = 700, stocks: int = 5, seed: int = 3) -> dict[str, object]:
    """Daily panels shaped like the engine's size_value_china inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=days, name="datetime")
    instruments = pd.Index([f"SH{600000 + index}" for index in range(stocks)], name="instrument")
    factors = {name: rng.normal(0.0, 0.008, days) for name in CH4_FACTORS}
    factors["mkt"] = rng.normal(0.0003, 0.012, days)
    factors["pmo"][:30] = np.nan
    loadings = rng.uniform(-0.5, 1.4, (4, stocks))
    exposures = np.column_stack([np.nan_to_num(factors[name]) for name in ("mkt", "smb4", "vmg", "pmo")])
    returns = exposures @ loadings + rng.normal(0.0, 0.015, (days, stocks))
    close = pd.DataFrame(10.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=instruments)
    close.iloc[200:215, 1] = np.nan
    close.iloc[:60, 2] = np.nan
    total_mv = close * rng.uniform(1e5, 1e7, stocks)
    pe = pd.DataFrame(rng.uniform(-20.0, 60.0, (days, stocks)), index=dates, columns=instruments)
    turnover = pd.DataFrame(rng.uniform(0.001, 0.05, (days, stocks)), index=dates, columns=instruments)
    inputs: dict[str, object] = {
        "close": close,
        "basic_total_mv": total_mv,
        "basic_pe_ttm": pe.where(pe > 0),
        "basic_turnover_total": turnover,
    }
    for name, values in factors.items():
        inputs[f"ch4_{name}"] = pd.Series(values, index=dates)
    return inputs


def truncated(inputs: dict[str, object], last: pd.Timestamp) -> dict[str, object]:
    return {key: value.loc[:last] for key, value in inputs.items()}


class SizeValueChinaCatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        catalog = get_catalog()
        names = catalog.canonical_names("size_value_china")
        self.assertEqual(21, len(names))
        self.assertEqual(("svc_001", "svc_021"), (names[0], names[-1]))
        self.assertEqual(len(set(SIZE_VALUE_CHINA_NAMES)), len(SIZE_VALUE_CHINA_NAMES))
        self.assertEqual(939, len(catalog.specs))

    def test_every_characteristic_has_a_calculator(self) -> None:
        for name in ("size", "ep", "abturn"):
            self.assertTrue(hasattr(SizeValueChina, f"_char_{name}"), name)


class SizeValueChinaCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_inputs()
        cls.outputs = SizeValueChinaLibrary().calculate(cls.inputs)

    def test_all_outputs_are_aligned_and_populated(self) -> None:
        close = self.inputs["close"]
        for name in SIZE_VALUE_CHINA_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertTrue(np.isfinite(panel.to_numpy()).any(), name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            self.assertTrue(panel.where(close.isna()).isna().all().all(), name)

    # Break caught: an output that reads a later quote, valuation or factor return than the row it is on.
    def test_no_output_uses_information_after_its_row(self) -> None:
        dates = self.inputs["close"].index
        for cut in (300, 520):
            last = dates[cut]
            past = SizeValueChinaLibrary().calculate(truncated(self.inputs, last))
            for name in SIZE_VALUE_CHINA_NAMES:
                expected = self.outputs[name].loc[:last].to_numpy()
                np.testing.assert_allclose(past[name].to_numpy(), expected, rtol=1e-9, atol=1e-12, equal_nan=True,
                                           err_msg=name)

    def test_outputs_match_direct_definitions(self) -> None:
        close = self.inputs["close"]
        row, column = 600, 0
        self.assertAlmostEqual(np.log(self.inputs["basic_total_mv"].iloc[row, column]),
                               self.outputs["size"].iloc[row, column])
        pe = self.inputs["basic_pe_ttm"].iloc[:, column]
        self.assertTrue(np.isnan(self.outputs["ep"].iloc[:, column][pe.isna() & close.iloc[:, column].notna()]).all())
        turnover = self.inputs["basic_turnover_total"].iloc[:, column]
        self.assertAlmostEqual(
            turnover.iloc[row - 19 : row + 1].mean() / turnover.iloc[row - 249 : row + 1].mean(),
            self.outputs["abturn"].iloc[row, column],
        )
        returns = (close / close.shift(1) - 1.0).iloc[row - 249 : row + 1, column].to_numpy()
        x = np.column_stack(
            [self.inputs[f"ch4_{name}"].iloc[row - 249 : row + 1] for name in ("mkt", "smb4", "vmg", "pmo")]
        )
        design = np.column_stack([np.ones(len(x)), x])
        coefficients, *_ = np.linalg.lstsq(design, returns, rcond=None)
        residuals = returns - design @ coefficients
        self.assertAlmostEqual(coefficients[0], self.outputs["ch4_alpha_y"].iloc[row, column])
        for position, name in enumerate(("mkt", "smb", "vmg", "pmo"), 1):
            self.assertAlmostEqual(coefficients[position], self.outputs[f"ch4_beta_{name}_y"].iloc[row, column])
        self.assertAlmostEqual(np.sqrt((residuals**2).sum() / (250 - 5)), self.outputs["ch4_ivol_y"].iloc[row, column])
        momentum = residuals[:229]
        self.assertAlmostEqual(momentum.mean() / momentum.std(ddof=1), self.outputs["ch4_resmom"].iloc[row, column])


class RollingOlsTests(unittest.TestCase):
    def test_matches_least_squares_with_missing_rows(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 2))
        y = (0.5 + x @ np.array([1.2, -0.7]))[:, None] + rng.normal(scale=0.3, size=(80, 1))
        y[[45, 63], 0] = np.nan
        x[50, 1] = np.nan
        fit = rolling_ols(y, x, ("a", "b"), 30, 20, {"tail": (20, 30)})
        rows = np.arange(50, 80)
        keep = np.isfinite(y[rows, 0]) & np.isfinite(x[rows]).all(axis=1)
        design = np.column_stack([np.ones(keep.sum()), x[rows][keep]])
        target = y[rows, 0][keep]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ coefficients
        self.assertAlmostEqual(coefficients[0], fit["alpha"][79, 0])
        self.assertAlmostEqual(coefficients[1], fit["beta_a"][79, 0])
        self.assertAlmostEqual(coefficients[2], fit["beta_b"][79, 0])
        self.assertAlmostEqual(np.sqrt((residuals**2).sum() / (keep.sum() - 3)), fit["ivol"][79, 0])
        total = ((target - target.mean()) ** 2).sum()
        self.assertAlmostEqual(1.0 - (residuals**2).sum() / total, fit["r2"][79, 0])
        tail = (y[rows, 0] - np.column_stack([np.ones(30), x[rows]]) @ coefficients)[20:]
        tail = tail[np.isfinite(tail)]
        self.assertAlmostEqual(tail.mean() / tail.std(ddof=1), fit["tail"][79, 0])
        self.assertTrue(np.isnan(fit["alpha"][28, 0]))


def market(days: int = 300, stocks: int = 400, seed: int = 1) -> dict[str, object]:
    """A full-market stand-in: returns, valuations and turnover for every stock and day, indexed by Tushare code."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=days)
    codes = [f"{600000 + index:06d}.SH" for index in range(stocks - 20)]
    codes += [f"{688000 + index:06d}.SH" for index in range(10)] + [f"{830000 + index:06d}.BJ" for index in range(10)]
    size = pd.Series(np.exp(rng.normal(14.0, 1.2, stocks)), index=codes)
    pe = pd.Series(rng.uniform(-30.0, 80.0, stocks), index=codes)
    turnover = pd.Series(rng.uniform(0.002, 0.03, stocks), index=codes)
    list_dates = {code: pd.Timestamp("2010-01-01") for code in codes}
    list_dates[codes[0]] = dates[200]  # listed late: excluded until seasoned
    days_data = []
    for day in dates:
        returns = pd.Series(rng.normal(0.0, 0.02, stocks), index=codes)
        traded = rng.random(stocks) > 0.03
        traded[1] = False  # never trades
        state = pd.DataFrame(
            {"total_mv": size * np.exp(rng.normal(0.0, 0.01, stocks)), "pe_ttm": pe.where(pe > 0)}, index=codes
        )
        daily_turnover = turnover * rng.uniform(0.5, 1.5, stocks)
        days_data.append((day, returns[traded], state, daily_turnover[traded]))
    return {"dates": dates, "codes": codes, "list_dates": list_dates, "days": days_data}


class CH4FactorBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.market = market()
        builder = CH4FactorBuilder(cls.market["list_dates"])
        cls.rows = []
        cls.groups = {}
        for day, returns, state, turnover in cls.market["days"]:
            cls.rows.append(builder.update(day, returns, state, turnover))
            cls.groups[day] = builder.groups
        cls.factors = pd.DataFrame(cls.rows).set_index("datetime")

    def test_factors_start_once_a_year_of_trading_history_exists(self) -> None:
        populated = self.factors["vmg"].dropna()
        self.assertFalse(populated.empty)
        # 120 trading days of history are needed before the first formation month.
        self.assertGreaterEqual(populated.index[0], self.market["dates"][120])
        self.assertTrue(np.isfinite(self.factors.loc[populated.index[0]:, list(CH4_FACTORS)]).all().all())

    def test_universe_drops_shells_beijing_star_unseasoned_and_untraded_stocks(self) -> None:
        day, _, state, _ = self.market["days"][-1]
        groups = self.groups[day]
        members = set(groups.index)
        codes = self.market["codes"]
        self.assertFalse(any(code.endswith(".BJ") or code.startswith("688") for code in members))
        self.assertNotIn(codes[1], members)
        board = [code for code in codes if code[:2] in ("60", "00", "30")]
        # Ranked on the total value of the last trading day before the month, before the trading filters.
        dates = self.market["dates"]
        formation = next(i for i, d in enumerate(dates) if (d.year, d.month) == (day.year, day.month))
        previous = self.market["days"][formation - 1][2]["total_mv"].reindex(board)
        cutoff = previous.quantile(0.3)
        self.assertTrue((previous.reindex(list(members)) > cutoff).all())
        self.assertGreater(len(members), 0.6 * len(board))

    def test_groups_change_only_at_month_starts(self) -> None:
        dates = self.market["dates"]
        for previous, current in zip(dates[150:-1], dates[151:], strict=True):
            if previous.month == current.month:
                self.assertIs(self.groups[previous], self.groups[current])

    def test_day_returns_match_a_direct_portfolio_calculation(self) -> None:
        position = len(self.market["days"]) - 5
        day, returns, _, _ = self.market["days"][position]
        weights = self.market["days"][position - 1][2]["total_mv"]
        groups = self.groups[day]
        today = returns.reindex(groups.index)
        valid = today.notna()
        groups, today, weights = groups[valid], today[valid], weights.reindex(groups.index)[valid]

        def vw(members: pd.Series) -> float:
            return float((today[members] * weights[members]).sum() / weights[members].sum())

        def cell(size: str, column: str, label: str) -> float:
            return vw(groups["size"].eq(size) & groups[column].eq(label))

        vmg = (cell("S", "value", "V") + cell("B", "value", "V")) / 2 - (cell("S", "value", "G")
                                                                           + cell("B", "value", "G")) / 2
        pmo = (cell("S", "sentiment", "P") + cell("B", "sentiment", "P")) / 2 - (
            cell("S", "sentiment", "O") + cell("B", "sentiment", "O")
        ) / 2
        smb = np.mean([cell("S", "value", v) for v in "VMG"]) - np.mean([cell("B", "value", v) for v in "VMG"])
        row = self.factors.loc[day]
        self.assertAlmostEqual(vw(pd.Series(True, index=groups.index)), row["mkt"])
        self.assertAlmostEqual(vmg, row["vmg"])
        self.assertAlmostEqual(pmo, row["pmo"])
        self.assertAlmostEqual(smb, row["smb"])

    # Break caught: a 14-day Spring Festival February emptying the next month's universe under a fixed 15-day rule.
    def test_short_month_keeps_the_next_month_populated(self) -> None:
        days = self.market["days"]
        short = [entry for entry in days if (entry[0].year, entry[0].month) == (2019, 8)][14:]
        builder = CH4FactorBuilder(self.market["list_dates"])
        rows = [builder.update(*entry) for entry in days if entry not in short]
        factors = pd.DataFrame(rows).set_index("datetime")
        september = factors[(factors.index.year == 2019) & (factors.index.month == 9)]
        self.assertFalse(september.empty)
        self.assertTrue(np.isfinite(september[list(CH4_FACTORS)]).all().all())

    def test_loss_makers_sort_into_growth(self) -> None:
        day = self.market["dates"][-1]
        groups = self.groups[day]
        pe = self.market["days"][-1][2]["pe_ttm"]
        losers = [code for code in groups.index if np.isnan(pe[code])]
        self.assertTrue(losers)
        self.assertTrue(groups.loc[losers, "value"].eq("G").all())


class SizeValueChinaEngineTests(unittest.TestCase):
    def test_engine_reads_audited_reference_files(self) -> None:
        inputs = synthetic_inputs(days=300, stocks=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            reference = canonical / "reference"
            dates = inputs["close"].index
            bars = inputs["close"].stack(future_stack=True).rename("close").reset_index()
            bars["in_csi300"] = True
            path = canonical / "bars" / "year=2015" / "bars.parquet"
            path.parent.mkdir(parents=True)
            bars.to_parquet(path, index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "canonical-fixture",
                                                            "raw_fingerprint": "raw-fixture"})
            files = {}
            daily = pd.concat(
                [inputs[f"basic_{field}"].stack().rename(field) for field in ("total_mv", "pe_ttm", "turnover_total")],
                axis=1,
            ).reset_index()
            for year_value, frame in daily.groupby(daily["datetime"].dt.year):
                destination = reference / "daily_basic" / f"year={year_value}" / "data.parquet"
                destination.parent.mkdir(parents=True)
                frame.to_parquet(destination, index=False)
                files[f"daily_basic/year={year_value}/data.parquet"] = sha256_file(destination)
            pd.DataFrame({"datetime": dates, **{name: inputs[f"ch4_{name}"].to_numpy() for name in CH4_FACTORS},
                          "stocks": 300}).to_parquet(reference / "ch4_factors.parquet", index=False)
            files["ch4_factors.parquet"] = sha256_file(reference / "ch4_factors.parquet")
            manifest = {"status": "complete", "raw_fingerprint": "raw-fixture", "files": files}
            manifest["fingerprint"] = stable_hash(manifest)
            atomic_write_json(reference / "manifest.json", manifest)

            engine = create_factor_engine(root / "cache", FactorBuildConfig(factor_set="size_value_china"))
            self.assertIsInstance(engine, PanelFactorEngine)
            result = engine.build_from_canonical(canonical, root / "factors")
            self.assertEqual(21, result["columns"])
            self.assertEqual(manifest["fingerprint"], result["reference_inputs"]["size_value_china_reference"])
            store = root / "factors" / "size_value_china"
            stored = pd.concat([pd.read_parquet(path) for path in sorted(store.glob("year=*/*.parquet"))])
            stored = stored.set_index(["datetime", "instrument"])
            direct = SizeValueChinaLibrary().calculate(inputs)
            for canonical_name, name in (("svc_001", "size"), ("svc_008", "ch4_beta_mkt")):
                values = stored[canonical_name].unstack()
                expected = direct[name].loc[values.index, values.columns]
                np.testing.assert_allclose(values.to_numpy(), expected.to_numpy(), rtol=1e-12, equal_nan=True)

            (reference / "ch4_factors.parquet").write_bytes(b"tampered")
            with self.assertRaisesRegex(Exception, "differs from its manifest"):
                engine.build_from_canonical(canonical, root / "factors")


if __name__ == "__main__":
    unittest.main()
