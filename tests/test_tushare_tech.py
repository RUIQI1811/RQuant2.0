from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rquant.factors.catalog import get_catalog
from rquant.factors.libraries.tushare_tech import (
    TUSHARE_TECH_NAMES,
    TushareTechLibrary,
    range_days,
    stock_indicators,
)

VOLUME_SCALED = ("obv",)


def synthetic_bars(days: int = 500, stocks: int = 5, seed: int = 9) -> dict[str, pd.DataFrame]:
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
    close[200, 3] = close[199, 3]
    panels = {
        field: pd.DataFrame(values, index=dates, columns=instruments)
        for field, values in (("open", open_), ("high", high), ("low", low), ("close", close), ("volume", volume))
    }
    for panel in panels.values():
        panel.iloc[120:140, 1] = np.nan
        panel.iloc[:90, 2] = np.nan
    panels["volume"].iloc[180, 2] = np.nan
    return panels


class TushareTechCatalogTests(unittest.TestCase):
    def test_catalog_is_registered_with_stable_names(self) -> None:
        names = get_catalog().canonical_names("tushare_tech")
        self.assertEqual(75, len(names))
        self.assertEqual(("tst_001", "tst_075"), (names[0], names[-1]))
        self.assertEqual(len(set(TUSHARE_TECH_NAMES)), len(TUSHARE_TECH_NAMES))
        for dropped in ("boll_mid", "xsii_td2", "xsii_td4"):
            self.assertNotIn(dropped, TUSHARE_TECH_NAMES)


class TushareTechCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = synthetic_bars()
        cls.outputs = TushareTechLibrary().calculate(cls.inputs)

    def test_all_outputs_are_aligned_populated_and_missing_without_a_quote(self) -> None:
        close = self.inputs["close"]
        quoted = pd.concat(self.inputs.values(), axis=1).notna().T.groupby(level=0).all().T
        for name in TUSHARE_TECH_NAMES:
            panel = self.outputs[name]
            self.assertEqual(close.shape, panel.shape, name)
            self.assertFalse(np.isinf(panel.to_numpy()).any(), name)
            self.assertTrue(panel.where(~quoted).isna().all().all(), name)
            self.assertGreater(np.isfinite(panel.to_numpy()).mean(), 0.4, name)

    # Break caught: any indicator that reads a later quote than the row it is published on.
    def test_no_indicator_uses_information_after_its_row(self) -> None:
        last = self.inputs["close"].index[300]
        past = TushareTechLibrary().calculate({field: panel.loc[:last] for field, panel in self.inputs.items()})
        for name in TUSHARE_TECH_NAMES:
            np.testing.assert_allclose(past[name].to_numpy(), self.outputs[name].loc[:last].to_numpy(), rtol=1e-9,
                                       atol=1e-12, equal_nan=True, err_msg=name)

    # Break caught: an output in price units, which would change with each stock's arbitrary adjustment level.
    def test_outputs_do_not_depend_on_the_price_level(self) -> None:
        scaled = {field: panel * (7.5 if field != "volume" else 1.0) for field, panel in self.inputs.items()}
        rescaled = TushareTechLibrary().calculate(scaled)
        volume = {field: panel * (3.0 if field == "volume" else 1.0) for field, panel in self.inputs.items()}
        revolumed = TushareTechLibrary().calculate(volume)
        for name in TUSHARE_TECH_NAMES:
            np.testing.assert_allclose(rescaled[name].to_numpy(), self.outputs[name].to_numpy(), rtol=1e-7,
                                       atol=1e-9, equal_nan=True, err_msg=name)
            np.testing.assert_allclose(revolumed[name].to_numpy(), self.outputs[name].to_numpy(), rtol=1e-7,
                                       atol=1e-9, equal_nan=True, err_msg=name)

    # Break caught: a suspension that resets recursive indicators or shortens a window.
    def test_suspended_days_are_skipped(self) -> None:
        column = 1
        quoted = self.inputs["close"].iloc[:, column].notna().to_numpy()
        arrays = {field: panel.iloc[:, column].to_numpy()[quoted] for field, panel in self.inputs.items()}
        direct = stock_indicators(arrays)
        for name in TUSHARE_TECH_NAMES:
            np.testing.assert_allclose(self.outputs[name].iloc[:, column].to_numpy()[quoted], direct[name],
                                       equal_nan=True, err_msg=name)

    def test_selected_indicators_match_direct_definitions(self) -> None:
        o, h, l, c, v = (  # noqa: E741
            self.inputs[field].iloc[:, 0].to_numpy() for field in ("open", "high", "low", "close", "volume")
        )
        row = 400
        np.testing.assert_allclose(self.outputs["ma_20"].iloc[row, 0], c[row - 19 : row + 1].mean() / c[row] - 1)
        np.testing.assert_allclose(
            self.outputs["bias2"].iloc[row, 0], (c[row] / c[row - 11 : row + 1].mean() - 1) * 100
        )
        rsv = (c[row] - l[row - 8 : row + 1].min()) / (h[row - 8 : row + 1].max() - l[row - 8 : row + 1].min())
        np.testing.assert_allclose(self.outputs["wr"].iloc[row, 0],
                                   100 * (h[row - 9 : row + 1].max() - c[row])
                                   / (h[row - 9 : row + 1].max() - l[row - 9 : row + 1].min()))
        k = d = 50.0
        rsv_series = [
            (c[t] - l[t - 8 : t + 1].min()) / (h[t - 8 : t + 1].max() - l[t - 8 : t + 1].min()) * 100
            for t in range(8, row + 1)
        ]
        k = d = rsv_series[0]
        for value in rsv_series[1:]:
            k = (2 * k + value) / 3
            d = (2 * d + k) / 3
        self.assertAlmostEqual(rsv * 100, rsv_series[-1])
        np.testing.assert_allclose(self.outputs["kdj_k"].iloc[row, 0], k, rtol=1e-9)
        np.testing.assert_allclose(self.outputs["kdj"].iloc[row, 0], 3 * k - 2 * d, rtol=1e-9)
        change = np.diff(c[: row + 1])
        gain = loss = None
        for value in change:
            gain = max(value, 0.0) if gain is None else (5 * gain + max(value, 0.0)) / 6
            loss = abs(value) if loss is None else (5 * loss + abs(value)) / 6
        np.testing.assert_allclose(self.outputs["rsi_6"].iloc[row, 0], gain / loss * 100, rtol=1e-9)
        up = v[row - 25 : row + 1][np.diff(c[row - 26 : row + 1]) > 0].sum()
        down = v[row - 25 : row + 1][np.diff(c[row - 26 : row + 1]) < 0].sum()
        flat = v[row - 25 : row + 1][np.diff(c[row - 26 : row + 1]) == 0].sum()
        np.testing.assert_allclose(self.outputs["vr"].iloc[row, 0], 100 * (2 * up + flat) / (2 * down + flat))
        np.testing.assert_allclose(
            self.outputs["brar_ar"].iloc[row, 0],
            (h[row - 25 : row + 1] - o[row - 25 : row + 1]).sum()
            / (o[row - 25 : row + 1] - l[row - 25 : row + 1]).sum() * 100,
        )

    def test_streak_and_range_counts(self) -> None:
        np.testing.assert_array_equal([0, 1, 0, 1, 2, 5, 0], range_days(np.array([3.0, 4, 1, 2, 3, 5, 1]), True))
        np.testing.assert_array_equal([0, 0, 2, 0, 0, 0, 6], range_days(np.array([3.0, 4, 1, 2, 3, 5, 0.5]), False))
        updays = self.outputs["updays"].iloc[:, 0].to_numpy()
        change = np.diff(self.inputs["close"].iloc[:, 0].to_numpy())
        self.assertEqual(0.0, updays[1 + int(np.flatnonzero(change <= 0)[3])])


if __name__ == "__main__":
    unittest.main()
