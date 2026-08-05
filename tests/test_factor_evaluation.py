from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rquant.factors.evaluation import evaluate_factor_frame, summarize_effectiveness


class FactorEvaluationTests(unittest.TestCase):
    def test_benchmark_adjustment_attributes_long_and_short_sides(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        rows = []
        for timestamp in dates:
            for ordinal in range(10):
                if ordinal >= 8:
                    label = 0.03
                elif ordinal <= 1:
                    label = -0.01
                else:
                    label = 0.01
                rows.append(
                    {
                        "datetime": timestamp,
                        "instrument": f"S{ordinal:02d}",
                        "label": label,
                        "benchmark": 0.01,
                        "factor": float(ordinal),
                        "constant": 1.0,
                    }
                )
        daily = evaluate_factor_frame(
            pd.DataFrame(rows),
            ["factor", "constant"],
            quantile=0.2,
            min_cross_section=5,
        )
        factor = daily[daily["factor"].eq("factor")]
        np.testing.assert_allclose(factor["top_return"], 0.03)
        np.testing.assert_allclose(factor["bottom_return"], -0.01)
        np.testing.assert_allclose(factor["long_excess"], 0.02)
        np.testing.assert_allclose(factor["short_excess"], 0.02)
        np.testing.assert_allclose(factor["long_short"], 0.04)
        self.assertTrue(daily[daily["factor"].eq("constant")]["ic"].isna().all())
        self.assertTrue(daily[daily["factor"].eq("constant")]["top_return"].isna().all())

    def test_annual_effectiveness_reports_which_side_worked_each_year(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "datetime": "2020-01-02",
                    "factor": "factor",
                    "ic": 0.10,
                    "rank_ic": 0.20,
                    "top_return": 0.03,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.00,
                    "long_excess": 0.02,
                    "short_excess": 0.01,
                },
                {
                    "datetime": "2020-01-03",
                    "factor": "factor",
                    "ic": 0.20,
                    "rank_ic": 0.30,
                    "top_return": 0.02,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.00,
                    "long_excess": 0.01,
                    "short_excess": 0.01,
                },
                {
                    "datetime": "2021-01-04",
                    "factor": "factor",
                    "ic": -0.10,
                    "rank_ic": -0.20,
                    "top_return": 0.00,
                    "benchmark_return": 0.01,
                    "bottom_return": -0.02,
                    "long_excess": -0.01,
                    "short_excess": 0.03,
                },
                {
                    "datetime": "2021-01-05",
                    "factor": "factor",
                    "ic": -0.20,
                    "rank_ic": -0.30,
                    "top_return": 0.00,
                    "benchmark_return": 0.01,
                    "bottom_return": -0.02,
                    "long_excess": -0.01,
                    "short_excess": 0.03,
                },
            ]
        )
        summary, annual = summarize_effectiveness(daily, ["factor"], min_effective_days=2)
        self.assertEqual(1, len(summary))
        year_2020 = annual[annual["year"].eq(2020)].iloc[0]
        year_2021 = annual[annual["year"].eq(2021)].iloc[0]
        self.assertEqual("both", year_2020["effective_side"])
        self.assertEqual("long", year_2020["dominant_side"])
        self.assertEqual("short", year_2021["effective_side"])
        self.assertEqual("short", year_2021["dominant_side"])
        self.assertGreater(year_2020["long_excess_return"], 0.0)
        self.assertGreater(year_2021["short_excess_return"], 0.0)

    def test_insufficient_days_are_not_called_effective(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "datetime": "2026-01-02",
                    "factor": "factor",
                    "ic": 0.2,
                    "rank_ic": 0.2,
                    "top_return": 0.03,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.0,
                    "long_excess": 0.02,
                    "short_excess": 0.01,
                }
            ]
        )
        _, annual = summarize_effectiveness(daily, ["factor"], min_effective_days=2)
        self.assertEqual("insufficient_data", annual.iloc[0]["effective_side"])
        self.assertFalse(bool(annual.iloc[0]["long_effective"]))
        self.assertFalse(bool(annual.iloc[0]["short_effective"]))
