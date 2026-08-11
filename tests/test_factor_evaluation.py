from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from rquant.factors import evaluation
from rquant.factors.evaluation import evaluate_factor_frame, summarize_effectiveness


class FactorEvaluationTests(unittest.TestCase):
    def test_five_day_evaluator_writes_every_trading_date_and_staggered_pocket(self) -> None:
        calendar = pd.date_range("2020-01-02", periods=10, freq="B")
        factor_rows = [
            {"datetime": timestamp, "instrument": instrument, "factor": float(ordinal)}
            for timestamp in calendar
            for ordinal, instrument in enumerate(("S00", "S01"))
        ]

        class Catalog:
            @staticmethod
            def canonical_names(factor_set: str) -> tuple[str, ...]:
                self.assertEqual("combined", factor_set)
                return ("factor",)

        def features(
            instruments: list[str],
            expressions: list[str],
            *,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            freq: str,
        ) -> pd.DataFrame:
            self.assertEqual("day", freq)
            selected_dates = calendar[(calendar >= start_time) & (calendar <= end_time)]
            index = pd.MultiIndex.from_product(
                [instruments, selected_dates],
                names=["instrument", "datetime"],
            )
            if instruments == ["SH000300"]:
                values = np.full(len(index), 0.01)
            else:
                values = np.tile([0.00, 0.02], len(selected_dates))
            return pd.DataFrame({expressions[0]: values}, index=index)

        class Data:
            @staticmethod
            def calendar(**_: object) -> pd.DatetimeIndex:
                return calendar

        Data.features = staticmethod(features)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qlib_root = root / "qlib"
            factor_root = root / "factors"
            factor_directory = factor_root / "combined"
            partition = factor_directory / "year=2020" / "factors.parquet"
            output = root / "run"
            qlib_root.mkdir()
            partition.parent.mkdir(parents=True)
            (qlib_root / "manifest.json").write_text(
                json.dumps({"status": "complete", "fingerprint": "qlib"}),
                encoding="utf-8",
            )
            (factor_directory / "manifest.json").write_text(
                json.dumps({"status": "complete", "factor_set": "combined", "fingerprint": "factors"}),
                encoding="utf-8",
            )
            pd.DataFrame(factor_rows).to_parquet(partition, index=False)
            config = evaluation.FactorEvaluationConfig(
                factor_set="combined",
                horizon="5d",
                quantile=0.5,
                min_cross_section=2,
                min_effective_days=1,
            )
            evaluator = evaluation.FactorEvaluator(
                qlib_root=qlib_root,
                factor_root=factor_root,
                output_directory=output,
                config=config,
            )

            with (
                mock.patch.object(evaluation, "get_catalog", return_value=Catalog()),
                mock.patch("qlib.init"),
                mock.patch("qlib.data.D", Data),
            ):
                result = evaluator.run(start=calendar[0], end=calendar[-1])

            daily = pd.read_parquet(output / "factor_daily.parquet")
            self.assertEqual(calendar.tolist(), daily["datetime"].tolist())
            self.assertEqual([1, 2, 3, 4, 5] * 2, daily["pocket"].tolist())
            self.assertEqual(1, result["anchor_step"])
            self.assertEqual(5, result["holding_period"])
            self.assertEqual(5, result["pocket_count"])
            self.assertEqual("staggered_pockets_v1", result["evaluation_method"])

    def test_assigns_every_daily_signal_to_staggered_horizon_pockets(self) -> None:
        for holding_period, observations in ((5, 10), (20, 40)):
            with self.subTest(holding_period=holding_period):
                calendar = pd.date_range("2020-01-02", periods=observations, freq="B")
                daily = pd.DataFrame({"datetime": calendar, "factor": "factor"})

                assigned = evaluation.assign_staggered_pockets(daily, calendar, holding_period=holding_period)

                self.assertEqual(observations, len(assigned))
                self.assertEqual(
                    list(range(1, holding_period + 1)) * 2,
                    assigned["pocket"].tolist(),
                )

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
        daily["pocket"] = 1
        summary, annual = summarize_effectiveness(
            daily,
            ["factor"],
            min_effective_days=2,
            holding_period=1,
        )
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
        daily["pocket"] = 1
        _, annual = summarize_effectiveness(
            daily,
            ["factor"],
            min_effective_days=2,
            holding_period=1,
        )
        self.assertEqual("insufficient_data", annual.iloc[0]["effective_side"])
        self.assertFalse(bool(annual.iloc[0]["long_effective"]))
        self.assertFalse(bool(annual.iloc[0]["short_effective"]))

    def test_staggered_pockets_compound_independently_then_combine_equally(self) -> None:
        rows = []
        for cycle in range(2):
            for pocket in range(1, 6):
                rows.append(
                    {
                        "datetime": pd.Timestamp("2020-01-02") + pd.offsets.BDay(cycle * 5 + pocket - 1),
                        "factor": "factor",
                        "pocket": pocket,
                        "ic": 0.10,
                        "rank_ic": 0.20,
                        "top_return": 0.10,
                        "benchmark_return": 0.01,
                        "bottom_return": 0.00,
                        "long_excess": 0.09,
                        "short_excess": 0.01,
                    }
                )

        summary, annual = summarize_effectiveness(
            pd.DataFrame(rows),
            ["factor"],
            min_effective_days=10,
            holding_period=5,
        )

        result = summary.iloc[0]
        self.assertEqual(10, result["return_days"])
        self.assertEqual(5, result["pocket_count"])
        self.assertEqual(5, result["required_pockets"])
        self.assertAlmostEqual(0.21, result["top_return"])
        self.assertAlmostEqual(0.0201, result["benchmark_return"])
        self.assertAlmostEqual(0.1899, result["long_excess_return"])
        self.assertAlmostEqual(0.0201, result["short_excess_return"])
        self.assertEqual("both", result["effective_side"])
        self.assertEqual(1, len(annual))

    def test_missing_pocket_is_insufficient_even_with_enough_daily_returns(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp("2020-01-02") + pd.offsets.BDay(index),
                    "factor": "factor",
                    "pocket": index % 4 + 1,
                    "ic": 0.10,
                    "rank_ic": 0.20,
                    "top_return": 0.03,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.00,
                    "long_excess": 0.02,
                    "short_excess": 0.01,
                }
                for index in range(20)
            ]
        )

        summary, _ = summarize_effectiveness(
            daily,
            ["factor"],
            min_effective_days=20,
            holding_period=5,
        )

        result = summary.iloc[0]
        self.assertEqual(4, result["pocket_count"])
        self.assertEqual(5, result["required_pockets"])
        self.assertEqual("insufficient_data", result["effective_side"])

    def test_one_day_horizon_preserves_single_pocket_compounding(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "datetime": "2020-01-02",
                    "factor": "factor",
                    "pocket": 1,
                    "ic": 0.10,
                    "rank_ic": 0.20,
                    "top_return": 0.10,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.00,
                    "long_excess": 0.09,
                    "short_excess": 0.01,
                },
                {
                    "datetime": "2020-01-03",
                    "factor": "factor",
                    "pocket": 1,
                    "ic": 0.20,
                    "rank_ic": 0.30,
                    "top_return": 0.20,
                    "benchmark_return": 0.01,
                    "bottom_return": 0.00,
                    "long_excess": 0.19,
                    "short_excess": 0.01,
                },
            ]
        )

        summary, _ = summarize_effectiveness(
            daily,
            ["factor"],
            min_effective_days=2,
            holding_period=1,
        )

        result = summary.iloc[0]
        self.assertAlmostEqual(0.32, result["top_return"])
        self.assertAlmostEqual(0.0201, result["benchmark_return"])
