from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rquant.errors import DataContractError
from rquant.io import atomic_write_json
from rquant.reporting import build_report

HAS_PARQUET = importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_PARQUET, "pandas and pyarrow are required")
class SignalAnalysisTests(unittest.TestCase):
    def _frame(self, rng):  # type: ignore[no-untyped-def]
        import numpy as np
        import pandas as pd

        dates = pd.bdate_range("2021-12-20", periods=12)
        instruments = [f"SH6{value:05d}" for value in range(30)]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        label = pd.Series(rng.normal(size=len(index)), index=index)
        score = 0.5 * label + pd.Series(rng.normal(size=len(index)), index=index)
        score.iloc[:3] = np.nan
        return score, label

    # Break caught: an IC that pools dates, drops the Spearman ranking, or counts days with too few stocks.
    def test_daily_ic_matches_qlib_calc_ic(self) -> None:
        import numpy as np
        from qlib.contrib.eva.alpha import calc_ic

        from rquant.workflow.signal_analysis import daily_ic

        score, label = self._frame(np.random.default_rng(1))
        daily = daily_ic(score, label).set_index("datetime")
        expected_ic, expected_rank_ic = calc_ic(score, label)
        np.testing.assert_allclose(expected_ic.to_numpy(), daily["ic"].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(expected_rank_ic.to_numpy(), daily["rank_ic"].to_numpy(), rtol=1e-12)
        self.assertEqual(27, int(daily["instruments"].iloc[0]))

        sparse = daily_ic(score, label, min_cross_section=28)
        self.assertNotIn(score.index.get_level_values("datetime")[0], set(sparse["datetime"]))

    def test_summary_splits_calendar_years_and_reports_icir(self) -> None:
        import numpy as np

        from rquant.workflow.signal_analysis import daily_ic, summarize_ic

        score, label = self._frame(np.random.default_rng(2))
        daily = daily_ic(score, label)
        summary = summarize_ic(daily)
        self.assertEqual([2021, 2022], [row["year"] for row in summary["years"]])
        self.assertEqual(len(daily), summary["overall"]["days"])
        self.assertAlmostEqual(daily["ic"].mean() / daily["ic"].std(), summary["overall"]["ic_ir"])
        self.assertAlmostEqual(daily["rank_ic"].mean(), summary["overall"]["rank_ic"])
        with self.assertRaises(DataContractError):
            summarize_ic(daily.iloc[0:0])

    def test_report_shows_prediction_ic_feature_importance_and_optimisation(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            pd.DataFrame(
                [[pd.Timestamp("2022-01-04"), "SH600000", 0.1]], columns=["datetime", "instrument", "score"]
            ).to_parquet(run_directory / "predictions.parquet", index=False)
            row = {"days": 2, "ic": 0.05, "ic_ir": 0.4, "rank_ic": 0.06, "rank_ic_ir": 0.5,
                   "rank_ic_positive_ratio": 0.6}
            atomic_write_json(
                run_directory / "signal_analysis.json",
                {"label": "Ref($open, -2) / Ref($open, -1) - 1", "overall": row, "years": [{"year": 2022, **row}]},
            )
            pd.DataFrame(
                {
                    "prediction_year": [2021, 2021, 2022, 2022],
                    "factor": ["a", "b", "a", "b"],
                    "gain": [3.0, 1.0, 1.0, 0.0],
                    "split": [3, 1, 1, 0],
                    "gain_share": [0.75, 0.25, 1.0, 0.0],
                }
            ).to_parquet(run_directory / "feature_importance.parquet", index=False)
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2022-01-05", "2022-01-06"]),
                    "universe": [300, 302],
                    "holdings": [80, 90],
                    "turnover": [1.0, 0.2],
                    "active_share": [0.4, 0.5],
                    "tracking_error": [0.05, 0.07],
                }
            ).to_parquet(run_directory / "optimizations.parquet", index=False)

            report = build_report(run_directory)

            top = report["feature_importance"]["top"]
            self.assertEqual(["a", "b"], [item["factor"] for item in top])
            self.assertAlmostEqual(0.875, top[0]["mean_gain_share"])
            self.assertEqual([2, 1], [item["years_used"] for item in top])
            self.assertAlmostEqual(85.0, report["optimizations"]["holdings"])
            markdown = (run_directory / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Prediction IC", markdown)
            self.assertIn("| All | 2 | 0.0500 | 0.400 | 0.0600 | 0.500 | 0.60 |", markdown)
            self.assertIn("| 1 | a | 87.50% | 2 |", markdown)
            self.assertIn("## Portfolio optimisation", markdown)
            with (run_directory / "report.json").open("r", encoding="utf-8") as handle:
                self.assertIn("signal_analysis", json.load(handle))


if __name__ == "__main__":
    unittest.main()
