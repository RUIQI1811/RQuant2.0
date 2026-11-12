from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rquant.errors import DataContractError
from rquant.io import atomic_write_json
from rquant.reporting import build_report

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("pandas", "pyarrow", "qlib"))


@unittest.skipUnless(HAS_STACK, "pandas, pyarrow and pyqlib are required")
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

    # Break caught: a summary that pools years, drops the Rank IC, or reports an annualised ICIR.
    def test_summary_of_qlib_daily_ic_splits_calendar_years(self) -> None:
        import numpy as np
        import pandas as pd
        from qlib.contrib.eva.alpha import calc_ic

        from rquant.workflow.signal_analysis import summarize_ic

        score, label = self._frame(np.random.default_rng(2))
        ic, rank_ic = calc_ic(score, label)
        daily = pd.DataFrame({"ic": ic, "rank_ic": rank_ic}).rename_axis("datetime").reset_index()
        summary = summarize_ic(daily)
        self.assertEqual([2021, 2022], [row["year"] for row in summary["years"]])
        self.assertEqual(len(daily), summary["overall"]["days"])
        self.assertAlmostEqual(ic.mean() / ic.std(), summary["overall"]["ic_ir"])
        self.assertAlmostEqual(rank_ic.mean(), summary["overall"]["rank_ic"])
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
                    "factor": ["a", "jq_001", "a", "jq_001"],
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
            self.assertEqual(["a", "jq_001"], [item["factor"] for item in top])
            # Catalog columns are shown with their source library's name; unknown columns keep their own.
            self.assertEqual(["a", "net_profit_to_total_operate_revenue_ttm"], [item["source_name"] for item in top])
            self.assertAlmostEqual(0.875, top[0]["mean_gain_share"])
            self.assertEqual([2, 1], [item["years_used"] for item in top])
            self.assertAlmostEqual(85.0, report["optimizations"]["holdings"])
            markdown = (run_directory / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Prediction IC", markdown)
            self.assertIn("| All | 2 | 0.0500 | 0.400 | 0.0600 | 0.500 | 0.60 |", markdown)
            self.assertIn("| 1 | a | a | 87.50% | 2 |", markdown)
            self.assertIn("## Portfolio optimisation", markdown)
            with (run_directory / "report.json").open("r", encoding="utf-8") as handle:
                self.assertIn("signal_analysis", json.load(handle))

    # Break caught: --signal-only scoring against a label other than the one the run trained on, or running the
    # portfolio backtest anyway.
    def test_signal_only_scores_predictions_with_sig_ana_record_and_no_backtest(self) -> None:
        import numpy as np
        import pandas as pd
        from qlib.contrib.eva.alpha import calc_ic
        from qlib.data import D
        from test_enhanced_indexing import write_provider

        from rquant.workflow.backtest import SignalAnalysisRunner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates, instruments, _ = write_provider(root / "qlib", np.random.default_rng(5))
            run_directory = root / "run"
            run_directory.mkdir()
            rng = np.random.default_rng(6)
            index = pd.MultiIndex.from_product([dates[20:50], instruments], names=["datetime", "instrument"])
            pd.Series(rng.normal(size=len(index)), index=index, name="score").reset_index().to_parquet(
                run_directory / "predictions.parquet", index=False
            )

            summary = SignalAnalysisRunner(
                qlib_root=root / "qlib", run_directory=run_directory, horizon="1d"
            ).run()

            score = pd.read_parquet(run_directory / "predictions.parquet").set_index(["datetime", "instrument"])
            label = D.features(instruments, ["Ref($open, -2) / Ref($open, -1) - 1"], dates[20], dates[49])
            label = label.iloc[:, 0].swaplevel().sort_index().reindex(score.index)
            ic, rank_ic = calc_ic(score["score"], label)
            self.assertAlmostEqual(ic.mean(), summary["overall"]["ic"])
            self.assertAlmostEqual(rank_ic.mean(), summary["overall"]["rank_ic"])
            self.assertTrue((run_directory / "signal_ic.parquet").exists())
            self.assertFalse((run_directory / "portfolio.parquet").exists())
            self.assertFalse((run_directory / "backtest_mlflow").exists())
            with self.assertRaises(DataContractError):
                SignalAnalysisRunner(qlib_root=root / "qlib", run_directory=run_directory, horizon="3d")


if __name__ == "__main__":
    unittest.main()
