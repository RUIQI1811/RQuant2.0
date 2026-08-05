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
class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        import pandas as pd

        self.pd = pd
        self.temporary = tempfile.TemporaryDirectory()
        self.run_directory = Path(self.temporary.name)
        pd.DataFrame(
            [
                [pd.Timestamp("2020-01-02"), "SZ000001", 0.1],
                [pd.Timestamp("2021-01-04"), "SZ000001", 0.2],
            ],
            columns=["datetime", "instrument", "score"],
        ).to_parquet(self.run_directory / "predictions.parquet", index=False)
        atomic_write_json(
            self.run_directory / "backtest.json",
            {"status": "complete", "metrics": {"total_return": 0.1}},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_report_includes_yearly_performance_in_json_and_markdown(self) -> None:
        pd = self.pd
        pd.DataFrame(
            [
                [pd.Timestamp("2020-01-02"), 0.10, 0.01, 0.02, 0.20, 10.0],
                [pd.Timestamp("2020-01-03"), -0.05, 0.01, -0.01, 0.30, 20.0],
                [pd.Timestamp("2021-01-04"), -0.10, 0.00, 0.01, 0.40, 25.0],
                [pd.Timestamp("2021-01-05"), 0.20, 0.02, 0.00, 0.10, 35.0],
            ],
            columns=["datetime", "return", "cost", "bench", "turnover", "total_cost"],
        ).to_parquet(self.run_directory / "portfolio.parquet", index=False)

        report = build_report(self.run_directory)

        years = report["yearly_performance"]["years"]
        self.assertEqual([2020, 2021], [row["year"] for row in years])
        self.assertAlmostEqual(0.0246, years[0]["strategy_return"])
        self.assertAlmostEqual(0.0098, years[0]["benchmark_return"])
        self.assertAlmostEqual(0.0148, years[0]["excess_return"])
        self.assertAlmostEqual(-0.06, years[0]["max_drawdown"])
        self.assertAlmostEqual(0.25, years[0]["average_daily_turnover"])
        self.assertAlmostEqual(0.50, years[0]["turnover_sum"])
        self.assertAlmostEqual(20.0, years[0]["cash_cost"])
        self.assertAlmostEqual(-0.10, years[1]["max_drawdown"])
        self.assertAlmostEqual(15.0, years[1]["cash_cost"])

        with (self.run_directory / "report.json").open("r", encoding="utf-8") as handle:
            written_report = json.load(handle)
        self.assertEqual(years, written_report["yearly_performance"]["years"])
        markdown = (self.run_directory / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Yearly stability", markdown)
        self.assertIn("| 2020 | 2020-01-02 to 2020-01-03 | +2.46% | +0.98% | +1.48 pp |", markdown)
        self.assertIn("| 2021 | 2021-01-04 to 2021-01-05 | +6.20% | +1.00% | +5.20 pp |", markdown)

    def test_report_rejects_incomplete_portfolio_columns(self) -> None:
        pd = self.pd
        pd.DataFrame(
            [[pd.Timestamp("2020-01-02"), 0.10]],
            columns=["datetime", "return"],
        ).to_parquet(self.run_directory / "portfolio.parquet", index=False)

        with self.assertRaisesRegex(DataContractError, "Portfolio columns are missing"):
            build_report(self.run_directory)


if __name__ == "__main__":
    unittest.main()
