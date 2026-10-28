import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "window_metrics.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("window_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowMetricsTests(unittest.TestCase):
    def test_window_slices_the_net_daily_path(self) -> None:
        window_metrics = _load_script().window_metrics
        dates = pd.bdate_range("2021-12-27", periods=8)
        portfolio = pd.DataFrame(
            {
                "datetime": dates,
                "return": [0.50, 0.50, 0.02, -0.01, 0.03, -0.02, 0.01, 0.50],
                "cost": [0.0, 0.0, 0.001, 0.0, 0.001, 0.0, 0.001, 0.0],
                "turnover": [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
                "bench": [0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, None],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            portfolio.to_parquet(Path(directory) / "portfolio.parquet", index=False)
            # 2021-12-29 .. 2022-01-04 keeps rows 2-6: the large first, second and last returns are outside.
            result = window_metrics(Path(directory), "2021-12-29", "2022-01-04")

        net = pd.Series([0.019, -0.01, 0.029, -0.02, 0.009])
        nav = (1 + net).cumprod()
        self.assertEqual(result["days"], 5)
        self.assertAlmostEqual(result["cumulative"], nav.iloc[-1] - 1)
        self.assertAlmostEqual(result["annual"], nav.iloc[-1] ** (252 / 5) - 1)
        self.assertAlmostEqual(result["sharpe"], net.mean() * 252 / (net.std(ddof=1) * math.sqrt(252)))
        self.assertAlmostEqual(result["max_drawdown"], (nav / nav.cummax() - 1).min())
        self.assertAlmostEqual(result["calmar"], result["annual"] / abs(result["max_drawdown"]))
        excess = net - pd.Series([0.01, 0.0, 0.01, 0.0, 0.01])
        self.assertAlmostEqual(result["excess_annual"], excess.mean() * 252)
        self.assertAlmostEqual(result["turnover"], 0.12)
        self.assertAlmostEqual(result["cost_annual"], 0.003 / 5 * 252)


if __name__ == "__main__":
    unittest.main()
