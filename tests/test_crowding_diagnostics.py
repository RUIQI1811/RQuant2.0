import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "crowding_diagnostics.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("crowding_diagnostics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel(days: pd.DatetimeIndex, stocks: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        score = rng.normal(size=stocks)
        cap = np.arange(1, stocks + 1, dtype=float)  # stock i is the i-th smallest every day
        rows.append(
            pd.DataFrame(
                {
                    "datetime": day,
                    "instrument": [f"S{i:03d}" for i in range(stocks)],
                    "score": score,
                    "circ_mv": cap,
                    # The overnight move before the buy follows the score; the captured label follows it only in
                    # small caps; the day after the holding period is noise.
                    "pre": 0.01 * score,
                    "day": np.where(cap <= stocks / 3, 0.002 * score + 0.001, 0.0),
                    "night": 0.0,
                    "next": rng.normal(size=stocks),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame["label"] = (1 + frame["day"]) * (1 + frame["night"]) - 1
    return frame


class CrowdingDiagnosticsTests(unittest.TestCase):
    # Break caught: a rank IC that mixes days, or ignores the per-day grouping and NaN rows.
    def test_daily_rank_ic_is_spearman_per_day(self) -> None:
        module = _load_script()
        frame = _panel(pd.bdate_range("2021-01-04", periods=3), 30)
        frame.loc[0, "label"] = np.nan
        ic = module.daily_rank_ic(frame, "score", "pre")
        np.testing.assert_allclose(ic.to_numpy(), 1.0)
        day = frame[frame["datetime"] == frame["datetime"].iloc[-1]]
        expected = day["score"].rank().corr(day["next"].rank())
        self.assertAlmostEqual(expected, module.daily_rank_ic(frame, "score", "next").iloc[-1])

    # Break caught: terciles or top-k picked on the wrong side, or the split of returns attributed to the wrong leg.
    def test_yearly_table_splits_ic_by_size_and_returns_by_leg(self) -> None:
        module = _load_script()
        days = pd.bdate_range("2021-12-20", "2022-01-10")
        table = module.yearly_table(_panel(days, 60), topk=10)

        self.assertEqual(["2021", "2022", "2021-2021", "2022+"], list(table.index))
        row = table.loc["2022+"]
        self.assertAlmostEqual(1.0, row["rank_ic_pre"])
        self.assertAlmostEqual(1.0, row["rank_ic_small"])
        self.assertTrue(np.isnan(row["rank_ic_mid"]))  # mid and large caps carry a constant label
        self.assertGreater(row["top_pre"], 0.0)
        self.assertAlmostEqual(0.0, row["top_night"])
        self.assertAlmostEqual(row["top_label"], row["top_day"], places=6)
        self.assertGreater(row["pre_share"], 0.5)
        self.assertEqual(len(days[days.year >= 2022]), row["days"])
        self.assertGreater(row["small_minus_large"], 5.0)  # +10 bp a day in small caps, plus score noise

    # Break caught: a factor that flips sign after 2022 counted as still working, or categories mixed up.
    def test_category_tables_align_sign_on_the_early_period(self) -> None:
        module = _load_script()
        annual = pd.DataFrame(
            [
                {"factor": "jq_001", "year": 2020, "rank_ic_mean": -0.04},
                {"factor": "jq_001", "year": 2023, "rank_ic_mean": -0.02},
                {"factor": "jq_002", "year": 2020, "rank_ic_mean": 0.03},
                {"factor": "jq_002", "year": 2023, "rank_ic_mean": -0.01},
                {"factor": "jq_003", "year": 2020, "rank_ic_mean": 0.05},
                {"factor": "other", "year": 2020, "rank_ic_mean": 0.9},
            ]
        )
        categories = {"jq_001": "momentum", "jq_002": "momentum", "jq_003": "quality"}
        summary, by_year = module.category_tables(annual, categories)

        momentum = summary.loc["momentum"]
        self.assertEqual(2, momentum["factors"])
        self.assertAlmostEqual(0.035, momentum["ic_before"])
        self.assertAlmostEqual(0.005, momentum["ic_after"])  # (0.02 - 0.01) / 2
        self.assertEqual(1, momentum["flipped"])
        self.assertEqual(3, summary.loc["ALL", "factors"])
        self.assertAlmostEqual(0.035, by_year.loc["momentum", 2020])
        self.assertEqual(233, len(module.jq240_categories()))


if __name__ == "__main__":
    unittest.main()
