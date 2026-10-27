from __future__ import annotations

import importlib.util
import unittest

from rquant.config import ProjectPaths
from rquant.errors import DataContractError
from rquant.universe import MAIN_NEW_LISTING_EXCLUDED_DAYS, universe_intervals, universe_mask

HAS_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas is required")
class UniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        import pandas as pd

        self.pd = pd
        seasoned = MAIN_NEW_LISTING_EXCLUDED_DAYS + 1
        self.frame = pd.DataFrame(
            [
                ["SH600000", "2020-01-02", False, seasoned, True],
                ["SH601318", "2020-01-02", False, seasoned, False],
                ["SH603000", "2020-01-02", False, seasoned, False],
                ["SH605000", "2020-01-02", False, seasoned, False],
                ["SZ000001", "2020-01-02", False, seasoned, True],
                ["SZ002001", "2020-01-02", False, seasoned, False],
                ["SZ003816", "2020-01-02", False, seasoned, False],
                ["SZ300750", "2020-01-02", False, seasoned, True],
                ["SZ301000", "2020-01-02", False, seasoned, False],
                ["SH688001", "2020-01-02", False, seasoned, False],
                ["SH689009", "2020-01-02", False, seasoned, False],
                ["BJ830799", "2020-01-02", False, seasoned, False],
                ["SH900901", "2020-01-02", False, seasoned, False],
                ["SZ200002", "2020-01-02", False, seasoned, False],
                ["SH600001", "2020-01-02", True, seasoned, False],
                ["SH600002", "2020-01-02", False, MAIN_NEW_LISTING_EXCLUDED_DAYS, False],
            ],
            columns=["instrument", "datetime", "is_st", "listed_days", "in_csi300"],
        )
        self.frame["datetime"] = pd.to_datetime(self.frame["datetime"])

    def test_main_keeps_seasoned_non_st_shanghai_and_shenzhen_main_board_only(self) -> None:
        members = self.frame.loc[universe_mask(self.frame, "main"), "instrument"].tolist()
        self.assertEqual(
            ["SH600000", "SH601318", "SH603000", "SH605000", "SZ000001", "SZ002001", "SZ003816"],
            members,
        )

    def test_sh_main_keeps_only_the_shanghai_main_board(self) -> None:
        members = self.frame.loc[universe_mask(self.frame, "sh_main"), "instrument"].tolist()
        self.assertEqual(["SH600000", "SH601318", "SH603000", "SH605000"], members)

    def test_csi300_and_all(self) -> None:
        self.assertEqual(
            ["SH600000", "SZ000001", "SZ300750"],
            self.frame.loc[universe_mask(self.frame, "csi300"), "instrument"].tolist(),
        )
        self.assertTrue(universe_mask(self.frame, "all").all())

    def test_unknown_st_history_and_unknown_universe_are_rejected(self) -> None:
        frame = self.frame.assign(is_st=self.pd.Series(self.pd.NA, index=self.frame.index, dtype="boolean"))
        with self.assertRaisesRegex(DataContractError, "namechange"):
            universe_mask(frame, "main")
        with self.assertRaisesRegex(DataContractError, "Unsupported universe"):
            universe_mask(self.frame, "csi500")

    def test_intervals_skip_suspensions_and_split_on_st_periods(self) -> None:
        pd = self.pd
        frame = pd.DataFrame(
            {
                "instrument": ["SH600000"] * 5 + ["SZ000001"] * 2,
                # SH600000 has no bar on 2020-01-06 (suspended) and is ST on 2020-01-08.
                "datetime": pd.to_datetime(
                    ["2020-01-02", "2020-01-03", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-02", "2020-01-03"]
                ),
                "is_st": [False, False, False, True, False, False, False],
                "listed_days": [100] * 7,
            }
        )
        intervals = [
            (instrument, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
            for instrument, start, end in universe_intervals(frame.sample(frac=1, random_state=0), "main")
        ]
        self.assertEqual(
            [
                ("SH600000", "2020-01-02", "2020-01-07"),
                ("SH600000", "2020-01-09", "2020-01-09"),
                ("SZ000001", "2020-01-02", "2020-01-03"),
            ],
            intervals,
        )
        self.assertEqual([], universe_intervals(frame.iloc[0:0], "main"))


class UniversePathTests(unittest.TestCase):
    def test_factor_stores_are_separated_by_universe(self) -> None:
        self.assertEqual("factors", ProjectPaths.from_root("/tmp/rquant").factors.name)
        self.assertEqual("factors-main", ProjectPaths.from_root("/tmp/rquant", universe="main").factors.name)


if __name__ == "__main__":
    unittest.main()
