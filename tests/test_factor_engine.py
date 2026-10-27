from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rquant.factors.engine import _atomic_replace_directory


class FactorDirectoryPublishTests(unittest.TestCase):
    def test_atomic_replace_directory_publishes_complete_staging_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "combined"
            staging = root / ".combined.staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "manifest.json").write_text("old", encoding="utf-8")
            (staging / "manifest.json").write_text("new", encoding="utf-8")

            _atomic_replace_directory(staging, destination)

            self.assertEqual("new", (destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(staging.exists())
            self.assertEqual([], list(root.glob(".combined.backup.*")))

    def test_atomic_replace_directory_restores_previous_tree_when_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "combined"
            staging = root / ".combined.staging"
            destination.mkdir()
            staging.mkdir()
            (destination / "manifest.json").write_text("old", encoding="utf-8")
            (staging / "manifest.json").write_text("new", encoding="utf-8")
            real_replace = os.replace

            def replace_with_publish_failure(source: str | Path, target: str | Path) -> None:
                if Path(source) == staging:
                    raise OSError("simulated publish failure")
                real_replace(source, target)

            with mock.patch("rquant.io.os.replace", side_effect=replace_with_publish_failure):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    _atomic_replace_directory(staging, destination)

            self.assertEqual("old", (destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("new", (staging / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([], list(root.glob(".combined.backup.*")))


if __name__ == "__main__":
    unittest.main()


class UniverseMembershipTests(unittest.TestCase):
    def test_panel_engine_uses_pre_entry_history_but_publishes_only_member_rows(self) -> None:
        import json

        import numpy as np
        import pandas as pd

        from rquant.factors.engine import FactorBuildConfig, create_factor_engine
        from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
        from rquant.factors.libraries.registry import FactorLibraryRegistry
        from rquant.io import atomic_write_json

        class RollingMean(PanelFactorLibrary):
            family = "rolling_test"
            required_inputs = ("close",)

            @property
            def specs(self) -> tuple[FactorSpec, ...]:
                return (
                    FactorSpec(
                        canonical_name="roll_001",
                        source_name="MA5",
                        family=self.family,
                        ordinal=1,
                        formula="mean(close, 5)",
                        max_lookback=5,
                        implementation="test",
                        catalog_version=1,
                    ),
                )

            def calculate(self, inputs):  # type: ignore[no-untyped-def]
                return {"MA5": inputs["close"].rolling(5).mean()}

        registry = FactorLibraryRegistry()
        registry.register_library(RollingMean())
        dates = pd.bdate_range("2021-01-04", periods=20)
        late_entry = 10
        bars = pd.concat(
            [
                pd.DataFrame(
                    {"datetime": dates, "instrument": "SH600000", "close": np.arange(20.0) + 1, "in_csi300": True}
                ),
                pd.DataFrame(
                    {
                        "datetime": dates,
                        "instrument": "SZ000001",
                        "close": np.arange(20.0) + 100,
                        "in_csi300": np.arange(20) >= late_entry,
                    }
                ),
                pd.DataFrame(
                    {"datetime": dates, "instrument": "SZ000002", "close": 5.0, "in_csi300": False}
                ),
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            (canonical / "bars").mkdir(parents=True)
            (canonical / "reference").mkdir(parents=True)
            bars.to_parquet(canonical / "bars" / "part.parquet", index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "test"})
            pd.DataFrame(
                {"ts_code": "000300.SH", "datetime": dates, "open": 1.0, "close": 1.0}
            ).to_parquet(canonical / "reference" / "index_daily.parquet", index=False)
            engine = create_factor_engine(
                root / "cache", FactorBuildConfig(factor_set="rolling_test"), registry=registry
            )
            manifest = engine.build_from_canonical(canonical, root / "factors")
            frame = pd.read_parquet(next((root / "factors" / "rolling_test").glob("year=*/factors.parquet")))
            stored = json.loads((root / "factors" / "rolling_test" / "manifest.json").read_text())

        self.assertEqual({"SH600000", "SZ000001"}, set(frame["instrument"]))
        late = frame[frame["instrument"].eq("SZ000001")].sort_values("datetime")
        self.assertEqual(20 - late_entry, len(late))
        self.assertEqual(pd.Timestamp(dates[late_entry]), late["datetime"].iloc[0])
        # The first member day already has a full five-day window from pre-entry history.
        expected_first = float(np.mean(np.arange(late_entry - 4, late_entry + 1) + 100))
        self.assertAlmostEqual(expected_first, late["roll_001"].iloc[0])
        self.assertEqual(20 + 20 - late_entry, manifest["rows"])
        self.assertEqual(2, stored["instruments"])
        self.assertEqual("csi300", stored["universe"])

    def test_main_universe_publishes_seasoned_non_st_main_board_rows_and_shares_membership(self) -> None:
        import numpy as np
        import pandas as pd

        from rquant.factors.engine import FactorBuildConfig, create_factor_engine
        from rquant.factors.libraries.base import FactorSpec, PanelFactorLibrary
        from rquant.factors.libraries.registry import FactorLibraryRegistry
        from rquant.io import atomic_write_json

        seen: dict[str, object] = {}

        class Close(PanelFactorLibrary):
            family = "close_test"
            required_inputs = ("close",)

            @property
            def specs(self) -> tuple[FactorSpec, ...]:
                return (
                    FactorSpec(
                        canonical_name="close_001",
                        source_name="CLOSE",
                        family=self.family,
                        ordinal=1,
                        formula="close",
                        max_lookback=1,
                        implementation="test",
                        catalog_version=1,
                    ),
                )

            def calculate(self, inputs):  # type: ignore[no-untyped-def]
                seen["universe"] = inputs["universe"].copy()
                return {"CLOSE": inputs["close"]}

        registry = FactorLibraryRegistry()
        registry.register_library(Close())
        dates = pd.bdate_range("2021-01-04", periods=4)
        frames = []
        for instrument, is_st, listed_days in (
            ("SH600000", [False, False, True, False], [100, 101, 102, 103]),
            ("SZ000002", [False] * 4, [59, 60, 61, 62]),
            ("SZ300750", [False] * 4, [500] * 4),
            ("SH688001", [False] * 4, [500] * 4),
        ):
            frames.append(
                pd.DataFrame(
                    {
                        "datetime": dates,
                        "instrument": instrument,
                        "close": np.arange(4.0) + 1,
                        "in_csi300": False,
                        "is_st": pd.array(is_st, dtype="boolean"),
                        "listed_days": np.asarray(listed_days, dtype="int32"),
                    }
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical"
            (canonical / "bars").mkdir(parents=True)
            pd.concat(frames, ignore_index=True).to_parquet(canonical / "bars" / "part.parquet", index=False)
            atomic_write_json(canonical / "manifest.json", {"status": "complete", "fingerprint": "test"})
            engine = create_factor_engine(
                root / "cache", FactorBuildConfig(factor_set="close_test", universe="main"), registry=registry
            )
            manifest = engine.build_from_canonical(canonical, root / "factors")
            frame = pd.read_parquet(next((root / "factors" / "close_test").glob("year=*/factors.parquet")))

        published = sorted(zip(frame["instrument"], frame["datetime"].dt.strftime("%m-%d"), strict=True))
        self.assertEqual(
            [
                ("SH600000", "01-04"),
                ("SH600000", "01-05"),
                ("SH600000", "01-07"),
                ("SZ000002", "01-06"),
                ("SZ000002", "01-07"),
            ],
            published,
        )
        self.assertEqual("main", manifest["universe"])
        self.assertEqual(["SH600000", "SZ000002"], list(seen["universe"].columns))
        self.assertEqual(5, int(seen["universe"].to_numpy().sum()))
