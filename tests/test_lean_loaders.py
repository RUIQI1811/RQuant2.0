"""The file-by-file canonical bar and daily_basic loaders give exactly what pivoting the whole table gave."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.errors import DataContractError
from rquant.factors.daily_panels import daily_basic_panels
from rquant.factors.engine import _CanonicalBars, _DiskArrays, _load_canonical_panels
from rquant.universe import universe_mask

FIELDS = ("open", "close", "volume")


def pivot_panels(bars_path: Path, fields, universe, *, start=None, end=None):
    """The loader as it was before: read every bar at once and pivot each field."""
    columns = ["datetime", "instrument", *fields, "in_csi300", "is_st", "listed_days"]
    bars = pd.read_parquet(bars_path, columns=columns)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    if start:
        bars = bars[bars["datetime"] >= pd.Timestamp(start)]
    if end:
        bars = bars[bars["datetime"] <= pd.Timestamp(end)]
    bars = bars.assign(in_universe=universe_mask(bars, universe).to_numpy())
    members = bars.loc[bars["in_universe"], "instrument"].unique()
    bars = bars[bars["instrument"].isin(members)]
    dates = pd.Index(sorted(bars["datetime"].unique()), name="datetime")
    instruments = pd.Index(sorted(members), name="instrument")
    panels = {
        field: bars.pivot(index="datetime", columns="instrument", values=field).reindex(
            index=dates, columns=instruments
        )
        for field in fields
    }
    mask = (
        bars.pivot(index="datetime", columns="instrument", values="in_universe")
        .reindex(index=dates, columns=instruments)
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    return dates, instruments, panels, mask


def write_bars(root: Path, seed: int = 3) -> Path:
    """Two years of bars in two files: gaps, late listings, ST days, non-member boards and a never-member stock."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-12-01", "2021-02-26")
    codes = ["SH600000", "SH600001", "SZ000002", "SZ000003", "SZ300750", "SH688001", "SZ000009"]
    rows = []
    for number, code in enumerate(codes):
        for day, stamp in enumerate(dates):
            if rng.random() < 0.15 or (code == "SZ000003" and day < 20):
                continue
            rows.append(
                {
                    "datetime": stamp,
                    "instrument": code,
                    "open": rng.normal(10, 1),
                    "close": rng.normal(10, 1) if rng.random() > 0.05 else np.nan,
                    "volume": float(rng.integers(1, 1000)),
                    "in_csi300": bool(number % 2 == 0 and day > 10),
                    # SZ000009 is ST throughout, so it is never a main-board member.
                    "is_st": code == "SZ000009" or (code == "SH600001" and 30 <= day < 35),
                    "listed_days": 40 + day + 10 * number,
                }
            )
    bars = pd.DataFrame(rows)
    bars["is_st"] = bars["is_st"].astype("boolean")
    bars["listed_days"] = bars["listed_days"].astype("int32")
    path = root / "bars"
    for year, frame in bars.groupby(bars["datetime"].dt.year):
        target = path / f"year={year}" / "bars.parquet"
        target.parent.mkdir(parents=True)
        frame.to_parquet(target, index=False)
    return path


class CanonicalBarsTests(unittest.TestCase):
    def test_file_by_file_panels_equal_the_pivoted_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bars_path = write_bars(Path(temporary))
            for universe, start, end in (
                ("main", None, None),
                ("sh_main", None, None),
                ("csi300", None, None),
                ("main", "2021-01-05", "2021-02-10"),
            ):
                with self.subTest(universe=universe, start=start, end=end):
                    expected = pivot_panels(bars_path, FIELDS, universe, start=start, end=end)
                    actual = _load_canonical_panels(pd, bars_path, FIELDS, universe, start=start, end=end)
                    pd.testing.assert_index_equal(expected[0], actual[0])
                    pd.testing.assert_index_equal(expected[1], actual[1])
                    for field in FIELDS:
                        pd.testing.assert_frame_equal(expected[2][field], actual[2][field], check_freq=False)
                    np.testing.assert_array_equal(expected[3], actual[3])

    def test_bars_written_to_disk_arrays_are_the_same(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bars_path = write_bars(root)
            expected = pivot_panels(bars_path, FIELDS, "main")
            bars = _CanonicalBars(pd, bars_path, "main", start=None, end=None)
            (root / "disk").mkdir()
            disk = _DiskArrays(np, root / "disk", np.float64)
            try:
                arrays = bars.fill(FIELDS, allocate=disk)
                for field in FIELDS:
                    self.assertIsInstance(arrays[field], np.memmap)
                    np.testing.assert_array_equal(expected[2][field].to_numpy(), np.asarray(arrays[field]))
            finally:
                disk.close()

    def test_a_repeated_bar_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bars_path = write_bars(Path(temporary))
            target = next(bars_path.glob("year=2021/*.parquet"))
            frame = pd.read_parquet(target)
            pd.concat([frame, frame.iloc[:1]]).to_parquet(target, index=False)
            with self.assertRaisesRegex(DataContractError, "repeat"):
                _load_canonical_panels(pd, bars_path, FIELDS, "main", start=None, end=None)

    def test_empty_range_and_empty_universe_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bars_path = write_bars(Path(temporary))
            with self.assertRaisesRegex(DataContractError, "No canonical rows in requested factor range"):
                _load_canonical_panels(pd, bars_path, FIELDS, "main", start="2030-01-01", end=None)
            with self.assertRaisesRegex(DataContractError, "No canonical rows for universe"):
                _load_canonical_panels(pd, bars_path, FIELDS, "csi300", start=None, end="2020-12-03")


class DailyBasicPanelsTests(unittest.TestCase):
    def test_panels_equal_pivot_and_reindex_times_scale(self) -> None:
        rng = np.random.default_rng(5)
        days = pd.bdate_range("2020-12-21", "2021-01-15")
        codes = ["SH600000", "SH600001", "SZ000002", "SZ300750"]
        daily = pd.DataFrame(
            [
                {"instrument": code, "datetime": day, "close": rng.normal(10, 1), "turnover_rate": rng.random()}
                for code in codes
                for day in days
                if rng.random() > 0.1
            ]
        )
        dates = pd.DatetimeIndex(days[3:-2], name="datetime")
        instruments = pd.Index(["SH600000", "SH600001", "SZ000002", "SZ000099"], name="instrument")
        scales = {"close": 1.0, "turnover_rate": 0.01}
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for year, frame in daily.groupby(daily["datetime"].dt.year):
                path = Path(temporary) / f"year={year}.parquet"
                frame.to_parquet(path, index=False)
                paths.append(path)
            actual = daily_basic_panels(paths, scales, dates, instruments)
            unscaled = daily_basic_panels(paths, ("close",), dates, instruments)
            frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
            frame = frame[frame["instrument"].isin(set(instruments))]
            for field, scale in scales.items():
                panel = frame.pivot(index="datetime", columns="instrument", values=field)
                expected = panel.reindex(index=dates, columns=instruments) * scale
                pd.testing.assert_frame_equal(expected, actual[f"basic_{field}"], check_freq=False)
            panel = frame.pivot(index="datetime", columns="instrument", values="close")
            pd.testing.assert_frame_equal(
                panel.reindex(index=dates, columns=instruments), unscaled["basic_close"], check_freq=False
            )

            repeated = pd.read_parquet(paths[-1])
            pd.concat([repeated, repeated.iloc[:1]]).to_parquet(paths[-1], index=False)
            with self.assertRaisesRegex(DataContractError, "repeats"):
                daily_basic_panels(paths, scales, dates, instruments)


if __name__ == "__main__":
    unittest.main()


class PartitionWriterTests(unittest.TestCase):
    def test_blocks_read_back_as_the_whole_year_frame(self) -> None:
        from unittest import mock

        from rquant.factors import engine

        rng = np.random.default_rng(9)
        dates = pd.DatetimeIndex(pd.bdate_range("2020-12-24", "2021-01-12"), name="datetime")
        instruments = pd.Index(["SH600000", "SH600001", "SZ000002"], name="instrument")
        names = ("f_001", "f_002")
        values = {name: rng.normal(size=(len(dates), len(instruments) + 1)) for name in names}
        values["f_002"][2, 1] = np.nan
        mask = rng.random((len(dates), len(instruments))) > 0.3
        mask[:, 0] = True

        def values_for(name, first, last):
            return values[name][first:last, : len(instruments)]

        for block_cells in (1, 7, 10**9):
            with self.subTest(block_cells=block_cells), tempfile.TemporaryDirectory() as temporary:
                hashes: dict[str, str] = {}
                with mock.patch.object(engine, "_PARTITION_BLOCK_CELLS", block_cells):
                    rows = engine._write_year_partitions(
                        pd, np, Path(temporary), dates, instruments, names, values_for, mask, hashes, label="t"
                    )
                self.assertEqual(int(mask.sum()), rows)
                self.assertEqual({"2020", "2021"}, set(hashes))
                for year in (2020, 2021):
                    stored = pd.read_parquet(Path(temporary) / f"year={year}" / "factors.parquet")
                    in_year = dates.year == year
                    keep = mask[in_year].reshape(-1)
                    expected = pd.DataFrame(
                        {
                            "datetime": np.repeat(dates[in_year].to_numpy(), len(instruments))[keep],
                            "instrument": np.tile(instruments.to_numpy(), int(in_year.sum()))[keep],
                            **{
                                name: values[name][in_year, : len(instruments)].reshape(-1)[keep]
                                for name in names
                            },
                        }
                    )
                    pd.testing.assert_frame_equal(expected, stored)


class ArrowPoolTests(unittest.TestCase):
    def test_builds_switch_pyarrow_to_the_system_pool(self) -> None:
        import pyarrow as pa

        from rquant.factors import engine

        previous = pa.default_memory_pool()
        try:
            pa.set_memory_pool(pa.mimalloc_memory_pool() if hasattr(pa, "mimalloc_memory_pool") else previous)
            engine._use_system_arrow_pool()
            self.assertEqual("system", pa.default_memory_pool().backend_name)
        finally:
            pa.set_memory_pool(previous)
