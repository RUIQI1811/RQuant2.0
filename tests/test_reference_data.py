from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rquant.data.collector import FUNDAMENTALS_MANIFEST, STATEMENT_ROW_GUARD, TushareCollector
from rquant.data.reference import ReferenceBuilder, load_reference_manifest
from rquant.errors import DataContractError
from rquant.io import atomic_write_json, sha256_file


def write(root: Path, endpoint: str, partition: str, frame: pd.DataFrame) -> None:
    path = root / endpoint / partition / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


class StatementClient:
    """Tushare stand-in that returns one statement row per quarter announced inside the window."""

    def __init__(self, crowded: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.crowded = crowded

    def __getattr__(self, endpoint: str):
        def fetch(**query: object) -> pd.DataFrame:
            self.calls.append((endpoint, query))
            first, last = int(str(query["start_date"])[:4]), int(str(query["end_date"])[:4])
            # A crowded stock fills a multi-year window up to the row cap with distinct correction rows.
            versions = STATEMENT_ROW_GUARD if query["ts_code"] == self.crowded and first != last else 1
            rows = [
                {
                    "ts_code": query["ts_code"],
                    "ann_date": f"{year}{month}20",
                    "f_ann_date": f"{year}{month}20",
                    "end_date": f"{year}{end}",
                    "report_type": "1",
                    "update_flag": "0",
                    "revenue": float(year + version),
                }
                for year in range(first, last + 1)
                for month, end in (("04", "0331"), ("08", "0630"), ("10", "0930"))
                for version in range(versions)
            ]
            return pd.DataFrame(rows)

        return fetch


class FundamentalSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.raw = Path(self.temporary.name) / "raw"
        atomic_write_json(self.raw / "sync_manifest.json", {"status": "complete", "index_code": "399300.SZ"})
        write(
            self.raw,
            "index_weight",
            "index=399300_SZ/month=2020-01",
            pd.DataFrame(
                {
                    "index_code": ["399300.SZ", "399300.SZ", "000905.SH"],
                    "con_code": ["000001.SZ", "600000.SH", "000002.SZ"],
                    "trade_date": ["20200102"] * 3,
                    "weight": [1.0, 1.0, 1.0],
                }
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def collector(self, client: StatementClient) -> TushareCollector:
        return TushareCollector(self.raw, client=client, retries=1, max_requests_per_minute=10_000)

    def test_windows_cache_closed_years_and_refetch_only_the_open_window(self) -> None:
        client = StatementClient()
        manifest = self.collector(client).sync_fundamentals(through=date(2020, 6, 30), start_year=2016, window_years=3)
        self.assertEqual(2, manifest["instruments"])
        # Two windows (2016-2018, 2019-2020) for three endpoints and two stocks of the configured index only.
        self.assertEqual(12, manifest["artifacts"])
        self.assertEqual({"000001.SZ", "600000.SH"}, {query["ts_code"] for _, query in client.calls})
        self.assertTrue((self.raw / FUNDAMENTALS_MANIFEST).exists())
        open_window = [query for _, query in client.calls if query["start_date"] == "20190101"]
        self.assertEqual({"20200630"}, {query["end_date"] for query in open_window})

        later = StatementClient()
        result = self.collector(later).sync_fundamentals(through=date(2020, 9, 30), start_year=2016, window_years=3)
        self.assertEqual(6, result["cached_artifacts"])
        self.assertEqual(6, result["fetched_artifacts"])
        self.assertEqual({"20190101"}, {query["start_date"] for _, query in later.calls})

    def test_window_at_the_row_cap_is_refetched_year_by_year(self) -> None:
        client = StatementClient(crowded="600000.SH")
        manifest = self.collector(client).sync_fundamentals(through=date(2018, 12, 31), start_year=2016, window_years=3)
        crowded = [item for item in manifest["files"] if "600000_SH" in item["path"]]
        self.assertEqual(9, len(crowded))
        self.assertTrue(all("/ann=2016-2016/" in item["path"] or "/ann=2017-2017/" in item["path"]
                            or "/ann=2018-2018/" in item["path"] for item in crowded))

    def test_requires_completed_daily_sync(self) -> None:
        atomic_write_json(self.raw / "sync_manifest.json", {"status": "running"})
        with self.assertRaisesRegex(DataContractError, "not complete"):
            self.collector(StatementClient()).sync_fundamentals(through=date(2020, 6, 30), start_year=2016)


class ReferenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.raw = root / "raw"
        self.canonical = root / "canonical"
        atomic_write_json(self.raw / "sync_manifest.json", {"status": "complete", "fingerprint": "raw-v1"})
        atomic_write_json(self.canonical / "manifest.json", {"status": "complete", "raw_fingerprint": "raw-v1"})
        rng = np.random.default_rng(5)
        self.codes = [f"{600000 + index:06d}.SH" for index in range(150)] + ["830001.BJ"]
        write(
            self.raw,
            "stock_basic",
            "list_status=L",
            pd.DataFrame({"ts_code": self.codes, "list_date": ["20100101"] * 150 + ["20100101"]}),
        )
        self.days = ["20200130", "20200131", "20200203", "20200204"]
        self.daily = {}
        self.basic = {}
        for day in self.days:
            close = rng.uniform(5.0, 50.0, len(self.codes))
            pre_close = close / (1.0 + rng.normal(0.0, 0.02, len(self.codes)))
            daily = pd.DataFrame(
                {"ts_code": self.codes, "trade_date": day, "close": close, "pre_close": pre_close}
            )
            basic = pd.DataFrame(
                {
                    "ts_code": self.codes,
                    "trade_date": day,
                    "close": close,
                    "turnover_rate": rng.uniform(0.1, 5.0, len(self.codes)),
                    "pb": rng.uniform(0.5, 8.0, len(self.codes)),
                    "total_mv": rng.uniform(1e5, 1e7, len(self.codes)),
                    "circ_mv": rng.uniform(1e5, 1e7, len(self.codes)),
                }
            )
            write(self.raw, "daily", f"trade_date={day}", daily)
            write(self.raw, "daily_basic", f"trade_date={day}", basic)
            self.daily[day] = daily
            self.basic[day] = basic

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_market_factors_use_previous_day_weights_and_skip_beijing(self) -> None:
        manifest = ReferenceBuilder(self.raw, self.canonical).build()
        factors = pd.read_parquet(self.canonical / "reference" / "market_factors.parquet")
        self.assertTrue(np.isnan(factors["mkt"].iloc[0]))
        day, previous = self.days[2], self.days[1]
        daily = self.daily[day].iloc[:150]
        weights = self.basic[previous]["circ_mv"].iloc[:150].to_numpy()
        returns = (daily["close"] / daily["pre_close"] - 1.0).to_numpy()
        self.assertAlmostEqual(float((returns * weights).sum() / weights.sum()), factors["mkt"].iloc[2])
        self.assertTrue(np.isfinite(factors[["smb", "hml"]].iloc[1:]).all().all())
        daily_basic = pd.read_parquet(self.canonical / "reference" / "daily_basic")
        self.assertLessEqual({"SH600000", "BJ830001"}, set(daily_basic["instrument"]))
        self.assertNotIn("fundamentals.parquet", manifest["files"])
        self.assertIn("turnover_total", daily_basic.columns)
        ch4 = pd.read_parquet(self.canonical / "reference" / "ch4_factors.parquet")
        self.assertEqual(["datetime", "mkt", "smb", "vmg", "pmo", "smb4", "stocks"], list(ch4.columns))
        self.assertEqual(len(self.days), len(ch4))
        self.assertIn("ch4_factors.parquet", manifest["files"])
        self.assertEqual(manifest, load_reference_manifest(self.canonical))

    def test_fundamentals_merge_first_reported_statements(self) -> None:
        files = []
        for endpoint, column, value in (
            ("income", "revenue", 10.0),
            ("balancesheet", "total_assets", 100.0),
            ("cashflow", "n_cashflow_act", 3.0),
        ):
            frame = pd.DataFrame(
                {
                    "ts_code": ["600000.SH", "600000.SH"],
                    "ann_date": ["20200425", "20200425"],
                    "f_ann_date": ["20200425", "20200820" if endpoint == "income" else "20200425"],
                    "end_date": ["20200331", "20200331"],
                    "report_type": ["1", "1"],
                    "update_flag": ["0", "1"],
                    column: [value, value * 2],
                }
            )
            partition = "ts_code=600000_SH/ann=2020-2020"
            write(self.raw, endpoint, partition, frame)
            path = self.raw / endpoint / partition / "data.parquet"
            files.append({"endpoint": endpoint, "path": path.relative_to(self.raw).as_posix(),
                          "sha256": sha256_file(path), "rows": 2})
        atomic_write_json(self.raw / FUNDAMENTALS_MANIFEST,
                          {"status": "complete", "files": files, "fingerprint": "fundamentals-v1"})

        manifest = ReferenceBuilder(self.raw, self.canonical).build()
        fundamentals = pd.read_parquet(self.canonical / "reference" / "fundamentals.parquet")
        row = fundamentals.iloc[0]
        self.assertEqual("SH600000", row["instrument"])
        self.assertEqual(pd.Timestamp("2020-04-25"), row["ann_date"])
        # Income keeps the original version; the corrected cash flow and balance sheet were public the same day.
        self.assertEqual(10.0, row["revenue"])
        self.assertEqual(200.0, row["total_assets"])
        self.assertEqual(6.0, row["n_cashflow_act"])
        self.assertEqual("fundamentals-v1", manifest["fundamentals_fingerprint"])

    def test_reference_from_another_raw_sync_is_rejected(self) -> None:
        ReferenceBuilder(self.raw, self.canonical).build()
        atomic_write_json(self.canonical / "manifest.json", {"status": "complete", "raw_fingerprint": "raw-v2"})
        with self.assertRaisesRegex(DataContractError, "different raw sync"):
            load_reference_manifest(self.canonical)


if __name__ == "__main__":
    unittest.main()
