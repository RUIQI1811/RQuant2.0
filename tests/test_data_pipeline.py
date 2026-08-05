from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path
from unittest.mock import patch

from rquant.data.canonical import CanonicalBuilder
from rquant.data.collector import RequestRateLimiter, TushareCollector
from rquant.data.qlib_builder import QlibProviderBuilder
from rquant.errors import DataContractError
from rquant.io import atomic_write_json
from rquant.workflow.backtest import PortfolioBacktestRunner, PortfolioConfig

HAS_PARQUET = importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_PARQUET, "pandas and pyarrow are required")
class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        import pandas as pd

        self.pd = pd
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.canonical = self.root / "canonical"
        self.qlib = self.root / "qlib"
        atomic_write_json(self.raw / "sync_manifest.json", {"status": "complete", "fingerprint": "fixture-v1"})
        self._write_raw_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, endpoint: str, frame: object) -> None:
        path = self.raw / endpoint / "fixture" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _write_raw_fixture(self) -> None:
        pd = self.pd
        self._write(
            "daily",
            pd.DataFrame(
                [
                    ["000001.SZ", "20200102", 10.0, 11.0, 9.0, 10.0, 9.5, 1.0, 2.0],
                    # 000001.SZ is suspended on 2020-01-03 and therefore absent from daily.
                    ["000002.SZ", "20200103", 20.0, 21.0, 19.0, 20.0, 20.0, 2.0, 4.0],
                    ["000001.SZ", "20200106", 12.0, 12.5, 11.5, 12.0, 10.0, 3.0, 6.0],
                    ["000002.SZ", "20200106", 20.5, 21.0, 20.0, 20.5, 20.0, 2.0, 4.1],
                ],
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"],
            ),
        )
        self._write(
            "adj_factor",
            pd.DataFrame(
                [
                    ["000001.SZ", "20200102", 2.0],
                    ["000002.SZ", "20200103", 1.0],
                    ["000001.SZ", "20200106", 2.4],
                    ["000002.SZ", "20200106", 1.0],
                ],
                columns=["ts_code", "trade_date", "adj_factor"],
            ),
        )
        self._write(
            "daily_basic",
            pd.DataFrame(
                [
                    ["000001.SZ", "20200102", 10.0],
                    ["000002.SZ", "20200103", 20.0],
                    ["000001.SZ", "20200106", 12.0],
                    ["000002.SZ", "20200106", 20.5],
                ],
                columns=["ts_code", "trade_date", "total_mv"],
            ),
        )
        self._write(
            "stk_limit",
            pd.DataFrame(
                [
                    ["000001.SZ", "20200102", 10.0, 9.0],
                    ["000002.SZ", "20200103", 22.0, 18.0],
                    ["000001.SZ", "20200106", 13.2, 10.8],
                    ["000002.SZ", "20200106", 22.0, 18.0],
                ],
                columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            ),
        )
        self._write("suspend_d", pd.DataFrame([["000001.SZ", "20200103"]], columns=["ts_code", "trade_date"]))
        self._write(
            "stock_basic",
            pd.DataFrame(
                [
                    ["000001.SZ", "000001", "A", "19910101", None, "L"],
                    ["000002.SZ", "000002", "B", "19910101", None, "L"],
                ],
                columns=["ts_code", "symbol", "name", "list_date", "delist_date", "list_status"],
            ),
        )
        self._write(
            "index_member_all",
            pd.DataFrame(
                [
                    ["000001.SZ", "20100101", None, "L1", "L2A", "L3A"],
                    ["000002.SZ", "20100101", None, "L1", "L2B", "L3B"],
                ],
                columns=["ts_code", "in_date", "out_date", "l1_code", "l2_code", "l3_code"],
            ),
        )
        self._write(
            "index_weight",
            pd.DataFrame(
                [
                    ["399300.SZ", "000001.SZ", "20200102", 1.0],
                    ["399300.SZ", "000002.SZ", "20200103", 1.0],
                    ["399300.SZ", "000001.SZ", "20200106", 1.0],
                    ["399300.SZ", "000002.SZ", "20200106", 1.0],
                ],
                columns=["index_code", "con_code", "trade_date", "weight"],
            ),
        )
        self._write(
            "trade_cal",
            pd.DataFrame(
                [["20200102", 1], ["20200103", 1], ["20200106", 1], ["20200107", 1]],
                columns=["cal_date", "is_open"],
            ),
        )
        self._write(
            "index_daily",
            pd.DataFrame(
                [
                    ["399300.SZ", "20200102", 4000.0, 4010.0, 3990.0, 4000.0, 1.0, 1.0, 0.0],
                    ["399300.SZ", "20200103", 4000.0, 4020.0, 3980.0, 4010.0, 1.0, 1.0, 0.25],
                    ["399300.SZ", "20200106", 4010.0, 4030.0, 4000.0, 4020.0, 1.0, 1.0, 0.25],
                ],
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"],
            ),
        )

    def test_units_adjustment_membership_and_limits(self) -> None:
        import numpy as np

        manifest = CanonicalBuilder(self.raw, self.canonical).build()
        self.assertEqual("complete", manifest["status"])
        bars = self.pd.read_parquet(self.canonical / "bars")
        first = bars[(bars["instrument"] == "SZ000001") & (bars["datetime"] == self.pd.Timestamp("2020-01-02"))].iloc[0]
        self.assertEqual(100.0, first.raw_volume)
        self.assertEqual(2_000.0, first.raw_amount)
        self.assertEqual(20.0, first.raw_vwap)
        self.assertAlmostEqual(1.0, first.close)
        self.assertAlmostEqual(first.raw_close, first.close / first.factor)
        self.assertEqual(100_000.0, first.cap)
        self.assertTrue(first.limit_buy)
        self.assertTrue(first.in_csi300)
        self.assertTrue(np.isfinite(first.sector))

    def test_invalid_ohlc_outside_csi300_is_excluded_and_audited(self) -> None:
        daily_path = self.raw / "daily" / "fixture" / "data.parquet"
        daily = self.pd.read_parquet(daily_path)
        daily.loc[len(daily)] = ["000003.SZ", "20200102", 10.0, 10.0, 10.0, 9.0, 10.0, 1.0, 1.0]
        daily.to_parquet(daily_path, index=False)

        adjustment_path = self.raw / "adj_factor" / "fixture" / "data.parquet"
        adjustment = self.pd.read_parquet(adjustment_path)
        adjustment.loc[len(adjustment)] = ["000003.SZ", "20200102", 1.0]
        adjustment.to_parquet(adjustment_path, index=False)

        manifest = CanonicalBuilder(self.raw, self.canonical).build()

        self.assertEqual(1, manifest["quality"]["excluded_invalid_ohlc_rows"])
        bars = self.pd.read_parquet(self.canonical / "bars")
        self.assertNotIn("SZ000003", set(bars["instrument"]))
        exclusions = self.pd.read_parquet(self.canonical / "quality" / "ohlc_exclusions.parquet")
        self.assertEqual("SZ000003", exclusions.iloc[0]["instrument"])
        self.assertEqual("invalid_ohlc_outside_csi300", exclusions.iloc[0]["reason"])

    def test_invalid_ohlc_inside_csi300_is_rejected_with_identity(self) -> None:
        daily_path = self.raw / "daily" / "fixture" / "data.parquet"
        daily = self.pd.read_parquet(daily_path)
        daily.loc[daily["ts_code"].eq("000001.SZ") & daily["trade_date"].eq("20200102"), "low"] = 10.5
        daily.to_parquet(daily_path, index=False)

        with self.assertRaisesRegex(DataContractError, "SZ000001@2020-01-02"):
            CanonicalBuilder(self.raw, self.canonical).build()

    def test_empty_partition_and_membership_merge_do_not_emit_future_warnings(self) -> None:
        empty_path = self.raw / "index_weight" / "empty" / "data.parquet"
        empty_path.parent.mkdir(parents=True)
        self.pd.DataFrame(columns=["index_code", "con_code", "trade_date", "weight"]).to_parquet(
            empty_path,
            index=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            CanonicalBuilder(self.raw, self.canonical).build()

    def test_canonical_uses_only_latest_industry_snapshot(self) -> None:
        columns = ["ts_code", "in_date", "out_date", "l1_code", "l2_code", "l3_code", "is_new"]
        older = self.pd.DataFrame(
            [["000001.SZ", "20100101", None, "OLD1", "OLD2", "OLD3", "Y"]],
            columns=columns,
        )
        latest = self.pd.DataFrame(
            [["000001.SZ", "20100101", None, "NEW1", "NEW2", "NEW3", "Y"]],
            columns=columns,
        )
        for snapshot, frame in (("20200101", older), ("20200106", latest)):
            path = self.raw / "index_member_all" / f"snapshot={snapshot}" / "status=Y" / "page=0000" / "data.parquet"
            path.parent.mkdir(parents=True)
            frame.to_parquet(path, index=False)

        membership = CanonicalBuilder(self.raw, self.canonical)._read_industry_membership("2020-01-06")

        self.assertEqual(["NEW2"], membership["l2_code"].tolist())

    def test_qlib_binary_has_suspension_gap_and_raw_price_recovery(self) -> None:
        import numpy as np
        from qlib.constant import REG_CN
        from qlib.data import D

        import qlib

        CanonicalBuilder(self.raw, self.canonical).build()
        manifest = QlibProviderBuilder(self.canonical, self.qlib).build()
        self.assertEqual("complete", manifest["status"])
        feature_root = self.qlib / "features" / "sz000001"
        close = np.fromfile(feature_root / "close.day.bin", dtype=np.float32)
        factor = np.fromfile(feature_root / "factor.day.bin", dtype=np.float32)
        self.assertEqual(0.0, close[0])
        self.assertAlmostEqual(1.0, close[1])
        self.assertTrue(np.isnan(close[2]))
        self.assertAlmostEqual(12.0, close[3] / factor[3], places=5)
        qlib.init(provider_uri=str(self.qlib), region=REG_CN)
        queried = D.features(
            ["SZ000001"],
            ["$close", "$factor"],
            start_time="2020-01-02",
            end_time="2020-01-06",
            freq="day",
        )
        self.assertEqual(3, len(queried))
        self.assertTrue(np.isnan(queried.loc[("SZ000001", self.pd.Timestamp("2020-01-03")), "$close"]))

    def test_small_qlib_portfolio_backtest_writes_auditable_artifacts(self) -> None:
        CanonicalBuilder(self.raw, self.canonical).build()
        QlibProviderBuilder(self.canonical, self.qlib).build()
        run_directory = self.root / "run"
        run_directory.mkdir()
        self.pd.DataFrame(
            [
                [self.pd.Timestamp("2020-01-02"), "SZ000002", 1.0],
                [self.pd.Timestamp("2020-01-03"), "SZ000001", 1.0],
            ],
            columns=["datetime", "instrument", "score"],
        ).to_parquet(run_directory / "predictions.parquet", index=False)
        result = PortfolioBacktestRunner(
            qlib_root=self.qlib,
            run_directory=run_directory,
            config=PortfolioConfig(initial_capital=100_000, topk=1, n_drop=1),
        ).run()
        self.assertEqual("complete", result["status"])
        self.assertTrue((run_directory / "portfolio.parquet").exists())
        self.assertTrue((run_directory / "positions.parquet").exists())
        self.assertTrue((run_directory / "trades.parquet").exists())
        self.assertTrue(result["terminal_liquidation"]["complete"])
        trades = self.pd.read_parquet(run_directory / "trades.parquet")
        self.assertEqual("sell", trades.iloc[-1]["direction"])


@unittest.skipUnless(HAS_PARQUET, "pandas and pyarrow are required")
class CollectorResumeTests(unittest.TestCase):
    def test_global_rate_limiter_enforces_rolling_minute(self) -> None:
        class Time:
            def __init__(self) -> None:
                self.now = 0.0
                self.sleeps: list[float] = []

            def clock(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.sleeps.append(seconds)
                self.now += seconds

        fake_time = Time()
        limiter = RequestRateLimiter(190, clock=fake_time.clock, sleeper=fake_time.sleep)
        for _ in range(190):
            limiter.acquire()
        self.assertEqual([], fake_time.sleeps)

        limiter.acquire()
        self.assertEqual([60.0], fake_time.sleeps)
        self.assertEqual(60.0, fake_time.now)

    def test_rate_limiter_rejects_invalid_limits(self) -> None:
        for value in (0, -1, True, 190.5):
            with self.subTest(value=value), self.assertRaises(DataContractError):
                RequestRateLimiter(value)  # type: ignore[arg-type]

    def test_retry_attempts_share_the_rate_limit(self) -> None:
        class Time:
            def __init__(self) -> None:
                self.now = 0.0
                self.sleeps: list[float] = []

            def clock(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.sleeps.append(seconds)
                self.now += seconds

        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def daily(self, **query: object) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("retry fixture")
                return str(query["trade_date"])

        fake_time = Time()
        limiter = RequestRateLimiter(1, clock=fake_time.clock, sleeper=fake_time.sleep)
        with tempfile.TemporaryDirectory() as temporary:
            client = Client()
            collector = TushareCollector(
                temporary,
                client=client,
                retries=2,
                base_backoff=0,
                rate_limiter=limiter,
            )
            with patch("rquant.data.collector.random.random", return_value=0):
                result = collector._call("daily", {"trade_date": "20200102"})

        self.assertEqual("20200102", result)
        self.assertEqual(2, client.calls)
        self.assertEqual([60.0], fake_time.sleeps)

    def test_hash_validated_resume_and_refetch(self) -> None:
        import pandas as pd

        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def daily(self, **query: object) -> object:
                self.calls += 1
                return pd.DataFrame([["000001.SZ", query["trade_date"]]], columns=["ts_code", "trade_date"])

        with tempfile.TemporaryDirectory() as temporary:
            client = Client()
            collector = TushareCollector(temporary, client=client)
            query = {"trade_date": "20200102"}
            first = collector.fetch_partition("daily", query, "trade_date=20200102")
            second = collector.fetch_partition("daily", query, "trade_date=20200102")
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(1, client.calls)
            Path(first.path).write_bytes(b"corrupt")
            repaired = collector.fetch_partition("daily", query, "trade_date=20200102")
            self.assertEqual(2, client.calls)
            with (Path(repaired.path).parent / "manifest.json").open("r", encoding="utf-8") as handle:
                self.assertEqual("complete", json.load(handle)["status"])

    def test_sync_progress_reports_partition_counts_and_cache_hits(self) -> None:
        import pandas as pd

        class Client:
            def __init__(self) -> None:
                self.industry_queries: list[dict[str, object]] = []

            def __getattr__(self, endpoint: str) -> object:
                def fetch(**query: object) -> object:
                    if endpoint == "trade_cal":
                        return pd.DataFrame(
                            [["20200102", 1], ["20200103", 1]],
                            columns=["cal_date", "is_open"],
                        )
                    if endpoint == "index_member_all":
                        self.industry_queries.append(query)
                        return pd.DataFrame([["000001.SZ"]], columns=["ts_code"])
                    return pd.DataFrame()

                return fetch

        with tempfile.TemporaryDirectory() as temporary:
            client = Client()
            collector = TushareCollector(temporary, client=client)
            first_stderr = io.StringIO()
            with redirect_stderr(first_stderr):
                first = collector.sync(
                    formal_start=date(2020, 1, 2),
                    through=date(2020, 1, 3),
                    warmup_trading_days=1,
                    show_progress=True,
                )
            self.assertEqual(first["artifacts"], first["fetched_artifacts"])
            self.assertEqual(0, first["cached_artifacts"])
            self.assertEqual(190, first["max_requests_per_minute"])
            self.assertIn("Tushare sync complete", first_stderr.getvalue())
            self.assertIn("100%", first_stderr.getvalue())
            self.assertEqual(["N", "Y"], [query["is_new"] for query in client.industry_queries])
            snapshot = Path(temporary, "index_member_all", "snapshot=20200103")
            self.assertTrue((snapshot / "status=N" / "page=0000" / "data.parquet").exists())
            self.assertTrue((snapshot / "status=Y" / "page=0000" / "data.parquet").exists())

            second_stderr = io.StringIO()
            with redirect_stderr(second_stderr):
                second = collector.sync(
                    formal_start=date(2020, 1, 2),
                    through=date(2020, 1, 3),
                    warmup_trading_days=1,
                    show_progress=True,
                )
            self.assertEqual(second["artifacts"], second["cached_artifacts"])
            self.assertEqual(0, second["fetched_artifacts"])
            self.assertIn(f"cached={second['artifacts']} fetched=0", second_stderr.getvalue())
            self.assertEqual(2, len(client.industry_queries))


if __name__ == "__main__":
    unittest.main()
