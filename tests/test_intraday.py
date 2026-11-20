from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rquant.data.canonical import CanonicalBuilder
from rquant.data.collector import TushareCollector
from rquant.data.minutes import (
    MinuteBarBuilder,
    monthly_requests,
    plan_minute_requests,
    qlib_to_ts_code,
    session_minutes,
    snap_to_session,
    sync_minutes,
)
from rquant.data.qlib_builder import QlibProviderBuilder
from rquant.errors import DataContractError
from rquant.io import atomic_write_json
from rquant.qlib_ext.minute import parse_window

HAS_QLIB = importlib.util.find_spec("qlib") is not None and importlib.util.find_spec("pyarrow") is not None

DAYS = ["20200130", "20200131", "20200203", "20200204", "20200205"]
BASE = {"000001.SZ": 10.0, "000002.SZ": 20.0}
# 000002.SZ opens at its up limit on 2020-02-03 and stays there through the 09:31 bar.
LIMIT_UP_OPEN = ("000002.SZ", "20200203")


def _minute_prices(ts_code: str, day: str) -> list[float]:
    """Raw close of each of the 241 Tushare bars of a day (the 09:30 auction print first): a rising line."""
    base = BASE[ts_code] + DAYS.index(day) * 0.1
    prices = [round(base + 0.01 * index, 2) for index in range(241)]
    if (ts_code, day) == LIMIT_UP_OPEN:
        prices[0] = prices[1] = prices[2] = _up_limit(ts_code, day)
    return prices


def _up_limit(ts_code: str, day: str) -> float:
    return round(BASE[ts_code] + DAYS.index(day) * 0.1 + 3.0, 2)


def _bar_end_times(day: str) -> list[str]:
    stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    times = ["09:30"]
    for hour, minute, count in ((9, 31, 120), (13, 1, 120)):
        for offset in range(count):
            total = hour * 60 + minute + offset
            times.append(f"{total // 60:02d}:{total % 60:02d}")
    return [f"{stamp} {value}:00" for value in times]


class FakeTushare:
    def __init__(self) -> None:
        import pandas as pd

        self.pd = pd
        self.calls: list[dict] = []

    def stk_mins(self, **query):
        pd = self.pd
        self.calls.append(query)
        rows = []
        first = pd.Timestamp(query["start_date"])
        last = pd.Timestamp(query["end_date"])
        for day in DAYS:
            if not first <= pd.Timestamp(day) <= last:
                continue
            for stamp, price in zip(_bar_end_times(day), _minute_prices(query["ts_code"], day), strict=True):
                rows.append([query["ts_code"], stamp, price, price, price, price, 1000.0, price * 1000.0])
        columns = ["ts_code", "trade_time", "close", "open", "high", "low", "vol", "amount"]
        # Tushare returns the latest bar first.
        frame = pd.DataFrame(rows, columns=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
        return frame[columns].iloc[::-1].reset_index(drop=True)


class MinuteHelperTests(unittest.TestCase):
    def test_session_has_240_bars(self) -> None:
        labels = session_minutes()
        self.assertEqual(240, len(labels))
        self.assertEqual("09:30", labels[0].strftime("%H:%M"))
        self.assertEqual("11:29", labels[119].strftime("%H:%M"))
        self.assertEqual("13:00", labels[120].strftime("%H:%M"))
        self.assertEqual("14:59", labels[-1].strftime("%H:%M"))

    @unittest.skipUnless(HAS_QLIB, "pandas is required")
    def test_bar_end_times_map_to_start_labels(self) -> None:
        import pandas as pd

        ends = pd.to_datetime(
            [
                "2020-01-02 09:25:00",
                "2020-01-02 09:30:00",
                "2020-01-02 09:31:00",
                "2020-01-02 11:30:00",
                "2020-01-02 13:00:00",
                "2020-01-02 13:01:00",
                "2020-01-02 15:00:00",
            ]
        )
        labels = [stamp.strftime("%H:%M") for stamp in snap_to_session(ends)]
        self.assertEqual(["09:30", "09:30", "09:30", "11:29", "13:00", "13:00", "14:59"], labels)

    def test_codes_and_windows(self) -> None:
        self.assertEqual("600000.SH", qlib_to_ts_code("SH600000"))
        self.assertEqual("000001.SZ", qlib_to_ts_code("sz000001"))
        with self.assertRaises(DataContractError):
            qlib_to_ts_code("600000")
        start, end = parse_window("9:30-10:00")
        self.assertEqual(("09:30", "10:00"), (start.strftime("%H:%M"), end.strftime("%H:%M")))
        with self.assertRaises(DataContractError):
            parse_window("15:00-15:00")
        with self.assertRaises(DataContractError):
            parse_window("10:00-09:30")

    @unittest.skipUnless(HAS_QLIB, "pandas is required")
    def test_plan_covers_top_ranked_and_held_stocks(self) -> None:
        import pandas as pd

        calendar = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"])
        predictions = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-30"] * 3 + ["2020-01-31"] * 3),
                "instrument": ["SH600000", "SH600001", "SH600002"] * 2,
                "score": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            }
        )
        positions = pd.DataFrame({"datetime": pd.to_datetime(["2020-01-31"]), "instrument": ["SZ000009"]})
        plan = plan_minute_requests(predictions, calendar, top=2, positions=positions)
        by_day = plan.groupby("day")["instrument"].apply(set).to_dict()
        self.assertEqual({"SH600000", "SH600001", "SZ000009"}, by_day[pd.Timestamp("2020-01-31")])
        self.assertEqual({"SH600001", "SH600002", "SZ000009"}, by_day[pd.Timestamp("2020-02-03")])
        requests = monthly_requests(plan)
        self.assertEqual({"202001", "202002"}, set(requests["month"]))
        self.assertEqual(6, len(requests))


@unittest.skipUnless(HAS_QLIB, "pyqlib and pyarrow are required")
class IntradayPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        import pandas as pd

        self.pd = pd
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.canonical = self.root / "canonical"
        self.qlib = self.root / "qlib"
        self.minutes = self.root / "minutes"
        atomic_write_json(self.raw / "sync_manifest.json", {"status": "complete", "fingerprint": "fixture-v1"})
        self._write_raw_fixture()
        CanonicalBuilder(self.raw, self.canonical).build()
        QlibProviderBuilder(self.canonical, self.qlib).build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, endpoint: str, frame, partition: str = "fixture") -> None:
        path = self.raw / endpoint / partition / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _write_raw_fixture(self) -> None:
        pd = self.pd
        daily, adjustment, basic = [], [], []
        for code in BASE:
            previous = None
            for day in DAYS:
                prices = _minute_prices(code, day)
                volume = 1000.0 * len(prices)
                amount = sum(price * 1000.0 for price in prices)
                daily.append(
                    [
                        code,
                        day,
                        prices[0],
                        max(prices),
                        min(prices),
                        prices[-1],
                        previous or prices[0],
                        volume / 100.0,
                        amount / 1000.0,
                    ]
                )
                adjustment.append([code, day, 1.0])
                basic.append([code, day, 100.0])
                previous = prices[-1]
        self._write(
            "daily",
            pd.DataFrame(
                daily,
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"],
            ),
        )
        self._write("adj_factor", pd.DataFrame(adjustment, columns=["ts_code", "trade_date", "adj_factor"]))
        self._write("daily_basic", pd.DataFrame(basic, columns=["ts_code", "trade_date", "total_mv"]))
        for day in DAYS:
            rows = []
            for code in BASE:
                opening = _minute_prices(code, day)[0]
                rows.append([code, day, _up_limit(code, day), round(opening * 0.9, 2)])
            self._write(
                "stk_limit",
                pd.DataFrame(rows, columns=["ts_code", "trade_date", "up_limit", "down_limit"]),
                partition=f"trade_date={day}",
            )
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
                [["399300.SZ", code, DAYS[0], 1.0] for code in BASE],
                columns=["index_code", "con_code", "trade_date", "weight"],
            ),
        )
        self._write(
            "trade_cal", pd.DataFrame([[day, 1] for day in [*DAYS, "20200206"]], columns=["cal_date", "is_open"])
        )
        self._write(
            "index_daily",
            pd.DataFrame(
                [
                    ["399300.SZ", day, 4000.0, 4010.0, 3990.0, 4000.0 + index, 1.0, 1.0, 0.0]
                    for index, day in enumerate(DAYS)
                ],
                columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg"],
            ),
        )

    def _predictions(self, run: Path) -> None:
        pd = self.pd
        rows = [
            ["2020-01-30", "SZ000001", 1.0],
            ["2020-01-30", "SZ000002", 0.0],
            ["2020-01-31", "SZ000001", 0.0],
            ["2020-01-31", "SZ000002", 1.0],
            ["2020-02-03", "SZ000001", 0.0],
            ["2020-02-03", "SZ000002", 1.0],
            ["2020-02-04", "SZ000001", 0.0],
            ["2020-02-04", "SZ000002", 1.0],
        ]
        frame = pd.DataFrame(rows, columns=["datetime", "instrument", "score"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        run.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(run / "predictions.parquet", index=False)

    def _sync_and_build(self, run: Path) -> FakeTushare:
        pd = self.pd
        calendar = pd.to_datetime(DAYS)
        plan = plan_minute_requests(pd.read_parquet(run / "predictions.parquet"), calendar, top=2)
        fake = FakeTushare()
        collector = TushareCollector(self.raw, client=fake, max_requests_per_minute=1000)
        sync = sync_minutes(collector, monthly_requests(plan))
        self.assertEqual(4, sync["requests"])
        self.assertEqual(4, sync["fetched"])
        self.assertEqual("1min", fake.calls[0]["freq"])
        built = MinuteBarBuilder(self.raw, self.canonical, self.minutes).build()
        self.assertEqual("complete", built["status"])
        self.assertEqual(["shares"], built["volume_units"])
        self.assertEqual(0, built["amount_mismatch_days"])
        return fake

    def test_minute_bars_restore_prices_and_flag_limits(self) -> None:
        pd = self.pd
        run = self.root / "run"
        self._predictions(run)
        self._sync_and_build(run)
        bars = pd.read_parquet(self.minutes / "month=202002" / "bars.parquet")
        day = bars[(bars["instrument"] == "SZ000002") & (bars["datetime"].dt.normalize() == pd.Timestamp("2020-02-03"))]
        self.assertEqual(240, len(day))
        daily = pd.read_parquet(self.canonical / "bars")
        factor = daily.loc[
            (daily["instrument"] == "SZ000002") & (daily["datetime"] == pd.Timestamp("2020-02-03")), "factor"
        ].iloc[0]
        prices = _minute_prices("000002.SZ", "20200203")
        first = day.iloc[0]
        # The 09:30 label holds the auction print and the 09:30-09:31 bar.
        self.assertAlmostEqual((prices[0] + prices[1]) / 2 * factor, float(first["vwap"]), places=5)
        self.assertAlmostEqual(prices[1] * factor, float(first["close"]), places=5)
        self.assertAlmostEqual(2000.0 / factor, float(first["volume"]), places=2)
        self.assertTrue(bool(first["limit_buy"]))
        self.assertTrue(bool(day.iloc[1]["limit_buy"]))
        self.assertFalse(bool(day.iloc[2]["limit_buy"]))
        self.assertTrue((self.minutes / "qlib" / "calendars" / "1min.txt").exists())

    def test_nested_backtest_executes_in_windows_across_months(self) -> None:
        pd = self.pd
        from rquant.workflow.intraday import IntradayBacktestRunner, IntradayConfig

        run = self.root / "run"
        self._predictions(run)
        self._sync_and_build(run)
        output = self.root / "intraday"
        manifest = IntradayBacktestRunner(
            qlib_root=self.qlib,
            minute_root=self.minutes,
            run_directory=run,
            output_directory=output,
            config=IntradayConfig(
                initial_capital=100_000.0,
                topk=1,
                n_drop=1,
                buy_window="09:31-09:32",
                sell_window="09:30-09:31",
            ),
        ).run()
        self.assertEqual("complete", manifest["status"])
        self.assertEqual(0, manifest["uncovered_holding_days"])
        trades = pd.read_parquet(output / "trades.parquet")
        daily = pd.read_parquet(self.canonical / "bars").set_index(["instrument", "datetime"])

        def factor(instrument: str, day: str) -> float:
            return float(daily.loc[(instrument, pd.Timestamp(day)), "factor"])

        buy_a = trades[(trades["instrument"] == "SZ000001") & (trades["direction"] == "buy")].iloc[0]
        self.assertEqual(pd.Timestamp("2020-01-31"), buy_a["datetime"])
        prices = _minute_prices("000001.SZ", "20200131")
        # Two equal slices: bars 09:31 and 09:32 (Tushare bars ending 09:32 and 09:33).
        expected = (prices[2] + prices[3]) / 2 * factor("SZ000001", "2020-01-31")
        self.assertAlmostEqual(expected, float(buy_a["price"]), places=5)

        sell_a = trades[(trades["instrument"] == "SZ000001") & (trades["direction"] == "sell")].iloc[0]
        self.assertEqual(pd.Timestamp("2020-02-03"), sell_a["datetime"])
        prices = _minute_prices("000001.SZ", "20200203")
        # Bar 09:30 holds the auction print and the minute to 09:31; bar 09:31 is the minute to 09:32.
        expected = ((prices[0] + prices[1]) / 2 + prices[2]) / 2 * factor("SZ000001", "2020-02-03")
        self.assertAlmostEqual(expected, float(sell_a["price"]), places=5)

        # 000002.SZ is at its up limit in the 09:31 bar, so the whole buy fills in the 09:32 bar.
        buy_b = trades[(trades["instrument"] == "SZ000002") & (trades["direction"] == "buy")].iloc[0]
        self.assertEqual(pd.Timestamp("2020-02-03"), buy_b["datetime"])
        prices = _minute_prices("000002.SZ", "20200203")
        self.assertAlmostEqual(prices[3] * factor("SZ000002", "2020-02-03"), float(buy_b["price"]), places=5)
        self.assertAlmostEqual(1.0, float(buy_b["fill_rate"]))

        # Held across the month boundary and liquidated on the last day.
        self.assertEqual("sell", trades.iloc[-1]["direction"])
        self.assertEqual(pd.Timestamp("2020-02-05"), trades.iloc[-1]["datetime"])
        portfolio = pd.read_parquet(output / "portfolio.parquet")
        self.assertEqual(len(DAYS), len(portfolio))
        # Daily returns chain the account across the month boundary.
        account = portfolio["account"].to_numpy()
        net = (portfolio["return"] - portfolio["cost"]).to_numpy()
        for day in range(1, len(account)):
            self.assertAlmostEqual(account[day] / account[day - 1] - 1.0, net[day], places=6)
        self.assertIn("net_return", manifest["metrics"])
        self.assertIsNotNone(manifest["execution_vs_open_bp"]["buy"])
        # The outer report books the inner trades' costs and turnover, and return - cost stays the net return.
        self.assertFalse(trades["fallback"].any())
        previous = portfolio["account"].shift(1).fillna(100_000.0)
        self.assertAlmostEqual(float(trades["cost"].sum()), float((portfolio["cost"] * previous).sum()), places=4)
        self.assertAlmostEqual(float(trades["value"].sum()), float((portfolio["turnover"] * previous).sum()), places=2)
        self.assertGreater(float(portfolio["cost"].iloc[1]), 0.0)  # the first buy, on 2020-01-31

    def test_starts_from_daily_book_and_falls_back_to_daily_bars(self) -> None:
        pd = self.pd
        from rquant.workflow.intraday import IntradayBacktestRunner, IntradayConfig

        run = self.root / "run"
        self._predictions(run)
        # Minute bars only for the top-ranked stock: SZ000001 on 2020-01-31, SZ000002 from 2020-02-03.
        plan = plan_minute_requests(pd.read_parquet(run / "predictions.parquet"), pd.to_datetime(DAYS), top=1)
        collector = TushareCollector(self.raw, client=FakeTushare(), max_requests_per_minute=1000)
        sync_minutes(collector, monthly_requests(plan))
        MinuteBarBuilder(self.raw, self.canonical, self.minutes).build()
        daily = pd.read_parquet(self.canonical / "bars").set_index(["instrument", "datetime"])

        def bar(instrument: str, day: str, field: str) -> float:
            return float(daily.loc[(instrument, pd.Timestamp(day)), field])

        # The daily book holds 1,000 shares of SZ000001 and 50,000 cash at the close of 2020-01-31.
        factor = bar("SZ000001", "2020-01-31", "factor")
        amount = 1000.0 / factor
        close = bar("SZ000001", "2020-01-31", "close")  # canonical prices are adjusted
        value = 50_000.0 + amount * close
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-30", "2020-01-31", "2020-01-31"]),
                "instrument": ["SZ000001", "SZ000001", "SZ000002"],
                "adjusted_amount": [amount, amount, 0.0],
            }
        ).iloc[:2].to_parquet(run / "positions.parquet", index=False)
        pd.DataFrame(
            {"datetime": pd.to_datetime(["2020-01-30", "2020-01-31"]), "cash": [50_000.0] * 2, "account": [value] * 2}
        ).to_parquet(run / "portfolio.parquet", index=False)

        output = self.root / "intraday-start"
        manifest = IntradayBacktestRunner(
            qlib_root=self.qlib,
            minute_root=self.minutes,
            run_directory=run,
            output_directory=output,
            config=IntradayConfig(
                initial_capital=1.0,
                topk=1,
                n_drop=1,
                buy_window="09:31-09:32",
                sell_window="09:30-09:31",
                start="2020-02-03",
            ),
        ).run()
        self.assertEqual({"date": "2020-01-31", "value": value, "holdings": 1}, manifest["start_book"])
        trades = pd.read_parquet(output / "trades.parquet")
        sell = trades.iloc[0]
        self.assertEqual(
            ("SZ000001", "sell", pd.Timestamp("2020-02-03")), tuple(sell[["instrument", "direction", "datetime"]])
        )
        # No minute bars for SZ000001 on 2020-02-03: the whole sell deals at the daily open.
        self.assertTrue(bool(sell["fallback"]))
        self.assertAlmostEqual(bar("SZ000001", "2020-02-03", "open"), float(sell["price"]), places=5)
        self.assertAlmostEqual(1.0, float(sell["fill_rate"]))
        self.assertEqual(1, manifest["fallback_trades"])
        self.assertEqual(1, manifest["execution_vs_open_bp"]["buy"]["trades"])
        portfolio = pd.read_parquet(output / "portfolio.parquet")
        first = portfolio.iloc[0]
        self.assertAlmostEqual(float(first["account"]) / value - 1.0, float(first["return"] - first["cost"]), places=6)


if __name__ == "__main__":
    unittest.main()
