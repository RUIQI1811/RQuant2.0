from __future__ import annotations

import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rquant.data.canonical import ts_code_to_qlib
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, stable_hash

QLIB_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "factor",
    "change",
    "limit_buy",
    "limit_sell",
)


class QlibProviderBuilder:
    """Write Qlib's local calendar/instrument/feature binary layout."""

    def __init__(self, canonical_root: str | Path, qlib_root: str | Path) -> None:
        self.canonical_root = Path(canonical_root)
        self.qlib_root = Path(qlib_root)

    def build(self) -> dict[str, Any]:
        pd, _ = _dependencies()
        canonical_manifest = self.canonical_root / "manifest.json"
        if not canonical_manifest.exists():
            raise DataContractError(f"Canonical manifest does not exist: {canonical_manifest}")
        import json

        with canonical_manifest.open("r", encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        if source_manifest.get("status") != "complete":
            raise DataContractError("Canonical dataset is not complete")

        bars_paths = sorted((self.canonical_root / "bars").glob("year=*/bars.parquet"))
        if not bars_paths:
            raise DataContractError("No canonical bar partitions found")
        bars = pd.concat([pd.read_parquet(path) for path in bars_paths], ignore_index=True)
        bars["datetime"] = pd.to_datetime(bars["datetime"])
        calendar = self._calendar(end_time=bars["datetime"].max())
        future_calendar = self._calendar()
        calendar_lookup = {timestamp: index for index, timestamp in enumerate(calendar)}

        self.qlib_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix="rquant-qlib-", dir=self.qlib_root.parent))
        backup: Path | None = None
        try:
            self._write_calendar(temporary_root, calendar, future_calendar)
            self._write_instruments(temporary_root, "all", self._all_intervals(calendar))
            self._write_instruments(temporary_root, "csi300", self._csi300_intervals(calendar))
            for instrument, frame in bars.groupby("instrument", sort=True):
                self._write_instrument(temporary_root, instrument, frame, calendar, calendar_lookup)
            benchmark = self._write_benchmark(temporary_root, calendar, calendar_lookup)

            if self.qlib_root.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = self.qlib_root.with_name(f"{self.qlib_root.name}.previous.{stamp}")
                self.qlib_root.rename(backup)
            temporary_root.rename(self.qlib_root)
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            if backup is not None and backup.exists() and not self.qlib_root.exists():
                backup.rename(self.qlib_root)
            raise

        manifest = {
            "status": "complete",
            "canonical_fingerprint": source_manifest.get("fingerprint"),
            "calendar_start": calendar[0].date().isoformat(),
            "calendar_end": calendar[-1].date().isoformat(),
            "calendar_days": len(calendar),
            "future_calendar_end": future_calendar[-1].date().isoformat(),
            "instruments": int(bars["instrument"].nunique()),
            "benchmark": benchmark,
            "fields": list(QLIB_FIELDS),
            "fingerprint": stable_hash(
                {
                    "canonical": source_manifest.get("fingerprint"),
                    "calendar": [str(calendar[0]), str(calendar[-1]), len(calendar)],
                    "future_calendar": [
                        str(future_calendar[0]),
                        str(future_calendar[-1]),
                        len(future_calendar),
                    ],
                    "fields": QLIB_FIELDS,
                }
            ),
        }
        atomic_write_json(self.qlib_root / "manifest.json", manifest)
        return manifest

    def _calendar(self, end_time: Any | None = None) -> Any:
        pd, _ = _dependencies()
        frame = pd.read_parquet(self.canonical_root / "reference" / "trade_cal.parquet")
        if not {"cal_date", "is_open"}.issubset(frame.columns):
            raise DataContractError("Canonical trade calendar lacks cal_date/is_open")
        values = frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"].astype(str).drop_duplicates()
        calendar = pd.DatetimeIndex(sorted(pd.to_datetime(values, format="%Y%m%d")))
        if end_time is not None:
            calendar = calendar[calendar <= pd.Timestamp(end_time)]
        if calendar.empty:
            raise DataContractError("Qlib calendar is empty")
        return calendar

    @staticmethod
    def _write_calendar(root: Path, calendar: Any, future_calendar: Any) -> None:
        path = root / "calendars" / "day.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{timestamp:%Y-%m-%d}\n" for timestamp in calendar), encoding="utf-8")
        future_path = root / "calendars" / "day_future.txt"
        future_path.write_text(
            "".join(f"{timestamp:%Y-%m-%d}\n" for timestamp in future_calendar),
            encoding="utf-8",
        )

    def _all_intervals(self, calendar: Any) -> list[tuple[str, Any, Any]]:
        pd, _ = _dependencies()
        stocks = pd.read_parquet(self.canonical_root / "reference" / "stock_basic.parquet")
        stocks = stocks.drop_duplicates("ts_code", keep="last")
        intervals = []
        for row in stocks.itertuples(index=False):
            start = pd.to_datetime(str(row.list_date), format="%Y%m%d", errors="coerce")
            end = pd.to_datetime(str(getattr(row, "delist_date", None)), format="%Y%m%d", errors="coerce")
            start = max(start, calendar[0]) if not pd.isna(start) else calendar[0]
            end = min(end, calendar[-1]) if not pd.isna(end) else calendar[-1]
            if start <= end:
                intervals.append((ts_code_to_qlib(row.ts_code), start, end))
        return sorted(intervals)

    def _csi300_intervals(self, calendar: Any) -> list[tuple[str, Any, Any]]:
        pd, _ = _dependencies()
        snapshots = pd.read_parquet(self.canonical_root / "reference" / "csi300_snapshots.parquet")
        if snapshots.empty:
            raise DataContractError("CSI300 snapshots are empty")
        snapshots["snapshot_date"] = pd.to_datetime(snapshots["trade_date"].astype(str), format="%Y%m%d")
        dates = sorted(snapshots["snapshot_date"].unique())
        by_date = {
            value: set(snapshots.loc[snapshots["snapshot_date"].eq(value), "con_code"].map(ts_code_to_qlib))
            for value in dates
        }
        raw: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        for index, snapshot_date in enumerate(dates):
            start = max(pd.Timestamp(snapshot_date), calendar[0])
            if index + 1 < len(dates):
                previous = calendar[calendar < pd.Timestamp(dates[index + 1])]
                if previous.empty:
                    continue
                end = previous[-1]
            else:
                end = calendar[-1]
            for instrument in by_date[snapshot_date]:
                raw[instrument].append((start, end))

        merged = []
        for instrument, values in raw.items():
            values.sort()
            current_start, current_end = values[0]
            for start, end in values[1:]:
                later = calendar[calendar > current_end]
                next_day = later[0] if len(later) else current_end
                if start <= next_day:
                    current_end = max(current_end, end)
                else:
                    merged.append((instrument, current_start, current_end))
                    current_start, current_end = start, end
            merged.append((instrument, current_start, current_end))
        return sorted(merged)

    @staticmethod
    def _write_instruments(root: Path, market: str, intervals: Iterable[tuple[str, Any, Any]]) -> None:
        path = root / "instruments" / f"{market}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f"{instrument}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}\n" for instrument, start, end in intervals),
            encoding="utf-8",
        )

    def _write_instrument(self, root: Path, instrument: str, frame: Any, calendar: Any, lookup: dict[Any, int]) -> None:
        _, np = _dependencies()
        frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last").set_index("datetime")
        valid_dates = frame.index.intersection(calendar)
        if valid_dates.empty:
            return
        start_index = lookup[valid_dates.min()]
        end_index = lookup[valid_dates.max()]
        dense = frame.reindex(calendar[start_index : end_index + 1])
        directory = root / "features" / instrument.lower()
        directory.mkdir(parents=True, exist_ok=True)
        for field in QLIB_FIELDS:
            values = (
                dense[field].astype(float).to_numpy(dtype=np.float32)
                if field in dense.columns
                else np.full(len(dense), np.nan, dtype=np.float32)
            )
            np.concatenate((np.asarray([start_index], dtype=np.float32), values)).tofile(directory / f"{field}.day.bin")

    def _write_benchmark(self, root: Path, calendar: Any, lookup: dict[Any, int]) -> str | None:
        pd, np = _dependencies()
        path = self.canonical_root / "reference" / "index_daily.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        frame = frame[frame["ts_code"].astype(str).isin(["000300.SH", "399300.SZ"])]
        if frame.empty:
            return None
        frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last")
        first_close = float(frame["close"].iloc[0])
        factor = 1.0 / first_close
        benchmark = "SH000300"
        prepared = pd.DataFrame(
            {
                "datetime": frame["datetime"],
                "open": frame["open"].astype(float) * factor,
                "high": frame["high"].astype(float) * factor,
                "low": frame["low"].astype(float) * factor,
                "close": frame["close"].astype(float) * factor,
                "volume": frame["vol"].astype(float) * 100.0 if "vol" in frame else np.nan,
                "amount": frame["amount"].astype(float) * 1_000.0 if "amount" in frame else np.nan,
                "vwap": frame["close"].astype(float) * factor,
                "factor": factor,
                "change": frame["pct_chg"].astype(float) / 100.0 if "pct_chg" in frame else 0.0,
                "limit_buy": False,
                "limit_sell": False,
            }
        )
        self._write_instrument(root, benchmark, prepared, calendar, lookup)
        return benchmark


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas, pyarrow and NumPy are required to build the Qlib provider") from exc
    return pd, np
