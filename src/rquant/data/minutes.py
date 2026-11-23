"""Tushare minute bars for the intraday-execution backtest (``rquant.workflow.intraday``).

Only the stock-days a daily run can trade are fetched, not the whole market: for every trading day, the top ``top``
stocks of the previous day's predictions (the buy candidates and the stocks a TopkDropout book keeps) plus the stocks
the daily backtest held or traded that day or the day before. Requests are one stock and one calendar month each
(about 5,500 one-minute rows, below Tushare's 8,000-row page).

``MinuteBarBuilder`` turns the raw responses into one Parquet file per month with Qlib's minute layout: 240 bars a
day labelled by their start (09:30-11:29, 13:00-14:59), prices restored with the canonical daily ``factor`` exactly as
the daily provider restores them, minute limit flags from the daily ``stk_limit`` prices, and a check of the minute
amounts against the daily bars. It also writes the 1-minute calendar Qlib's inner executor steps through; no
per-minute feature bins are written, so the backtest reads quotes from the monthly files directly.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from rquant.data.canonical import ts_code_to_qlib
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, stable_hash

MINUTE_ENDPOINT = "stk_mins"
MINUTE_FREQ = "1min"
# Tushare pages hold at most 8,000 rows; a month of one stock is about 23 x 241.
MINUTE_ROW_GUARD = 8000
MINUTE_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "amount",
    "factor",
    "change",
    "limit_buy",
    "limit_sell",
)
PLAN_FILE = "minute_plan.parquet"
SYNC_MANIFEST = "stk_mins_manifest.json"


def session_minutes() -> list[time]:
    """Qlib's 240 China A-share minute labels: the start of each one-minute bar."""
    labels = []
    for first, count in ((datetime(2000, 1, 1, 9, 30), 120), (datetime(2000, 1, 1, 13, 0), 120)):
        labels.extend((first + timedelta(minutes=offset)).time() for offset in range(count))
    return labels


SESSION_MINUTES = session_minutes()


def qlib_to_ts_code(instrument: str) -> str:
    value = str(instrument).upper()
    exchange, symbol = value[:2], value[2:]
    if exchange not in {"SH", "SZ", "BJ"} or not symbol.isdigit():
        raise DataContractError(f"Invalid Qlib instrument: {instrument}")
    return f"{symbol}.{exchange}"


def snap_to_session(stamps: Any) -> Any:
    """Map Tushare bar end times to Qlib bar start labels.

    A bar that ends at ``t`` starts at ``t - 1 minute``. Bars that then fall outside the session are merged into the
    nearest session bar: the opening-auction print (09:25-09:30) into 09:30, a 13:00 print into 13:00, and anything
    after 11:29 or 14:59 into the last bar of its session.
    """
    pd, _ = _dependencies()
    starts = pd.DatetimeIndex(stamps) - pd.Timedelta(minutes=1)
    day = starts.normalize()
    offset = (starts - day) / pd.Timedelta(minutes=1)
    morning_open, morning_last = 9 * 60 + 30, 11 * 60 + 29
    afternoon_open, afternoon_last = 13 * 60, 14 * 60 + 59
    minutes = offset.to_numpy(dtype=float)
    import numpy as np

    snapped = np.where(minutes < morning_open, morning_open, minutes)
    snapped = np.where(
        (snapped > morning_last) & (snapped < (morning_last + afternoon_open) / 2), morning_last, snapped
    )
    snapped = np.where(
        (snapped >= (morning_last + afternoon_open) / 2) & (snapped < afternoon_open), afternoon_open, snapped
    )
    snapped = np.where(snapped > afternoon_last, afternoon_last, snapped)
    return day + pd.to_timedelta(snapped, unit="min")


def minute_calendar(days: Iterable[Any]) -> Any:
    """Every Qlib minute label of ``days``."""
    pd, _ = _dependencies()
    offsets = pd.to_timedelta([m.hour * 60 + m.minute for m in SESSION_MINUTES], unit="min")
    stamps = [pd.Timestamp(day).normalize() + offset for day in sorted(set(days)) for offset in offsets]
    return pd.DatetimeIndex(stamps)


def plan_minute_requests(
    predictions: Any,
    calendar: Any,
    *,
    top: int,
    positions: Any | None = None,
    trades: Any | None = None,
    start: Any | None = None,
    end: Any | None = None,
) -> Any:
    """The stock-days whose minute bars the intraday backtest of a run can need, as (instrument, day) rows.

    ``predictions`` holds ``datetime, instrument, score``; a prediction dated on one trading day is traded on the next
    (``calendar`` is the daily provider calendar). ``positions`` (``datetime, instrument``) and ``trades`` are the daily
    backtest's ``positions.parquet`` and ``trades.parquet``: a held stock is needed on its day and on the next one, when
    it may be sold.
    """
    pd, _ = _dependencies()
    if top < 1:
        raise DataContractError("top must be at least 1")
    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(calendar))))
    following = pd.Series(days[1:], index=days[:-1])
    frame = predictions[["datetime", "instrument", "score"]].dropna(subset=["score"]).copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame[frame["datetime"].isin(following.index)]
    ranked = frame.sort_values(["datetime", "score"], ascending=[True, False]).groupby("datetime").head(top)
    needed = [pd.DataFrame({"day": ranked["datetime"].map(following).to_numpy(), "instrument": ranked["instrument"]})]
    if positions is not None and len(positions):
        held = positions[["datetime", "instrument"]].copy()
        held["datetime"] = pd.to_datetime(held["datetime"])
        needed.append(held.rename(columns={"datetime": "day"}))
        later = held[held["datetime"].isin(following.index)]
        needed.append(
            pd.DataFrame({"day": later["datetime"].map(following).to_numpy(), "instrument": later["instrument"]})
        )
    if trades is not None and len(trades):
        traded = trades[["datetime", "instrument"]].copy()
        traded["datetime"] = pd.to_datetime(traded["datetime"])
        needed.append(traded.rename(columns={"datetime": "day"}))
    plan = pd.concat(needed, ignore_index=True).dropna().drop_duplicates()
    if start is not None:
        plan = plan[plan["day"] >= pd.Timestamp(start)]
    if end is not None:
        plan = plan[plan["day"] <= pd.Timestamp(end)]
    plan = plan[plan["instrument"].str.match(r"^(SH|SZ|BJ)\d{6}$")]
    return plan.sort_values(["day", "instrument"]).reset_index(drop=True)


def monthly_requests(plan: Any) -> Any:
    """One request per instrument and month of ``plan``: its first and last needed day in that month."""
    frame = plan.copy()
    frame["month"] = frame["day"].dt.strftime("%Y%m")
    grouped = frame.groupby(["instrument", "month"], sort=True)["day"].agg(["min", "max"]).reset_index()
    return grouped.rename(columns={"min": "first_day", "max": "last_day"})


def sync_minutes(
    collector: Any,
    requests: Any,
    *,
    workers: int = 1,
    force: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Fetch ``stk_mins`` (1-minute) for each request row through ``collector``'s cached, rate-limited partitions.

    A request covers its month from the first trading day of the month through its last needed day, so a later run
    that needs more of the same month refetches it once instead of trusting a shorter cached response.
    """
    from tqdm import tqdm

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise DataContractError("sync workers must be a positive integer")
    rows = list(requests.itertuples(index=False))
    progress = tqdm(
        total=len(rows), desc="Tushare stk_mins", unit="request", disable=not show_progress, file=sys.stderr
    )
    lock = threading.Lock()
    counts = {"cached": 0, "fetched": 0}

    def fetch(row: Any) -> dict[str, Any]:
        ts_code = qlib_to_ts_code(row.instrument)
        month = str(row.month)
        query = {
            "ts_code": ts_code,
            "freq": MINUTE_FREQ,
            "start_date": f"{month[:4]}-{month[4:]}-01 09:00:00",
            "end_date": f"{row.last_day:%Y-%m-%d} 15:30:00",
        }
        partition = f"freq={MINUTE_FREQ}/ts_code={ts_code.replace('.', '_')}/month={month}"
        artifact, cached = collector._fetch_partition(MINUTE_ENDPOINT, query, partition, force=force)
        if artifact.rows >= MINUTE_ROW_GUARD:
            raise DataContractError(
                f"Tushare {MINUTE_ENDPOINT} returned {artifact.rows} rows for {ts_code} in {month}; "
                "the page may be truncated"
            )
        with lock:
            counts["cached" if cached else "fetched"] += 1
            progress.set_postfix_str(f"cached={counts['cached']} fetched={counts['fetched']}", refresh=False)
            progress.update()
        return {
            "instrument": row.instrument,
            "month": month,
            "path": Path(artifact.path).relative_to(collector.raw_root).as_posix(),
            "rows": artifact.rows,
            "sha256": artifact.sha256,
        }

    try:
        if workers == 1:
            files = [fetch(row) for row in rows]
        else:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare-minutes")
            futures = [executor.submit(fetch, row) for row in rows]
            try:
                files = [future.result() for future in futures]
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
    finally:
        progress.close()
    manifest = {
        "status": "complete",
        "endpoint": MINUTE_ENDPOINT,
        "freq": MINUTE_FREQ,
        "requests": len(files),
        "cached": counts["cached"],
        "fetched": counts["fetched"],
        "empty": sum(1 for item in files if item["rows"] == 0),
        "rows": sum(item["rows"] for item in files),
        "files": files,
        "fingerprint": stable_hash([[item["path"], item["sha256"]] for item in files]),
    }
    atomic_write_json(collector.raw_root / SYNC_MANIFEST, manifest)
    return manifest


class MinuteBarBuilder:
    """Normalise the synced minute responses into ``<minute_root>/month=YYYYMM/bars.parquet`` and a minute calendar.

    Prices are raw minute prices times the canonical daily ``factor`` of the same day, volume is raw shares divided by
    it, so minute and daily quotes share one restoration and Qlib's trade unit (100 raw shares) works on both. A bar
    without trades carries the last close forward with zero volume; ``vwap`` is the bar's amount over its volume (its
    close when it has none). ``limit_buy`` / ``limit_sell`` mark bars whose raw close sits at the day's up / down limit.
    """

    def __init__(self, raw_root: str | Path, canonical_root: str | Path, minute_root: str | Path) -> None:
        self.raw_root = Path(raw_root)
        self.canonical_root = Path(canonical_root)
        self.minute_root = Path(minute_root)

    def build(self, *, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        pd, _ = _dependencies()
        manifest_path = self.raw_root / SYNC_MANIFEST
        if not manifest_path.exists():
            raise DataContractError(f"Minute sync manifest does not exist: {manifest_path}")
        sync = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sync.get("status") != "complete":
            raise DataContractError("Minute sync is not complete")
        by_month: dict[str, list[str]] = {}
        for item in sync["files"]:
            by_month.setdefault(item["month"], []).append(item["path"])
        self.minute_root.mkdir(parents=True, exist_ok=True)
        months = []
        for month in sorted(by_month):
            summary = self._build_month(month, by_month[month])
            months.append(summary)
            if log is not None:
                log(
                    f"{month}: {summary['instruments']} instruments, {summary['stock_days']} stock-days, "
                    f"amount ratio median {summary['amount_ratio_median']}"
                )
        calendar_days = self._calendar_days()
        first = pd.Timestamp(f"{months[0]['month']}01") if months else calendar_days[0]
        last_bar = max((pd.Timestamp(item["last_day"]) for item in months if item["last_day"]), default=first)
        self._write_calendar(calendar_days[(calendar_days >= first) & (calendar_days <= last_bar)], "1min")
        # Qlib's step times look one bar past the last one, so the inner executor also needs the future calendar.
        self._write_calendar(calendar_days[calendar_days >= first], "1min_future")
        stock_days = sum(item["stock_days"] for item in months)
        mismatched = sum(item["amount_mismatch_days"] for item in months)
        manifest = {
            "status": "complete",
            "freq": MINUTE_FREQ,
            "sync_fingerprint": sync.get("fingerprint"),
            "months": len(months),
            "stock_days": stock_days,
            "amount_mismatch_days": mismatched,
            "amount_mismatch_share": (mismatched / stock_days) if stock_days else None,
            "volume_units": sorted({item["volume_unit"] for item in months if item["volume_unit"]}),
            "by_month": months,
            "fields": list(MINUTE_FIELDS),
        }
        atomic_write_json(self.minute_root / "manifest.json", manifest)
        return manifest

    def _build_month(self, month: str, paths: list[str]) -> dict[str, Any]:
        pd, np = _dependencies()
        frames = [pd.read_parquet(self.raw_root / path) for path in paths]
        frames = [frame for frame in frames if not frame.empty]
        target = self.minute_root / f"month={month}" / "bars.parquet"
        empty = {
            "month": month,
            "instruments": 0,
            "stock_days": 0,
            "amount_mismatch_days": 0,
            "amount_ratio_median": None,
            "volume_unit": None,
            "last_day": None,
        }
        if not frames:
            return empty
        raw = pd.concat(frames, ignore_index=True)
        required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
        if not required.issubset(raw.columns):
            raise DataContractError(f"stk_mins response lacks {sorted(required - set(raw.columns))}")
        raw = raw.drop_duplicates(["ts_code", "trade_time"], keep="last")
        raw["instrument"] = raw["ts_code"].map(ts_code_to_qlib)
        raw["datetime"] = snap_to_session(pd.to_datetime(raw["trade_time"]))
        bars = (
            raw.sort_values(["instrument", "trade_time"])
            .groupby(["instrument", "datetime"], sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                vol=("vol", "sum"),
                amount=("amount", "sum"),
            )
            .reset_index()
        )
        bars["day"] = bars["datetime"].dt.normalize()
        daily = self._daily(sorted(bars["day"].unique()), sorted(bars["instrument"].unique()))
        unit, ratio = _volume_unit(bars, daily)
        bars["vol"] = bars["vol"] * unit

        # Every session minute of every stock-day with any minute bar and a daily bar.
        keys = bars[["instrument", "day"]].drop_duplicates().merge(daily, on=["instrument", "day"], how="inner")
        minute_offsets = pd.to_timedelta([m.hour * 60 + m.minute for m in SESSION_MINUTES], unit="min")
        grid = keys.loc[keys.index.repeat(len(minute_offsets))].reset_index(drop=True)
        grid["datetime"] = grid["day"] + np.tile(minute_offsets.to_numpy(), len(keys))
        dense = grid.merge(bars.drop(columns=["day"]), on=["instrument", "datetime"], how="left")
        dense = dense.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        group = dense.groupby(["instrument", "day"], sort=False)
        dense["close"] = group["close"].ffill()
        # Minutes before the first print of the day take the first print's open.
        dense["close"] = dense["close"].fillna(group["open"].bfill())
        for column in ("open", "high", "low"):
            dense[column] = dense[column].fillna(dense["close"])
        dense["vol"] = dense["vol"].fillna(0.0)
        dense["amount"] = dense["amount"].fillna(0.0)

        raw_vwap = (dense["amount"] / dense["vol"].where(dense["vol"] > 0)).fillna(dense["close"])
        out = pd.DataFrame({"instrument": dense["instrument"], "datetime": dense["datetime"]})
        factor = dense["factor"].astype(float)
        for column in ("open", "high", "low", "close"):
            out[column] = dense[column].astype(float) * factor
        out["vwap"] = raw_vwap.astype(float) * factor
        out["volume"] = dense["vol"].astype(float) / factor
        out["amount"] = dense["amount"].astype(float)
        out["factor"] = factor
        previous = out.groupby("instrument", sort=False)["close"].shift(1)
        out["change"] = (out["close"] / previous - 1.0).fillna(0.0)
        raw_close = dense["close"].astype(float).round(2)
        out["limit_buy"] = dense["up_limit"].notna() & raw_close.ge(dense["up_limit"].round(2))
        out["limit_sell"] = dense["down_limit"].notna() & raw_close.le(dense["down_limit"].round(2))
        for column in ("open", "high", "low", "close", "vwap", "volume", "amount", "factor", "change"):
            out[column] = out[column].astype("float32")

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(".bars.parquet.tmp")
        out.to_parquet(temporary, index=False)
        temporary.replace(target)

        per_day = dense.groupby(["instrument", "day"], sort=False).agg(
            amount=("amount", "sum"), daily_amount=("raw_amount", "first")
        )
        amount_ratio = per_day["amount"] / per_day["daily_amount"].where(per_day["daily_amount"] > 0)
        mismatch = int((amount_ratio - 1.0).abs().gt(0.05).sum())
        return {
            "month": month,
            "instruments": int(out["instrument"].nunique()),
            "stock_days": len(keys),
            "amount_mismatch_days": mismatch,
            "amount_ratio_median": _rounded(amount_ratio.median()),
            "volume_unit": "shares" if unit == 1.0 else "lots",
            "volume_ratio_median": _rounded(ratio),
            "last_day": str(keys["day"].max().date()) if len(keys) else None,
        }

    def _daily(self, days: list[Any], instruments: list[str]) -> Any:
        """Canonical factor, raw volume and amount and the stk_limit prices of ``instruments`` on ``days``."""
        pd, _ = _dependencies()
        years = sorted({pd.Timestamp(day).year for day in days})
        columns = ["instrument", "datetime", "factor", "raw_volume", "raw_amount"]
        frames = []
        for year in years:
            path = self.canonical_root / "bars" / f"year={year}" / "bars.parquet"
            if not path.exists():
                raise DataContractError(f"Canonical bars are missing for {year}: {path}")
            frame = pd.read_parquet(path, columns=columns)
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            frames.append(frame[frame["datetime"].isin(days) & frame["instrument"].isin(instruments)])
        daily = pd.concat(frames, ignore_index=True).rename(columns={"datetime": "day"})
        limits = []
        for day in days:
            path = self.raw_root / "stk_limit" / f"trade_date={pd.Timestamp(day):%Y%m%d}" / "data.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["ts_code", "up_limit", "down_limit"])
            frame["instrument"] = frame["ts_code"].map(ts_code_to_qlib)
            frame["day"] = pd.Timestamp(day)
            limits.append(frame[frame["instrument"].isin(instruments)].drop(columns=["ts_code"]))
        if limits:
            daily = daily.merge(pd.concat(limits, ignore_index=True), on=["instrument", "day"], how="left")
        else:
            daily["up_limit"] = float("nan")
            daily["down_limit"] = float("nan")
        return daily.dropna(subset=["factor"])

    def _calendar_days(self) -> Any:
        pd, _ = _dependencies()
        frame = pd.read_parquet(self.canonical_root / "reference" / "trade_cal.parquet")
        values = frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"].astype(str).drop_duplicates()
        return pd.DatetimeIndex(sorted(pd.to_datetime(values, format="%Y%m%d")))

    def _write_calendar(self, days: Any, name: str) -> None:
        """``<minute_root>/qlib/calendars/<name>.txt``: the minute steps of Qlib's inner executor."""
        stamps = minute_calendar(days)
        path = self.minute_root / "qlib" / "calendars" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{stamp:%Y-%m-%d %H:%M:%S}\n" for stamp in stamps), encoding="utf-8")
        instruments = self.minute_root / "qlib" / "instruments" / "all.txt"
        instruments.parent.mkdir(parents=True, exist_ok=True)
        instruments.write_text("", encoding="utf-8")


def load_minute_quotes(minute_root: str | Path, start: Any, end: Any, codes: Iterable[str] | None = None) -> Any:
    """Minute bars of ``codes`` (all when None) from ``start`` through ``end`` from the monthly files."""
    pd, _ = _dependencies()
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    months = pd.period_range(first.to_period("M"), last.to_period("M"), freq="M")
    wanted = set(codes) if codes is not None else None
    frames = []
    for month in months:
        path = Path(minute_root) / f"month={month.strftime('%Y%m')}" / "bars.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[(frame["datetime"] >= first) & (frame["datetime"] <= last)]
        if wanted is not None:
            frame = frame[frame["instrument"].isin(wanted)]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["instrument", "datetime", *MINUTE_FIELDS])
    return pd.concat(frames, ignore_index=True)


def _volume_unit(bars: Any, daily: Any) -> tuple[float, float | None]:
    """1.0 when ``stk_mins`` volume is in shares, 100.0 when in lots, from the median ratio to the daily volume."""
    per_day = bars.groupby(["instrument", "day"])["vol"].sum().rename("minute").reset_index()
    joined = per_day.merge(daily[["instrument", "day", "raw_volume"]], on=["instrument", "day"], how="inner")
    ratio = (joined["minute"] / joined["raw_volume"].where(joined["raw_volume"] > 0)).median()
    if ratio != ratio:
        return 1.0, None
    if 0.5 <= ratio <= 2.0:
        return 1.0, float(ratio)
    if 0.005 <= ratio <= 0.02:
        return 100.0, float(ratio)
    raise DataContractError(f"Minute volume does not match daily volume (median ratio {ratio:.4f})")


def _rounded(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if number == number else None


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas, pyarrow and NumPy are required for minute data") from exc
    return pd, np
