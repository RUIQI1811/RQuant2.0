from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rquant.data.contracts import CANONICAL_BAR_COLUMNS
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_replace_directory, atomic_write_json, stable_hash


class CanonicalBuilder:
    """Transform immutable Tushare artifacts into one audited daily contract."""

    def __init__(self, raw_root: str | Path, canonical_root: str | Path) -> None:
        self.raw_root = Path(raw_root)
        self.canonical_root = Path(canonical_root)

    def build(self) -> dict[str, Any]:
        pd, np = _dependencies()
        sync_manifest = self.raw_root / "sync_manifest.json"
        if not sync_manifest.exists():
            raise DataContractError(f"Raw sync manifest does not exist: {sync_manifest}")
        with sync_manifest.open("r", encoding="utf-8") as handle:
            raw_manifest = json.load(handle)
        if raw_manifest.get("status") != "complete":
            raise DataContractError("Raw sync is not complete")

        stock_basic = self._read_endpoint("stock_basic")
        industry = self._read_industry_membership(raw_manifest.get("industry_snapshot"))
        index_weight = self._read_endpoint("index_weight")
        trade_cal = self._read_endpoint("trade_cal")
        namechange = self._read_namechange(raw_manifest.get("namechange_snapshot"))

        bars = self._read_daily_bars()
        bars = bars.sort_values(["datetime", "instrument"]).reset_index(drop=True)
        # Rows are in date order within each instrument, so "first" is the earliest quote.
        first = bars.groupby("instrument", sort=False).agg(
            first_adj=("adj_factor", "first"), first_close=("close", "first")
        )
        bars["factor"] = bars["adj_factor"] / (
            bars["instrument"].map(first["first_adj"]) * bars["instrument"].map(first["first_close"])
        )

        bars["raw_open"] = bars["open"]
        bars["raw_high"] = bars["high"]
        bars["raw_low"] = bars["low"]
        bars["raw_close"] = bars["close"]
        bars["raw_pre_close"] = bars["pre_close"]
        bars["raw_volume"] = bars["vol"] * 100.0
        bars["raw_amount"] = bars["amount"] * 1_000.0
        bars["raw_vwap"] = bars["raw_amount"] / bars["raw_volume"].replace(0.0, np.nan)

        for column in ("open", "high", "low", "close"):
            bars[column] = bars[f"raw_{column}"] * bars["factor"]
        bars["volume"] = bars["raw_volume"] / bars["factor"]
        bars["amount"] = bars["raw_amount"]
        bars["vwap"] = bars["raw_vwap"] * bars["factor"]
        bars["change"] = bars["raw_close"] / bars["raw_pre_close"] - 1.0
        bars["limit_buy"] = bars["up_limit"].notna() & bars["raw_open"].round(2).ge(bars["up_limit"].round(2))
        bars["limit_sell"] = bars["down_limit"].notna() & bars["raw_open"].round(2).le(bars["down_limit"].round(2))
        for column in ("pre_close", "vol", "adj_factor", "up_limit", "down_limit"):
            del bars[column]

        bars = self._attach_index_membership(bars, index_weight)
        bars = self._attach_industry(bars, industry)
        bars = self._attach_listing_age(bars, stock_basic, trade_cal)
        bars = self._attach_st(bars, namechange)
        invalid_ohlc = self._invalid_ohlc_mask(bars)
        invalid_research_rows = invalid_ohlc & bars["in_csi300"]
        if invalid_research_rows.any():
            examples = self._format_ohlc_examples(bars.loc[invalid_research_rows])
            raise DataContractError(
                f"Canonical bars contain {int(invalid_research_rows.sum())} invalid OHLC rows inside CSI300: "
                f"{examples}"
            )
        ohlc_exclusions = bars.loc[
            invalid_ohlc,
            ["ts_code", "instrument", "datetime", "raw_open", "raw_high", "raw_low", "raw_close"],
        ].copy()
        ohlc_exclusions["reason"] = "invalid_ohlc_outside_csi300"
        # Excluded rows are dropped per year instead of filtering (and copying) the full-history frame.
        kept = ~invalid_ohlc
        if not kept.any():
            raise DataContractError("Canonical bars are empty")

        # Publish all yearly partitions at once so a narrower rebuild cannot leave stale years behind.
        bars_root = self.canonical_root / "bars"
        self.canonical_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".bars.staging.", dir=self.canonical_root))
        try:
            for year, frame in bars.groupby(bars["datetime"].dt.year, sort=True):
                frame = frame.loc[kept.loc[frame.index]]
                if frame.empty:
                    continue
                self._validate_bars(frame)
                destination = staging / f"year={year}" / "bars.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.loc[:, CANONICAL_BAR_COLUMNS].to_parquet(destination, index=False)
            atomic_replace_directory(staging, bars_root)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        reference_root = self.canonical_root / "reference"
        reference_root.mkdir(parents=True, exist_ok=True)
        stock_basic.to_parquet(reference_root / "stock_basic.parquet", index=False)
        industry.to_parquet(reference_root / "industry_membership.parquet", index=False)
        index_weight.to_parquet(reference_root / "csi300_snapshots.parquet", index=False)
        trade_cal.to_parquet(reference_root / "trade_cal.parquet", index=False)
        if namechange is not None:
            namechange.to_parquet(reference_root / "namechange.parquet", index=False)
        self._build_index_daily(reference_root)

        quality_root = self.canonical_root / "quality"
        quality_root.mkdir(parents=True, exist_ok=True)
        ohlc_exclusions.to_parquet(quality_root / "ohlc_exclusions.parquet", index=False)

        quality = {
            "excluded_invalid_ohlc_rows": len(ohlc_exclusions),
            "excluded_invalid_ohlc_fingerprint": stable_hash(
                ohlc_exclusions.assign(datetime=ohlc_exclusions["datetime"].astype(str)).to_dict("records")
            ),
        }

        rows = int(kept.sum())
        kept_dates = bars.loc[kept, "datetime"]
        manifest = {
            "status": "complete",
            "raw_fingerprint": raw_manifest.get("fingerprint"),
            "rows": rows,
            "instruments": int(bars.loc[kept, "instrument"].nunique()),
            "start": kept_dates.min().date().isoformat(),
            "end": kept_dates.max().date().isoformat(),
            "columns": list(CANONICAL_BAR_COLUMNS),
            "st_history": namechange is not None,
            "quality": quality,
            "fingerprint": stable_hash(
                {
                    "raw": raw_manifest.get("fingerprint"),
                    "rows": rows,
                    "columns": list(CANONICAL_BAR_COLUMNS),
                    "start": str(kept_dates.min()),
                    "end": str(kept_dates.max()),
                    "quality": quality,
                }
            ),
        }
        atomic_write_json(self.canonical_root / "manifest.json", manifest)
        return manifest

    def _read_daily_bars(self) -> Any:
        """Join daily, adj_factor, daily_basic and stk_limit one raw partition at a time.

        The per-day endpoints share partition names, so joining partition by partition gives the same rows as joining
        the concatenated endpoints, without holding every endpoint's full history (and its per-row key strings) in
        memory at once.
        """
        pd, np = _dependencies()
        keys = ["ts_code", "trade_date"]
        required_daily = [*keys, "open", "high", "low", "close", "pre_close", "vol", "amount"]
        daily_paths = sorted((self.raw_root / "daily").glob("**/data.parquet"))
        if not daily_paths:
            raise DataContractError("No raw artifacts for endpoint: daily")
        # One shared string object per code keeps the object columns at pointer size across ~15M rows.
        ts_codes: dict[str, str] = {}
        instruments: dict[str, str] = {}
        frames = []
        for daily_path in daily_paths:
            partition = daily_path.parent.relative_to(self.raw_root / "daily")
            daily = pd.read_parquet(daily_path).drop_duplicates()
            if daily.empty:
                continue
            missing = set(required_daily) - set(daily.columns)
            if missing:
                raise DataContractError(f"daily is missing fields: {sorted(missing)}")
            bars = daily[required_daily].drop_duplicates(keys, keep="last")

            adjustment = self._read_partition("adj_factor", partition)
            if adjustment is not None and not adjustment.empty:
                if not {"ts_code", "trade_date", "adj_factor"}.issubset(adjustment.columns):
                    raise DataContractError("adj_factor response lacks ts_code/trade_date/adj_factor")
                adjustment = adjustment.drop_duplicates(keys, keep="last")
                bars = bars.merge(adjustment[[*keys, "adj_factor"]], on=keys, how="left")
            else:
                bars["adj_factor"] = np.nan

            daily_basic = self._read_partition("daily_basic", partition)
            if daily_basic is not None and "total_mv" in daily_basic.columns:
                cap = daily_basic[[*keys, "total_mv"]].drop_duplicates(keys, keep="last")
                cap["cap"] = pd.to_numeric(cap["total_mv"], errors="coerce") * 10_000.0
                bars = bars.merge(cap[[*keys, "cap"]], on=keys, how="left")
            else:
                bars["cap"] = np.nan

            limits = self._read_partition("stk_limit", partition)
            limit_columns = [*keys, "up_limit", "down_limit"]
            if limits is not None and set(limit_columns).issubset(limits.columns):
                bars = bars.merge(limits[limit_columns].drop_duplicates(keys, keep="last"), on=keys, how="left")
            else:
                bars["up_limit"] = np.nan
                bars["down_limit"] = np.nan

            for code in bars["ts_code"].unique():
                if code not in ts_codes:
                    ts_codes[code] = code
                    instruments[code] = ts_code_to_qlib(code)
            bars["instrument"] = bars["ts_code"].map(instruments)
            bars["ts_code"] = bars["ts_code"].map(ts_codes)
            bars["datetime"] = pd.to_datetime(bars["trade_date"], format="%Y%m%d", errors="raise")
            del bars["trade_date"]
            for column in (
                "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor", "up_limit", "down_limit"
            ):
                bars[column] = pd.to_numeric(bars[column], errors="coerce")
            frames.append(bars)
        if not frames:
            raise DataContractError("No raw artifacts for endpoint: daily")
        return pd.concat(frames, ignore_index=True, sort=False)

    def _read_partition(self, endpoint: str, partition: Path) -> Any | None:
        pd, _ = _dependencies()
        path = self.raw_root / endpoint / partition / "data.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path).drop_duplicates()

    def _read_endpoint(self, endpoint: str) -> Any:
        pd, _ = _dependencies()
        paths = sorted((self.raw_root / endpoint).glob("**/data.parquet"))
        if not paths:
            raise DataContractError(f"No raw artifacts for endpoint: {endpoint}")
        frames = [pd.read_parquet(path) for path in paths]
        populated = [frame for frame in frames if not frame.empty]
        if populated:
            return pd.concat(populated, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
        with_schema = [frame for frame in frames if len(frame.columns) > 0]
        if not with_schema:
            return pd.DataFrame()
        return with_schema[0].iloc[0:0].copy()

    def _read_industry_membership(self, snapshot: str | None = None) -> Any:
        """Read the N/Y industry snapshot named by the complete sync manifest, with a legacy fallback."""
        pd, _ = _dependencies()
        endpoint_root = self.raw_root / "index_member_all"
        if snapshot:
            snapshot_root = endpoint_root / f"snapshot={snapshot.replace('-', '')}"
            paths = sorted(snapshot_root.glob("**/data.parquet"))
            if not paths:
                raise DataContractError(f"Industry snapshot from sync manifest does not exist: {snapshot_root}")
        else:
            paths = sorted(endpoint_root.glob("**/data.parquet"))
        if not paths:
            raise DataContractError("No raw artifacts for endpoint: index_member_all")
        frames = [pd.read_parquet(path) for path in paths]
        populated = [frame for frame in frames if not frame.empty]
        if populated:
            return pd.concat(populated, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
        with_schema = [frame for frame in frames if len(frame.columns) > 0]
        if not with_schema:
            return pd.DataFrame()
        return with_schema[0].iloc[0:0].copy()

    def _attach_index_membership(self, bars: Any, snapshots: Any) -> Any:
        """Add ``in_csi300`` in place from the latest CSI300 snapshot on or before each bar's date."""
        pd, _ = _dependencies()
        if snapshots.empty or not {"trade_date", "con_code"}.issubset(snapshots.columns):
            bars["in_csi300"] = False
            return bars
        membership = snapshots[["trade_date", "con_code"]].drop_duplicates().copy()
        membership["snapshot_date"] = pd.to_datetime(membership["trade_date"], format="%Y%m%d")
        membership["instrument"] = membership["con_code"].map(ts_code_to_qlib)
        snapshot_dates = pd.DataFrame({"snapshot_date": sorted(membership["snapshot_date"].unique())})
        date_map = pd.DataFrame({"datetime": sorted(bars["datetime"].unique())})
        date_map = pd.merge_asof(
            date_map, snapshot_dates, left_on="datetime", right_on="snapshot_date", direction="backward"
        )
        row_snapshot = bars["datetime"].map(date_map.set_index("datetime")["snapshot_date"])
        members = pd.MultiIndex.from_frame(membership[["snapshot_date", "instrument"]])
        bars["in_csi300"] = pd.MultiIndex.from_arrays([row_snapshot, bars["instrument"]]).isin(members)
        return bars

    def _read_namechange(self, snapshot: str | None) -> Any | None:
        """Read the stock name history named by the sync manifest; None when the sync predates namechange."""
        pd, _ = _dependencies()
        if not snapshot:
            return None
        snapshot_root = self.raw_root / "namechange" / f"snapshot={snapshot.replace('-', '')}"
        # One partition per stock; older paged partitions in the same snapshot are incomplete and ignored.
        paths = sorted(snapshot_root.glob("ts_code=*/data.parquet"))
        if not paths:
            raise DataContractError(f"Name-change snapshot from sync manifest does not exist: {snapshot_root}")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)
        missing = {"ts_code", "name", "start_date"} - set(frame.columns)
        if missing:
            raise DataContractError(f"namechange is missing fields: {sorted(missing)}")
        if frame.empty:
            raise DataContractError("namechange history is empty")
        return frame.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _attach_listing_age(bars: Any, stock_basic: Any, trade_cal: Any) -> Any:
        """Add ``listed_days``: open trading days from the listing date through each bar's date, inclusive.

        Listings before the first synced calendar day are aged with weekday counts for the part before the calendar.
        A stock without a usable listing date is aged from its first canonical bar.
        """
        pd, np = _dependencies()
        open_days = trade_cal.loc[trade_cal["is_open"].astype(int).eq(1), "cal_date"].astype(str).drop_duplicates()
        calendar = np.sort(pd.to_datetime(open_days, format="%Y%m%d").to_numpy(dtype="datetime64[D]"))
        if not len(calendar):
            raise DataContractError("Trade calendar has no open days")
        listing = pd.Series(dtype="datetime64[ns]")
        if {"ts_code", "list_date"}.issubset(stock_basic.columns):
            stocks = stock_basic[["ts_code", "list_date"]].dropna().drop_duplicates("ts_code", keep="last")
            listing = pd.Series(
                pd.to_datetime(stocks["list_date"].astype(str), format="%Y%m%d", errors="coerce").to_numpy(),
                index=stocks["ts_code"].map(ts_code_to_qlib).to_numpy(),
            ).dropna()
        first_bar = bars.groupby("instrument", sort=False)["datetime"].min()
        listed = first_bar.index.to_series().map(listing).fillna(first_bar)
        listed = listed.to_numpy(dtype="datetime64[D]")
        list_position = np.searchsorted(calendar, listed, side="left")
        before_calendar = np.busday_count(np.minimum(listed, calendar[0]), calendar[0])
        offsets = pd.Series(list_position - before_calendar, index=first_bar.index)
        bar_position = np.searchsorted(calendar, bars["datetime"].to_numpy(dtype="datetime64[D]"), side="left")
        bars["listed_days"] = (bar_position - bars["instrument"].map(offsets).to_numpy() + 1).astype("int32")
        return bars

    @staticmethod
    def _attach_st(bars: Any, namechange: Any | None) -> Any:
        """Add ``is_st`` from the name in force on each bar's date: ST, *ST, PT and delisting (退) names count."""
        pd, _ = _dependencies()
        if namechange is None:
            bars["is_st"] = pd.Series(pd.NA, index=bars.index, dtype="boolean")
            return bars
        names = namechange[["ts_code", "name", "start_date", *(["ann_date"] if "ann_date" in namechange else [])]]
        names = names[names["ts_code"].astype(str).str.upper().str.endswith((".SH", ".SZ", ".BJ"))]
        names = names.assign(
            instrument=names["ts_code"].map(ts_code_to_qlib),
            start=pd.to_datetime(names["start_date"].astype(str), format="%Y%m%d", errors="coerce"),
        ).dropna(subset=["start"])
        # A later announcement for the same effective date corrects the earlier one.
        sort_keys = ["instrument", "start", *(["ann_date"] if "ann_date" in names else [])]
        names = names.sort_values(sort_keys).drop_duplicates(["instrument", "start"], keep="last")
        flagged = names["name"].astype(str).str.upper().str.contains(r"ST|^PT|退", regex=True)
        names = pd.DataFrame({"instrument": names["instrument"], "start": names["start"], "is_st": flagged.to_numpy()})
        rows = bars[["instrument", "datetime"]].reset_index().sort_values("datetime")
        matched = pd.merge_asof(
            rows,
            names.sort_values("start"),
            left_on="datetime",
            right_on="start",
            by="instrument",
            direction="backward",
        )
        bars["is_st"] = pd.Series(matched["is_st"].fillna(False).to_numpy(dtype=bool), index=matched["index"]).astype(
            "boolean"
        )
        return bars

    def _attach_industry(self, bars: Any, industry: Any) -> Any:
        """Add point-in-time sector/industry/subindustry codes to ``bars`` in place."""
        pd, np = _dependencies()
        outputs = {output: np.full(len(bars), np.nan) for output in ("sector", "industry", "subindustry")}
        required = {"ts_code", "in_date", "out_date", "l1_code", "l2_code", "l3_code"}
        if industry.empty or not required.issubset(industry.columns):
            for output, values in outputs.items():
                bars[output] = values
            return bars

        history = industry[list(required)].drop_duplicates().copy()
        history["instrument"] = history["ts_code"].map(ts_code_to_qlib)
        history["in_date"] = pd.to_datetime(history["in_date"], format="%Y%m%d", errors="coerce")
        history["out_date"] = pd.to_datetime(history["out_date"], format="%Y%m%d", errors="coerce").fillna(
            pd.Timestamp.max.normalize()
        )
        levels = {"sector": "l1_code", "industry": "l2_code", "subindustry": "l3_code"}
        code_maps = {
            output: {
                code: index + 1 for index, code in enumerate(sorted(history[source].dropna().astype(str).unique()))
            }
            for output, source in levels.items()
        }
        all_dates = bars["datetime"].to_numpy(dtype="datetime64[ns]")
        for instrument, rows in bars.groupby("instrument", sort=False).indices.items():
            intervals = history[history["instrument"].eq(instrument)].sort_values("in_date")
            if intervals.empty:
                continue
            dates = all_dates[rows]
            starts = intervals["in_date"].to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(starts, dates, side="right") - 1
            valid_position = positions >= 0
            safe_positions = positions.clip(min=0)
            ends = intervals["out_date"].to_numpy(dtype="datetime64[ns]")[safe_positions]
            valid = valid_position & (dates <= ends)
            for output, source in levels.items():
                values = intervals[source].astype(str).map(code_maps[output]).to_numpy(dtype=float)[safe_positions]
                values[~valid] = np.nan
                outputs[output][rows] = values
        for output, values in outputs.items():
            bars[output] = values
        return bars

    def _build_index_daily(self, reference_root: Path) -> None:
        pd, _ = _dependencies()
        frame = self._read_endpoint("index_daily")
        if frame.empty:
            return
        frame["datetime"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
        frame["instrument"] = frame["ts_code"].map(ts_code_to_qlib)
        frame.to_parquet(reference_root / "index_daily.parquet", index=False)

    @staticmethod
    def _invalid_ohlc_mask(bars: Any) -> Any:
        return bars["raw_high"].lt(bars[["raw_open", "raw_close", "raw_low"]].max(axis=1)) | bars["raw_low"].gt(
            bars[["raw_open", "raw_close", "raw_high"]].min(axis=1)
        )

    @staticmethod
    def _format_ohlc_examples(rows: Any, limit: int = 5) -> str:
        return ", ".join(
            f"{row.instrument}@{row.datetime:%Y-%m-%d}"
            for row in rows.head(limit).itertuples(index=False)
        )

    @staticmethod
    def _validate_bars(bars: Any) -> None:
        pd, np = _dependencies()
        if bars.empty:
            raise DataContractError("Canonical bars are empty")
        if bars.duplicated(["instrument", "datetime"]).any():
            raise DataContractError("Canonical bars contain duplicate instrument/datetime rows")
        invalid_ohlc = CanonicalBuilder._invalid_ohlc_mask(bars)
        if invalid_ohlc.any():
            examples = CanonicalBuilder._format_ohlc_examples(bars.loc[invalid_ohlc])
            raise DataContractError(
                f"Canonical bars contain {int(invalid_ohlc.sum())} invalid OHLC rows: {examples}"
            )
        # Tushare daily rows exist only for traded days, so every canonical bar is a live quote.
        if bars["factor"].isna().any() or bars["factor"].le(0).any():
            raise DataContractError("Live quotes require a positive restoration factor")
        restored = bars["close"] / bars["factor"]
        if not np.allclose(restored, bars["raw_close"], rtol=1e-10, atol=1e-10, equal_nan=True):
            raise DataContractError("Qlib close/factor does not restore the raw close")
        if tuple(column for column in CANONICAL_BAR_COLUMNS if column not in bars.columns):
            missing = [column for column in CANONICAL_BAR_COLUMNS if column not in bars.columns]
            raise DataContractError(f"Canonical columns missing: {missing}")


def ts_code_to_qlib(ts_code: str) -> str:
    value = str(ts_code).upper()
    if "." not in value:
        raise DataContractError(f"Invalid Tushare code: {ts_code}")
    symbol, exchange = value.split(".", 1)
    if exchange not in {"SH", "SZ", "BJ"}:
        raise DataContractError(f"Unsupported A-share exchange: {ts_code}")
    return f"{exchange}{symbol}"


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas, pyarrow and NumPy are required for canonical data") from exc
    return pd, np
