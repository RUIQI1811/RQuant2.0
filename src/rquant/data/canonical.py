from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rquant.data.contracts import CANONICAL_BAR_COLUMNS
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, stable_hash


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

        daily = self._read_endpoint("daily")
        adjustment = self._read_endpoint("adj_factor")
        daily_basic = self._read_endpoint("daily_basic")
        limits = self._read_endpoint("stk_limit")
        stock_basic = self._read_endpoint("stock_basic")
        industry = self._read_industry_membership(raw_manifest.get("industry_snapshot"))
        index_weight = self._read_endpoint("index_weight")

        required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"}
        if not required_daily.issubset(daily.columns):
            raise DataContractError(f"daily is missing fields: {sorted(required_daily - set(daily.columns))}")
        if not {"ts_code", "trade_date", "adj_factor"}.issubset(adjustment.columns):
            raise DataContractError("adj_factor response lacks ts_code/trade_date/adj_factor")

        daily = daily.drop_duplicates(["ts_code", "trade_date"], keep="last")
        adjustment = adjustment.drop_duplicates(["ts_code", "trade_date"], keep="last")
        bars = daily.merge(
            adjustment[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left"
        )
        if "total_mv" in daily_basic.columns:
            cap = daily_basic[["ts_code", "trade_date", "total_mv"]].drop_duplicates(
                ["ts_code", "trade_date"], keep="last"
            )
            cap["cap"] = pd.to_numeric(cap["total_mv"], errors="coerce") * 10_000.0
            bars = bars.merge(cap[["ts_code", "trade_date", "cap"]], on=["ts_code", "trade_date"], how="left")
        else:
            bars["cap"] = np.nan

        limit_columns = [
            column for column in ("ts_code", "trade_date", "up_limit", "down_limit") if column in limits.columns
        ]
        if set(limit_columns) == {"ts_code", "trade_date", "up_limit", "down_limit"}:
            bars = bars.merge(
                limits[limit_columns].drop_duplicates(["ts_code", "trade_date"], keep="last"),
                on=["ts_code", "trade_date"],
                how="left",
            )
        else:
            bars["up_limit"] = np.nan
            bars["down_limit"] = np.nan

        bars["datetime"] = pd.to_datetime(bars["trade_date"], format="%Y%m%d", errors="raise")
        bars["instrument"] = bars["ts_code"].map(ts_code_to_qlib)
        numeric = ("open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor", "up_limit", "down_limit")
        for column in numeric:
            bars[column] = pd.to_numeric(bars[column], errors="coerce")

        bars = bars.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        first = bars.groupby("instrument", sort=False).agg(
            first_adj=("adj_factor", "first"), first_close=("close", "first")
        )
        bars = bars.join(first, on="instrument")
        bars["factor"] = bars["adj_factor"] / (bars["first_adj"] * bars["first_close"])

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
        bars["suspended"] = False
        bars["limit_buy"] = bars["up_limit"].notna() & bars["raw_open"].round(2).ge(bars["up_limit"].round(2))
        bars["limit_sell"] = bars["down_limit"].notna() & bars["raw_open"].round(2).le(bars["down_limit"].round(2))

        bars = self._attach_index_membership(bars, index_weight)
        bars = self._attach_industry(bars, industry)
        bars = bars.sort_values(["datetime", "instrument"]).reset_index(drop=True)
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
        bars = bars.loc[~invalid_ohlc].reset_index(drop=True)
        self._validate_bars(bars)

        bars_root = self.canonical_root / "bars"
        for year, frame in bars.groupby(bars["datetime"].dt.year, sort=True):
            destination = bars_root / f"year={year}" / "bars.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.loc[:, CANONICAL_BAR_COLUMNS].to_parquet(destination, index=False)

        reference_root = self.canonical_root / "reference"
        reference_root.mkdir(parents=True, exist_ok=True)
        stock_basic.to_parquet(reference_root / "stock_basic.parquet", index=False)
        industry.to_parquet(reference_root / "industry_membership.parquet", index=False)
        index_weight.to_parquet(reference_root / "csi300_snapshots.parquet", index=False)
        self._read_endpoint("trade_cal").to_parquet(reference_root / "trade_cal.parquet", index=False)
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

        manifest = {
            "status": "complete",
            "raw_fingerprint": raw_manifest.get("fingerprint"),
            "rows": len(bars),
            "instruments": int(bars["instrument"].nunique()),
            "start": bars["datetime"].min().date().isoformat(),
            "end": bars["datetime"].max().date().isoformat(),
            "columns": list(CANONICAL_BAR_COLUMNS),
            "quality": quality,
            "fingerprint": stable_hash(
                {
                    "raw": raw_manifest.get("fingerprint"),
                    "rows": len(bars),
                    "columns": list(CANONICAL_BAR_COLUMNS),
                    "start": str(bars["datetime"].min()),
                    "end": str(bars["datetime"].max()),
                    "quality": quality,
                }
            ),
        }
        atomic_write_json(self.canonical_root / "manifest.json", manifest)
        return manifest

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
        pd, _ = _dependencies()
        bars = bars.copy()
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
        bars = bars.merge(date_map, on="datetime", how="left")
        membership["in_csi300"] = True
        bars = bars.merge(
            membership[["snapshot_date", "instrument", "in_csi300"]],
            on=["snapshot_date", "instrument"],
            how="left",
        )
        bars["in_csi300"] = bars["in_csi300"].astype("boolean").fillna(False).astype(bool)
        return bars.drop(columns="snapshot_date")

    def _attach_industry(self, bars: Any, industry: Any) -> Any:
        pd, np = _dependencies()
        result = bars.copy()
        for output in ("sector", "industry", "subindustry"):
            result[output] = np.nan
        required = {"ts_code", "in_date", "out_date", "l1_code", "l2_code", "l3_code"}
        if industry.empty or not required.issubset(industry.columns):
            return result

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
        for instrument, row_index in result.groupby("instrument", sort=False).groups.items():
            intervals = history[history["instrument"].eq(instrument)].sort_values("in_date")
            if intervals.empty:
                continue
            dates = result.loc[row_index, "datetime"]
            starts = intervals["in_date"].to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(starts, dates.to_numpy(dtype="datetime64[ns]"), side="right") - 1
            valid_position = positions >= 0
            safe_positions = positions.clip(min=0)
            ends = intervals["out_date"].to_numpy(dtype="datetime64[ns]")[safe_positions]
            valid = valid_position & (dates.to_numpy(dtype="datetime64[ns]") <= ends)
            for output, source in levels.items():
                values = intervals[source].astype(str).map(code_maps[output]).to_numpy(dtype=float)[safe_positions]
                values[~valid] = np.nan
                result.loc[row_index, output] = values
        return result

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
        live = ~bars["suspended"]
        if bars.loc[live, "factor"].isna().any() or bars.loc[live, "factor"].le(0).any():
            raise DataContractError("Live quotes require a positive restoration factor")
        restored = bars.loc[live, "close"] / bars.loc[live, "factor"]
        if not np.allclose(restored, bars.loc[live, "raw_close"], rtol=1e-10, atol=1e-10, equal_nan=True):
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
