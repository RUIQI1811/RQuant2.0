"""Point-in-time reference data built from raw Tushare artifacts next to the canonical bars.

The reference set never changes the canonical bar contract or its fingerprint, so existing factor stores stay current.
Factor libraries that read it record the hashes of the files they consumed in their own manifests.

* ``daily_basic/year=YYYY/data.parquet``: per-stock valuation, share-count and turnover fields for every A share.
* ``market_factors.parquet``: daily value-weighted A-share market return plus Fama-French SMB and HML, all built from
  the full A-share cross-section with previous-day weights and month-start portfolio formation.
* ``ch4_factors.parquet``: daily Liu-Stambaugh-Yuan CH-3/CH-4 factor returns (``rquant.data.ch4_factors``).
* ``fundamentals.parquet``: one first-reported row per stock and fiscal quarter, with the trading-day-independent
  announcement date after which the whole row is known.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rquant.data.canonical import ts_code_to_qlib
from rquant.data.ch4_factors import CH4_FACTORS, CH4FactorBuilder
from rquant.data.collector import FUNDAMENTALS_MANIFEST
from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS, STATEMENT_KEY_FIELDS
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_replace_directory, atomic_write_json, sha256_file, stable_hash

DAILY_BASIC_FIELDS = (
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
)
REFERENCE_VERSION = 2
# Newly listed stocks trade outside normal price limits; they join the factor portfolios after this many calendar days.
MARKET_FACTOR_SEASONING_DAYS = 30
MARKET_FACTOR_MIN_STOCKS = 100
QUARTER_ENDS = ("0331", "0630", "0930", "1231")


class ReferenceBuilder:
    """Build the extended point-in-time reference set under ``<canonical>/reference``."""

    def __init__(self, raw_root: str | Path, canonical_root: str | Path) -> None:
        self.raw_root = Path(raw_root)
        self.reference_root = Path(canonical_root) / "reference"

    def build(self) -> dict[str, Any]:
        raw_manifest = _complete_manifest(self.raw_root / "sync_manifest.json", "Raw sync")
        self.reference_root.mkdir(parents=True, exist_ok=True)
        list_dates = self._list_dates()

        files: dict[str, str] = {}
        daily_root = self.reference_root / "daily_basic"
        staging = Path(tempfile.mkdtemp(prefix=".daily_basic.staging.", dir=self.reference_root))
        try:
            factors, ch4 = self._stream_daily(staging, list_dates)
            atomic_replace_directory(staging, daily_root)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        for path in sorted(daily_root.glob("year=*/data.parquet")):
            files[path.relative_to(self.reference_root).as_posix()] = sha256_file(path)
        _write_parquet(factors, self.reference_root / "market_factors.parquet")
        files["market_factors.parquet"] = sha256_file(self.reference_root / "market_factors.parquet")
        _write_parquet(ch4, self.reference_root / "ch4_factors.parquet")
        files["ch4_factors.parquet"] = sha256_file(self.reference_root / "ch4_factors.parquet")

        fundamentals_manifest = self.raw_root / FUNDAMENTALS_MANIFEST
        fundamentals_fingerprint = None
        fundamentals_path = self.reference_root / "fundamentals.parquet"
        if fundamentals_manifest.exists():
            source = _complete_manifest(fundamentals_manifest, "Fundamental sync")
            _write_parquet(self._fundamentals(source), fundamentals_path)
            files["fundamentals.parquet"] = sha256_file(fundamentals_path)
            fundamentals_fingerprint = source.get("fingerprint")
        else:
            # A stale table from an earlier sync must not outlive the manifest that described it.
            fundamentals_path.unlink(missing_ok=True)

        manifest = {
            "status": "complete",
            "version": REFERENCE_VERSION,
            "raw_fingerprint": raw_manifest.get("fingerprint"),
            "fundamentals_fingerprint": fundamentals_fingerprint,
            "market_factor_days": int(factors["mkt"].notna().sum()),
            "ch4_factor_days": int(ch4["pmo"].notna().sum()),
            "files": files,
        }
        manifest["fingerprint"] = stable_hash(manifest)
        atomic_write_json(self.reference_root / "manifest.json", manifest)
        return manifest

    def _list_dates(self) -> dict[str, Any]:
        pd, _ = _dependencies()
        paths = sorted((self.raw_root / "stock_basic").glob("**/data.parquet"))
        frames = [pd.read_parquet(path) for path in paths]
        frames = [frame for frame in frames if {"ts_code", "list_date"}.issubset(frame.columns) and not frame.empty]
        if not frames:
            raise DataContractError("No raw stock_basic rows with list_date")
        basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
        listed = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
        return dict(zip(basic["ts_code"], listed, strict=True))

    def _stream_daily(self, staging: Path, list_dates: dict[str, Any]) -> tuple[Any, Any]:
        """Write yearly daily_basic partitions and return daily market and CH-4 factors, one trading date at a time.

        Published daily_basic rows gain ``turnover_total``: share volume over total shares, as a fraction.
        """
        pd, np = _dependencies()
        daily_paths = sorted((self.raw_root / "daily").glob("**/data.parquet"))
        if not daily_paths:
            raise DataContractError("No raw artifacts for endpoint: daily")
        rows: list[dict[str, Any]] = []
        ch4_rows: list[dict[str, Any]] = []
        ch4 = CH4FactorBuilder(list_dates)
        year_frames: list[Any] = []
        current_year: int | None = None
        previous: Any = None
        assignment: Any = None
        assignment_month: tuple[int, int] | None = None

        def flush(year: int | None) -> None:
            if year is None or not year_frames:
                return
            frame = pd.concat(year_frames, ignore_index=True)
            destination = staging / f"year={year}" / "data.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False)
            year_frames.clear()

        for daily_path in daily_paths:
            partition = daily_path.parent.relative_to(self.raw_root / "daily")
            basic_path = self.raw_root / "daily_basic" / partition / "data.parquet"
            daily = pd.read_parquet(daily_path)
            daily = daily[[column for column in ("ts_code", "trade_date", "close", "pre_close", "vol")
                           if column in daily.columns]]
            if daily.empty:
                continue
            if not basic_path.exists():
                raise DataContractError(f"daily_basic partition is missing for {partition}")
            basic = pd.read_parquet(basic_path).drop_duplicates(["ts_code", "trade_date"], keep="last")
            if daily["trade_date"].nunique() != 1:
                raise DataContractError(f"daily partition {partition} spans more than one trade date")
            trade_date = pd.Timestamp(str(daily["trade_date"].iloc[0]))

            fields = [field for field in DAILY_BASIC_FIELDS if field in basic.columns]
            published = basic[["ts_code", *fields]].copy()
            for field in fields:
                published[field] = pd.to_numeric(published[field], errors="coerce")
            today = daily.drop_duplicates("ts_code", keep="last").set_index("ts_code")
            volume = pd.to_numeric(today["vol"], errors="coerce") if "vol" in today.columns else today["close"] * np.nan
            if "total_share" in published.columns:
                # Tushare vol is in lots of 100 shares and total_share in units of 10,000 shares.
                shares = published.set_index("ts_code")["total_share"]
                turnover = (volume.reindex(shares.index) / (shares.where(shares > 0) * 100.0)).astype(float)
            else:
                turnover = pd.Series(np.nan, index=published["ts_code"].to_numpy(), dtype=float)
            published["turnover_total"] = turnover.to_numpy()
            published.insert(0, "datetime", trade_date)
            published.insert(0, "instrument", published.pop("ts_code").map(ts_code_to_qlib))
            if current_year != trade_date.year:
                flush(current_year)
                current_year = trade_date.year
            year_frames.append(published)

            today["ret"] = pd.to_numeric(today["close"], errors="coerce") / pd.to_numeric(
                today["pre_close"], errors="coerce"
            ) - 1.0
            state = basic.set_index("ts_code")[
                [f for f in ("circ_mv", "total_mv", "pb", "pe_ttm") if f in basic.columns]
            ]
            state = state.apply(pd.to_numeric, errors="coerce")
            if "total_mv" in state.columns:
                ch4_rows.append(ch4.update(trade_date, today["ret"], state, turnover.reindex(today.index)))
            else:
                ch4_rows.append({"datetime": trade_date, **{name: np.nan for name in CH4_FACTORS}, "ch4_stocks": 0})
            listed = pd.Series(list_dates).reindex(today.index)
            seasoned = (trade_date - listed).dt.days >= MARKET_FACTOR_SEASONING_DAYS
            eligible = ~today.index.str.endswith(".BJ") & seasoned.fillna(False).to_numpy()
            row: dict[str, Any] = {"datetime": trade_date, "mkt": np.nan, "smb": np.nan, "hml": np.nan, "stocks": 0}
            if previous is not None and "circ_mv" in previous.columns:
                month = (trade_date.year, trade_date.month)
                if assignment_month != month:
                    assignment = _size_value_groups(previous, pd, np)
                    assignment_month = month
                weights = previous["circ_mv"].reindex(today.index)
                returns = today["ret"].where(eligible)
                valid = returns.notna() & weights.gt(0)
                row["stocks"] = int(valid.sum())
                if row["stocks"] >= MARKET_FACTOR_MIN_STOCKS:
                    row["mkt"] = float((returns[valid] * weights[valid]).sum() / weights[valid].sum())
                    row["smb"], row["hml"] = _size_value_returns(returns[valid], weights[valid], assignment, np)
            rows.append(row)
            previous = state
        flush(current_year)
        if not rows:
            raise DataContractError("No daily partitions to build reference data from")
        return (
            pd.DataFrame(rows, columns=["datetime", "mkt", "smb", "hml", "stocks"]),
            pd.DataFrame(ch4_rows, columns=["datetime", *CH4_FACTORS, "ch4_stocks"]).rename(
                columns={"ch4_stocks": "stocks"}
            ),
        )

    def _fundamentals(self, source: dict[str, Any]) -> Any:
        """Merge first-reported statements into one row per stock and quarter end."""
        pd, np = _dependencies()
        listed = source.get("files") or []
        merged = None
        for endpoint, fields in FUNDAMENTAL_ENDPOINT_FIELDS.items():
            frames = []
            for item in listed:
                if item["endpoint"] != endpoint:
                    continue
                path = self.raw_root / item["path"]
                if not path.exists() or sha256_file(path) != item["sha256"]:
                    raise DataContractError(f"Fundamental artifact is missing or changed: {path}")
                frames.append(pd.read_parquet(path))
            values = [field for field in fields if field not in STATEMENT_KEY_FIELDS]
            frames = [frame for frame in frames if not frame.empty]
            if not frames:
                raise DataContractError(f"Fundamental sync has no {endpoint} rows")
            frame = first_reported(pd.concat(frames, ignore_index=True), values, pd, np)
            frame = frame.rename(columns={"available": f"available_{endpoint}"})
            merged = frame if merged is None else merged.merge(frame, on=["ts_code", "end_date"], how="outer")
        assert merged is not None
        available = [f"available_{endpoint}" for endpoint in FUNDAMENTAL_ENDPOINT_FIELDS]
        # A quarter counts as announced once every statement published for it is out.
        merged["ann_date"] = merged[available].max(axis=1)
        merged = merged.drop(columns=available)
        merged.insert(0, "instrument", merged.pop("ts_code").map(ts_code_to_qlib))
        merged["end_date"] = pd.to_datetime(merged["end_date"], format="%Y%m%d")
        merged["ann_date"] = pd.to_datetime(merged["ann_date"], format="%Y%m%d")
        return merged.sort_values(["instrument", "end_date"]).reset_index(drop=True)


def first_reported(frame: Any, values: list[str], pd: Any, np: Any) -> Any:
    """Keep the earliest-announced consolidated version of every stock and quarter-end statement.

    A version becomes known on the later of ``ann_date`` and ``f_ann_date``. Later corrections are dropped so a value is
    never used before it was published; when two versions share the first date, the corrected one is kept because both
    were public that day.
    """
    frame = frame.copy()
    for column in ("ann_date", "f_ann_date", "end_date"):
        if column not in frame.columns:
            frame[column] = None
        frame[column] = frame[column].astype("string")
    if "report_type" in frame.columns:
        report_type = frame["report_type"].astype("string")
        frame = frame[report_type.isna() | report_type.eq("1")]
    frame = frame[frame["end_date"].str[4:].isin(QUARTER_ENDS)]
    frame["available"] = frame[["ann_date", "f_ann_date"]].max(axis=1, skipna=True)
    frame = frame[frame["available"].notna()]
    if "update_flag" in frame.columns:
        frame["_flag"] = pd.to_numeric(frame["update_flag"], errors="coerce").fillna(0)
    else:
        frame["_flag"] = 0
    frame = frame.sort_values(["ts_code", "end_date", "available", "_flag"], ascending=[True, True, True, False])
    frame = frame.drop_duplicates(["ts_code", "end_date"], keep="first")
    for column in values:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["ts_code", "end_date", "available", *values]].reset_index(drop=True)


def _size_value_groups(previous: Any, pd: Any, np: Any) -> Any:
    """Assign Fama-French 2x3 size/book-to-market groups from the previous day's total value and P/B."""
    if not {"total_mv", "pb"}.issubset(previous.columns):
        return pd.Series(dtype=object)
    frame = previous[["total_mv", "pb"]].dropna()
    frame = frame[(frame["total_mv"] > 0) & (frame["pb"] > 0)]
    if frame.empty:
        return pd.Series(dtype=object)
    size = np.where(frame["total_mv"] <= frame["total_mv"].median(), "S", "B")
    book_to_market = 1.0 / frame["pb"]
    low, high = book_to_market.quantile([0.3, 0.7])
    value = np.where(book_to_market <= low, "L", np.where(book_to_market > high, "H", "M"))
    return pd.Series(np.char.add(size.astype(str), value.astype(str)), index=frame.index)


def _size_value_returns(returns: Any, weights: Any, assignment: Any, np: Any) -> tuple[float, float]:
    groups = assignment.reindex(returns.index)
    portfolio: dict[str, float] = {}
    for name in ("SL", "SM", "SH", "BL", "BM", "BH"):
        members = groups.eq(name).to_numpy()
        if not members.any():
            return np.nan, np.nan
        portfolio[name] = float((returns[members] * weights[members]).sum() / weights[members].sum())
    smb = (portfolio["SL"] + portfolio["SM"] + portfolio["SH"]) / 3.0 - (
        portfolio["BL"] + portfolio["BM"] + portfolio["BH"]
    ) / 3.0
    hml = (portfolio["SH"] + portfolio["BH"]) / 2.0 - (portfolio["SL"] + portfolio["BL"]) / 2.0
    return smb, hml


def load_reference_manifest(canonical_root: str | Path) -> dict[str, Any]:
    """Return the reference manifest after checking it was built from the canonical dataset's raw sync."""
    canonical_root = Path(canonical_root)
    reference = _complete_manifest(canonical_root / "reference" / "manifest.json", "Reference data")
    canonical = _complete_manifest(canonical_root / "manifest.json", "Canonical dataset")
    if reference.get("raw_fingerprint") != canonical.get("raw_fingerprint"):
        raise DataContractError("Reference data was built from a different raw sync; run `rquant data build-reference`")
    payload = dict(reference)
    if payload.pop("fingerprint", None) != stable_hash(payload):
        raise DataContractError("Reference manifest fingerprint is invalid")
    return reference


def _complete_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise DataContractError(f"{label} manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise DataContractError(f"{label} is not complete: {path}")
    return manifest


def _write_parquet(frame: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise DependencyError("pandas, pyarrow and NumPy are required for reference data") from exc
    return pd, np
