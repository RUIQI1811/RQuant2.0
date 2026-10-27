from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rquant.data.canonical import ts_code_to_qlib
from rquant.data.contracts import FUNDAMENTAL_ENDPOINT_FIELDS, RAW_DAILY_ENDPOINTS
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file, stable_hash
from rquant.universe import BOARD_PREFIXES

# Tushare caps some statement responses at 100 rows without saying so; a window that reaches the cap is refetched
# year by year instead of being trusted.
STATEMENT_ROW_GUARD = 100
FUNDAMENTALS_MANIFEST = "fundamentals_manifest.json"


@dataclass(frozen=True)
class FetchArtifact:
    endpoint: str
    query: dict[str, Any]
    path: str
    rows: int
    min_date: str | None
    max_date: str | None
    sha256: str
    status: str = "complete"


class RequestRateLimiter:
    """Enforce a sliding-window request cap shared by one collector."""

    def __init__(
        self,
        max_requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(max_requests_per_minute, bool) or not isinstance(max_requests_per_minute, int):
            raise DataContractError("max_requests_per_minute must be a positive integer")
        if max_requests_per_minute <= 0:
            raise DataContractError("max_requests_per_minute must be a positive integer")
        self.max_requests_per_minute = max_requests_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._requests: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                window_start = now - 60.0
                while self._requests and self._requests[0] <= window_start:
                    self._requests.popleft()
                if len(self._requests) < self.max_requests_per_minute:
                    self._requests.append(now)
                    return
                delay = max(0.0, self._requests[0] + 60.0 - now)
            self._sleeper(delay)


class TushareCollector:
    """Independent, resumable Tushare Pro collector.

    Each successful query owns one immutable Parquet artifact and one adjacent manifest. A query is skipped only when
    both files exist and the current file hash matches the completed manifest.
    """

    def __init__(
        self,
        raw_root: str | Path,
        *,
        token: str | None = None,
        retries: int = 5,
        base_backoff: float = 1.0,
        client: Any | None = None,
        max_requests_per_minute: int = 190,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        if client is None and not self.token:
            raise DataContractError("TUSHARE_TOKEN is required; tokens are never read from project files")
        self.retries = retries
        self.base_backoff = base_backoff
        self._client = client
        self.rate_limiter = rate_limiter or RequestRateLimiter(max_requests_per_minute)

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import tushare as ts
            except ImportError as exc:
                raise DependencyError("tushare==1.4.29 is required for data sync") from exc
            self._client = ts.pro_api(self.token)
        return self._client

    def _call(self, endpoint: str, query: dict[str, Any]) -> Any:
        method = getattr(self.client, endpoint, None)
        if method is None:

            def method(**kwargs: Any) -> Any:
                return self.client.query(endpoint, **kwargs)

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                # Every outbound attempt consumes capacity, including retries after provider errors.
                self.rate_limiter.acquire()
                return method(**query)
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self.retries:
                    break
                delay = min(30.0, self.base_backoff * (2**attempt)) + random.random() * 0.25
                time.sleep(delay)
        raise DataContractError(
            f"Tushare {endpoint} failed after {self.retries} attempts: {last_error}"
        ) from last_error

    def fetch_partition(
        self,
        endpoint: str,
        query: dict[str, Any],
        partition: str,
        *,
        force: bool = False,
        require_rows: bool = False,
    ) -> FetchArtifact:
        artifact, _ = self._fetch_partition(endpoint, query, partition, force=force, require_rows=require_rows)
        return artifact

    def _fetch_partition(
        self,
        endpoint: str,
        query: dict[str, Any],
        partition: str,
        *,
        force: bool = False,
        require_rows: bool = False,
    ) -> tuple[FetchArtifact, bool]:
        """Fetch one partition; with ``require_rows`` an empty response is an error and is never cached."""
        target_dir = self.raw_root / endpoint / partition
        target = target_dir / "data.parquet"
        manifest_path = target_dir / "manifest.json"
        if not force:
            cached = self._validated_cache(target, manifest_path, endpoint, query)
            # Empty trading-day partitions cached by earlier versions are refetched instead of trusted.
            if cached is not None and not (require_rows and cached.rows == 0):
                return cached, True

        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required for Parquet collection") from exc

        frame = self._call(endpoint, query)
        if frame is None:
            frame = pd.DataFrame()
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        frame = frame.copy()
        frame.columns = [str(column) for column in frame.columns]
        if not frame.empty:
            frame = frame.drop_duplicates().reset_index(drop=True)
        elif require_rows:
            raise DataContractError(
                f"Tushare {endpoint} returned no rows for {query}; the data may not be published yet. "
                "Retry later or choose an earlier --through date"
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".data.", suffix=".parquet", dir=target_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        date_column = next(
            (name for name in ("trade_date", "cal_date", "in_date", "ann_date") if name in frame.columns), None
        )
        # Some endpoints (namechange) leave the date empty on a few rows.
        dates = frame[date_column].dropna().astype(str) if date_column else None
        min_date = dates.min() if dates is not None and len(dates) else None
        max_date = dates.max() if dates is not None and len(dates) else None
        artifact = FetchArtifact(
            endpoint=endpoint,
            query=query,
            path=str(target),
            rows=len(frame),
            min_date=min_date,
            max_date=max_date,
            sha256=sha256_file(target),
        )
        atomic_write_json(manifest_path, {**asdict(artifact), "query_fingerprint": stable_hash(query)})
        return artifact, False

    def _validated_cache(
        self,
        target: Path,
        manifest_path: Path,
        endpoint: str,
        query: dict[str, Any],
    ) -> FetchArtifact | None:
        if not target.exists() or not manifest_path.exists():
            return None
        import json

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete":
            return None
        if manifest.get("endpoint") != endpoint or manifest.get("query_fingerprint") != stable_hash(query):
            return None
        if manifest.get("sha256") != sha256_file(target):
            return None
        return FetchArtifact(**{key: manifest[key] for key in FetchArtifact.__dataclass_fields__})

    def sync(
        self,
        *,
        formal_start: date,
        through: date,
        warmup_trading_days: int = 300,
        index_code: str = "399300.SZ",
        force: bool = False,
        show_progress: bool = False,
        workers: int = 1,
    ) -> dict[str, Any]:
        if through >= date.today() + timedelta(days=1):
            raise DataContractError("--through cannot be in the future")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise DataContractError("sync workers must be a positive integer")
        calendar_buffer_days = max(450, warmup_trading_days * 2)
        fetch_start = formal_start - timedelta(days=calendar_buffer_days)
        artifacts: list[FetchArtifact] = []
        cached_artifacts = 0
        fetched_artifacts = 0
        progress_lock = threading.Lock()
        progress = tqdm(
            total=4,
            desc="Tushare preparing",
            unit="partition",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
            file=sys.stderr,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        def fetch(
            endpoint: str, query: dict[str, Any], partition: str, *, require_rows: bool = False
        ) -> FetchArtifact:
            nonlocal cached_artifacts, fetched_artifacts
            artifact, cached = self._fetch_partition(
                endpoint, query, partition, force=force, require_rows=require_rows
            )
            with progress_lock:
                if cached:
                    cached_artifacts += 1
                else:
                    fetched_artifacts += 1
                progress.set_description_str(f"Tushare {endpoint} {partition}", refresh=False)
                progress.set_postfix_str(
                    f"cached={cached_artifacts} fetched={fetched_artifacts}",
                    refresh=False,
                )
                progress.update()
            return artifact

        try:
            for status in ("L", "D", "P"):
                artifacts.append(
                    fetch(
                        "stock_basic",
                        {
                            "exchange": "",
                            "list_status": status,
                            "fields": "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
                        },
                        f"list_status={status}",
                    )
                )

            calendar_artifact = fetch(
                "trade_cal",
                {
                    "exchange": "SSE",
                    "start_date": _date_text(fetch_start),
                    # Qlib needs a future calendar boundary to execute the final completed trading day.
                    "end_date": _date_text(through + timedelta(days=31)),
                },
                f"range={fetch_start:%Y%m%d}-{through + timedelta(days=31):%Y%m%d}",
            )
            artifacts.append(calendar_artifact)
            completed_open_dates = [
                value for value in self._open_dates(Path(calendar_artifact.path)) if value <= through
            ]
            open_dates = completed_open_dates[-(warmup_trading_days + _formal_day_count(formal_start, through)) :]
            month_starts = _month_starts(fetch_start, through)
            # One historical and one current industry-membership page are known up front. The total grows only
            # when either query requires another page, and once per stock for the name histories.
            progress.total = 4 + len(open_dates) * len(RAW_DAILY_ENDPOINTS) + 1 + len(month_starts) + 2
            progress.refresh()

            daily_requests = [
                (endpoint, trade_date.strftime("%Y%m%d"))
                for trade_date in open_dates
                for endpoint in RAW_DAILY_ENDPOINTS
            ]
            daily_artifacts = self._fetch_daily_partitions(daily_requests, fetch=fetch, workers=workers)
            artifacts.extend(daily_artifacts)

            artifacts.append(
                fetch(
                    "index_daily",
                    {"ts_code": index_code, "start_date": _date_text(fetch_start), "end_date": _date_text(through)},
                    f"index={index_code.replace('.', '_')}",
                )
            )
            for month_start in month_starts:
                month_end = _month_end(month_start)
                artifacts.append(
                    fetch(
                        "index_weight",
                        {
                            "index_code": index_code,
                            "start_date": _date_text(month_start),
                            "end_date": _date_text(min(month_end, through)),
                        },
                        f"index={index_code.replace('.', '_')}/month={month_start:%Y-%m}",
                    )
                )

            artifacts.extend(self._fetch_industry_history(fetch=fetch, progress=progress, snapshot_date=through))
            codes = self._stock_codes([*artifacts[:3], *daily_artifacts])
            progress.total += len(codes)
            progress.refresh()
            artifacts.extend(self._fetch_name_history(codes, fetch=fetch, workers=workers, snapshot_date=through))
            manifest = {
                "status": "complete",
                "formal_start": formal_start.isoformat(),
                "fetch_start": fetch_start.isoformat(),
                "through": through.isoformat(),
                "warmup_trading_days": warmup_trading_days,
                "index_code": index_code,
                "industry_snapshot": through.isoformat(),
                "namechange_snapshot": through.isoformat(),
                "max_requests_per_minute": self.rate_limiter.max_requests_per_minute,
                "artifacts": len(artifacts),
                "cached_artifacts": cached_artifacts,
                "fetched_artifacts": fetched_artifacts,
                "rows": sum(item.rows for item in artifacts),
                "fingerprint": stable_hash([asdict(item) for item in artifacts]),
            }
            atomic_write_json(self.raw_root / "sync_manifest.json", manifest)
            progress.set_description_str("Tushare sync complete", refresh=False)
            progress.refresh()
            return manifest
        except BaseException:
            progress.set_description_str("Tushare sync stopped", refresh=False)
            progress.refresh()
            raise
        finally:
            progress.close()

    def sync_fundamentals(
        self,
        *,
        through: date,
        start_year: int,
        universe: str = "csi300",
        window_years: int = 6,
        force: bool = False,
        show_progress: bool = False,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Fetch income, balance-sheet and cash-flow statements for every stock the research universe can contain.

        Statements are queried per stock by announcement-date window. Closed windows never change, so a later sync only
        refetches the open window that ends at ``through``. The completed artifact list is written to
        ``fundamentals_manifest.json``; ``data sync`` and its manifest are not touched.
        """
        if through >= date.today() + timedelta(days=1):
            raise DataContractError("--through cannot be in the future")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise DataContractError("sync workers must be a positive integer")
        if window_years <= 0 or start_year > through.year:
            raise DataContractError("Fundamental windows need a positive length and a start year before --through")
        codes = self._fundamental_codes(universe)
        windows = [
            (first, min(first + window_years - 1, through.year))
            for first in range(start_year, through.year + 1, window_years)
        ]
        tasks = [(endpoint, code) for endpoint in FUNDAMENTAL_ENDPOINT_FIELDS for code in codes]
        cached_artifacts = 0
        fetched_artifacts = 0
        progress_lock = threading.Lock()
        progress = tqdm(
            total=len(tasks) * len(windows),
            desc="Tushare statements",
            unit="partition",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
            file=sys.stderr,
            bar_format=(
                "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        def fetch(endpoint: str, code: str, first: int, last: int) -> tuple[FetchArtifact, bool]:
            query = {
                "ts_code": code,
                "start_date": f"{first}0101",
                "end_date": min(f"{last}1231", _date_text(through)),
                "fields": ",".join(FUNDAMENTAL_ENDPOINT_FIELDS[endpoint]),
            }
            partition = f"ts_code={code.replace('.', '_')}/ann={first}-{last}"
            return self._fetch_partition(endpoint, query, partition, force=force)

        def record(endpoint: str, code: str, cached: bool, *, extra: int = 0) -> None:
            nonlocal cached_artifacts, fetched_artifacts
            with progress_lock:
                if cached:
                    cached_artifacts += 1
                else:
                    fetched_artifacts += 1
                progress.total += extra
                progress.set_description_str(f"Tushare {endpoint} {code}", refresh=False)
                progress.set_postfix_str(f"cached={cached_artifacts} fetched={fetched_artifacts}", refresh=False)
                progress.update()

        def fetch_stock(task: tuple[str, str]) -> list[tuple[str, FetchArtifact]]:
            endpoint, code = task
            artifacts = []
            for first, last in windows:
                artifact, cached = fetch(endpoint, code, first, last)
                if artifact.rows < STATEMENT_ROW_GUARD:
                    record(endpoint, code, cached)
                    artifacts.append((endpoint, artifact))
                    continue
                if first == last:
                    raise DataContractError(
                        f"Tushare {endpoint} returned {artifact.rows} rows for {code} in {first}; "
                        "the response may be truncated"
                    )
                # The superseded multi-year artifact stays on disk but is not listed in the manifest.
                record(endpoint, code, cached, extra=last - first + 1)
                for year in range(first, last + 1):
                    yearly, yearly_cached = fetch(endpoint, code, year, year)
                    if yearly.rows >= STATEMENT_ROW_GUARD:
                        raise DataContractError(
                            f"Tushare {endpoint} returned {yearly.rows} rows for {code} in {year}; "
                            "the response may be truncated"
                        )
                    record(endpoint, code, yearly_cached)
                    artifacts.append((endpoint, yearly))
            return artifacts

        try:
            if workers == 1:
                results = [fetch_stock(task) for task in tasks]
            else:
                executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare-statements")
                futures = [executor.submit(fetch_stock, task) for task in tasks]
                try:
                    results = [future.result() for future in futures]
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
            listed = [item for result in results for item in result]
            files = [
                {
                    "endpoint": endpoint,
                    "path": Path(artifact.path).relative_to(self.raw_root).as_posix(),
                    "sha256": artifact.sha256,
                    "rows": artifact.rows,
                }
                for endpoint, artifact in listed
            ]
            manifest = {
                "status": "complete",
                "through": through.isoformat(),
                "universe": universe,
                "start_year": start_year,
                "window_years": window_years,
                "endpoints": {endpoint: list(fields) for endpoint, fields in FUNDAMENTAL_ENDPOINT_FIELDS.items()},
                "instruments": len(codes),
                "artifacts": len(files),
                "cached_artifacts": cached_artifacts,
                "fetched_artifacts": fetched_artifacts,
                "rows": sum(item["rows"] for item in files),
                "files": files,
                "fingerprint": stable_hash(
                    [
                        {"endpoint": endpoint, "query": artifact.query, "sha256": artifact.sha256}
                        for endpoint, artifact in listed
                    ]
                ),
            }
            atomic_write_json(self.raw_root / FUNDAMENTALS_MANIFEST, manifest)
            progress.set_description_str("Tushare statements complete", refresh=False)
            progress.refresh()
            return manifest
        except BaseException:
            progress.set_description_str("Tushare statements stopped", refresh=False)
            progress.refresh()
            raise
        finally:
            progress.close()

    def _fundamental_codes(self, universe: str) -> list[str]:
        """Return every Tushare code the universe can contain, read from the completed daily sync."""
        import json

        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required for fundamental collection") from exc
        manifest_path = self.raw_root / "sync_manifest.json"
        if not manifest_path.exists():
            raise DataContractError("Run `rquant data sync` before syncing fundamentals")
        with manifest_path.open("r", encoding="utf-8") as handle:
            sync_manifest = json.load(handle)
        if sync_manifest.get("status") != "complete":
            raise DataContractError("Raw sync is not complete; finish `rquant data sync` first")
        if universe == "csi300":
            endpoint, column = "index_weight", "con_code"
        elif universe in (*BOARD_PREFIXES, "all"):
            endpoint, column = "stock_basic", "ts_code"
        else:
            raise DataContractError(f"Unsupported fundamental universe: {universe!r}")
        paths = sorted((self.raw_root / endpoint).glob("**/data.parquet"))
        frames = [pd.read_parquet(path) for path in paths]
        frames = [frame for frame in frames if column in frame.columns and not frame.empty]
        if not frames:
            raise DataContractError(f"No raw {endpoint} rows to derive the fundamental universe from")
        frame = pd.concat(frames, ignore_index=True)
        index_code = sync_manifest.get("index_code")
        if endpoint == "index_weight" and index_code and "index_code" in frame.columns:
            frame = frame[frame["index_code"] == index_code]
        codes = sorted(str(code) for code in frame[column].dropna().unique())
        if universe in BOARD_PREFIXES:
            # ST and listing-age filters change over time, so every stock on the boards may enter the universe.
            codes = [code for code in codes if ts_code_to_qlib(code).startswith(BOARD_PREFIXES[universe])]
        if not codes:
            raise DataContractError(f"No stocks found for fundamental universe {universe}")
        return codes

    @staticmethod
    def _fetch_daily_partitions(
        requests: list[tuple[str, str]], *, fetch: Any, workers: int
    ) -> list[FetchArtifact]:
        """Fetch per-trading-day partitions, overlapping request latency across ``workers`` threads.

        The shared rate limiter still caps total throughput, and results keep the request order so the sync
        fingerprint does not depend on the worker count.
        """

        def fetch_one(request: tuple[str, str]) -> FetchArtifact:
            endpoint, query_date = request
            # Every open trading day has quotes; an empty answer means "not published yet".
            return fetch(endpoint, {"trade_date": query_date}, f"trade_date={query_date}", require_rows=True)

        return _map_in_order(fetch_one, requests, workers)

    def _fetch_industry_history(self, *, fetch: Any, progress: Any, snapshot_date: date) -> list[FetchArtifact]:
        artifacts: list[FetchArtifact] = []
        limit = 2000
        snapshot_root = f"snapshot={snapshot_date:%Y%m%d}"
        for is_new in ("N", "Y"):
            for offset in range(0, 200_000, limit):
                artifact = fetch(
                    "index_member_all",
                    {"is_new": is_new, "offset": offset, "limit": limit},
                    f"{snapshot_root}/status={is_new}/page={offset // limit:04d}",
                )
                artifacts.append(artifact)
                if artifact.rows < limit:
                    break
                progress.total += 1
                progress.refresh()
            else:
                raise DataContractError(
                    f"index_member_all is_new={is_new} pagination exceeded 200,000 rows; "
                    "refusing a silently truncated result"
                )
        return artifacts

    @staticmethod
    def _stock_codes(artifacts: list[FetchArtifact]) -> list[str]:
        """Every Tushare code in the given stock_basic and daily partitions.

        The daily partitions add stocks listed after the cached stock_basic snapshot.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required for stock code collection") from exc
        codes: set[str] = set()
        for artifact in artifacts:
            if artifact.rows:
                codes.update(pd.read_parquet(artifact.path, columns=["ts_code"])["ts_code"].dropna().astype(str))
        if not codes:
            raise DataContractError("No stock codes to fetch name histories for")
        return sorted(codes)

    @staticmethod
    def _fetch_name_history(
        codes: list[str], *, fetch: Any, workers: int, snapshot_date: date
    ) -> list[FetchArtifact]:
        """Fetch each stock's name history, which marks its point-in-time ST and delisting periods.

        The all-stock query pages by offset over an unstable order and silently drops rows, so every stock is asked
        for on its own.
        """

        def fetch_one(code: str) -> FetchArtifact:
            return fetch(
                "namechange",
                {"ts_code": code, "fields": "ts_code,name,start_date,end_date,ann_date,change_reason"},
                f"snapshot={snapshot_date:%Y%m%d}/ts_code={code}",
            )

        return _map_in_order(fetch_one, codes, workers)

    @staticmethod
    def _open_dates(calendar_path: Path) -> list[date]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required for calendar parsing") from exc
        frame = pd.read_parquet(calendar_path)
        if not {"cal_date", "is_open"}.issubset(frame.columns):
            raise DataContractError("trade_cal response lacks cal_date/is_open")
        values = frame.loc[frame["is_open"].astype(int).eq(1), "cal_date"].astype(str)
        return sorted(datetime.strptime(value, "%Y%m%d").date() for value in values)


def _map_in_order(function: Callable[[Any], FetchArtifact], items: list[Any], workers: int) -> list[FetchArtifact]:
    """Run ``function`` over ``items`` on ``workers`` threads and return results in item order.

    The shared rate limiter still caps total throughput, and the order keeps the sync fingerprint independent of the
    worker count.
    """
    if workers == 1:
        return [function(item) for item in items]
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tushare-sync")
    futures = [executor.submit(function, item) for item in items]
    try:
        return [future.result() for future in futures]
    finally:
        # On the first failure, stop queued requests instead of draining the whole history.
        executor.shutdown(wait=True, cancel_futures=True)


def _date_text(value: date) -> str:
    return value.strftime("%Y%m%d")


def _formal_day_count(start: date, end: date) -> int:
    return max(1, int((end - start).days * 5 / 7) + 30)


def _month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    values = []
    while current <= end:
        values.append(current)
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return values


def _month_end(value: date) -> date:
    next_month = date(value.year + (value.month == 12), value.month % 12 + 1, 1)
    return next_month - timedelta(days=1)
