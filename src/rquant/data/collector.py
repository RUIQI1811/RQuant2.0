from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rquant.data.contracts import RAW_DAILY_ENDPOINTS
from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file, stable_hash


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

        last_error: BaseException | None = None
        for attempt in range(self.retries):
            try:
                # Every outbound attempt consumes capacity, including retries after provider errors.
                self.rate_limiter.acquire()
                return method(**query)
            except BaseException as exc:
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
    ) -> FetchArtifact:
        artifact, _ = self._fetch_partition(endpoint, query, partition, force=force)
        return artifact

    def _fetch_partition(
        self,
        endpoint: str,
        query: dict[str, Any],
        partition: str,
        *,
        force: bool = False,
    ) -> tuple[FetchArtifact, bool]:
        target_dir = self.raw_root / endpoint / partition
        target = target_dir / "data.parquet"
        manifest_path = target_dir / "manifest.json"
        if not force:
            cached = self._validated_cache(target, manifest_path, endpoint, query)
            if cached is not None:
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

        date_column = next((name for name in ("trade_date", "cal_date", "in_date") if name in frame.columns), None)
        min_date = str(frame[date_column].min()) if date_column and not frame.empty else None
        max_date = str(frame[date_column].max()) if date_column and not frame.empty else None
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
    ) -> dict[str, Any]:
        if through >= date.today() + timedelta(days=1):
            raise DataContractError("--through cannot be in the future")
        calendar_buffer_days = max(450, warmup_trading_days * 2)
        fetch_start = formal_start - timedelta(days=calendar_buffer_days)
        artifacts: list[FetchArtifact] = []
        cached_artifacts = 0
        fetched_artifacts = 0
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

        def fetch(endpoint: str, query: dict[str, Any], partition: str) -> FetchArtifact:
            nonlocal cached_artifacts, fetched_artifacts
            artifact, cached = self._fetch_partition(endpoint, query, partition, force=force)
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
            # when either query requires another page.
            progress.total = 4 + len(open_dates) * len(RAW_DAILY_ENDPOINTS) + 1 + len(month_starts) + 2
            progress.refresh()

            for trade_date in open_dates:
                query_date = trade_date.strftime("%Y%m%d")
                for endpoint in RAW_DAILY_ENDPOINTS:
                    artifacts.append(
                        fetch(
                            endpoint,
                            {"trade_date": query_date},
                            f"trade_date={query_date}",
                        )
                    )

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
            manifest = {
                "status": "complete",
                "formal_start": formal_start.isoformat(),
                "fetch_start": fetch_start.isoformat(),
                "through": through.isoformat(),
                "warmup_trading_days": warmup_trading_days,
                "index_code": index_code,
                "industry_snapshot": through.isoformat(),
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
