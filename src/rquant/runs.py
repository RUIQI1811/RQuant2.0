from __future__ import annotations

import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rquant.io import atomic_write_json, stable_hash


@dataclass
class RunManifest:
    run_id: str
    command: list[str]
    status: str
    started_at: str
    completed_at: str | None = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    config: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RunContext:
    def __init__(self, runs_root: str | Path, command: list[str], config: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        config_fingerprint = stable_hash(config)
        self.directory = Path(runs_root) / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.directory / "run.json"
        self.log_path = self.directory / "run.log"
        self.log_path.touch(exist_ok=False)
        self.manifest = RunManifest(
            run_id=run_id,
            command=command,
            status="running",
            started_at=now.isoformat(),
            config=config,
            config_fingerprint=config_fingerprint,
        )
        self._write()
        self.log(f"started command={command!r}")

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    def add_input(self, name: str, value: Any) -> None:
        self.manifest.inputs[name] = value
        self._write()

    def add_output(self, name: str, value: Any) -> None:
        self.manifest.outputs[name] = value
        self._write()

    def complete(self) -> None:
        self.manifest.status = "complete"
        self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
        self.log("completed")
        self._write()

    def fail(self, exc: BaseException) -> None:
        self.manifest.status = "failed"
        self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
        self.manifest.error = f"{type(exc).__name__}: {exc}"
        self.log(self.manifest.error)
        self._write()

    def _write(self) -> None:
        atomic_write_json(self.manifest_path, asdict(self.manifest))
