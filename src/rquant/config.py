from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rquant.errors import ConfigurationError, DependencyError


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    canonical: Path
    qlib: Path
    factors: Path
    cache: Path
    runs: Path

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        data_root: str | Path = "data",
        runs_root: str | Path = "runs",
    ) -> ProjectPaths:
        root_path = Path(root).expanduser().resolve()
        data_path = _under_root(root_path, data_root)
        runs_path = _under_root(root_path, runs_root)
        return cls(
            root=root_path,
            data=data_path,
            raw=data_path / "raw",
            canonical=data_path / "canonical",
            qlib=data_path / "qlib",
            factors=data_path / "factors",
            cache=data_path / "cache",
            runs=runs_path,
        )

    def create_runtime_dirs(self) -> None:
        for path in (self.data, self.raw, self.canonical, self.qlib, self.factors, self.cache, self.runs):
            path.mkdir(parents=True, exist_ok=True)


def _under_root(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(f"Runtime path must stay inside project root: {candidate}") from exc
    return candidate


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise DependencyError("PyYAML is required to load configuration") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("Top-level configuration must be a mapping")
    return loaded


def resolve_paths(config: dict[str, Any], root: str | Path) -> ProjectPaths:
    project = config.get("project", {})
    return ProjectPaths.from_root(
        root,
        data_root=project.get("data_root", "data"),
        runs_root=project.get("runs_root", "runs"),
    )
