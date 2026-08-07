from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError, FactorContractError
from rquant.factors.catalog import FactorSet, catalog_from_registry, get_catalog
from rquant.factors.extensions import kunquant as kunquant_extensions
from rquant.factors.libraries.base import CANONICAL_INPUT_FIELDS
from rquant.factors.libraries.registry import FactorLibraryRegistry, get_library_registry
from rquant.io import atomic_write_json, sha256_file, stable_hash

INPUT_FIELDS = CANONICAL_INPUT_FIELDS
FACTOR_ENGINE_VERSION = 2


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    """Publish a fully built directory while preserving the previous version on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup.", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


@dataclass(frozen=True)
class FactorBuildConfig:
    factor_set: FactorSet = "combined"
    dtype: str = "double"
    input_layout: str = "TS"
    output_layout: str = "TS"
    workers: int = 4
    universe: str = "csi300"
    external_inputs: tuple[tuple[str, str], ...] = ()


class KunQuantFactorEngine:
    """Compile and execute the locked factor graph against canonical daily panels."""

    def __init__(
        self,
        cache_dir: str | Path,
        config: FactorBuildConfig | None = None,
        *,
        registry: FactorLibraryRegistry | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config or FactorBuildConfig()
        self.registry = registry or get_library_registry()
        self.catalog = catalog_from_registry(self.registry)
        self.libraries = self.registry.select(self.config.factor_set)
        self.input_fields = self.registry.required_inputs(self.config.factor_set)
        if any(library.backend != "kunquant" for library in self.libraries):
            raise FactorContractError(f"Factor set {self.config.factor_set} is not a KunQuant-only factor set")

    def compilation_fingerprint(self) -> str:
        try:
            version = importlib.metadata.version("KunQuant")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        return stable_hash(
            {
                "kunquant": version,
                "catalog": self.catalog.fingerprint_for(self.config.factor_set),
                "factor_set": self.config.factor_set,
                "dtype": self.config.dtype,
                "input_layout": self.config.input_layout,
                "output_layout": self.config.output_layout,
                "factor_engine_version": FACTOR_ENGINE_VERSION,
                "factor_libraries": self.registry.implementation_fingerprint(self.config.factor_set),
                "kunquant_extensions": inspect.getsource(kunquant_extensions),
            }
        )

    def build_graph(self) -> Any:
        try:
            from KunQuant.Driver import KunCompilerConfig
            from KunQuant.Op import Builder, Input, Output
            from KunQuant.Stage import Function
        except ImportError as exc:
            raise DependencyError("KunQuant==0.1.11 is required for factor computation") from exc

        builder = Builder()
        with builder:
            inputs = {name: Input(name) for name in self.input_fields}
            outputs: dict[str, Any] = {}
            for library in self.libraries:
                source_outputs = dict(library.build(inputs))
                source_to_canonical = {spec.source_name: spec.canonical_name for spec in library.specs}
                if set(source_outputs) != set(source_to_canonical):
                    missing_names = sorted(set(source_to_canonical) - set(source_outputs))
                    extra_names = sorted(set(source_outputs) - set(source_to_canonical))
                    raise FactorContractError(
                        f"Factor library {library.family} differs from its catalog; "
                        f"missing={missing_names}, extra={extra_names}"
                    )
                for source_name, canonical_name in source_to_canonical.items():
                    if canonical_name in outputs:
                        raise FactorContractError(f"Duplicate canonical factor output: {canonical_name}")
                    outputs[canonical_name] = source_outputs[source_name]

            expected = self.catalog.canonical_names(self.config.factor_set)
            if tuple(outputs) != expected:
                raise FactorContractError("Factor-library output order differs from the locked catalog")
            for canonical_name, expression in outputs.items():
                Output(expression, canonical_name)

        return Function(builder.ops), KunCompilerConfig(
            dtype=self.config.dtype,
            input_layout=self.config.input_layout,
            output_layout=self.config.output_layout,
            # None lets KunQuant select the architecture-safe value (unaligned is unsupported on Apple Silicon).
            allow_unaligned=None,
            # Prefer deterministic double-precision rolling statistics over the faster incremental approximation.
            options={"no_fast_stat": True},
        )

    def compile(self) -> Any:
        try:
            from KunQuant.jit import cfake
        except ImportError as exc:
            raise DependencyError("KunQuant==0.1.11 is required for factor computation") from exc

        fingerprint = self.compilation_fingerprint()
        target = self.cache_dir / "kunquant" / fingerprint
        target.mkdir(parents=True, exist_ok=True)
        function, compiler_config = self.build_graph()
        library_name = f"rquant_{self.config.factor_set}_{fingerprint[:12]}"
        library = cfake.compileit(
            [("rquant_factors", function, compiler_config)],
            library_name,
            cfake.CppCompilerConfig(),
            tempdir=str(target),
            keep_files=True,
        )
        atomic_write_json(
            target / "compile.json",
            {
                "fingerprint": fingerprint,
                "factor_set": self.config.factor_set,
                "catalog_fingerprint": self.catalog.fingerprint_for(self.config.factor_set),
                "factor_libraries": [library.family for library in self.libraries],
                "input_fields": list(self.input_fields),
                "output_names": list(self.catalog.canonical_names(self.config.factor_set)),
            },
        )
        return library.getModule("rquant_factors")

    def run(
        self,
        inputs: dict[str, Any],
        module: Any | None = None,
        *,
        output_buffers: dict[str, Any] | None = None,
        trim_outputs: bool = True,
    ) -> dict[str, Any]:
        try:
            import numpy as np
            from KunQuant.runner import KunRunner as kr
        except ImportError as exc:
            raise DependencyError("NumPy and KunQuant are required for factor computation") from exc

        absent = sorted(set(self.input_fields) - set(inputs))
        if absent:
            raise DataContractError(f"Missing KunQuant inputs: {absent}")
        numpy_dtype = np.float64 if self.config.dtype == "double" else np.float32
        normalized = {
            name: np.ascontiguousarray(np.asarray(inputs[name], dtype=numpy_dtype)) for name in self.input_fields
        }
        shapes = {tuple(normalized[name].shape) for name in self.input_fields}
        if len(shapes) != 1:
            raise DataContractError(f"KunQuant input shapes differ: {sorted(shapes)}")
        shape = next(iter(shapes))
        if len(shape) != 2 or min(shape) == 0:
            raise DataContractError(f"KunQuant TS inputs must be non-empty 2D arrays; got {shape}")
        time_count, stock_count = shape
        module = module or self.compile()
        blocking_len = int(module.blocking_len)
        pad_stocks = (-stock_count) % blocking_len
        if pad_stocks:
            if self.config.input_layout != "TS" or self.config.output_layout != "TS":
                raise DataContractError(
                    "Unaligned stock counts are only supported for KunQuant TS input/output layouts"
                )
            normalized = {
                name: np.pad(values, ((0, 0), (0, pad_stocks)), constant_values=np.nan)
                for name, values in normalized.items()
            }
        expected = self.catalog.canonical_names(self.config.factor_set)
        if output_buffers is not None and set(output_buffers) != set(expected):
            raise DataContractError("KunQuant output buffers do not match the locked catalog")
        executor = kr.createMultiThreadExecutor(self.config.workers)
        output = kr.runGraph(executor, module, normalized, 0, time_count, output_buffers or {})
        if pad_stocks and trim_outputs:
            output = {name: np.ascontiguousarray(values[:, :stock_count]) for name, values in output.items()}
        if tuple(output) != expected:
            if set(output) != set(expected):
                raise FactorContractError("KunQuant runtime output columns do not match the locked catalog")
            output = {name: output[name] for name in expected}
        return output

    def build_from_canonical(
        self,
        canonical_root: str | Path,
        output_root: str | Path,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas, pyarrow and NumPy are required to read canonical Parquet") from exc

        bars_path = Path(canonical_root) / "bars"
        canonical_manifest_path = Path(canonical_root) / "manifest.json"
        if not canonical_manifest_path.exists():
            raise DataContractError(f"Canonical manifest does not exist: {canonical_manifest_path}")
        with canonical_manifest_path.open("r", encoding="utf-8") as handle:
            canonical_manifest = json.load(handle)
        if canonical_manifest.get("status") != "complete":
            raise DataContractError("Canonical dataset is not complete")
        if not bars_path.exists():
            raise DataContractError(f"Canonical bars do not exist: {bars_path}")
        columns = ["datetime", "instrument", *self.input_fields]
        if self.config.universe == "csi300":
            columns.append("in_csi300")
        bars = pd.read_parquet(bars_path, columns=columns)
        bars["datetime"] = pd.to_datetime(bars["datetime"])
        if start:
            bars = bars[bars["datetime"] >= pd.Timestamp(start)]
        if end:
            bars = bars[bars["datetime"] <= pd.Timestamp(end)]
        if bars.empty:
            raise DataContractError("No canonical rows in requested factor range")
        if self.config.universe == "csi300":
            bars = bars[bars["in_csi300"].fillna(False)]
        if bars.empty:
            raise DataContractError(f"No canonical rows for universe {self.config.universe}")

        dates = pd.Index(sorted(bars["datetime"].unique()), name="datetime")
        instruments = pd.Index(sorted(bars["instrument"].unique()), name="instrument")
        numpy_dtype = np.dtype(np.float64 if self.config.dtype == "double" else np.float32)
        matrices: dict[str, Any] = {}
        for field in self.input_fields:
            panel = bars.pivot(index="datetime", columns="instrument", values=field).reindex(
                index=dates, columns=instruments
            )
            matrices[field] = np.ascontiguousarray(panel.to_numpy(dtype=numpy_dtype))
        del bars, panel

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        final_root = output_root / self.config.factor_set
        staging_root = Path(tempfile.mkdtemp(prefix=f".{self.config.factor_set}.staging.", dir=output_root))
        expected = self.catalog.canonical_names(self.config.factor_set)
        module = self.compile()
        blocking_len = int(module.blocking_len)
        padded_stock_count = len(instruments) + (-len(instruments)) % blocking_len
        raw_output_bytes = len(expected) * len(dates) * padded_stock_count * numpy_dtype.itemsize
        required_free_bytes = raw_output_bytes * 2 + 1024**3
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < required_free_bytes:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise DataContractError(
                "Insufficient free disk for atomic factor build: "
                f"need at least {required_free_bytes / 1024**3:.1f} GiB, "
                f"have {free_bytes / 1024**3:.1f} GiB"
            )

        print(
            f"factors {self.config.factor_set}: computing dates={len(dates)} "
            f"stocks={len(instruments)} padded_stocks={padded_stock_count} outputs={len(expected)} "
            f"temporary_output={raw_output_bytes / 1024**3:.2f} GiB",
            file=sys.stderr,
            flush=True,
        )
        partition_hashes: dict[str, str] = {}
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{self.config.factor_set}.runtime.", dir=output_root
            ) as runtime_directory:
                output_path = Path(runtime_directory) / "kunquant_outputs.dat"
                output_store = np.memmap(
                    output_path,
                    mode="w+",
                    dtype=numpy_dtype,
                    shape=(len(expected), len(dates), padded_stock_count),
                )
                output_buffers = {name: output_store[index] for index, name in enumerate(expected)}
                result = self.run(
                    matrices,
                    module=module,
                    output_buffers=output_buffers,
                    trim_outputs=False,
                )
                del matrices
                output_store.flush()
                print(
                    f"factors {self.config.factor_set}: KunQuant computation complete; writing yearly partitions",
                    file=sys.stderr,
                    flush=True,
                )

                date_years = dates.year
                total_rows = 0
                for year in sorted(set(date_years)):
                    positions = np.flatnonzero(date_years == year)
                    first = int(positions[0])
                    last = int(positions[-1]) + 1
                    year_dates = dates[first:last]
                    factor_data: dict[str, Any] = {
                        "datetime": np.repeat(year_dates.to_numpy(), len(instruments)),
                        "instrument": np.tile(instruments.to_numpy(), len(year_dates)),
                    }
                    for name in expected:
                        factor_data[name] = result[name][first:last, : len(instruments)].reshape(-1)
                    frame = pd.DataFrame(factor_data)
                    destination = staging_root / f"year={year}" / "factors.parquet"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(destination, index=False)
                    partition_hashes[str(year)] = sha256_file(destination)
                    total_rows += len(frame)
                    print(
                        f"factors {self.config.factor_set}: year={year} dates={len(year_dates)} rows={len(frame)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    del factor_data, frame

                output_buffers.clear()
                result.clear()
                output_store.flush()
                output_store._mmap.close()
                del output_store

            metadata = {
                "status": "complete",
                "factor_set": self.config.factor_set,
                "catalog_fingerprint": self.catalog.fingerprint_for(self.config.factor_set),
                "factor_libraries": [library.family for library in self.libraries],
                "input_fields": list(self.input_fields),
                "compilation_fingerprint": self.compilation_fingerprint(),
                "canonical_fingerprint": canonical_manifest.get("fingerprint"),
                "start": str(dates.min()),
                "end": str(dates.max()),
                "instruments": len(instruments),
                "rows": total_rows,
                "columns": len(expected),
                "partitions": partition_hashes,
                "execution": {
                    "output_storage": "temporary_memmap",
                    "raw_output_bytes": raw_output_bytes,
                    "padded_instruments": padded_stock_count,
                },
            }
            metadata["fingerprint"] = stable_hash(metadata)
            atomic_write_json(staging_root / "manifest.json", metadata)
            _atomic_replace_directory(staging_root, final_root)
            return metadata
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)


class PanelFactorEngine:
    """Evaluate registered Pandas panel libraries and publish the standard factor store."""

    def __init__(
        self,
        cache_dir: str | Path,
        config: FactorBuildConfig,
        *,
        registry: FactorLibraryRegistry | None = None,
    ) -> None:
        from rquant.factors.catalog import catalog_from_registry

        self.cache_dir = Path(cache_dir)
        self.config = config
        self.registry = registry or get_library_registry()
        self.catalog = catalog_from_registry(self.registry)
        self.libraries = self.registry.select(config.factor_set)
        if any(library.backend != "panel" for library in self.libraries):
            raise FactorContractError(f"Factor set {config.factor_set} is not a panel-only factor set")
        self.input_fields = self.registry.required_inputs(config.factor_set)

    def compilation_fingerprint(self) -> str:
        return stable_hash(
            {
                "backend": "pandas_panel_v1",
                "catalog": self.catalog.fingerprint_for(self.config.factor_set),
                "factor_set": self.config.factor_set,
                "dtype": self.config.dtype,
                "factor_libraries": self.registry.implementation_fingerprint(self.config.factor_set),
            }
        )

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        absent = sorted(set(self.input_fields) - set(inputs))
        if absent:
            raise DataContractError(f"Missing panel inputs: {absent}")
        outputs: dict[str, Any] = {}
        for library in self.libraries:
            source_outputs = dict(library.calculate(inputs))  # type: ignore[attr-defined]
            expected_sources = {spec.source_name for spec in library.specs}
            if set(source_outputs) != expected_sources:
                raise FactorContractError(f"Panel library {library.family} output differs from its catalog")
            for spec in library.specs:
                outputs[spec.canonical_name] = source_outputs[spec.source_name]
        expected = self.catalog.canonical_names(self.config.factor_set)
        if tuple(outputs) != expected:
            raise FactorContractError("Panel-library output order differs from the locked catalog")
        return outputs

    def build_from_canonical(
        self,
        canonical_root: str | Path,
        output_root: str | Path,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas, pyarrow and NumPy are required for panel factor libraries") from exc

        canonical_root = Path(canonical_root)
        bars_path = canonical_root / "bars"
        canonical_manifest_path = canonical_root / "manifest.json"
        if not canonical_manifest_path.exists():
            raise DataContractError(f"Canonical manifest does not exist: {canonical_manifest_path}")
        with canonical_manifest_path.open("r", encoding="utf-8") as handle:
            canonical_manifest = json.load(handle)
        if canonical_manifest.get("status") != "complete":
            raise DataContractError("Canonical dataset is not complete")
        if not bars_path.exists():
            raise DataContractError(f"Canonical bars do not exist: {bars_path}")

        columns = ["datetime", "instrument", *self.input_fields]
        if self.config.universe == "csi300":
            columns.append("in_csi300")
        bars = pd.read_parquet(bars_path, columns=columns)
        bars["datetime"] = pd.to_datetime(bars["datetime"])
        if start:
            bars = bars[bars["datetime"] >= pd.Timestamp(start)]
        if end:
            bars = bars[bars["datetime"] <= pd.Timestamp(end)]
        if self.config.universe == "csi300":
            bars = bars[bars["in_csi300"].fillna(False)]
        if bars.empty:
            raise DataContractError("No canonical rows in requested panel-factor range and universe")

        dates = pd.Index(sorted(bars["datetime"].unique()), name="datetime")
        instruments = pd.Index(sorted(bars["instrument"].unique()), name="instrument")
        panels = {
            field: bars.pivot(index="datetime", columns="instrument", values=field).reindex(
                index=dates, columns=instruments
            )
            for field in self.input_fields
        }
        del bars
        external, reference_hashes = _load_gtja191_external_inputs(
            canonical_root,
            dates,
            dict(self.config.external_inputs),
            pd,
        )
        panels.update(external)

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        final_root = output_root / self.config.factor_set
        staging_root = Path(tempfile.mkdtemp(prefix=f".{self.config.factor_set}.staging.", dir=output_root))
        expected = self.catalog.canonical_names(self.config.factor_set)
        numpy_dtype = np.dtype(np.float64 if self.config.dtype == "double" else np.float32)
        raw_output_bytes = len(expected) * len(dates) * len(instruments) * numpy_dtype.itemsize
        required_free_bytes = raw_output_bytes * 2 + 1024**3
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < required_free_bytes:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise DataContractError(
                "Insufficient free disk for atomic factor build: "
                f"need at least {required_free_bytes / 1024**3:.1f} GiB, "
                f"have {free_bytes / 1024**3:.1f} GiB"
            )

        partition_hashes: dict[str, str] = {}
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{self.config.factor_set}.runtime.", dir=output_root
            ) as runtime_directory:
                output_store = np.memmap(
                    Path(runtime_directory) / "panel_outputs.dat",
                    mode="w+",
                    dtype=numpy_dtype,
                    shape=(len(expected), len(dates), len(instruments)),
                )
                output_buffers = {name: output_store[index] for index, name in enumerate(expected)}
                for library in self.libraries:
                    library_buffers = {
                        spec.canonical_name: output_buffers[spec.canonical_name] for spec in library.specs
                    }
                    library.calculate_into(panels, library_buffers)  # type: ignore[attr-defined]
                del panels
                output_store.flush()

                total_rows = 0
                for year in sorted(set(dates.year)):
                    positions = np.flatnonzero(dates.year == year)
                    first = int(positions[0])
                    last = int(positions[-1]) + 1
                    year_dates = dates[first:last]
                    factor_data: dict[str, Any] = {
                        "datetime": np.repeat(year_dates.to_numpy(), len(instruments)),
                        "instrument": np.tile(instruments.to_numpy(), len(year_dates)),
                    }
                    for name in expected:
                        factor_data[name] = output_buffers[name][first:last].reshape(-1)
                    frame = pd.DataFrame(factor_data)
                    destination = staging_root / f"year={year}" / "factors.parquet"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_parquet(destination, index=False)
                    partition_hashes[str(year)] = sha256_file(destination)
                    total_rows += len(frame)
                    del factor_data, frame

                output_buffers.clear()
                output_store.flush()
                output_store._mmap.close()
                del output_store

            metadata = {
                "status": "complete",
                "factor_set": self.config.factor_set,
                "catalog_fingerprint": self.catalog.fingerprint_for(self.config.factor_set),
                "compilation_fingerprint": self.compilation_fingerprint(),
                "canonical_fingerprint": canonical_manifest.get("fingerprint"),
                "factor_libraries": [library.family for library in self.libraries],
                "input_fields": list(self.input_fields),
                "reference_inputs": reference_hashes,
                "start": str(dates.min()),
                "end": str(dates.max()),
                "instruments": len(instruments),
                "rows": total_rows,
                "columns": len(expected),
                "partitions": partition_hashes,
                "execution": {
                    "backend": "pandas_panel_v1",
                    "output_storage": "temporary_memmap",
                    "raw_output_bytes": raw_output_bytes,
                },
            }
            metadata["fingerprint"] = stable_hash(metadata)
            atomic_write_json(staging_root / "manifest.json", metadata)
            _atomic_replace_directory(staging_root, final_root)
            return metadata
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)


def _load_gtja191_external_inputs(
    canonical_root: Path,
    dates: Any,
    external_inputs: dict[str, str],
    pd: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    unexpected = sorted(set(external_inputs) - {"style_factors"})
    if unexpected:
        raise DataContractError(f"GTJA191 received unknown external inputs: {unexpected}")
    benchmark_path = canonical_root / "reference" / "index_daily.parquet"
    if not benchmark_path.exists():
        raise DataContractError(f"GTJA191 benchmark input does not exist: {benchmark_path}")
    benchmark = pd.read_parquet(benchmark_path)
    benchmark_columns = {"ts_code", "datetime", "open", "close"}
    if not benchmark_columns.issubset(benchmark.columns):
        raise DataContractError(
            f"GTJA191 benchmark input is missing columns: {sorted(benchmark_columns - set(benchmark.columns))}"
        )
    preferred_codes = ("000300.SH", "399300.SZ")
    selected_code = next((code for code in preferred_codes if code in set(benchmark["ts_code"])), None)
    if selected_code is None:
        raise DataContractError("GTJA191 benchmark input has no CSI300 series")
    benchmark = benchmark[benchmark["ts_code"] == selected_code].copy()
    benchmark["datetime"] = pd.to_datetime(benchmark["datetime"])
    if benchmark["datetime"].duplicated().any():
        raise DataContractError(f"GTJA191 benchmark input contains duplicate dates for {selected_code}")
    benchmark = benchmark.set_index("datetime").sort_index()
    benchmark = benchmark.reindex(dates)

    external = {
        "benchmark_open": benchmark["open"],
        "benchmark_close": benchmark["close"],
    }
    reference_hashes = {"benchmark": sha256_file(benchmark_path)}
    style_value = external_inputs.get("style_factors")
    if not style_value:
        return external, reference_hashes
    style_path = Path(style_value)
    if not style_path.exists():
        raise DataContractError(f"GTJA191 style-factor input does not exist: {style_path}")
    style = pd.read_parquet(style_path) if style_path.suffix.lower() == ".parquet" else pd.read_csv(style_path)
    date_column = "datetime" if "datetime" in style.columns else "trade_date"
    required = {date_column, "mkt", "smb", "hml"}
    if not required.issubset(style.columns):
        raise DataContractError(
            f"GTJA191 style-factor input is missing columns: {sorted(required - set(style.columns))}"
        )
    style[date_column] = pd.to_datetime(style[date_column])
    if style[date_column].duplicated().any():
        raise DataContractError("GTJA191 style-factor input contains duplicate dates")
    style = style.set_index(date_column).sort_index().reindex(dates)

    external.update({"mkt": style["mkt"], "smb": style["smb"], "hml": style["hml"]})
    reference_hashes["style_factors"] = sha256_file(style_path)
    return external, reference_hashes


def create_factor_engine(
    cache_dir: str | Path,
    config: FactorBuildConfig,
    *,
    registry: FactorLibraryRegistry | None = None,
) -> KunQuantFactorEngine | PanelFactorEngine:
    selected_registry = registry or get_library_registry()
    backends = {library.backend for library in selected_registry.select(config.factor_set)}
    if backends == {"kunquant"}:
        return KunQuantFactorEngine(cache_dir, config, registry=selected_registry)
    if backends == {"panel"}:
        return PanelFactorEngine(cache_dir, config, registry=selected_registry)
    raise FactorContractError(
        f"Factor set {config.factor_set} mixes unsupported execution backends: {sorted(backends)}"
    )


def validate_factor_frame(frame: Any, factor_set: FactorSet) -> dict[str, Any]:
    catalog = get_catalog()
    expected = catalog.canonical_names(factor_set)
    columns = tuple(column for column in frame.columns if column not in {"datetime", "instrument"})
    if columns != expected:
        raise FactorContractError(
            "Factor frame columns are missing, duplicated, exposed by source name, or out of order"
        )
    forbidden = {spec.source_name for spec in catalog.select(factor_set)} & set(frame.columns)
    if forbidden:
        raise FactorContractError(f"Factor frame exposes raw source names: {sorted(forbidden)}")
    return {
        "factor_set": factor_set,
        "columns": len(columns),
        "rows": len(frame),
        "catalog": catalog.fingerprint_for(factor_set),
    }
