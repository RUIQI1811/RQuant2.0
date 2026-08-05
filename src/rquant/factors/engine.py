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
from rquant.factors import alpha101_missing
from rquant.factors.alpha101_missing import ExtendedAllData, build_missing
from rquant.factors.catalog import FactorSet, get_catalog
from rquant.io import atomic_write_json, sha256_file, stable_hash

INPUT_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "cap", "sector", "industry", "subindustry")
ALPHA101_TS_RANK_SEMANTICS = "percentile_v1"


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


class KunQuantFactorEngine:
    """Compile and execute the locked factor graph against canonical daily panels."""

    def __init__(self, cache_dir: str | Path, config: FactorBuildConfig | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config or FactorBuildConfig()
        self.catalog = get_catalog()

    def compilation_fingerprint(self) -> str:
        try:
            version = importlib.metadata.version("KunQuant")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        return stable_hash(
            {
                "kunquant": version,
                "catalog": self.catalog.fingerprint,
                "factor_set": self.config.factor_set,
                "dtype": self.config.dtype,
                "input_layout": self.config.input_layout,
                "output_layout": self.config.output_layout,
                "missing_alpha101_source": inspect.getsource(alpha101_missing),
                "alpha101_ts_rank_semantics": ALPHA101_TS_RANK_SEMANTICS,
            }
        )

    def build_graph(self) -> Any:
        try:
            from KunQuant.Driver import KunCompilerConfig
            from KunQuant.Op import Builder, Input, Output
            from KunQuant.predefined import Alpha101, Alpha158
            from KunQuant.Stage import Function
        except ImportError as exc:
            raise DependencyError("KunQuant==0.1.11 is required for factor computation") from exc

        builder = Builder()
        with builder:
            inputs = {name: Input(name) for name in INPUT_FIELDS}
            outputs: dict[str, Any] = {}
            if self.config.factor_set in {"qlib_alpha158", "combined"}:
                alpha158_data = Alpha158.AllData(
                    open=inputs["open"],
                    close=inputs["close"],
                    high=inputs["high"],
                    low=inputs["low"],
                    volume=inputs["volume"],
                    amount=inputs["amount"],
                )
                fields, names = alpha158_data.build(
                    {
                        "kbar": {},
                        "price": {
                            "windows": [0],
                            "feature": [
                                ("OPEN", inputs["open"]),
                                ("HIGH", inputs["high"]),
                                ("LOW", inputs["low"]),
                                ("VWAP", inputs["vwap"]),
                            ],
                        },
                        "rolling": {},
                    }
                )
                expected = [spec.source_name for spec in self.catalog.select("qlib_alpha158")]
                if names != expected:
                    raise FactorContractError(
                        "KunQuant Alpha158 output order differs from the locked catalog; "
                        "upgrade requires explicit remapping"
                    )
                outputs.update(dict(zip(names, fields, strict=True)))

            if self.config.factor_set in {"wq_alpha101", "combined"}:
                builtin_data = Alpha101.AllData(
                    open=inputs["open"],
                    close=inputs["close"],
                    high=inputs["high"],
                    low=inputs["low"],
                    volume=inputs["volume"],
                    amount=inputs["amount"],
                    vwap=inputs["vwap"],
                )
                extended = ExtendedAllData(**inputs)
                missing = build_missing(extended)
                original_ts_rank = Alpha101.ts_rank

                def percentile_ts_rank(value: Any, window: int) -> Any:
                    return Alpha101.TsRank(value, window) / float(window)

                Alpha101.ts_rank = percentile_ts_rank
                try:
                    for ordinal in range(1, 102):
                        name = f"alpha{ordinal:03d}"
                        if hasattr(Alpha101, name):
                            outputs[name] = getattr(Alpha101, name)(builtin_data)
                        elif name in missing:
                            outputs[name] = missing[name]
                        else:
                            raise FactorContractError(f"No implementation for {name}")
                finally:
                    Alpha101.ts_rank = original_ts_rank

            mapping = self.catalog.source_to_canonical(self.config.factor_set)
            if set(outputs) != set(mapping):
                missing_names = sorted(set(mapping) - set(outputs))
                extra_names = sorted(set(outputs) - set(mapping))
                raise FactorContractError(
                    f"Factor graph/catalog mismatch; missing={missing_names}, extra={extra_names}"
                )
            for source_name, canonical_name in mapping.items():
                Output(outputs[source_name], canonical_name)

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
                "catalog_fingerprint": self.catalog.fingerprint,
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

        absent = sorted(set(INPUT_FIELDS) - set(inputs))
        if absent:
            raise DataContractError(f"Missing KunQuant inputs: {absent}")
        numpy_dtype = np.float64 if self.config.dtype == "double" else np.float32
        normalized = {name: np.ascontiguousarray(np.asarray(inputs[name], dtype=numpy_dtype)) for name in INPUT_FIELDS}
        shapes = {tuple(normalized[name].shape) for name in INPUT_FIELDS}
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
        columns = ["datetime", "instrument", *INPUT_FIELDS]
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
        for field in INPUT_FIELDS:
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
                        f"factors {self.config.factor_set}: year={year} dates={len(year_dates)} "
                        f"rows={len(frame)}",
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
                "catalog_fingerprint": self.catalog.fingerprint,
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


def validate_factor_frame(frame: Any, factor_set: FactorSet) -> dict[str, Any]:
    catalog = get_catalog()
    expected = catalog.canonical_names(factor_set)
    columns = tuple(column for column in frame.columns if column.startswith(("a158_", "a101_")))
    if columns != expected:
        raise FactorContractError(
            "Factor frame columns are missing, duplicated, exposed by source name, or out of order"
        )
    forbidden = {spec.source_name for spec in catalog.select(factor_set)} & set(frame.columns)
    if forbidden:
        raise FactorContractError(f"Factor frame exposes raw source names: {sorted(forbidden)}")
    return {"factor_set": factor_set, "columns": len(columns), "rows": len(frame), "catalog": catalog.fingerprint}
