from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rquant.errors import DataContractError, DependencyError, FactorContractError
from rquant.factors.catalog import FactorSet, catalog_from_registry, get_catalog
from rquant.factors.extensions import kunquant as kunquant_extensions
from rquant.factors.libraries.base import CANONICAL_INPUT_FIELDS
from rquant.factors.libraries.registry import FactorLibraryRegistry, get_library_registry
from rquant.io import atomic_replace_directory as _atomic_replace_directory
from rquant.io import atomic_write_json, sha256_file, stable_hash
from rquant.universe import UNIVERSE_COLUMNS, universe_mask, validate_universe

if TYPE_CHECKING:
    from rquant.factors.statements import StatementResults

INPUT_FIELDS = CANONICAL_INPUT_FIELDS
FACTOR_ENGINE_VERSION = 3


def cpp_compiler_probe() -> tuple[str, list[str], dict[str, str] | None]:
    """Return the name, version-probe argv and environment of the C++ compiler KunQuant's JIT uses here."""
    if sys.platform != "win32":
        return "clang++", ["clang++", "--version"], None
    # KunQuant compiles with MSVC on Windows and finds cl.exe through setuptools when it is not on PATH.
    env: dict[str, str] | None = None
    try:
        from KunQuant.jit.env import get_compiler_env

        # The discovery helper prints diagnostics to stdout, which must stay machine-readable for `doctor`.
        with contextlib.redirect_stdout(sys.stderr):
            env = dict(get_compiler_env())
    except Exception:
        env = None
    compiler = shutil.which("cl.exe", path=(env or {}).get("PATH")) or "cl.exe"
    # cl.exe has no version flag; invoked bare it prints its version banner and exits successfully.
    return "cl.exe", [compiler], env


def expose_msvc_compiler() -> None:
    """Put KunQuant's discovered cl.exe on this process's PATH.

    KunQuant launches a bare ``cl.exe`` with a child environment that names the MSVC directory, but Windows resolves
    the executable from the parent's PATH, so compilation fails whenever that PATH was not already extended.
    """
    if sys.platform != "win32" or shutil.which("cl.exe"):
        return
    _, command, _ = cpp_compiler_probe()
    if os.path.isabs(command[0]):
        os.environ["PATH"] = os.path.dirname(command[0]) + os.pathsep + os.environ.get("PATH", "")


@contextlib.contextmanager
def _stdout_to_stderr() -> Any:
    """Send child-process output to stderr; compiler messages would otherwise corrupt the JSON printed on stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def load_kunquant_library(path: str) -> Any:
    """Load a compiled KunQuant library, staging it at an ASCII path on Windows when needed.

    KunRunner opens the DLL through a narrow-string Windows API, which cannot address non-ASCII paths such as a
    project folder with Chinese characters; the cached build stays in the project and only a copy is loaded.
    """
    from KunQuant.runner import KunRunner

    if sys.platform == "win32" and not path.isascii():
        source = Path(path)
        loadable = Path(tempfile.gettempdir()) / "rquant-kunquant" / source.parent.name / source.name
        if not str(loadable).isascii():
            raise DependencyError(
                f"KunQuant cannot load libraries from non-ASCII paths; set TEMP to an ASCII directory (got {loadable})"
            )
        if not loadable.exists() or sha256_file(loadable) != sha256_file(source):
            loadable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, loadable)
        path = str(loadable)
    return KunRunner.Library.load(path)


def cpp_compiler_available() -> bool:
    _, command, env = cpp_compiler_probe()
    try:
        return subprocess.run(command, capture_output=True, env=env, check=False).returncode == 0
    except OSError:
        return False


@dataclass(frozen=True)
class FactorBuildConfig:
    factor_set: FactorSet = "qlib_alpha158"
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
        self.canonical_fields = self.registry.required_inputs(self.config.factor_set)
        self.reference_fields = self.registry.reference_inputs(self.config.factor_set)
        self.input_fields = self.canonical_fields + self.reference_fields
        if any(library.backend != "kunquant" for library in self.libraries):
            raise FactorContractError(f"Factor set {self.config.factor_set} is not a KunQuant-only factor set")
        quote_fields = {library.quote_fields for library in self.libraries}
        if len(quote_fields) > 1:
            raise FactorContractError(
                f"Factor set {self.config.factor_set} mixes libraries on calendar days and on their own quotes"
            )
        self.quote_fields: tuple[str, ...] = next(iter(quote_fields), ())
        # Factors published unchanged from a statement-graph result skip the daily graph: ``run`` writes them straight
        # into their output buffers.
        self.statement_factors = {
            spec.canonical_name: library.statement_factors[spec.source_name]
            for library in self.libraries
            for spec in library.specs
            if spec.source_name in library.statement_factors
        }

    @property
    def strict_float(self) -> bool:
        """Whether the graph must round every operation like NumPy, without fused multiply-adds."""
        return any(library.strict_float for library in self.libraries)

    def compiler_config(self) -> Any:
        from KunQuant.jit import cfake

        # GCC and Clang fuse a * b + c into one FMA instruction by default, which rounds differently from the
        # pandas reference; MSVC does not contract under its default /fp:precise.
        flags = ("-ffp-contract=off",) if self.strict_float and sys.platform != "win32" else ()
        preludes = list(dict.fromkeys(library.cpp_prelude for library in self.libraries if library.cpp_prelude))
        if not preludes:
            return cfake.CppCompilerConfig(other_flags=flags)
        prelude = "#include <algorithm>\n#include <cmath>\n#include <cstddef>\n#include <limits>\n#include <vector>\n"
        prelude += "\n".join(preludes) + "\n"

        def compile_with_prelude(sources: list[tuple[str, str]], compile_one: Any) -> Any:
            return cfake.multi_thread_compile([(name, prelude + source) for name, source in sources], compile_one)

        return cfake.CppCompilerConfig(other_flags=flags, for_each=compile_with_prelude)

    def compilation_fingerprint(self) -> str:
        try:
            version = importlib.metadata.version("KunQuant")
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        strict = {"strict_float": True} if self.strict_float else {}
        return stable_hash(
            {
                **strict,
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
                source_to_canonical = {
                    spec.source_name: spec.canonical_name
                    for spec in library.specs
                    if spec.source_name not in library.statement_factors
                }
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

            expected = self.graph_outputs
            if tuple(outputs) != expected:
                raise FactorContractError("Factor-library output order differs from the locked catalog")
            for canonical_name, expression in outputs.items():
                Output(expression, canonical_name)

        return Function(builder.ops), self._kun_compiler_config()

    def _kun_compiler_config(self) -> Any:
        from KunQuant.Driver import KunCompilerConfig

        return KunCompilerConfig(
            dtype=self.config.dtype,
            input_layout=self.config.input_layout,
            output_layout=self.config.output_layout,
            # None lets KunQuant select the architecture-safe value (unaligned is unsupported on Apple Silicon).
            allow_unaligned=None,
            split_source=max(library.compile_split_source for library in self.libraries),
            # Prefer deterministic double-precision rolling statistics over the faster incremental approximation.
            options={"no_fast_stat": True},
        )

    @property
    def graph_outputs(self) -> tuple[str, ...]:
        """Catalog outputs the daily graph computes, in catalog order."""
        names = self.catalog.canonical_names(self.config.factor_set)
        return tuple(name for name in names if name not in self.statement_factors)

    @property
    def statement_libraries(self) -> tuple[Any, ...]:
        return tuple(library for library in self.libraries if library.statement_outputs)

    def build_statement_graphs(self) -> list[tuple[str, Any, Any]]:
        """One KunQuant function per library with a statement graph, over fiscal quarters x instruments."""
        try:
            from KunQuant.Op import Builder, Input, Output
            from KunQuant.Stage import Function
        except ImportError as exc:
            raise DependencyError("KunQuant==0.1.11 is required for factor computation") from exc

        functions = []
        for library in self.statement_libraries:
            if any(not name.startswith(("q_", "y_")) for name in library.statement_outputs):
                raise FactorContractError(f"Statement outputs of {library.family} must start with q_ or y_")
            published = set(library.reference_inputs) | set(library.statement_factors.values())
            if not set(library.statement_outputs) <= published:
                raise FactorContractError(f"Statement outputs of {library.family} are neither inputs nor factors")
            builder = Builder()
            with builder:
                inputs = {name: Input(name) for name in (*library.statement_fields, "fiscal_quarter")}
                outputs = dict(library.build_statements(inputs))
                if tuple(outputs) != library.statement_outputs:
                    raise FactorContractError(f"Statement graph of {library.family} differs from its declared outputs")
                for name, expression in outputs.items():
                    Output(expression, name)
            functions.append((f"statements_{library.family}", Function(builder.ops), self._kun_compiler_config()))
        return functions

    def compile(self) -> Any:
        try:
            from KunQuant.jit import cfake
        except ImportError as exc:
            raise DependencyError("KunQuant==0.1.11 is required for factor computation") from exc

        expose_msvc_compiler()
        fingerprint = self.compilation_fingerprint()
        target = self.cache_dir / "kunquant" / fingerprint
        target.mkdir(parents=True, exist_ok=True)
        function, compiler_config = self.build_graph()
        library_name = f"rquant_{self.config.factor_set}_{fingerprint[:12]}"
        with _stdout_to_stderr():
            compiled = cfake.compileit(
                [("rquant_factors", function, compiler_config)],
                library_name,
                self.compiler_config(),
                tempdir=str(target),
                keep_files=True,
                load=False,
            )
        library = load_kunquant_library(compiled)
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

    def compile_statements(self) -> dict[str, Any]:
        """Compile the statement graphs; returns each library family's KunQuant module."""
        functions = self.build_statement_graphs()
        if not functions:
            return {}
        from KunQuant.jit import cfake

        expose_msvc_compiler()
        fingerprint = self.compilation_fingerprint()
        target = self.cache_dir / "kunquant" / fingerprint
        target.mkdir(parents=True, exist_ok=True)
        with _stdout_to_stderr():
            compiled = cfake.compileit(
                functions,
                f"rquant_{self.config.factor_set}_{fingerprint[:12]}_statements",
                self.compiler_config(),
                tempdir=str(target),
                keep_files=True,
                load=False,
            )
        library = load_kunquant_library(compiled)
        return {name.removeprefix("statements_"): library.getModule(name) for name, _, _ in functions}

    def run_statements(
        self, inputs: dict[str, Any], dates: Any, instruments: Any, modules: dict[str, Any] | None = None
    ) -> StatementResults:
        """Run every library's statement graph over its first-reported statements in ``inputs["fundamentals"]``."""
        from rquant.factors.statements import QuarterGrid, StatementResults

        results = StatementResults()
        if not self.statement_libraries:
            return results
        try:
            import numpy as np
            import pandas as pd
            from KunQuant.runner import KunRunner as kr
        except ImportError as exc:
            raise DependencyError("NumPy, pandas and KunQuant are required for factor computation") from exc

        if "fundamentals" not in inputs:
            raise DataContractError("Statement graphs need the reference input `fundamentals`")
        modules = self.compile_statements() if modules is None else modules
        dtype = np.float64 if self.config.dtype == "double" else np.float32
        executor = kr.createMultiThreadExecutor(self.config.workers)
        dates, instruments = pd.DatetimeIndex(dates), pd.Index(instruments)
        for library in self.statement_libraries:
            grid = QuarterGrid(library.prepare_statements(inputs["fundamentals"]), dates, instruments)
            module = modules[library.family]
            pad = (-len(instruments)) % int(module.blocking_len)
            arrays = {
                name: np.pad(values.astype(dtype), ((0, 0), (0, pad)), constant_values=np.nan)
                for name, values in grid.statement_arrays(library.statement_fields).items()
            }
            output = kr.runGraph(executor, module, arrays, 0, grid.quarters)
            for name in library.statement_outputs:
                results.add(name, grid, output[name][:, : len(instruments)])
        return results

    def run(
        self,
        inputs: dict[str, Any],
        module: Any | None = None,
        *,
        statements: StatementResults | None = None,
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
        if self.statement_factors and statements is None:
            raise DataContractError(f"Factor set {self.config.factor_set} needs its statement-graph results")
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
        for library in self.libraries:
            library.validate_inputs(normalized)
        time_count, stock_count = shape
        packing = _QuotePacking(np, normalized, self.quote_fields) if self.quote_fields else None
        if packing is not None:
            normalized = {name: packing.pack(values) for name, values in normalized.items()}
        run_length = packing.length if packing is not None else time_count
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
        run_buffers = {name: output_buffers[name][:run_length] for name in self.graph_outputs} if output_buffers else {}
        output = kr.runGraph(executor, module, normalized, 0, run_length, run_buffers)
        if packing is not None:
            output = {
                name: packing.unpack(values, output_buffers[name] if output_buffers else None, time_count)
                for name, values in output.items()
            }
        if self.statement_factors:
            # Published on the days with a close, like every other factor; infinite values are missing.
            trading = np.isfinite(np.asarray(inputs["close"], dtype=float))
            for name, result in self.statement_factors.items():
                values = statements.daily(result)
                target = output_buffers[name] if output_buffers else np.empty((time_count, stock_count + pad_stocks))
                target[:, :stock_count] = np.where(trading & np.isfinite(values), values, np.nan)
                target[:, stock_count:] = np.nan
                output[name] = target
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
        dates, instruments, panels, output_mask = _load_canonical_panels(
            pd, bars_path, self.canonical_fields, self.config.universe, start=start, end=end
        )
        external, reference_hashes = _load_panel_references(
            self.libraries, Path(canonical_root), dates, instruments, dict(self.config.external_inputs), pd
        )
        statements = self.run_statements(external, dates, instruments)
        external.update({name: statements.daily(name) for name in self.reference_fields if name in statements})
        # Cross-sectional operators use the research universe's members of each day.
        external["universe"] = pd.DataFrame(output_mask, index=dates, columns=instruments)
        missing_references = sorted(set(self.reference_fields) - set(external))
        if missing_references:
            raise DataContractError(f"Factor libraries did not load their reference inputs: {missing_references}")
        numpy_dtype = np.dtype(np.float64 if self.config.dtype == "double" else np.float32)
        matrices = {field: np.ascontiguousarray(panel.to_numpy(dtype=numpy_dtype)) for field, panel in panels.items()}
        for field in self.reference_fields:
            matrices[field] = _aligned_matrix(np, external[field], dates, instruments, numpy_dtype)
        del panels, external

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
                    statements=statements,
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

                total_rows = _write_year_partitions(
                    pd,
                    np,
                    staging_root,
                    dates,
                    instruments,
                    expected,
                    lambda name, first, last: result[name][first:last, : len(instruments)],
                    output_mask,
                    partition_hashes,
                    label=self.config.factor_set,
                )

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
                "reference_inputs": reference_hashes,
                "compilation_fingerprint": self.compilation_fingerprint(),
                "canonical_fingerprint": canonical_manifest.get("fingerprint"),
                "start": str(dates[output_mask.any(axis=1)].min()),
                "end": str(dates[output_mask.any(axis=1)].max()),
                "instruments": len(instruments),
                "universe": self.config.universe,
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
                "factor_engine_version": FACTOR_ENGINE_VERSION,
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

        dates, instruments, panels, output_mask = _load_canonical_panels(
            pd, bars_path, self.input_fields, self.config.universe, start=start, end=end
        )
        external, reference_hashes = _load_panel_references(
            self.libraries,
            canonical_root,
            dates,
            instruments,
            dict(self.config.external_inputs),
            pd,
        )
        panels.update(external)
        # Cross-sectional panel operators use the research universe's members of each day.
        panels["universe"] = pd.DataFrame(output_mask, index=dates, columns=instruments)

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

                total_rows = _write_year_partitions(
                    pd,
                    np,
                    staging_root,
                    dates,
                    instruments,
                    expected,
                    lambda name, first, last: output_buffers[name][first:last],
                    output_mask,
                    partition_hashes,
                    label=self.config.factor_set,
                )

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
                "start": str(dates[output_mask.any(axis=1)].min()),
                "end": str(dates[output_mask.any(axis=1)].max()),
                "instruments": len(instruments),
                "universe": self.config.universe,
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


class SelectionFactorEngine:
    """Publish a registered factor selection by copying columns from current source factor stores.

    No formula is recomputed, so members may come from KunQuant and Pandas-panel libraries alike. Every source store
    must be complete, built from the current canonical dataset and match the current source engine fingerprint;
    otherwise the build fails instead of mixing stale artifacts into the selection.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        config: FactorBuildConfig,
        *,
        registry: FactorLibraryRegistry | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.registry = registry or get_library_registry()
        self.catalog = catalog_from_registry(self.registry)
        selection = self.registry.selection(config.factor_set)
        if selection is None:
            raise FactorContractError(f"Factor set {config.factor_set} is not a registered factor selection")
        self.selection = selection
        self.libraries = self.registry.select(config.factor_set)
        self.input_fields = self.registry.required_inputs(config.factor_set)
        self.sources = {name: self.registry.source_factor_set(selection, name) for name in selection.canonical_names}
        self.source_engines = {
            source: create_factor_engine(
                cache_dir, dataclasses.replace(config, factor_set=source), registry=self.registry
            )
            for source in selection.source_factor_sets
            if source in self.sources.values()
        }

    def compilation_fingerprint(self) -> str:
        return stable_hash(
            {
                "backend": "selection_v1",
                "factor_engine_version": FACTOR_ENGINE_VERSION,
                "catalog": self.catalog.fingerprint_for(self.config.factor_set),
                "factor_set": self.config.factor_set,
                "selection": self.selection.definition(),
                "sources": {source: engine.compilation_fingerprint() for source, engine in self.source_engines.items()},
            }
        )

    def build_from_canonical(
        self,
        canonical_root: str | Path,
        output_root: str | Path,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise DependencyError("pandas and pyarrow are required to publish a factor selection") from exc

        canonical_manifest_path = Path(canonical_root) / "manifest.json"
        if not canonical_manifest_path.exists():
            raise DataContractError(f"Canonical manifest does not exist: {canonical_manifest_path}")
        with canonical_manifest_path.open("r", encoding="utf-8") as handle:
            canonical_manifest = json.load(handle)
        if canonical_manifest.get("status") != "complete":
            raise DataContractError("Canonical dataset is not complete")

        output_root = Path(output_root)
        source_manifests = {
            source: _current_source_manifest(
                output_root / source, source, engine, self.catalog, canonical_manifest, self.config.universe
            )
            for source, engine in self.source_engines.items()
        }
        years = {source: set(manifest["partitions"]) for source, manifest in source_manifests.items()}
        if len({frozenset(value) for value in years.values()}) != 1:
            raise DataContractError(f"Source factor stores cover different years: {years}")

        expected = self.catalog.canonical_names(self.config.factor_set)
        by_source: dict[str, list[str]] = {}
        for name in expected:
            by_source.setdefault(self.sources[name], []).append(name)
        start_timestamp = pd.Timestamp(start) if start else None
        end_timestamp = pd.Timestamp(end) if end else None

        output_root.mkdir(parents=True, exist_ok=True)
        rows = max(int(manifest["rows"]) for manifest in source_manifests.values())
        required_free_bytes = len(expected) * rows * 8 * 2 + 1024**3
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes < required_free_bytes:
            raise DataContractError(
                "Insufficient free disk for atomic factor build: "
                f"need at least {required_free_bytes / 1024**3:.1f} GiB, "
                f"have {free_bytes / 1024**3:.1f} GiB"
            )
        final_root = output_root / self.config.factor_set
        staging_root = Path(tempfile.mkdtemp(prefix=f".{self.config.factor_set}.staging.", dir=output_root))
        partition_hashes: dict[str, str] = {}
        total_rows = 0
        instruments: set[str] = set()
        first_datetime = last_datetime = None
        try:
            for year in sorted(next(iter(years.values())), key=int):
                frame = _read_selected_columns(pd, output_root, source_manifests, by_source, year)
                frame = frame.loc[:, ["datetime", "instrument", *expected]]
                if start_timestamp is not None:
                    frame = frame[pd.to_datetime(frame["datetime"]) >= start_timestamp]
                if end_timestamp is not None:
                    frame = frame[pd.to_datetime(frame["datetime"]) <= end_timestamp]
                if frame.empty:
                    continue
                destination = staging_root / f"year={year}" / "factors.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(destination, index=False)
                partition_hashes[year] = sha256_file(destination)
                total_rows += len(frame)
                instruments.update(frame["instrument"].unique())
                year_first, year_last = frame["datetime"].min(), frame["datetime"].max()
                first_datetime = year_first if first_datetime is None else min(first_datetime, year_first)
                last_datetime = year_last if last_datetime is None else max(last_datetime, year_last)
                print(
                    f"factors {self.config.factor_set}: year={year} rows={len(frame)} columns={len(expected)}",
                    file=sys.stderr,
                    flush=True,
                )
                del frame
            if not total_rows:
                raise DataContractError(f"Factor build for {self.config.factor_set} produced no rows")

            metadata = {
                "status": "complete",
                "factor_set": self.config.factor_set,
                "catalog_fingerprint": self.catalog.fingerprint_for(self.config.factor_set),
                "compilation_fingerprint": self.compilation_fingerprint(),
                "canonical_fingerprint": canonical_manifest.get("fingerprint"),
                "factor_libraries": [library.family for library in self.libraries],
                "input_fields": list(self.input_fields),
                "source_factor_sets": {
                    source: {
                        "fingerprint": manifest["fingerprint"],
                        "compilation_fingerprint": manifest["compilation_fingerprint"],
                        "columns": by_source[source],
                    }
                    for source, manifest in source_manifests.items()
                },
                "selection": self.selection.definition(),
                "start": str(first_datetime),
                "end": str(last_datetime),
                "instruments": len(instruments),
                "universe": self.config.universe,
                "rows": total_rows,
                "columns": len(expected),
                "partitions": partition_hashes,
                "execution": {"backend": "selection_v1", "output_storage": "copied_source_columns"},
            }
            metadata["fingerprint"] = stable_hash(metadata)
            atomic_write_json(staging_root / "manifest.json", metadata)
            _atomic_replace_directory(staging_root, final_root)
            return metadata
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)


def _current_source_manifest(
    directory: Path,
    factor_set: str,
    engine: Any,
    catalog: Any,
    canonical_manifest: dict[str, Any],
    universe: str,
) -> dict[str, Any]:
    """Load a source factor-store manifest and refuse it unless it describes the current code and data."""
    path = directory / "manifest.json"
    if not path.exists():
        raise DataContractError(f"Source factor set {factor_set} is not built: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" or manifest.get("factor_set") != factor_set:
        raise DataContractError(f"Source factor manifest is not complete for {factor_set}: {path}")
    payload = dict(manifest)
    if payload.pop("fingerprint", None) != stable_hash(payload):
        raise DataContractError(f"Source factor manifest fingerprint is invalid: {path}")
    if manifest.get("catalog_fingerprint") != catalog.fingerprint_for(factor_set):
        raise DataContractError(f"Source factor set {factor_set} was built from a stale catalog; rebuild it")
    if manifest.get("compilation_fingerprint") != engine.compilation_fingerprint():
        raise DataContractError(f"Source factor set {factor_set} was built by stale factor code; rebuild it")
    if manifest.get("canonical_fingerprint") != canonical_manifest.get("fingerprint"):
        raise DataContractError(f"Source factor set {factor_set} was not built from the current canonical dataset")
    if manifest.get("universe") != universe:
        raise DataContractError(f"Source factor set {factor_set} uses universe {manifest.get('universe')!r}")
    return manifest


def _read_selected_columns(
    pd: Any,
    output_root: Path,
    source_manifests: dict[str, dict[str, Any]],
    by_source: dict[str, list[str]],
    year: str,
) -> Any:
    """Read one year of the selected columns, requiring every source to publish identical row keys."""
    frame = None
    parts = []
    for source, names in by_source.items():
        partition = output_root / source / f"year={year}" / "factors.parquet"
        if not partition.exists() or sha256_file(partition) != source_manifests[source]["partitions"][year]:
            raise DataContractError(f"Factor partition is missing or differs from its manifest: {partition}")
        part = pd.read_parquet(partition, columns=["datetime", "instrument", *names])
        if frame is None:
            frame = part
            parts.append(part)
            continue
        same_keys = (
            len(part) == len(frame)
            and (part["datetime"].to_numpy() == frame["datetime"].to_numpy()).all()
            and (part["instrument"].to_numpy() == frame["instrument"].to_numpy()).all()
        )
        if not same_keys:
            raise DataContractError(f"Source factor stores have different rows in year={year}: {source}")
        parts.append(part.loc[:, names].set_axis(frame.index))
    if frame is None:
        raise DataContractError("Factor selection has no source columns")
    # One concat instead of a column-by-column insert, which fragments the frame for hundreds of members.
    return pd.concat(parts, axis=1)


def _load_canonical_panels(
    pd: Any,
    bars_path: Path,
    input_fields: tuple[str, ...],
    universe: str,
    *,
    start: str | None,
    end: str | None,
) -> tuple[Any, Any, dict[str, Any], Any]:
    """Build full-history input panels and the mask of rows that belong in the published store.

    Time-series operators need each instrument's history before it entered the research universe, so the panels keep
    every canonical bar of every instrument that is ever in the universe. Membership only decides which computed rows
    are published.
    """
    validate_universe(universe)
    columns = ["datetime", "instrument", *input_fields, *UNIVERSE_COLUMNS[universe]]
    bars = pd.read_parquet(bars_path, columns=list(dict.fromkeys(columns)))
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    if start:
        bars = bars[bars["datetime"] >= pd.Timestamp(start)]
    if end:
        bars = bars[bars["datetime"] <= pd.Timestamp(end)]
    if bars.empty:
        raise DataContractError("No canonical rows in requested factor range")
    bars = bars.assign(in_universe=universe_mask(bars, universe).to_numpy())
    members = bars.loc[bars["in_universe"], "instrument"].unique()
    if not len(members):
        raise DataContractError(f"No canonical rows for universe {universe}")
    bars = bars[bars["instrument"].isin(members)]

    dates = pd.Index(sorted(bars["datetime"].unique()), name="datetime")
    instruments = pd.Index(sorted(members), name="instrument")
    panels = {
        field: bars.pivot(index="datetime", columns="instrument", values=field).reindex(
            index=dates, columns=instruments
        )
        for field in input_fields
    }
    output_mask = (
        bars.pivot(index="datetime", columns="instrument", values="in_universe")
        .reindex(index=dates, columns=instruments)
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    return dates, instruments, panels, output_mask


def _write_year_partitions(
    pd: Any,
    np: Any,
    staging_root: Path,
    dates: Any,
    instruments: Any,
    expected: tuple[str, ...],
    values_for: Any,
    output_mask: Any,
    partition_hashes: dict[str, str],
    *,
    label: str,
) -> int:
    """Write one Parquet partition per year containing only rows inside the research universe."""
    total_rows = 0
    date_years = dates.year
    for year in sorted(set(date_years)):
        positions = np.flatnonzero(date_years == year)
        first = int(positions[0])
        last = int(positions[-1]) + 1
        year_dates = dates[first:last]
        keep = output_mask[first:last].reshape(-1)
        if not keep.any():
            continue
        factor_data: dict[str, Any] = {
            "datetime": np.repeat(year_dates.to_numpy(), len(instruments))[keep],
            "instrument": np.tile(instruments.to_numpy(), len(year_dates))[keep],
        }
        for name in expected:
            factor_data[name] = np.asarray(values_for(name, first, last)).reshape(-1)[keep]
        frame = pd.DataFrame(factor_data)
        destination = staging_root / f"year={year}" / "factors.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        partition_hashes[str(year)] = sha256_file(destination)
        total_rows += len(frame)
        print(
            f"factors {label}: year={year} dates={len(year_dates)} rows={len(frame)}",
            file=sys.stderr,
            flush=True,
        )
        del factor_data, frame
    if not total_rows:
        raise DataContractError(f"Factor build for {label} produced no rows inside the research universe")
    return total_rows


def _aligned_matrix(np: Any, value: Any, dates: Any, instruments: Any, dtype: Any) -> Any:
    """A dates x instruments array from a reference panel, or from a daily series repeated for every instrument."""
    if getattr(value, "ndim", 2) == 1:
        series = np.asarray(value.reindex(dates) if hasattr(value, "reindex") else value, dtype=dtype)
        return np.ascontiguousarray(np.repeat(series[:, None], len(instruments), axis=1))
    if hasattr(value, "reindex"):
        value = value.reindex(index=dates, columns=instruments)
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


class _QuotePacking:
    """Moves each stock's quoted rows to the top of the time axis and puts results back on their dates."""

    def __init__(self, np: Any, inputs: dict[str, Any], fields: tuple[str, ...]) -> None:
        self.np = np
        quoted = np.logical_and.reduce([np.isfinite(inputs[field]) for field in fields])
        self.times, self.stocks = np.nonzero(quoted)
        self.slots = (np.cumsum(quoted, axis=0) - 1)[self.times, self.stocks]
        self.length = max(int(quoted.sum(axis=0).max(initial=0)), 1)

    def pack(self, values: Any) -> Any:
        packed = self.np.full((self.length, values.shape[1]), self.np.nan, dtype=values.dtype)
        packed[self.slots, self.stocks] = values[self.times, self.stocks]
        return packed

    def unpack(self, packed: Any, target: Any | None, time_count: int) -> Any:
        """Scatter packed rows to their dates; ``target`` may share memory with ``packed``."""
        values = packed[self.slots, self.stocks]
        if target is None:
            target = self.np.empty((time_count, packed.shape[1]), dtype=packed.dtype)
        target[...] = self.np.nan
        target[self.times, self.stocks] = values
        return target


def _load_panel_references(
    libraries: tuple[Any, ...],
    canonical_root: Path,
    dates: Any,
    instruments: Any,
    external_inputs: dict[str, str],
    pd: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the reference inputs every selected panel library declares, with the hashes of the files read."""
    inputs: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    if any(library.family == "gtja191" for library in libraries):
        inputs, hashes = _load_gtja191_external_inputs(canonical_root, dates, external_inputs, pd)
    elif external_inputs:
        raise DataContractError(f"Factor libraries {[lib.family for lib in libraries]} accept no external inputs")
    for library in libraries:
        loaded, loaded_hashes = library.load_reference_inputs(canonical_root, dates, instruments)
        overlap = (set(loaded) & set(inputs)) | (set(loaded_hashes) & set(hashes))
        if overlap:
            raise FactorContractError(f"Panel reference inputs are declared twice: {sorted(overlap)}")
        inputs.update(loaded)
        hashes.update(loaded_hashes)
    return inputs, hashes


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
) -> KunQuantFactorEngine | PanelFactorEngine | SelectionFactorEngine:
    selected_registry = registry or get_library_registry()
    if selected_registry.selection(config.factor_set) is not None:
        return SelectionFactorEngine(cache_dir, config, registry=selected_registry)
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
