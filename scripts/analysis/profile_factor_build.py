"""Find where a factor build's memory goes: phase timings, private memory and the biggest allocation sites.

Runs ``factors build`` for one factor set the same way the CLI does, but logs the process's private memory every
few seconds together with the build phase it is in, and once private memory passes ``--stop-gib`` writes the top
allocation sites (``tracemalloc``, which also sees NumPy buffers) and exits at once. Nothing is published: the store
is written under a temporary directory next to the real one (left behind if the script exits early; delete
``data/factors-<universe>/.<set>.profile.*`` afterwards).

    python scripts/analysis/profile_factor_build.py --root D:\\自己瞎搞\\RQuant2.0 --universe main --factor-set jq240
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path


def private_gib() -> float:
    """Private (committed) memory of this process in GiB; resident anonymous memory on Linux."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return float(getattr(info, "private", getattr(info, "rss", 0))) / 1024**3
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return counters.PrivateUsage / 1024**3
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def report(snapshot, top: int, frames: int) -> None:  # type: ignore[no-untyped-def]
    statistics = snapshot.statistics("traceback")
    total = sum(stat.size for stat in statistics)
    print(f"\ntraced live allocations: {total / 1024**3:.2f} GiB; top {top} sites:", flush=True)
    for stat in statistics[:top]:
        print(f"\n{stat.size / 1024**2:9.1f} MiB in {stat.count} blocks")
        for line in stat.traceback.format(limit=frames):
            print("   ", line)
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="RQuant data root (the folder that holds data/)")
    parser.add_argument("--universe", default="main")
    parser.add_argument("--factor-set", default="jq240")
    parser.add_argument("--start", default=None)
    parser.add_argument("--stop-gib", type=float, default=4.0, help="stop once private memory passes this")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between memory samples")
    parser.add_argument("--frames", type=int, default=12, help="stack frames kept per allocation")
    parser.add_argument("--top", type=int, default=25, help="allocation sites to print")
    args = parser.parse_args()

    import rquant.factors.engine as engine_module
    from rquant.config import ProjectPaths, load_config
    from rquant.factors.engine import FactorBuildConfig, create_factor_engine

    root = Path(args.root)
    paths = ProjectPaths.from_root(root, universe=args.universe)
    config_path = root / "config" / "default.yaml"
    factor_config = load_config(config_path).get("factors", {}) if config_path.exists() else {}
    started = time.time()
    state = {"phase": "start", "peak": 0.0, "stop": False}
    lock = threading.Lock()

    def log(message: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    def sampler() -> None:
        while not state["stop"]:
            private = private_gib()
            with lock:
                state["peak"] = max(state["peak"], private)
                phase = state["phase"]
            log(f"private={private:.2f} GiB phase={phase}")
            if private > args.stop_gib:
                log(f"private memory passed {args.stop_gib} GiB in phase {phase}; allocation sites:")
                report(tracemalloc.take_snapshot(), args.top, args.frames)
                os._exit(3)
            time.sleep(args.interval)

    def phase(name: str, function):  # type: ignore[no-untyped-def]
        @functools.wraps(function)
        def wrapped(*positional, **keywords):  # type: ignore[no-untyped-def]
            with lock:
                previous, state["phase"] = state["phase"], name
            log(f"enter {name} private={private_gib():.2f} GiB")
            try:
                return function(*positional, **keywords)
            finally:
                log(f"leave {name} private={private_gib():.2f} GiB")
                with lock:
                    state["phase"] = previous

        return wrapped

    for name in ("_CanonicalBars", "_load_panel_references", "_write_year_partitions"):
        if hasattr(engine_module, name):
            setattr(engine_module, name, phase(name, getattr(engine_module, name)))
    for name in ("run_statements", "run", "compile", "compile_statements"):
        method = getattr(engine_module.KunQuantFactorEngine, name, None)
        if method is not None:
            setattr(engine_module.KunQuantFactorEngine, name, phase(name, method))
    fill = engine_module._CanonicalBars.__wrapped__.fill
    engine_module._CanonicalBars.__wrapped__.fill = phase("_CanonicalBars.fill", fill)

    tracemalloc.start(args.frames)
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    paths.factors.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f".{args.factor_set}.profile.", dir=paths.factors))
    engine = create_factor_engine(
        paths.cache,
        FactorBuildConfig(
            factor_set=args.factor_set,  # type: ignore[arg-type]
            dtype=str(factor_config.get("dtype", "double")),
            input_layout=str(factor_config.get("input_layout", "TS")),
            output_layout=str(factor_config.get("output_layout", "TS")),
            workers=int(factor_config.get("workers", 4)),
            graph_outputs=int(factor_config.get("graph_outputs", 20)),
            universe=args.universe,
        ),
    )
    try:
        engine.build_from_canonical(paths.canonical, output, start=args.start)
        log("build finished below the stop threshold")
    finally:
        state["stop"] = True
        thread.join(timeout=10)
        shutil.rmtree(output, ignore_errors=True)
    log(f"peak private={state['peak']:.2f} GiB")
    report(tracemalloc.take_snapshot(), args.top, args.frames)
    tracemalloc.stop()


if __name__ == "__main__":
    main()
