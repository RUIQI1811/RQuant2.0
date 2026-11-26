"""Bundle the main-board blend (jq240 + size_value_china on RQuant's ``main`` universe) into pasteable PTrade files.

The blend scores Shanghai 60xxxx and Shenzhen 00xxxx stocks with two exported LightGBM windows,
``qmt/models/lgb_jq240_main.json`` and ``qmt/models/lgb_size_value_china_main.json`` (``qmt/export_model.py``), and
averages their daily z-scores as ``scripts/analysis/blend_runs.py`` does. The bundles reuse the jq240 port
(``qmt/build_jq240.py``: the jq240 pandas reference, the PTrade data adapter, ``lgb_numpy`` and the live strategy) and
add, rewritten the same way:

* the size_value_china constants (``rquant.factors.libraries.size_value_china``) and its pandas reference
  (``tests/panel_reference/size_value_china.py``, kept equal to the KunQuant library by the test suite);
* ``rquant.data.ch4_factors.CH4FactorBuilder``, which turns PTrade's all-A data into the CH-3/CH-4 factor returns;
* ``ptrade/runtime/svc_inputs.py`` (the size_value_china inputs, scoring and the blend).

Outputs:

* ``ptrade/generated/main_blend_research_parity.py``: for one PTrade research notebook cell; scores AS_OF and compares
  with RQuant's backtest scores of that day (``tests/data/main_blend_parity_202610.parquet``);
* ``ptrade/generated/main_blend_ptrade_live.py``: the live strategy (``ptrade/runtime/ptrade_live.py`` with
  LIVE_UNIVERSE "main", top 50, drop 5, no rank exit, planned the afternoon before from the closing snapshots, and
  ``ptrade/runtime/main_live.py``'s scoring), dry run.

Usage: ``python ptrade/build_main_blend.py``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qmt"))

import build_jq240 as jq  # noqa: E402

JQ240_MODEL = ROOT / "qmt" / "models" / "lgb_jq240_main.json"
SVC_MODEL = ROOT / "qmt" / "models" / "lgb_size_value_china_main.json"
SVC_LIBRARY = ROOT / "src" / "rquant" / "factors" / "libraries" / "size_value_china.py"
SVC_CONSTANTS = ("MONTH", "HALF_YEAR", "YEAR", "TURNOVER_DAYS", "CH3", "CH4", "FITS", "SIZE_VALUE_CHINA",
                 "SIZE_VALUE_CHINA_NAMES")
SVC_REFERENCE = ROOT / "tests" / "panel_reference" / "size_value_china.py"
CH4_SOURCE = ROOT / "src" / "rquant" / "data" / "ch4_factors.py"
PARITY_SCORES = ROOT / "tests" / "data" / "main_blend_parity_202610.parquet"
PARITY_DAY = "2026-10-21"
RUNTIME = ROOT / "ptrade" / "runtime"
PARITY_OUTPUT = ROOT / "ptrade" / "generated" / "main_blend_research_parity.py"
LIVE_OUTPUT = ROOT / "ptrade" / "generated" / "main_blend_ptrade_live.py"
# The live strategy's settings for the blend, as backtested (top 50, at most 5 swaps a day, no rank exit), and the
# afternoon plan from the closing snapshots: once the bars are in (09:10), the morning plan of 3,200 stocks and two
# models takes about 16 minutes, too late for the 09:21 auction (2026-11-25 and 11-26: done 09:26).
LIVE_SETTINGS = (
    ('LIVE_UNIVERSE = "sh_main"', 'LIVE_UNIVERSE = "main"'),
    ("LIVE_TOPK = 30", "LIVE_TOPK = 50"),
    ("LIVE_N_DROP = 3", "LIVE_N_DROP = 5"),
    ("LIVE_EXIT_RANK_PCT = 0.7", "LIVE_EXIT_RANK_PCT = None"),
    ("LIVE_SNAPSHOT_AT = ()", 'LIVE_SNAPSHOT_AT = ("15:10", "15:40", "16:10")'),
)


def svc_constants() -> str:
    """The size_value_china catalog constants, as plain assignments, in file order."""
    tree = ast.parse(SVC_LIBRARY.read_text(encoding="utf-8"))
    kept = []
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = getattr(node.targets[0], "id", None)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = getattr(node.target, "id", None)
            node = ast.Assign(targets=[node.target], value=node.value)
        if target in SVC_CONSTANTS:
            kept.append(node)
    found = {node.targets[0].id for node in kept}
    if found != set(SVC_CONSTANTS):
        raise ValueError(f"{SVC_LIBRARY.name} lacks {sorted(set(SVC_CONSTANTS) - found)}")
    module = ast.fix_missing_locations(ast.Module(body=kept, type_ignores=[]))
    return f"# From {SVC_LIBRARY.relative_to(ROOT).as_posix()}\n" + ast.unparse(module)


def svc_core() -> str:
    """size_value_china (constants and pandas reference) and the CH-4 factor builder, for Python 3.11."""
    parts = [svc_constants(), jq._module_source(SVC_REFERENCE), jq._module_source(CH4_SOURCE)]
    return "\n\n".join(parts)


def expected_scores(path: Path = PARITY_SCORES, day: str = PARITY_DAY) -> str:
    """EXPECTED = {model: {instrument: score}} for ``day`` from RQuant's backtest scores, as compact literals."""
    import pandas as pd

    frame = pd.read_parquet(path)
    frame = frame[pd.to_datetime(frame["datetime"]) == pd.Timestamp(day)].sort_values("instrument")
    if frame.empty:
        raise ValueError(f"No scores for {day} in {path}")
    lines = [
        "def _scores(text):",
        "    return dict((item.split(':')[0], float(item.split(':')[1])) for item in text.split(','))",
        "",
        f"# RQuant's walk-forward scores of {day} (jq240 and size_value_china on main, and their blend).",
        "EXPECTED = {",
    ]
    for name in ("jq240", "svc", "blend"):
        text = ",".join(f"{code}:{value:.7g}" for code, value in zip(frame["instrument"], frame[name], strict=True))
        lines.append(f"    {name!r}: _scores({text!r}),")
    lines.append("}")
    return "\n".join(lines)


def models() -> str:
    return f"MODEL = {jq.model_literal(JQ240_MODEL)!r}\n\nSVC_MODEL = {jq.model_literal(SVC_MODEL)!r}"


def build_parity() -> str:
    parts = [
        "# Generated by ptrade/build_main_blend.py; paste the whole file into one PTrade research notebook cell.",
        jq.build(),
        svc_core(),
        models(),
        (RUNTIME / "ptrade_inputs.py").read_text(encoding="utf-8"),
        (ROOT / "qmt" / "runtime" / "lgb_numpy.py").read_text(encoding="utf-8"),
        jq.pick(ROOT / "qmt" / "runtime" / "paper.py", ("ranked",)),
        jq.pick(ROOT / "qmt" / "runtime" / "paper_shell.py", ("daily_scores",)),
        (RUNTIME / "svc_inputs.py").read_text(encoding="utf-8"),
        expected_scores(),
        (RUNTIME / "main_parity.py").read_text(encoding="utf-8"),
    ]
    source = jq._ascii("\n\n".join(parts)) + "\n"
    ast.parse(source)
    jq._check_ptrade(source, jq.PTRADE_BANNED)
    return source


def live_shell() -> str:
    source = (RUNTIME / "ptrade_live.py").read_text(encoding="utf-8")
    for old, new in LIVE_SETTINGS:
        if source.count(old) != 1:
            raise ValueError(f"ptrade_live.py must set {old!r} exactly once")
        source = source.replace(old, new)
    return source


def build_live() -> str:
    parts = [
        "# Generated by ptrade/build_main_blend.py; paste the whole file into a PTrade strategy (daily frequency).",
        jq.build(),
        svc_core(),
        models(),
        (RUNTIME / "ptrade_inputs.py").read_text(encoding="utf-8"),
        (ROOT / "qmt" / "runtime" / "lgb_numpy.py").read_text(encoding="utf-8"),
        (ROOT / "qmt" / "runtime" / "paper.py").read_text(encoding="utf-8"),
    ]
    parts.extend(jq.pick(path, names) for path, names in jq.PTRADE_LIVE_PICKS)
    parts.extend([
        (RUNTIME / "svc_inputs.py").read_text(encoding="utf-8"),
        live_shell(),
        (RUNTIME / "main_live.py").read_text(encoding="utf-8"),
    ])
    source = jq._ascii("\n\n".join(parts)) + "\n"
    ast.parse(source)
    jq._check_ptrade(source, jq.PTRADE_LIVE_BANNED)
    if "\ndef log(" in source or "\nlog = " in source:
        raise ValueError("the live strategy must not replace PTrade's log object")
    return source


def main() -> None:
    for path, source in ((PARITY_OUTPUT, build_parity()), (LIVE_OUTPUT, build_live())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="ascii")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    for path in (JQ240_MODEL, SVC_MODEL):
        model = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name}: run {model['run']} year {model['prediction_year']} trees {len(model['trees'])}")


if __name__ == "__main__":
    main()
