"""Bundle the jq240 pandas reference into Python 3.6 source that runs inside the Guojin QMT sandbox.

QMT's built-in interpreter is Python 3.6 with numpy 1.19 and pandas 0.22, and a strategy is a single file that may not
read other files. This script takes the independent pandas reference of jq240 (``tests/panel_reference/jq240.py``,
which the test suite keeps equal to the KunQuant production library) and the quarter grid it uses
(``rquant.factors.statements``), and rewrites them for that interpreter:

* drops ``from __future__ import annotations``, typing-only imports and every annotation;
* drops the RQuant library wrapper class (manifests, hashing, file IO) and keeps the calculator;
* empties the Chinese factor descriptions and docstrings, and escapes what is left, so the file is ASCII.

``qmt/runtime/compat.py`` is prepended; it back-fills the few newer numpy/pandas calls the reference makes
(``to_numpy``, ``sliding_window_view``). The result is written to ``qmt/generated/jq240_core.py``, whose
``JQFactors`` takes the same input dict as the reference. ``tests/test_qmt_bundle.py`` runs the bundle against the
reference.

Usage: ``python qmt/build_jq240.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "src" / "rquant" / "factors" / "statements.py",
    ROOT / "tests" / "panel_reference" / "jq240.py",
)
COMPAT = ROOT / "qmt" / "runtime" / "compat.py"
OUTPUT = ROOT / "qmt" / "generated" / "jq240_core.py"
# The parity strategy: the core, the QMT field map, the QMT data adapter and a strategy shell, in one pasteable file.
STRATEGY_PARTS = (
    ROOT / "qmt" / "jq240_field_map.py",
    ROOT / "qmt" / "runtime" / "qmt_inputs.py",
    ROOT / "qmt" / "runtime" / "lgb_numpy.py",
    ROOT / "qmt" / "runtime" / "parity_shell.py",
)
# The LightGBM window the strategy scores with, as qmt/export_model.py wrote it on Windows; absent -> no scoring.
MODEL_PATH = ROOT / "qmt" / "models" / "lgb_jq240_sh_main.json"
MODEL_KEYS = ("run", "prediction_year", "fit_period", "features", "mean", "std", "trees")
PARITY_OUTPUT = ROOT / "qmt" / "generated" / "jq240_parity.py"
# The daily paper-trading strategy: the same core, adapter and model, with the paper book and its shell.
PAPER_PARTS = (
    ROOT / "qmt" / "jq240_field_map.py",
    ROOT / "qmt" / "runtime" / "qmt_inputs.py",
    ROOT / "qmt" / "runtime" / "lgb_numpy.py",
    ROOT / "qmt" / "runtime" / "paper.py",
    ROOT / "qmt" / "runtime" / "paper_shell.py",
)
PAPER_OUTPUT = ROOT / "qmt" / "generated" / "jq240_paper.py"
# The live strategy: the paper strategy's parts plus the live shell, whose init/handlebar replace the paper ones.
LIVE_PARTS = PAPER_PARTS + (ROOT / "qmt" / "runtime" / "live_shell.py",)
LIVE_OUTPUT = ROOT / "qmt" / "generated" / "jq240_live.py"
# PTrade (Python 3.11, research notebook): the same core and model with the PTrade data adapter and a parity shell.
PTRADE_PARITY_PARTS = (
    ROOT / "ptrade" / "runtime" / "ptrade_inputs.py",
    ROOT / "qmt" / "runtime" / "lgb_numpy.py",
    ROOT / "ptrade" / "runtime" / "research_parity.py",
)
PTRADE_PARITY_OUTPUT = ROOT / "ptrade" / "generated" / "jq240_research_parity.py"
# Words PTrade's research notebook rejects anywhere in a cell.
PTRADE_BANNED = ("import os", "import sys", "from os", "from sys", "os.", "sys.")
DROPPED_CLASSES = {"JQ240Library", "StatementResults"}
DROPPED_MODULES = {"__future__", "typing", "collections.abc", "importlib", "inspect", "pathlib"}


class _Rewriter(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if module in DROPPED_MODULES or module.startswith("rquant"):
            return None
        return node

    def visit_Import(self, node: ast.Import):
        kept = [alias for alias in node.names if alias.name not in DROPPED_MODULES]
        if not kept:
            return None
        node.names = kept
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name in DROPPED_CLASSES:
            return None
        node.bases = [base for base in node.bases if getattr(base, "id", "") != "PanelFactorLibrary"]
        _drop_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.returns = None
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            argument.annotation = None
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None:
                argument.annotation = None
        _drop_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)

    def visit_Assign(self, node: ast.Assign):
        # Panel = pd.DataFrame and Formula = Callable[...] aliases: only the first is needed at run time.
        if len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "Formula":
            return None
        self.generic_visit(node)
        return node


def _drop_docstring(node) -> None:
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        if isinstance(body[0].value.value, str):
            body.pop(0)
            if not body:
                body.append(ast.Pass())


def _strip_descriptions(tree: ast.Module) -> None:
    """Blank the description column of the JQ240 table (third tuple field), the only long non-ASCII text."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "JQ240":
            for entry in node.value.elts:
                entry.elts[2] = ast.Constant("")


def _module_source(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    _drop_docstring(tree)
    tree = _Rewriter().visit(tree)
    _strip_descriptions(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _ascii(source: str) -> str:
    return "".join(ch if ord(ch) < 128 else f"\\u{ord(ch):04x}" for ch in source)


def build() -> str:
    parts = [
        "# Generated by qmt/build_jq240.py from rquant.factors.statements and tests/panel_reference/jq240.py.",
        "# Do not edit; rebuild instead.",
        COMPAT.read_text(encoding="utf-8"),
        "class DataContractError(Exception):\n    pass",
        "def replace_inf(value):\n    return value.replace([np.inf, -np.inf], np.nan)",
    ]
    parts.extend(_module_source(path) for path in SOURCES)
    source = _ascii("\n\n".join(parts)) + "\n"
    ast.parse(source, feature_version=(3, 6))
    return source


def build_strategy(parts_paths: tuple[Path, ...] = STRATEGY_PARTS) -> str:
    """The core plus ``parts_paths`` (default ``STRATEGY_PARTS``), headed by the encoding line QMT's editor expects."""
    parts = ["#encoding:gbk", "# Generated by qmt/build_jq240.py; paste the whole file into a QMT strategy.", build()]
    parts.append("MODEL = %r" % (model_literal(),))
    for path in parts_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        parts.append("\n".join(line for line in lines if not line.startswith("#encoding")))
    source = _ascii("\n\n".join(parts)) + "\n"
    ast.parse(source, feature_version=(3, 6))
    return source


def build_ptrade(parts_paths: tuple[Path, ...] = PTRADE_PARITY_PARTS) -> str:
    """The core, the model and ``parts_paths`` as one file to paste into a PTrade research notebook cell."""
    parts = ["# Generated by qmt/build_jq240.py; paste the whole file into one PTrade research notebook cell.",
             build(), "MODEL = %r" % (model_literal(),)]
    parts.extend(path.read_text(encoding="utf-8") for path in parts_paths)
    source = _ascii("\n\n".join(parts)) + "\n"
    ast.parse(source)
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        for word in PTRADE_BANNED:
            if word in code and not any(f"{prefix}{word}" in code for prefix in "abcdefghijklmnopqrstuvwxyz_"):
                raise ValueError(f"PTrade rejects {word!r}: {line.strip()}")
    return source


def model_literal(path: Path = MODEL_PATH) -> dict | None:
    """The exported model reduced to what the strategy reads, or None when no model has been exported."""
    if not path.exists():
        return None
    model = json.loads(path.read_text(encoding="utf-8"))
    return {key: model[key] for key in MODEL_KEYS}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PTRADE_PARITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    outputs = ((OUTPUT, build()), (PARITY_OUTPUT, build_strategy()), (PAPER_OUTPUT, build_strategy(PAPER_PARTS)),
               (LIVE_OUTPUT, build_strategy(LIVE_PARTS)), (PTRADE_PARITY_OUTPUT, build_ptrade()))
    for path, source in outputs:
        path.write_text(source, encoding="ascii")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
