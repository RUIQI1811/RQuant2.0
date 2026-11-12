from __future__ import annotations

import ast
import runpy
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "qmt"))

import build_jq240  # noqa: E402
from panel_reference.jq240 import JQ240_NAMES  # noqa: E402
from panel_reference.jq240 import JQFactors as ReferenceFactors  # noqa: E402
from test_jq240 import synthetic_inputs  # noqa: E402


@lru_cache(None)
def bundle() -> dict:
    return runpy.run_path(str(build_jq240.OUTPUT))


def test_committed_bundle_is_current() -> None:
    assert build_jq240.OUTPUT.read_text(encoding="ascii") == build_jq240.build()


def test_bundle_is_ascii_python36() -> None:
    source = build_jq240.OUTPUT.read_bytes().decode("ascii")
    tree = ast.parse(source, feature_version=(3, 6))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= {"numpy", "pandas"}
    assert not any(isinstance(node, (ast.AnnAssign, ast.NamedExpr)) for node in ast.walk(tree))
    assert "open(" not in source and "read_parquet" not in source


def test_bundle_matches_the_reference_on_every_factor() -> None:
    inputs = synthetic_inputs(days=700, stocks=6)
    bundled = bundle()["JQFactors"](inputs)
    reference = ReferenceFactors(inputs)
    assert tuple(bundle()["JQ240_NAMES"]) == JQ240_NAMES
    for name in JQ240_NAMES:
        pd.testing.assert_frame_equal(bundled.calculate(name), reference.calculate(name), check_names=False)


def test_sliding_window_shim_matches_numpy() -> None:
    shim = bundle()["_sliding_window_view"]
    values = np.arange(60, dtype=float).reshape(12, 5)
    expected = np.lib.stride_tricks.sliding_window_view(values, 4, axis=0)
    np.testing.assert_array_equal(shim(values, 4, axis=0), expected)


def test_to_numpy_shim() -> None:
    to_numpy = bundle()["_to_numpy"]
    frame = pd.DataFrame({"a": [1, 2]})
    result = to_numpy(frame, dtype=float, copy=True)
    assert result.dtype == float and result.tolist() == [[1.0], [2.0]]
