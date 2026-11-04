from __future__ import annotations

import ast
import re
import runpy
from pathlib import Path

FIELD_MAP = Path(__file__).resolve().parents[1] / "qmt" / "jq240_field_map.py"
CONTRACTS = Path(__file__).resolve().parents[1] / "src" / "rquant" / "data" / "contracts.py"
KEY_FIELDS = {"ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "update_flag"}


def _contract_fields() -> dict[str, tuple[str, ...]]:
    # contracts.py imports the data package, so read the literal instead of importing it.
    tree = ast.parse(CONTRACTS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FUNDAMENTAL_ENDPOINT_FIELDS":
            result = {}
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                names = [elt.value for elt in value.elts if isinstance(elt, ast.Constant)]
                result[key.value] = tuple(name for name in names if name not in KEY_FIELDS)
            return result
    raise AssertionError("FUNDAMENTAL_ENDPOINT_FIELDS not found")


def test_field_map_is_ascii_python36() -> None:
    source = FIELD_MAP.read_bytes().decode("ascii")
    ast.parse(source, feature_version=(3, 6))


def test_every_statement_field_is_mapped() -> None:
    mapping = runpy.run_path(str(FIELD_MAP))["STATEMENT_FIELDS"]
    for endpoint, fields in _contract_fields().items():
        assert set(mapping[endpoint]) == set(fields), endpoint


def test_mapped_names_use_known_qmt_tables() -> None:
    mapping = runpy.run_path(str(FIELD_MAP))["STATEMENT_FIELDS"]
    pattern = re.compile(r"^(ASHAREINCOME|ASHAREBALANCESHEET|ASHARECASHFLOW|CAPITALSTRUCTURE)\.[a-z_0-9]+$")
    for fields in mapping.values():
        for target in fields.values():
            assert target is None or pattern.match(target), target
