from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = Path(__file__).resolve().parents[1] / "qmt" / "probe_env.py"
TRADING_OR_IO_NAMES = {
    "passorder",
    "order_shares",
    "order_value",
    "order_target_value",
    "order_target_percent",
    "algo_passorder",
    "cancel",
    "get_trade_detail_data",
    "open",
    "socket",
    "urllib",
    "requests",
    "http",
}


class FakeContext:
    def is_last_bar(self) -> bool:
        return True

    def get_stock_list_in_sector(self, name: str) -> list[str]:
        return ["600000.SH", "600519.SH", "000001.SZ"]

    def set_universe(self, codes: list[str]) -> None:
        self.universe = list(codes)

    def get_history_data(self, length: int, period: str, field: str, dividend: int) -> dict[str, list[float]]:
        return {code: [float(i) for i in range(length)] for code in self.universe}

    def get_market_data_ex(self, fields, codes, **kwargs):
        count = kwargs.get("count", 30)
        count = 30 if count in (-1, None) else count
        index = [f"2026{i:04d}" for i in range(count)]
        frame = pd.DataFrame({field: np.arange(count, dtype=float) for field in fields}, index=index)
        return {code: frame for code in codes}

    def get_instrumentdetail(self, code: str) -> dict[str, object]:
        return {"InstrumentName": "X", "OpenDate": "19991110", "FloatVolume": 1.0}

    def get_financial_data(self, fields, codes, start, end, report_type):
        raise KeyError(fields[0])


def test_probe_is_ascii_and_python36_syntax() -> None:
    source = PROBE.read_bytes()
    source.decode("ascii")
    ast.parse(source.decode("ascii"), feature_version=(3, 6))


def test_probe_never_trades_or_touches_files_or_network() -> None:
    tree = ast.parse(PROBE.read_text(encoding="ascii"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not (names | attributes | imported) & TRADING_OR_IO_NAMES


def test_probe_runs_against_a_fake_context(capsys) -> None:
    module = runpy.run_path(str(PROBE))
    context = FakeContext()
    module["init"](context)
    module["handlebar"](context)
    module["handlebar"](context)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[RQ]")]
    assert lines[0] == "[RQ] probe=start"
    assert lines[-1] == "[RQ] probe=end"
    assert sum(line == "[RQ] probe=end" for line in lines) == 1
    assert any(line.startswith("[RQ] bars.universe_320d.codes_returned=2") for line in lines)
    assert any(line.startswith("[RQ] score_speed.seconds=") for line in lines)
    assert "[RQ] universe.set=2" in lines
    assert "[RQ] hist.close.div0.codes=2" in lines
