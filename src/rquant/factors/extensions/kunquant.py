"""Minimal RQuant extensions and adapters for the KunQuant backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from rquant.errors import DependencyError


def kunquant_symbols(*names: str) -> dict[str, Any]:
    try:
        from KunQuant import Op, ops
        from KunQuant.Op import ConstantOp
    except ImportError as exc:
        raise DependencyError("KunQuant==0.1.11 is required to build factor graphs") from exc
    symbols = {"ConstantOp": ConstantOp}
    for name in names:
        symbols[name] = getattr(ops, name) if hasattr(ops, name) else getattr(Op, name)
    return symbols


@lru_cache(maxsize=1)
def group_neutralize_operator() -> type:
    symbols = kunquant_symbols("GenericCrossSectionalOp")
    base = symbols["GenericCrossSectionalOp"]

    class GroupNeutralize(base):
        def __init__(self, value: Any, group_id: Any) -> None:
            super().__init__([value, group_id], None)

        def generate_head(self) -> str:
            return "std::vector<T> group_sums;\nstd::vector<size_t> group_counts;\n"

        def generate_body(self) -> str:
            return r"""
            size_t max_group = 0;
            for (size_t i = 0; i < num_stocks; ++i) {
                const T group = input_1[i];
                if (std::isfinite(group) && group >= T(0)) {
                    max_group = std::max(max_group, static_cast<size_t>(group));
                }
            }
            group_sums.assign(max_group + 1, T(0));
            group_counts.assign(max_group + 1, 0);
            for (size_t i = 0; i < num_stocks; ++i) {
                const T value = input_0[i];
                const T group = input_1[i];
                if (!std::isfinite(value) || !std::isfinite(group) || group < T(0)) continue;
                const size_t key = static_cast<size_t>(group);
                group_sums[key] += value;
                group_counts[key] += 1;
            }
            for (size_t i = 0; i < num_stocks; ++i) {
                const T value = input_0[i];
                const T group = input_1[i];
                if (!std::isfinite(value) || !std::isfinite(group) || group < T(0)) {
                    output_0[i] = std::numeric_limits<T>::quiet_NaN();
                    continue;
                }
                const size_t key = static_cast<size_t>(group);
                output_0[i] = group_counts[key] == 0
                    ? std::numeric_limits<T>::quiet_NaN()
                    : value - group_sums[key] / static_cast<T>(group_counts[key]);
            }
            """

    GroupNeutralize.__name__ = "RQuantGroupNeutralize"
    return GroupNeutralize


def group_neutralize(value: Any, group_id: Any) -> Any:
    return group_neutralize_operator()(value, group_id)


def minimum(left: Any, right: Any) -> Any:
    try:
        from KunQuant.ops import Min
    except ImportError as exc:
        raise DependencyError("KunQuant==0.1.11 is required to build factor graphs") from exc
    return Min(left, right)
