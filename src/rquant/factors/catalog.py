from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rquant.errors import FactorContractError
from rquant.io import stable_hash

FactorFamily = Literal["qlib_alpha158", "wq_alpha101"]
FactorSet = Literal["qlib_alpha158", "wq_alpha101", "combined"]
CATALOG_VERSION = 1


@dataclass(frozen=True)
class FactorSpec:
    canonical_name: str
    source_name: str
    family: FactorFamily
    ordinal: int
    formula: str
    max_lookback: int
    implementation: str
    catalog_version: int = CATALOG_VERSION


class FactorCatalog:
    def __init__(self, specs: Iterable[FactorSpec]) -> None:
        self.specs = tuple(specs)
        self.validate()

    def select(self, factor_set: FactorSet) -> tuple[FactorSpec, ...]:
        if factor_set == "combined":
            return self.specs
        if factor_set not in {"qlib_alpha158", "wq_alpha101"}:
            raise FactorContractError(f"Unknown factor set: {factor_set}")
        return tuple(spec for spec in self.specs if spec.family == factor_set)

    def canonical_names(self, factor_set: FactorSet = "combined") -> tuple[str, ...]:
        return tuple(spec.canonical_name for spec in self.select(factor_set))

    def source_to_canonical(self, factor_set: FactorSet) -> dict[str, str]:
        return {spec.source_name: spec.canonical_name for spec in self.select(factor_set)}

    @property
    def fingerprint(self) -> str:
        return stable_hash([asdict(spec) for spec in self.specs])

    def validate(self) -> None:
        expected_158 = tuple(f"a158_{index:03d}" for index in range(1, 159))
        expected_101 = tuple(f"a101_{index:03d}" for index in range(1, 102))
        actual_158 = tuple(spec.canonical_name for spec in self.specs if spec.family == "qlib_alpha158")
        actual_101 = tuple(spec.canonical_name for spec in self.specs if spec.family == "wq_alpha101")
        if actual_158 != expected_158:
            raise FactorContractError("Alpha158 catalog must be exactly a158_001..a158_158 in locked order")
        if actual_101 != expected_101:
            raise FactorContractError("Alpha101 catalog must be exactly a101_001..a101_101 in locked order")
        canonical = [spec.canonical_name for spec in self.specs]
        if len(canonical) != 259 or len(set(canonical)) != 259:
            raise FactorContractError("Combined catalog must contain 259 unique canonical names")
        for spec in self.specs:
            if spec.source_name == spec.canonical_name:
                raise FactorContractError(f"Source name leaked as canonical name: {spec.source_name}")

    def rows(self, factor_set: FactorSet = "combined") -> list[dict[str, object]]:
        return [asdict(spec) for spec in self.select(factor_set)]

    def export(self, destination: str | Path, *, factor_set: FactorSet = "combined", format: str = "json") -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows(factor_set)
        if format == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"catalog_version": CATALOG_VERSION, "fingerprint": self.fingerprint, "factors": rows},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            return
        if format == "csv":
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            return
        raise FactorContractError(f"Unsupported catalog export format: {format}")


_ALPHA158_KBAR = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
_ALPHA158_PRICE = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
_ALPHA158_ROLLING = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
_ALPHA158_WINDOWS = (5, 10, 20, 30, 60)
ALPHA158_SOURCE_NAMES = (
    _ALPHA158_KBAR
    + _ALPHA158_PRICE
    + tuple(f"{operator}{window}" for operator in _ALPHA158_ROLLING for window in _ALPHA158_WINDOWS)
)

ALPHA101_KUNQUANT_MISSING = frozenset({48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100})


def _alpha158_specs() -> list[FactorSpec]:
    if len(ALPHA158_SOURCE_NAMES) != 158:
        raise FactorContractError(f"Locked Alpha158 source order has {len(ALPHA158_SOURCE_NAMES)} names, expected 158")
    specs = []
    for ordinal, source_name in enumerate(ALPHA158_SOURCE_NAMES, 1):
        lookback = next((window for window in _ALPHA158_WINDOWS if source_name.endswith(str(window))), 1)
        specs.append(
            FactorSpec(
                canonical_name=f"a158_{ordinal:03d}",
                source_name=source_name,
                family="qlib_alpha158",
                ordinal=ordinal,
                formula=f"Qlib Alpha158 source expression: {source_name}",
                max_lookback=lookback,
                implementation="KunQuant.predefined.Alpha158",
            )
        )
    return specs


def _alpha101_specs() -> list[FactorSpec]:
    specs = []
    for ordinal in range(1, 102):
        missing = ordinal in ALPHA101_KUNQUANT_MISSING
        specs.append(
            FactorSpec(
                canonical_name=f"a101_{ordinal:03d}",
                source_name=f"alpha{ordinal:03d}",
                family="wq_alpha101",
                ordinal=ordinal,
                formula=f"WorldQuant Formulaic Alpha #{ordinal:03d}",
                max_lookback=250,
                implementation=(
                    f"rquant.factors.alpha101_missing.alpha{ordinal:03d}"
                    if missing
                    else f"KunQuant.predefined.Alpha101.alpha{ordinal:03d}"
                ),
            )
        )
    return specs


_CATALOG = FactorCatalog((*_alpha158_specs(), *_alpha101_specs()))


def get_catalog() -> FactorCatalog:
    return _CATALOG
