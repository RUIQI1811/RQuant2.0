from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from rquant.errors import FactorContractError
from rquant.factors.libraries.alpha101 import ALPHA101_UPSTREAM_MISSING
from rquant.factors.libraries.base import FactorFamily, FactorSet, FactorSpec
from rquant.factors.libraries.registry import FactorLibraryRegistry, get_library_registry
from rquant.io import stable_hash

CATALOG_VERSION = 2


class FactorCatalog:
    def __init__(self, specs: Iterable[FactorSpec], factor_sets: Mapping[str, tuple[str, ...]]) -> None:
        self.specs = tuple(specs)
        self._factor_sets = dict(factor_sets)
        self.validate()

    def factor_sets(self) -> tuple[str, ...]:
        return tuple(self._factor_sets)

    def select(self, factor_set: FactorSet) -> tuple[FactorSpec, ...]:
        try:
            families = self._factor_sets[factor_set]
        except KeyError as exc:
            raise FactorContractError(f"Unknown factor set: {factor_set}") from exc
        by_family = {family: [] for family in families}
        for spec in self.specs:
            if spec.family in by_family:
                by_family[spec.family].append(spec)
        return tuple(spec for family in families for spec in by_family[family])

    def canonical_names(self, factor_set: FactorSet = "combined") -> tuple[str, ...]:
        return tuple(spec.canonical_name for spec in self.select(factor_set))

    def source_to_canonical(self, factor_set: FactorSet) -> dict[str, str]:
        specs = self.select(factor_set)
        source_names = [spec.source_name for spec in specs]
        if len(source_names) != len(set(source_names)):
            raise FactorContractError(f"Factor set {factor_set} has ambiguous source names")
        return {spec.source_name: spec.canonical_name for spec in specs}

    @property
    def fingerprint(self) -> str:
        return stable_hash([asdict(spec) for spec in self.specs])

    def fingerprint_for(self, factor_set: FactorSet) -> str:
        return stable_hash([asdict(spec) for spec in self.select(factor_set)])

    def validate(self) -> None:
        if not self.specs:
            raise FactorContractError("Factor catalog must not be empty")
        registered_families = {family for families in self._factor_sets.values() for family in families}
        spec_families = {spec.family for spec in self.specs}
        if spec_families != registered_families:
            raise FactorContractError("Factor catalog families differ from the registered factor sets")
        canonical = [spec.canonical_name for spec in self.specs]
        if len(canonical) != len(set(canonical)):
            raise FactorContractError("Factor catalog contains duplicate canonical names")
        for spec in self.specs:
            if not spec.canonical_name or not spec.source_name:
                raise FactorContractError("Factor names must not be empty")
            if spec.source_name == spec.canonical_name:
                raise FactorContractError(f"Source name leaked as canonical name: {spec.source_name}")
            if spec.ordinal < 1 or spec.max_lookback < 1:
                raise FactorContractError(f"Invalid factor metadata: {spec.canonical_name}")
        for factor_set in self._factor_sets:
            selected = self.select(factor_set)
            if not selected:
                raise FactorContractError(f"Factor set is empty: {factor_set}")

    def rows(self, factor_set: FactorSet = "combined") -> list[dict[str, object]]:
        return [asdict(spec) for spec in self.select(factor_set)]

    def export(self, destination: str | Path, *, factor_set: FactorSet = "combined", format: str = "json") -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows(factor_set)
        if format == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "catalog_version": CATALOG_VERSION,
                        "fingerprint": self.fingerprint_for(factor_set),
                        "factors": rows,
                    },
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


def catalog_from_registry(registry: FactorLibraryRegistry) -> FactorCatalog:
    factor_sets = {
        factor_set: tuple(library.family for library in registry.select(factor_set))
        for factor_set in registry.factor_sets()
    }
    specs = tuple(spec for library in registry.libraries() for spec in library.specs)
    return FactorCatalog(specs, factor_sets)


def get_catalog() -> FactorCatalog:
    return catalog_from_registry(get_library_registry())


__all__ = [
    "ALPHA101_UPSTREAM_MISSING",
    "CATALOG_VERSION",
    "FactorCatalog",
    "FactorFamily",
    "FactorSet",
    "FactorSpec",
    "catalog_from_registry",
    "get_catalog",
]
