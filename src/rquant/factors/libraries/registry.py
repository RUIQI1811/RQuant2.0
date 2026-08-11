"""Explicit, auditable registry for factor libraries and factor-set bundles."""

from __future__ import annotations

from collections.abc import Iterable

from rquant.errors import FactorContractError
from rquant.factors.libraries.base import CANONICAL_INPUT_FIELDS, FactorLibrary, FactorSet
from rquant.io import stable_hash


class FactorLibraryRegistry:
    def __init__(self) -> None:
        self._libraries: dict[str, FactorLibrary] = {}
        self._factor_sets: dict[str, tuple[str, ...]] = {}

    def register_library(self, library: FactorLibrary, *, standalone: bool = True) -> None:
        family = library.family
        if not family or family == "combined":
            raise FactorContractError(f"Invalid factor-library family: {family!r}")
        if library.backend not in {"kunquant", "panel"}:
            raise FactorContractError(f"Factor library {family} has unsupported backend: {library.backend}")
        if family in self._libraries:
            raise FactorContractError(f"Factor library is already registered: {family}")
        unknown_inputs = sorted(set(library.required_inputs) - set(CANONICAL_INPUT_FIELDS))
        if unknown_inputs:
            raise FactorContractError(f"Factor library {family} requires unknown inputs: {unknown_inputs}")
        if not library.specs:
            raise FactorContractError(f"Factor library is empty: {family}")
        if any(spec.family != family for spec in library.specs):
            raise FactorContractError(f"Factor library {family} contains specs assigned to another family")
        canonical_names = [spec.canonical_name for spec in library.specs]
        source_names = [spec.source_name for spec in library.specs]
        if len(canonical_names) != len(set(canonical_names)):
            raise FactorContractError(f"Factor library {family} contains duplicate canonical names")
        if len(source_names) != len(set(source_names)):
            raise FactorContractError(f"Factor library {family} contains duplicate source names")
        self._libraries[family] = library
        if standalone:
            self.register_factor_set(family, (family,))

    def register_factor_set(self, name: FactorSet, families: Iterable[str]) -> None:
        selected = tuple(families)
        if not name or name in self._factor_sets:
            raise FactorContractError(f"Factor set is already registered or invalid: {name!r}")
        if not selected or len(set(selected)) != len(selected):
            raise FactorContractError(f"Factor set {name} must contain unique registered libraries")
        missing = tuple(family for family in selected if family not in self._libraries)
        if missing:
            raise FactorContractError(f"Factor set {name} references unregistered libraries: {missing}")
        self._factor_sets[name] = selected

    def select(self, factor_set: FactorSet) -> tuple[FactorLibrary, ...]:
        try:
            families = self._factor_sets[factor_set]
        except KeyError as exc:
            raise FactorContractError(f"Unknown factor set: {factor_set}") from exc
        return tuple(self._libraries[family] for family in families)

    def factor_sets(self) -> tuple[str, ...]:
        return tuple(self._factor_sets)

    def libraries(self) -> tuple[FactorLibrary, ...]:
        return tuple(self._libraries.values())

    def required_inputs(self, factor_set: FactorSet) -> tuple[str, ...]:
        required = {field for library in self.select(factor_set) for field in library.required_inputs}
        return tuple(field for field in CANONICAL_INPUT_FIELDS if field in required)

    def implementation_fingerprint(self, factor_set: FactorSet) -> str:
        return stable_hash(
            [
                {"family": library.family, "fingerprint": library.implementation_fingerprint}
                for library in self.select(factor_set)
            ]
        )


def _default_registry() -> FactorLibraryRegistry:
    from rquant.factors.libraries.alpha101 import WorldQuantAlpha101Library
    from rquant.factors.libraries.alpha158 import QlibAlpha158Library
    from rquant.factors.libraries.alpha360 import QlibAlpha360Library
    from rquant.factors.libraries.gtja191 import GTJA191Library

    registry = FactorLibraryRegistry()
    registry.register_library(QlibAlpha158Library())
    registry.register_library(WorldQuantAlpha101Library())
    registry.register_library(GTJA191Library())
    registry.register_library(QlibAlpha360Library())
    registry.register_factor_set("combined", ("qlib_alpha158", "wq_alpha101"))
    return registry


_REGISTRY = _default_registry()


def get_library_registry() -> FactorLibraryRegistry:
    return _REGISTRY
