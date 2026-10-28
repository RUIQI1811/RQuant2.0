"""Explicit, auditable registry for factor libraries and factor-set bundles."""

from __future__ import annotations

from collections.abc import Iterable

from rquant.errors import FactorContractError
from rquant.factors.libraries.base import CANONICAL_INPUT_FIELDS, FactorLibrary, FactorSelection, FactorSet
from rquant.io import stable_hash


class FactorLibraryRegistry:
    def __init__(self) -> None:
        self._libraries: dict[str, FactorLibrary] = {}
        self._factor_sets: dict[str, tuple[str, ...]] = {}
        self._selections: dict[str, FactorSelection] = {}

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
        if set(library.reference_inputs) & set(CANONICAL_INPUT_FIELDS):
            raise FactorContractError(f"Factor library {family} declares canonical fields as reference inputs")
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

    def register_factor_selection(self, selection: FactorSelection) -> None:
        name = selection.name
        if not name or name in self._factor_sets:
            raise FactorContractError(f"Factor set is already registered or invalid: {name!r}")
        if not selection.members:
            raise FactorContractError(f"Factor selection is empty: {name}")
        names = selection.canonical_names
        if len(names) != len(set(names)):
            raise FactorContractError(f"Factor selection {name} contains duplicate factors")
        if any(member.direction not in (1, -1) for member in selection.members):
            raise FactorContractError(f"Factor selection {name} directions must be +1 or -1")
        for source in selection.source_factor_sets:
            if source not in self._factor_sets or source in self._selections:
                raise FactorContractError(f"Factor selection {name} needs a registered library factor set: {source}")
        family_by_name = {spec.canonical_name: spec.family for lib in self._libraries.values() for spec in lib.specs}
        unknown = [member for member in names if member not in family_by_name]
        if unknown:
            raise FactorContractError(f"Factor selection {name} references unknown factors: {unknown[:5]}")
        for member in names:
            if self.source_factor_set(selection, member) is None:
                raise FactorContractError(f"Factor selection {name} has no source factor set for {member}")
        families = tuple(dict.fromkeys(family_by_name[member] for member in names))
        self._factor_sets[name] = families
        self._selections[name] = selection

    def selection(self, factor_set: FactorSet) -> FactorSelection | None:
        return self._selections.get(factor_set)

    def source_factor_set(self, selection: FactorSelection, canonical_name: str) -> FactorSet | None:
        """Return the first declared source factor set whose store publishes ``canonical_name``."""
        for source in selection.source_factor_sets:
            if any(
                spec.canonical_name == canonical_name
                for family in self._factor_sets[source]
                for spec in self._libraries[family].specs
            ):
                return source
        return None

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

    def reference_inputs(self, factor_set: FactorSet) -> tuple[str, ...]:
        """Non-canonical inputs of the factor set's libraries, in library declaration order."""
        names = (name for library in self.select(factor_set) for name in library.reference_inputs)
        return tuple(dict.fromkeys(names))

    def implementation_fingerprint(self, factor_set: FactorSet) -> str:
        libraries = [
            {"family": library.family, "fingerprint": library.implementation_fingerprint}
            for library in self.select(factor_set)
        ]
        selection = self._selections.get(factor_set)
        if selection is None:
            return stable_hash(libraries)
        return stable_hash({"libraries": libraries, "selection": selection.definition()})


def _default_registry() -> FactorLibraryRegistry:
    from rquant.factors.libraries.alpha101 import WorldQuantAlpha101Library
    from rquant.factors.libraries.alpha158 import QlibAlpha158Library
    from rquant.factors.libraries.alpha360 import QlibAlpha360Library
    from rquant.factors.libraries.barra_cne6 import BarraCNE6Library
    from rquant.factors.libraries.china_chars import ChinaCharsLibrary
    from rquant.factors.libraries.gtja191 import GTJA191Library
    from rquant.factors.libraries.hquality import HQUALITY_SELECTION
    from rquant.factors.libraries.jkp153 import JKP153Library
    from rquant.factors.libraries.jq240 import JQ240Library
    from rquant.factors.libraries.oap import OAPLibrary
    from rquant.factors.libraries.oap_jq_cnc import OAP_JQ_CNC_SELECTION
    from rquant.factors.libraries.recent import RECENT_SELECTION
    from rquant.factors.libraries.recent_sh_main import RECENT_SH_MAIN_SELECTION
    from rquant.factors.libraries.remove_duplicates import REMOVE_DUPLICATES_SELECTION
    from rquant.factors.libraries.size_value_china import SizeValueChinaLibrary
    from rquant.factors.libraries.talib_indicators import TALibLibrary
    from rquant.factors.libraries.tushare_tech import TushareTechLibrary

    registry = FactorLibraryRegistry()
    registry.register_library(QlibAlpha158Library())
    registry.register_library(WorldQuantAlpha101Library())
    registry.register_library(GTJA191Library())
    registry.register_library(QlibAlpha360Library())
    registry.register_factor_selection(HQUALITY_SELECTION)
    registry.register_library(ChinaCharsLibrary())
    registry.register_library(SizeValueChinaLibrary())
    registry.register_library(TALibLibrary())
    registry.register_factor_selection(RECENT_SELECTION)
    registry.register_library(JQ240Library())
    registry.register_library(OAPLibrary())
    registry.register_library(BarraCNE6Library())
    registry.register_library(TushareTechLibrary())
    registry.register_library(JKP153Library())
    registry.register_factor_selection(RECENT_SH_MAIN_SELECTION)
    registry.register_factor_selection(REMOVE_DUPLICATES_SELECTION)
    registry.register_factor_selection(OAP_JQ_CNC_SELECTION)
    return registry


_REGISTRY = _default_registry()


def get_library_registry() -> FactorLibraryRegistry:
    return _REGISTRY
