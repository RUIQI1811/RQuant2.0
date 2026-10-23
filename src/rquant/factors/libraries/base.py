"""Contracts shared by registered factor-library backends."""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from rquant.io import stable_hash

FactorFamily: TypeAlias = str
FactorSet: TypeAlias = str

CANONICAL_INPUT_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "cap",
    "sector",
    "industry",
    "subindustry",
)


@dataclass(frozen=True)
class FactorSpec:
    canonical_name: str
    source_name: str
    family: FactorFamily
    ordinal: int
    formula: str
    max_lookback: int
    implementation: str
    catalog_version: int


@dataclass(frozen=True)
class SelectedFactor:
    canonical_name: str
    # +1 when larger values predict higher returns, -1 when the factor is used inverted. Stored values are never
    # negated, so a column keeps the same meaning in every factor set; consumers that need orientation read it here.
    direction: int


@dataclass(frozen=True)
class FactorSelection:
    """A named factor set published from columns of already-built source factor sets.

    Members can come from libraries with different execution backends, because the selection never recomputes a
    formula: it copies the published source columns after checking that every source store is current.
    """

    name: FactorSet
    source_factor_sets: tuple[FactorSet, ...]
    members: tuple[SelectedFactor, ...]
    criteria: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(member.canonical_name for member in self.members)

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_factor_sets": list(self.source_factor_sets),
            "members": [{"canonical_name": m.canonical_name, "direction": m.direction} for m in self.members],
            "criteria": dict(self.criteria),
            "provenance": dict(self.provenance),
        }


class FactorLibrary(ABC):
    """A named group of formulas with a declared execution backend."""

    backend = "kunquant"
    family: FactorFamily
    required_inputs: tuple[str, ...]

    @property
    @abstractmethod
    def specs(self) -> tuple[FactorSpec, ...]:
        """Return the library's factors in stable public output order."""

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build source-name to KunQuant expression mappings."""
        raise NotImplementedError(f"{self.family} does not implement the KunQuant backend")

    @property
    def implementation_fingerprint(self) -> str:
        """Fingerprint the complete defining module, not only the provider class."""
        module = importlib.import_module(self.__class__.__module__)
        return stable_hash(
            {
                "family": self.family,
                "required_inputs": self.required_inputs,
                "source": inspect.getsource(module),
            }
        )


class PanelFactorLibrary(FactorLibrary):
    """A factor library evaluated on aligned Pandas date-by-instrument panels."""

    backend = "panel"

    @abstractmethod
    def calculate(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return source-name to aligned panel mappings."""

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        """Load inputs beyond the canonical bars, aligned to ``dates`` x ``instruments``, and the read files' hashes."""
        return {}, {}

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        outputs = self.calculate(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = outputs[spec.source_name].to_numpy(dtype=float)
