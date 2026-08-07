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

    def calculate_into(self, inputs: Mapping[str, Any], output_buffers: Mapping[str, Any]) -> None:
        outputs = self.calculate(inputs)
        for spec in self.specs:
            output_buffers[spec.canonical_name][...] = outputs[spec.source_name].to_numpy(dtype=float)
