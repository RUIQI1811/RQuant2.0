"""Contracts shared by registered factor-library backends."""

from __future__ import annotations

import importlib
import inspect
import sys
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
    # Inputs beyond the canonical bars, loaded by ``load_reference_inputs`` and aligned to the dates x instruments grid.
    reference_inputs: tuple[str, ...] = ()
    # KunQuant libraries ported from a pandas reference compile without fused multiply-adds to round like NumPy.
    strict_float = False
    # Fields that make a row one of a stock's own trading days. When set, the engine moves every stock's rows where
    # all of them are present to the top of the time axis before the graph runs and puts the outputs back on their
    # dates, so time-series operators run over consecutive quotes across suspensions.
    quote_fields: tuple[str, ...] = ()
    # C++ source (helper functions the graph's kernels call) placed before every generated KunQuant source file.
    cpp_prelude = ""
    # A statement graph (``build_statements``) runs over fiscal quarters x instruments of the first-reported statements
    # in the reference input ``fundamentals``, reading ``statement_fields`` and ``fiscal_quarter`` (0 for Q1). The
    # engine publishes each result named in ``statement_outputs`` as a daily input of ``build``: ``q_`` results from
    # the latest quarter known on the day, ``y_`` results from the latest known fiscal year
    # (``rquant.factors.statements``). Statement outputs are listed in ``reference_inputs`` too.
    statement_fields: tuple[str, ...] = ()
    statement_outputs: tuple[str, ...] = ()

    @property
    @abstractmethod
    def specs(self) -> tuple[FactorSpec, ...]:
        """Return the library's factors in stable public output order."""

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build source-name to KunQuant expression mappings."""
        raise NotImplementedError(f"{self.family} does not implement the KunQuant backend")

    def build_statements(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build ``statement_outputs`` to KunQuant expressions over fiscal quarters x instruments."""
        raise NotImplementedError(f"{self.family} has no statement graph")

    def load_reference_inputs(self, canonical_root: Any, dates: Any, instruments: Any) -> tuple[dict[str, Any], dict]:
        """Load inputs beyond the canonical bars, aligned to ``dates`` x ``instruments``, and the read files' hashes."""
        return {}, {}

    def validate_inputs(self, inputs: Mapping[str, Any]) -> None:  # noqa: B027 - optional hook
        """Raise when the aligned input arrays cannot produce trustworthy factors."""

    @property
    def implementation_fingerprint(self) -> str:
        """Fingerprint the complete defining module, not only the provider class."""
        module = importlib.import_module(self.__class__.__module__)
        payload: dict[str, Any] = {
            "family": self.family,
            "required_inputs": self.required_inputs,
            "source": inspect.getsource(module),
        }
        # Operator modules imported whole (``from rquant.factors.extensions import kunquant_panel``) define formula
        # semantics too, so their source is part of the fingerprint.
        extensions = sorted(
            value.__name__
            for value in vars(module).values()
            if inspect.ismodule(value) and value.__name__.startswith("rquant.factors.extensions.")
        )
        if extensions:
            payload["extensions"] = {name: inspect.getsource(sys.modules[name]) for name in extensions}
        if self.reference_inputs:
            payload["reference_inputs"] = self.reference_inputs
        return stable_hash(payload)


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
