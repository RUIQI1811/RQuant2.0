"""Reusable provider for small user-defined KunQuant factor libraries.

Define formula callables in a dedicated module, wrap them as ``CustomFactor``
objects, then register one ``CustomFactorLibrary`` in ``libraries/registry.py``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rquant.factors.libraries.base import FactorLibrary, FactorSpec
from rquant.io import stable_hash

FormulaBuilder = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class CustomFactor:
    canonical_name: str
    source_name: str
    formula: str
    max_lookback: int
    builder: FormulaBuilder


class CustomFactorLibrary(FactorLibrary):
    def __init__(
        self,
        *,
        family: str,
        required_inputs: tuple[str, ...],
        factors: tuple[CustomFactor, ...],
        catalog_version: int = 1,
    ) -> None:
        self.family = family
        self.required_inputs = required_inputs
        self.factors = factors
        self.catalog_version = catalog_version

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        return tuple(
            FactorSpec(
                canonical_name=factor.canonical_name,
                source_name=factor.source_name,
                family=self.family,
                ordinal=ordinal,
                formula=factor.formula,
                max_lookback=factor.max_lookback,
                implementation=f"{factor.builder.__module__}.{factor.builder.__qualname__}",
                catalog_version=self.catalog_version,
            )
            for ordinal, factor in enumerate(self.factors, 1)
        )

    def build(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return {factor.source_name: factor.builder(inputs) for factor in self.factors}

    @property
    def implementation_fingerprint(self) -> str:
        return stable_hash(
            {
                "family": self.family,
                "required_inputs": self.required_inputs,
                "catalog_version": self.catalog_version,
                "factors": [
                    {
                        "spec": {
                            "canonical_name": factor.canonical_name,
                            "source_name": factor.source_name,
                            "formula": factor.formula,
                            "max_lookback": factor.max_lookback,
                        },
                        "source": inspect.getsource(factor.builder),
                    }
                    for factor in self.factors
                ],
            }
        )
