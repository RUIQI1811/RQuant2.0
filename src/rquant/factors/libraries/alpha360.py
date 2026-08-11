"""RQuant-owned Qlib Alpha360 feature definitions for KunQuant execution."""

from __future__ import annotations

from rquant.errors import FactorContractError
from rquant.factors.libraries.base import FactorLibrary, FactorSpec

CATALOG_VERSION = 1
_LAGS = tuple(range(59, -1, -1))
_GROUPS = (
    ("CLOSE", "close"),
    ("OPEN", "open"),
    ("HIGH", "high"),
    ("LOW", "low"),
    ("VWAP", "vwap"),
    ("VOLUME", "volume"),
)


def _source_formula(field: str, lag: int) -> str:
    field = field.lower()
    numerator = f"Ref(${field}, {lag})" if lag else f"${field}"
    denominator = "($volume+1e-12)" if field == "volume" else "$close"
    return f"{numerator}/{denominator}"


ALPHA360_SOURCE_NAMES = tuple(f"{label}{lag}" for label, _ in _GROUPS for lag in _LAGS)
ALPHA360_SOURCE_EXPRESSIONS = tuple(
    _source_formula(field, lag) for _, field in _GROUPS for lag in _LAGS
)


class QlibAlpha360Library(FactorLibrary):
    family = "qlib_alpha360"
    required_inputs = ("open", "high", "low", "close", "volume", "vwap")

    @property
    def specs(self) -> tuple[FactorSpec, ...]:
        if len(ALPHA360_SOURCE_NAMES) != 360:
            raise FactorContractError("Locked Alpha360 catalog must contain exactly 360 features")
        return tuple(
            FactorSpec(
                canonical_name=f"a360_{ordinal:03d}",
                source_name=source_name,
                family=self.family,
                ordinal=ordinal,
                formula=formula,
                max_lookback=max(int(source_name.removeprefix(source_name.rstrip("0123456789"))), 1),
                implementation=f"rquant.factors.libraries.alpha360.QlibAlpha360Library.build[{source_name}]",
                catalog_version=CATALOG_VERSION,
            )
            for ordinal, (source_name, formula) in enumerate(
                zip(ALPHA360_SOURCE_NAMES, ALPHA360_SOURCE_EXPRESSIONS, strict=True), 1
            )
        )
