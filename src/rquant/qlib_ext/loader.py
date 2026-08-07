from __future__ import annotations

from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.factors.catalog import FactorSet, get_catalog

try:
    from qlib.data.dataset.loader import DataLoader as _QlibDataLoaderBase
except ImportError:

    class _QlibDataLoaderBase:  # type: ignore[no-redef]
        pass


LABEL_EXPRESSIONS = {
    "1d": "Ref($open, -2) / Ref($open, -1) - 1",
    "5d": "Ref($open, -6) / Ref($open, -1) - 1",
    "20d": "Ref($open, -21) / Ref($open, -1) - 1",
}

HORIZON_STEPS = {"1d": 1, "5d": 5, "20d": 20}


class KunQuantDataLoader(_QlibDataLoaderBase):
    """Load a registered RQuant factor store and a Qlib-evaluated next-open label."""

    def __init__(self, factor_root: str, factor_set: FactorSet = "combined", horizon: str = "1d") -> None:
        if horizon not in LABEL_EXPRESSIONS:
            raise DataContractError(f"Unsupported horizon: {horizon}")
        self.factor_root = Path(factor_root)
        self.factor_set = factor_set
        self.horizon = horizon
        self.catalog = get_catalog()
        self.catalog.select(factor_set)

    def load(
        self,
        instruments: Any = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> Any:
        try:
            import pandas as pd
            from qlib.data import D
        except ImportError as exc:
            raise DependencyError("pandas, pyarrow and pyqlib==0.9.7 are required by KunQuantDataLoader") from exc

        paths = sorted((self.factor_root / self.factor_set).glob("year=*/factors.parquet"))
        if not paths:
            raise DataContractError(f"No factor partitions for {self.factor_set}: {self.factor_root}")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        if start_time is not None:
            frame = frame[frame["datetime"] >= pd.Timestamp(start_time)]
        if end_time is not None:
            frame = frame[frame["datetime"] <= pd.Timestamp(end_time)]
        instrument_list = _resolve_instruments(D, instruments, start_time, end_time)
        if instrument_list:
            frame = frame[frame["instrument"].isin(instrument_list)]
        expected = list(self.catalog.canonical_names(self.factor_set))
        factor_columns = [column for column in frame.columns if column not in {"datetime", "instrument"}]
        if factor_columns != expected:
            raise DataContractError("Factor store columns do not match the locked catalog")

        # DataHandlerLP follows QlibDataLoader's canonical <datetime, instrument> index order.
        features = frame.set_index(["datetime", "instrument"])[expected].sort_index()
        query_instruments = sorted(features.index.get_level_values("instrument").unique())
        labels = D.features(
            query_instruments,
            [LABEL_EXPRESSIONS[self.horizon]],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        labels.columns = ["LABEL0"]
        labels = labels.swaplevel().sort_index()
        labels.index = labels.index.set_names(["datetime", "instrument"])
        merged = features.join(labels, how="left")
        merged.columns = pd.MultiIndex.from_tuples(
            [("feature", column) if column != "LABEL0" else ("label", column) for column in merged.columns]
        )
        return merged.sort_index()


def _resolve_instruments(D: Any, instruments: Any, start_time: Any, end_time: Any) -> list[str] | None:
    if instruments is None:
        return None
    if isinstance(instruments, (list, tuple, set)):
        return list(instruments)
    if isinstance(instruments, str):
        config = D.instruments(market=instruments)
        return D.list_instruments(config, start_time=start_time, end_time=end_time, freq="day", as_list=True)
    return D.list_instruments(instruments, start_time=start_time, end_time=end_time, freq="day", as_list=True)
