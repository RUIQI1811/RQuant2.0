"""Qlib's enhanced-indexing portfolio construction on RQuant data.

``EnhancedIndexingStrategy`` follows ``qlib.contrib.strategy.EnhancedIndexingStrategy`` step for step and solves the
same ``EnhancedIndexingOptimizer`` problem (maximise benchmark-relative expected return minus ``lamb`` times the
tracking variance, long only, fully invested, per-stock benchmark deviation and total turnover limits). Two inputs
come from RQuant instead of files Qlib expects to be prepared beforehand:

- The risk model is Qlib's ``StructuredCovEstimator`` (10 PCA statistical factors) fitted, on each signal date, to
  the trailing ``lookback`` daily returns of that date's universe, as ``examples/portfolio/prepare_riskdata.py``
  prepares Qlib's risk data: returns clipped per stock at their 2.5% and 97.5% quantiles.
- The benchmark is the universe's own float-cap-weighted index on the signal date (Tushare ``circ_mv``), so the same
  construction works for csi300, main and sh_main; Qlib reads a ``$<market>_weight`` provider field instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rquant.errors import DataContractError

try:
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase as _WeightStrategyBase
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _WeightStrategyBase:  # type: ignore[no-redef]
        pass


class RiskModelData:
    """Daily prices, volumes and float caps behind the risk model, benchmark weights and tradability on a date."""

    def __init__(
        self,
        *,
        close: Any,
        volume: Any,
        float_cap: Any,
        lookback: int = 240,
        num_factors: int = 10,
    ) -> None:
        """``close`` (adjusted), ``volume`` and ``float_cap`` are frames of dates by instruments."""
        if lookback < 2:
            raise ValueError("lookback must be at least 2 trading days")
        if num_factors < 1:
            raise ValueError("num_factors must be at least 1")
        self.close = close.sort_index()
        self.volume = volume.reindex(index=self.close.index, columns=self.close.columns)
        self.float_cap = float_cap.reindex(index=self.close.index, columns=self.close.columns)
        self.lookback = lookback
        self.num_factors = num_factors
        self._cache: dict[Any, tuple[Any, Any, Any]] = {}

    def risk(self, date: Any, codes: list[str]) -> tuple[Any, Any, Any]:
        """Return factor exposures (codes x factors), factor covariance and specific volatility for ``codes``.

        The window is the ``lookback`` price rows ending on ``date``. Prices are carried over suspensions, so a
        suspended day returns zero and the move across it lands on the day trading resumes.
        """
        key = (date, tuple(codes))
        if key in self._cache:
            return self._cache[key]
        import numpy as np
        from qlib.model.riskmodel import StructuredCovEstimator

        position = self.close.index.get_indexer([date])[0]
        if position < 0:
            raise DataContractError(f"No prices on {date}")
        if position + 1 < self.lookback:
            raise DataContractError(f"Fewer than {self.lookback} price rows end on {date}")
        price = self.close.iloc[position + 1 - self.lookback : position + 1].reindex(columns=codes)
        returns = price.ffill().pct_change(fill_method=None)
        returns = returns.clip(returns.quantile(0.025), returns.quantile(0.975), axis=1)
        exposures, factor_cov, specific_var = StructuredCovEstimator(num_factors=self.num_factors).predict(
            returns, is_price=False, return_decomposed_components=True
        )
        result = (exposures, factor_cov, np.sqrt(specific_var))
        self._cache = {key: result}
        return result

    def benchmark_weight(self, date: Any, codes: list[str]) -> Any:
        """Float-cap weights of ``codes`` on ``date``, summing to one; a code without a cap weighs zero."""
        caps = self.float_cap.loc[date].reindex(codes)
        caps = caps.where(caps > 0).fillna(0.0)
        total = float(caps.sum())
        if total <= 0:
            raise DataContractError(f"No float market cap for the universe on {date}")
        return (caps / total).to_numpy(dtype=float)

    def tradable(self, date: Any, codes: list[str]) -> Any:
        """Whether each code traded on ``date`` (positive volume), Qlib's test for a stock it may rebalance."""
        return self.volume.loc[date].reindex(codes).gt(0).to_numpy()


class EnhancedIndexingStrategy(_WeightStrategyBase):  # type: ignore[misc,valid-type]
    """Qlib's enhanced-indexing strategy with an in-memory risk model and the universe's float-cap benchmark."""

    def __init__(self, *, risk_data: RiskModelData, optimizer_kwargs: dict[str, Any] | None = None, **kwargs: Any):
        from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer

        super().__init__(**kwargs)
        self.risk_data = risk_data
        self.optimizer = EnhancedIndexingOptimizer(**(optimizer_kwargs or {}))
        self.optimizations: list[dict[str, Any]] = []

    def generate_target_weight_position(
        self, score: Any, current: Any, trade_start_time: Any, trade_end_time: Any
    ) -> dict[str, float] | None:
        import numpy as np

        trade_step = self.trade_calendar.get_trade_step()
        signal_date, _ = self.trade_calendar.get_step_time(trade_step, shift=1)
        if hasattr(score, "columns"):
            score = score.iloc[:, 0]
        score = score.dropna()
        universe = sorted(set(score.index).intersection(self.risk_data.close.columns))
        if not universe:
            return None
        exposures, factor_cov, specific_risk = self.risk_data.risk(signal_date, universe)
        expected = score.reindex(universe).to_numpy(dtype=float)

        # A holding outside today's universe gets no target weight, so it is sold when it can be.
        held = current.get_stock_weight_dict(only_stock=False)
        current_weight = np.array([held.get(stock, 0.0) for stock in universe], dtype=float)
        current_weight = current_weight / self.get_risk_degree(trade_step)
        benchmark = self.risk_data.benchmark_weight(signal_date, universe)
        force_hold = ~self.risk_data.tradable(signal_date, universe)

        weight = self.optimizer(
            r=expected,
            F=exposures,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=current_weight,
            wb=benchmark,
            mfh=force_hold,
            mfs=np.zeros(len(universe), dtype=bool),
        )
        active = weight - benchmark
        self.optimizations.append(
            {
                "datetime": trade_start_time,
                "universe": len(universe),
                "holdings": int((weight > 0).sum()),
                "turnover": float(np.abs(weight - current_weight).sum()),
                "active_share": float(np.abs(active).sum() / 2.0),
                "tracking_error": float(
                    np.sqrt(
                        max(
                            0.0,
                            (active @ exposures) @ factor_cov @ (active @ exposures)
                            + (specific_risk**2) @ (active**2),
                        )
                    )
                    / 100.0
                    * np.sqrt(252.0)
                ),
            }
        )
        return {stock: float(value) for stock, value in zip(universe, weight, strict=True) if value > 0}


def load_float_cap(reference_root: str | Path, start: Any, end: Any, instruments: list[str]) -> Any:
    """Float market cap (``circ_mv``, 10,000 CNY) of ``instruments`` by date from the reference daily_basic set."""
    import pandas as pd

    root = Path(reference_root) / "daily_basic"
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for year in range(first.year, last.year + 1):
        path = root / f"year={year}" / "data.parquet"
        if not path.exists():
            raise DataContractError(
                f"Reference daily_basic partition is missing: {path}; run `rquant data build-reference`"
            )
        frame = pd.read_parquet(path, columns=["datetime", "instrument", "circ_mv"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frames.append(frame[frame["instrument"].isin(instruments)])
    caps = pd.concat(frames, ignore_index=True)
    caps = caps[(caps["datetime"] >= first) & (caps["datetime"] <= last)]
    return caps.pivot_table(index="datetime", columns="instrument", values="circ_mv", aggfunc="last")
