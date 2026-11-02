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
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy as _TopkDropoutStrategy
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase as _WeightStrategyBase
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _WeightStrategyBase:  # type: ignore[no-redef]
        pass

    class _TopkDropoutStrategy:  # type: ignore[no-redef]
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


class OptimizationLog:
    """Rows ``EnhancedIndexingStrategy`` appends per rebalance. Qlib copies the lists and dicts of a strategy config
    before building the strategy, so a log the caller reads back afterwards has to be an object of its own."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []


class EnhancedIndexingStrategy(_WeightStrategyBase):  # type: ignore[misc,valid-type]
    """Qlib's enhanced-indexing strategy with an in-memory risk model and the universe's float-cap benchmark."""

    def __init__(
        self,
        *,
        risk_data: RiskModelData,
        optimizer_kwargs: dict[str, Any] | None = None,
        optimization_log: OptimizationLog | None = None,
        **kwargs: Any,
    ):
        from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer

        super().__init__(**kwargs)
        self.risk_data = risk_data
        self.optimizer = EnhancedIndexingOptimizer(**(optimizer_kwargs or {}))
        self.optimization_log = optimization_log or OptimizationLog()
        self.optimizations = self.optimization_log.rows

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
        failed = not np.isfinite(weight).all() or weight.sum() <= 0
        if failed:
            # Qlib's optimiser returns the current weights when ECOS fails twice; on an empty book that is nothing to
            # hold, so the benchmark weights (feasible under every limit) stand in.
            weight = benchmark.copy()
        active = weight - benchmark
        exposure = active @ exposures
        names = getattr(self.risk_data, "factor_names", None)
        named = {}
        if names is not None:
            industries = [position for position, name in enumerate(names) if name.startswith("industry_")]
            named = {
                "max_industry_active": float(np.abs(exposure[industries]).max()),
                "size_active": float(exposure[names.index("size")]),
            }
        self.optimizations.append(
            {
                "datetime": trade_start_time,
                "universe": len(universe),
                "solved": not failed,
                "holdings": int((weight > 0).sum()),
                "turnover": float(np.abs(weight - current_weight).sum()),
                "active_share": float(np.abs(active).sum() / 2.0),
                "tracking_error": float(
                    np.sqrt(
                        max(
                            0.0,
                            exposure @ factor_cov @ exposure
                            + (specific_risk**2) @ (active**2),
                        )
                    )
                    / 100.0
                    * np.sqrt(252.0)
                ),
                **named,
            }
        )
        return {stock: float(value) for stock, value in zip(universe, weight, strict=True) if value > 0}


class RebalancingLiquidationMixin:
    """Trade every ``rebalance_days`` trading days and sell everything on the last step.

    Rebalancing keeps the signal of every ``rebalance_days``-th trading day from its first date: Qlib strategies place
    no orders on a step whose previous day has no signal, so the book stays untouched between rebalances. On the final
    step every holding held at least ``hold_thresh`` steps (one when the strategy has none) is sold if it can trade.
    """

    def __init__(self, *, rebalance_days: int = 1, **kwargs: Any) -> None:
        if rebalance_days < 1:
            raise ValueError(f"rebalance_days must be at least 1, got {rebalance_days}")
        if rebalance_days > 1 and hasattr(kwargs.get("signal"), "index"):
            kwargs["signal"] = keep_rebalance_dates(kwargs["signal"], rebalance_days)
        self.rebalance_days = rebalance_days
        super().__init__(**kwargs)

    def generate_trade_decision(self, execute_result: Any = None) -> Any:
        from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

        trade_step = self.trade_calendar.get_trade_step()
        if trade_step + 1 < self.trade_calendar.get_trade_len():
            return super().generate_trade_decision(execute_result)

        trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
        orders = []
        for stock in sorted(self.trade_position.get_stock_list()):
            held_steps = self.trade_position.get_stock_count(stock, bar=self.trade_calendar.get_freq())
            if held_steps < getattr(self, "hold_thresh", 1):
                continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock,
                start_time=trade_start,
                end_time=trade_end,
                direction=OrderDir.SELL,
            ):
                continue
            amount = self.trade_position.get_stock_amount(stock)
            order = Order(
                stock_id=stock,
                amount=amount,
                start_time=trade_start,
                end_time=trade_end,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(order):
                orders.append(order)
        return TradeDecisionWO(orders, self)


def keep_rebalance_dates(signal: Any, every: int) -> Any:
    """The rows of ``signal`` (on a <datetime, instrument> index) dated on every ``every``-th trading day of the
    provider calendar, counted from the signal's first date."""
    import pandas as pd
    from qlib.data import D

    days = signal.index.get_level_values("datetime")
    calendar = D.calendar(start_time=days.min(), end_time=days.max(), freq="day")
    kept = signal[days.isin(pd.DatetimeIndex(pd.to_datetime(list(calendar))[::every]))]
    if len(kept) == 0:
        raise DataContractError("No predictions fall on a rebalance date")
    return kept


class LiquidatingTopkDropoutStrategy(RebalancingLiquidationMixin, _TopkDropoutStrategy):  # type: ignore[misc]
    """Qlib's ``TopkDropoutStrategy`` with a rebalance interval and a final liquidation."""


class LiquidatingEnhancedIndexingStrategy(RebalancingLiquidationMixin, EnhancedIndexingStrategy):  # type: ignore[misc]
    """``EnhancedIndexingStrategy`` with a rebalance interval and a final liquidation."""


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
