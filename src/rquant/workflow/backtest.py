from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file
from rquant.qlib_ext.exchange import AshareCostModel, AshareExchange
from rquant.qlib_ext.runtime import qlib_init_kwargs

STRATEGIES = ("topk-dropout", "enhanced-indexing")


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    topk: int = 50
    n_drop: int = 5
    benchmark: str = "SH000300"
    trade_unit: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_before: float = 0.001
    stamp_tax_after: float = 0.0005
    impact_cost: float = 0.0
    universe: str = "csi300"
    # Trade only every `rebalance_days` trading days, counted from the first prediction date. With n_drop == topk each
    # rebalance replaces every holding that fell out of the latest top-K, so the book becomes that top-K.
    rebalance_days: int = 1
    # "topk-dropout" holds the equal-weighted top K. "enhanced-indexing" weights the whole universe with Qlib's
    # EnhancedIndexingOptimizer against the universe's float-cap index (rquant.qlib_ext.strategy); topk and n_drop are
    # then unused.
    strategy: str = "topk-dropout"
    risk_lookback_days: int = 240
    risk_factors: int = 10
    risk_aversion: float = 1.0
    # Qlib's optimiser needs a number here; 2.0 (sell the whole book and buy a new one) leaves turnover unlimited.
    turnover_limit: float = 0.2
    benchmark_deviation: float | None = 0.01
    risk_degree: float = 0.95

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError(f"topk must be at least 1, got {self.topk}")
        if not 1 <= self.n_drop <= self.topk:
            raise ValueError(f"n_drop must be between 1 and topk={self.topk}, got {self.n_drop}")
        if self.rebalance_days < 1:
            raise ValueError(f"rebalance_days must be at least 1, got {self.rebalance_days}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy}")
        if self.risk_lookback_days < 2 or self.risk_factors < 1:
            raise ValueError("risk_lookback_days must be at least 2 and risk_factors at least 1")
        if self.risk_aversion < 0:
            raise ValueError(f"risk_aversion must not be negative, got {self.risk_aversion}")
        if self.turnover_limit < 0:
            raise ValueError(f"turnover_limit must not be negative, got {self.turnover_limit}")
        if self.benchmark_deviation is not None and self.benchmark_deviation < 0:
            raise ValueError(f"benchmark_deviation must not be negative, got {self.benchmark_deviation}")
        if not 0 < self.risk_degree <= 1:
            raise ValueError(f"risk_degree must be in (0, 1], got {self.risk_degree}")


class PortfolioBacktestRunner:
    """Run Qlib TopkDropoutStrategy or EnhancedIndexingStrategy on the saved out-of-sample predictions."""

    def __init__(
        self,
        *,
        qlib_root: str | Path,
        run_directory: str | Path,
        config: PortfolioConfig | None = None,
        reference_root: str | Path | None = None,
    ) -> None:
        self.qlib_root = Path(qlib_root)
        self.run_directory = Path(run_directory)
        self.config = config or PortfolioConfig()
        self.reference_root = Path(reference_root) if reference_root is not None else None
        if self.config.strategy == "enhanced-indexing" and self.reference_root is None:
            raise DataContractError("enhanced-indexing needs the reference data root for float market caps")

    def run(self) -> dict[str, Any]:
        qlib, pd, np = _dependencies()
        predictions_path = self.run_directory / "predictions.parquet"
        if not predictions_path.exists():
            raise DataContractError(f"Predictions do not exist: {predictions_path}")
        predictions = pd.read_parquet(predictions_path)
        required = {"datetime", "instrument", "score"}
        if not required.issubset(predictions.columns):
            raise DataContractError(f"Prediction columns must include {sorted(required)}")
        predictions = predictions.dropna(subset=["score"]).copy()
        predictions["datetime"] = pd.to_datetime(predictions["datetime"])
        if predictions.empty:
            raise DataContractError("No finite predictions are available for backtesting")
        if predictions.duplicated(["datetime", "instrument"]).any():
            raise DataContractError("Predictions contain duplicate datetime/instrument rows")

        from qlib.backtest import backtest
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.constant import REG_CN
        from qlib.contrib.strategy import TopkDropoutStrategy
        from qlib.data import D

        from rquant.qlib_ext.strategy import EnhancedIndexingStrategy

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN, **qlib_init_kwargs())
        start_time = predictions["datetime"].min()
        signal_end = predictions["datetime"].max()
        later = D.calendar(start_time=signal_end, freq="day")
        end_time = next((value for value in later if pd.Timestamp(value) > signal_end), signal_end)
        signal = predictions.set_index(["datetime", "instrument"])["score"].sort_index()
        rebalance_dates = _rebalance_dates(
            pd,
            D.calendar(start_time=start_time, end_time=signal_end, freq="day"),
            self.config.rebalance_days,
        )
        # Qlib strategies place no orders on a step whose previous day has no signal, so dropping the signal on
        # non-rebalance days leaves the book untouched until the next rebalance.
        signal = signal[signal.index.get_level_values("datetime").isin(rebalance_dates)]
        if signal.empty:
            raise DataContractError("No predictions fall on a rebalance date")
        cost_model = AshareCostModel(
            commission_rate=self.config.commission_rate,
            minimum_commission=self.config.minimum_commission,
            stamp_tax_before=self.config.stamp_tax_before,
            stamp_tax_after=self.config.stamp_tax_after,
        )
        exchange = AshareExchange(
            freq="day",
            start_time=start_time,
            end_time=end_time,
            # Quotes must cover every instrument the strategy can hold, for its whole history: a "csi300" market only
            # returns quotes inside membership spans, so holdings that left the index could never be sold again.
            # The signal itself is already restricted to point-in-time universe members, so buys stay in-universe.
            codes=sorted(predictions["instrument"].unique()),
            cost_model=cost_model,
            trade_unit=self.config.trade_unit,
            impact_cost=self.config.impact_cost,
        )
        if self.config.strategy == "enhanced-indexing":
            strategy = _liquidating_strategy_class(EnhancedIndexingStrategy)(
                signal=signal,
                risk_data=self._risk_data(D, pd, predictions, start_time, end_time),
                optimizer_kwargs={
                    "lamb": self.config.risk_aversion,
                    "delta": self.config.turnover_limit,
                    "b_dev": self.config.benchmark_deviation,
                },
                risk_degree=self.config.risk_degree,
            )
        else:
            strategy = _liquidating_strategy_class(TopkDropoutStrategy)(
                signal=signal,
                topk=self.config.topk,
                n_drop=self.config.n_drop,
                risk_degree=1.0,
                hold_thresh=1,
                only_tradable=True,
                forbid_all_trade_at_limit=False,
            )
        executor_class = _zero_order_safe_executor_class(SimulatorExecutor)
        executor = executor_class(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
        portfolio, indicators = backtest(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy,
            executor=executor,
            benchmark=self.config.benchmark,
            account=self.config.initial_capital,
            exchange_kwargs={"exchange": exchange},
        )
        if "1day" not in portfolio:
            raise DataContractError(f"Qlib did not return daily portfolio metrics: {sorted(portfolio)}")
        report, positions = portfolio["1day"]
        indicator_frame, indicator_object = indicators["1day"]

        portfolio_path = self.run_directory / "portfolio.parquet"
        report.reset_index().to_parquet(portfolio_path, index=False)
        indicators_path = self.run_directory / "trade_indicators.parquet"
        indicator_frame.reset_index().to_parquet(indicators_path, index=False)
        holdings = _position_artifact(pd, positions, exchange)
        trades = _trade_artifact(pd, indicator_object, exchange)
        holdings_path = self.run_directory / "positions.parquet"
        trades_path = self.run_directory / "trades.parquet"
        holdings.to_parquet(holdings_path, index=False)
        trades.to_parquet(trades_path, index=False)

        terminal_position = positions[max(positions)]
        unliquidated = sorted(
            stock
            for stock in terminal_position.get_stock_list()
            if abs(float(terminal_position.get_stock_amount(stock))) > 1e-12
        )

        optimizations_path = None
        if self.config.strategy == "enhanced-indexing":
            optimizations_path = self.run_directory / "optimizations.parquet"
            pd.DataFrame(
                strategy.optimizations,
                columns=["datetime", "universe", "holdings", "turnover", "active_share", "tracking_error"],
            ).to_parquet(optimizations_path, index=False)

        net_returns = report["return"].astype(float) - report["cost"].astype(float)
        benchmark_returns = report["bench"].astype(float)
        metrics = _metrics(np, net_returns, benchmark_returns)
        manifest = {
            "status": "complete",
            "engine": "qlib.backtest",
            "strategy": type(strategy).__name__,
            "start": str(pd.Timestamp(start_time).date()),
            "end": str(pd.Timestamp(end_time).date()),
            "config": asdict(self.config),
            "signal_days": int(signal.index.get_level_values("datetime").nunique()),
            "constraints": {
                "rebalance": _rebalance_description(self.config),
                "deal_price": "next trading day open",
                "long_only": True,
                "equal_weight": self.config.strategy == "topk-dropout",
                "weights": _weighting_description(self.config),
                "t_plus_one": (
                    "TopkDropoutStrategy hold_thresh=1"
                    if self.config.strategy == "topk-dropout"
                    else "daily steps: a stock bought at one open is sold at the earliest on the next"
                ),
                "suspension": "Qlib $close NaN",
                "open_limit": "direction-aware $limit_buy/$limit_sell",
                "market_impact": (
                    f"impact_cost * (trade_value / daily_value)^2, impact_cost={self.config.impact_cost}"
                    if self.config.impact_cost
                    else False
                ),
                "queue_model": False,
                "capacity_model": False,
                "terminal_liquidation": "attempted on the final available open under the same trading constraints",
            },
            "terminal_liquidation": {
                "complete": not unliquidated,
                "unliquidated_instruments": unliquidated,
            },
            "rows": len(report),
            "trade_rows": len(trades),
            "metrics": metrics,
            "artifacts": {
                "portfolio": sha256_file(portfolio_path),
                "positions": sha256_file(holdings_path),
                "trades": sha256_file(trades_path),
                "trade_indicators": sha256_file(indicators_path),
                **({"optimizations": sha256_file(optimizations_path)} if optimizations_path is not None else {}),
            },
        }
        atomic_write_json(self.run_directory / "backtest.json", manifest)
        return manifest


    def _risk_data(self, D: Any, pd: Any, predictions: Any, start_time: Any, end_time: Any) -> Any:
        """Prices, volumes and float caps of every predicted instrument, from ``risk_lookback_days`` rows before the
        first signal through the last trading day."""
        from rquant.qlib_ext.strategy import RiskModelData, load_float_cap

        assert self.reference_root is not None
        calendar = D.calendar(end_time=start_time, freq="day")
        if len(calendar) < self.config.risk_lookback_days:
            raise DataContractError(
                f"The provider calendar holds fewer than {self.config.risk_lookback_days} days before {start_time}"
            )
        history_start = calendar[-self.config.risk_lookback_days]
        instruments = sorted(predictions["instrument"].unique())
        quotes = D.features(instruments, ["$close", "$volume"], start_time=history_start, end_time=end_time, freq="day")
        quotes.columns = ["close", "volume"]
        close = quotes["close"].unstack(level="instrument").sort_index()
        volume = quotes["volume"].unstack(level="instrument").sort_index()
        float_cap = load_float_cap(self.reference_root, history_start, end_time, instruments)
        return RiskModelData(
            close=close,
            volume=volume,
            float_cap=float_cap,
            lookback=self.config.risk_lookback_days,
            num_factors=self.config.risk_factors,
        )


def _weighting_description(config: PortfolioConfig) -> str:
    if config.strategy == "topk-dropout":
        return f"equal weight across the top {config.topk}"
    return (
        "Qlib EnhancedIndexingOptimizer against the universe float-cap index: "
        f"risk aversion {config.risk_aversion}, turnover limit {config.turnover_limit}, "
        f"per-stock deviation {config.benchmark_deviation}, {config.risk_factors} PCA risk factors over "
        f"{config.risk_lookback_days} days, risk degree {config.risk_degree}"
    )


def _rebalance_dates(pd: Any, calendar: Any, every: int) -> Any:
    """Return every `every`-th trading day of `calendar`, starting with its first day."""
    return pd.DatetimeIndex(pd.to_datetime(list(calendar))[::every])


def _rebalance_description(config: PortfolioConfig) -> str:
    cadence = "daily" if config.rebalance_days == 1 else f"every {config.rebalance_days} trading days"
    if config.strategy == "enhanced-indexing":
        return f"{cadence}, re-optimised weights"
    if config.n_drop == config.topk:
        return f"{cadence}, full rebalance into the latest top {config.topk}"
    return f"{cadence}, drop at most {config.n_drop} of the top {config.topk}"


def _liquidating_strategy_class(base_class: Any) -> type[Any]:
    """Add a constrained final-step liquidation to a Qlib signal strategy."""
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

    class LiquidatingStrategy(base_class):  # type: ignore[misc,valid-type]
        def generate_trade_decision(self, execute_result: Any = None) -> Any:
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

    LiquidatingStrategy.__name__ = LiquidatingStrategy.__qualname__ = f"Liquidating{base_class.__name__}"
    return LiquidatingStrategy


def _zero_order_safe_executor_class(base_class: Any) -> type[Any]:
    """Keep Qlib trade indicators defined when a daily decision contains no orders."""
    from qlib.backtest.report import Indicator

    class ZeroOrderSafeIndicator(Indicator):
        def cal_trade_indicators(
            self,
            trade_start_time: Any,
            freq: str,
            indicator_config: dict[str, Any] | None = None,
        ) -> None:
            if self.order_indicator.get_index_data("amount").empty:
                self.trade_indicator.update(
                    {
                        "ffr": None,
                        "pa": None,
                        "pos": None,
                        "deal_amount": 0.0,
                        "value": 0.0,
                        "count": 0,
                    }
                )
                return
            super().cal_trade_indicators(trade_start_time, freq, indicator_config or {})

    class ZeroOrderSafeSimulatorExecutor(base_class):  # type: ignore[misc,valid-type]
        def reset_common_infra(self, common_infra: Any, copy_trade_account: bool = False) -> None:
            super().reset_common_infra(common_infra, copy_trade_account)
            self.trade_account.indicator = ZeroOrderSafeIndicator()

    return ZeroOrderSafeSimulatorExecutor


def _position_artifact(pd: Any, positions: dict[Any, Any], exchange: Any) -> Any:
    holding_rows: list[dict[str, Any]] = []
    last_factors: dict[str, float] = {}
    for timestamp, position in sorted(positions.items()):
        current = {stock: float(position.get_stock_amount(stock)) for stock in position.get_stock_list()}
        for stock, amount in sorted(current.items()):
            current_factor = exchange.get_factor(stock, timestamp, timestamp)
            if current_factor is not None:
                last_factors[stock] = float(current_factor)
            factor = last_factors.get(stock, float("nan"))
            holding_rows.append(
                {
                    "datetime": timestamp,
                    "instrument": stock,
                    "adjusted_amount": amount,
                    "shares": amount * factor,
                    "factor": factor,
                }
            )
    holding_columns = ["datetime", "instrument", "adjusted_amount", "shares", "factor"]
    return pd.DataFrame(holding_rows, columns=holding_columns)


def _trade_artifact(pd: Any, indicator: Any, exchange: Any) -> Any:
    trade_rows: list[dict[str, Any]] = []
    for timestamp, order_indicator in sorted(indicator.order_indicator_his.items()):
        metrics = {
            name: order_indicator.get_index_data(name).to_dict()
            for name in ("deal_amount", "trade_price", "trade_value", "trade_cost", "trade_dir")
        }
        instruments = sorted(set().union(*(metric.keys() for metric in metrics.values())))
        for instrument in instruments:
            amount = metrics["deal_amount"].get(instrument)
            if amount is None or pd.isna(amount) or abs(float(amount)) <= 1e-12:
                continue
            factor = exchange.get_factor(instrument, timestamp, timestamp)
            adjusted_amount = abs(float(amount))
            trade_rows.append(
                {
                    "datetime": timestamp,
                    "instrument": instrument,
                    "direction": "sell" if int(metrics["trade_dir"].get(instrument)) == 0 else "buy",
                    "adjusted_amount": adjusted_amount,
                    "shares": adjusted_amount * float(factor),
                    "price": float(metrics["trade_price"].get(instrument)),
                    "value": abs(float(metrics["trade_value"].get(instrument))),
                    "cost": float(metrics["trade_cost"].get(instrument)),
                }
            )
    trade_columns = ["datetime", "instrument", "direction", "adjusted_amount", "shares", "price", "value", "cost"]
    return pd.DataFrame(trade_rows, columns=trade_columns)


def _metrics(np: Any, returns: Any, benchmark: Any) -> dict[str, float | None]:
    values = returns.fillna(0.0).to_numpy(dtype=float)
    benchmark_values = benchmark.fillna(0.0).to_numpy(dtype=float)
    nav = np.cumprod(1.0 + values)
    excess = values - benchmark_values
    annual_return = float(nav[-1] ** (252.0 / len(values)) - 1.0) if len(values) else float("nan")
    volatility = float(np.std(values, ddof=1) * math.sqrt(252)) if len(values) > 1 else float("nan")
    # Conventional annualized Sharpe: arithmetic mean daily return scaled by 252 over annualized volatility.
    sharpe = float(np.mean(values) * 252.0) / volatility if volatility > 0 else float("nan")
    running_max = np.maximum.accumulate(nav) if len(nav) else np.asarray([])
    max_drawdown = float(np.min(nav / running_max - 1.0)) if len(nav) else float("nan")
    return {
        "total_return": _finite(float(nav[-1] - 1.0)) if len(nav) else None,
        "annual_return": _finite(annual_return),
        "annual_volatility": _finite(volatility),
        "sharpe_zero_rf": _finite(sharpe),
        "max_drawdown": _finite(max_drawdown),
        "benchmark_total_return": _finite(float(np.prod(1.0 + benchmark_values) - 1.0)),
        "excess_total_return_additive": _finite(float(np.sum(excess))),
    }


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import qlib
    except ImportError as exc:
        raise DependencyError(
            "pyqlib==0.9.7, pandas, NumPy and pyarrow are required for portfolio backtesting"
        ) from exc
    return qlib, pd, np
