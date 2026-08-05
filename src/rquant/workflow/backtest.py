from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file
from rquant.qlib_ext.exchange import AshareCostModel, AshareExchange


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


class PortfolioBacktestRunner:
    """Run Qlib TopkDropoutStrategy on the saved out-of-sample predictions."""

    def __init__(
        self,
        *,
        qlib_root: str | Path,
        run_directory: str | Path,
        config: PortfolioConfig | None = None,
    ) -> None:
        self.qlib_root = Path(qlib_root)
        self.run_directory = Path(run_directory)
        self.config = config or PortfolioConfig()

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

        qlib.init(provider_uri=str(self.qlib_root), region=REG_CN)
        start_time = predictions["datetime"].min()
        signal_end = predictions["datetime"].max()
        later = D.calendar(start_time=signal_end, freq="day")
        end_time = next((value for value in later if pd.Timestamp(value) > signal_end), signal_end)
        signal = predictions.set_index(["datetime", "instrument"])["score"].sort_index()
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
            codes="csi300",
            cost_model=cost_model,
            trade_unit=self.config.trade_unit,
            impact_cost=0.0,
        )
        strategy_class = _liquidating_strategy_class(TopkDropoutStrategy)
        strategy = strategy_class(
            signal=signal,
            topk=self.config.topk,
            n_drop=self.config.n_drop,
            risk_degree=1.0,
            hold_thresh=1,
            only_tradable=True,
            forbid_all_trade_at_limit=False,
        )
        executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, verbose=False)
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

        net_returns = report["return"].astype(float) - report["cost"].astype(float)
        benchmark_returns = report["bench"].astype(float)
        metrics = _metrics(np, net_returns, benchmark_returns)
        manifest = {
            "status": "complete",
            "engine": "qlib.backtest",
            "strategy": "LiquidatingTopkDropoutStrategy",
            "start": str(pd.Timestamp(start_time).date()),
            "end": str(pd.Timestamp(end_time).date()),
            "config": asdict(self.config),
            "constraints": {
                "deal_price": "next trading day open",
                "long_only": True,
                "equal_weight": True,
                "t_plus_one": "TopkDropoutStrategy hold_thresh=1",
                "suspension": "Qlib $close NaN",
                "open_limit": "direction-aware $limit_buy/$limit_sell",
                "market_impact": False,
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
            },
        }
        atomic_write_json(self.run_directory / "backtest.json", manifest)
        return manifest


def _liquidating_strategy_class(base_class: Any) -> type[Any]:
    """Add a constrained final-step liquidation to Qlib's TopkDropoutStrategy."""
    from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

    class LiquidatingTopkDropoutStrategy(base_class):  # type: ignore[misc,valid-type]
        def generate_trade_decision(self, execute_result: Any = None) -> Any:
            trade_step = self.trade_calendar.get_trade_step()
            if trade_step + 1 < self.trade_calendar.get_trade_len():
                return super().generate_trade_decision(execute_result)

            trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
            orders = []
            for stock in sorted(self.trade_position.get_stock_list()):
                if self.trade_position.get_stock_count(stock, bar=self.trade_calendar.get_freq()) < self.hold_thresh:
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

    return LiquidatingTopkDropoutStrategy


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
    sharpe = annual_return / volatility if volatility > 0 else float("nan")
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
