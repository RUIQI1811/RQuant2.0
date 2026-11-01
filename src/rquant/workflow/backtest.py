from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rquant.errors import DataContractError, DependencyError
from rquant.io import atomic_write_json, sha256_file
from rquant.qlib_ext.exchange import AshareCostModel, AshareExchange
from rquant.qlib_ext.loader import LABEL_EXPRESSIONS

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
    """Backtest the saved out-of-sample predictions through Qlib's record templates.

    The predictions (and, given the run's horizon, their labels) are saved as ``SignalRecord`` saves them, in a fresh
    recorder under ``<run>/backtest_mlflow``; ``SigAnaRecord`` scores the signal, and ``PortAnaRecord`` runs the
    backtest (TopkDropoutStrategy or EnhancedIndexingStrategy on the A-share exchange) and its ``risk_analysis``.
    """

    def __init__(
        self,
        *,
        qlib_root: str | Path,
        run_directory: str | Path,
        config: PortfolioConfig | None = None,
        reference_root: str | Path | None = None,
        horizon: str | None = None,
    ) -> None:
        """``horizon`` names the label the predictions were trained on; without it the signal is not scored."""
        self.qlib_root = Path(qlib_root)
        self.run_directory = Path(run_directory)
        self.config = config or PortfolioConfig()
        self.reference_root = Path(reference_root) if reference_root is not None else None
        self.horizon = horizon
        if self.config.strategy == "enhanced-indexing" and self.reference_root is None:
            raise DataContractError("enhanced-indexing needs the reference data root for float market caps")
        if horizon is not None and horizon not in LABEL_EXPRESSIONS:
            raise DataContractError(f"Unsupported horizon: {horizon}")

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

        from qlib.constant import REG_CN
        from qlib.contrib.evaluate import risk_analysis
        from qlib.workflow import R
        from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

        from rquant.qlib_ext.strategy import OptimizationLog, keep_rebalance_dates
        from rquant.workflow.runner import init_run_tracking, release_run_tracking

        store = self.run_directory / "backtest_mlflow"
        # A fresh store on every report: a copied run directory would otherwise write into the original's artifacts.
        if store.exists():
            shutil.rmtree(store)
        store.mkdir()
        tracking_uri = init_run_tracking(
            store, BACKTEST_EXPERIMENT, provider_uri=self.qlib_root, region=REG_CN, qlib_module=qlib
        )
        from qlib.data import D

        start_time = predictions["datetime"].min()
        signal_end = predictions["datetime"].max()
        later = D.calendar(start_time=signal_end, freq="day")
        end_time = next((value for value in later if pd.Timestamp(value) > signal_end), signal_end)
        signal = predictions.set_index(["datetime", "instrument"])["score"].sort_index()
        signal_days = int(
            keep_rebalance_dates(signal, self.config.rebalance_days).index.get_level_values("datetime").nunique()
        )
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
        optimization_log = OptimizationLog()
        if self.config.strategy == "enhanced-indexing":
            strategy_class = "LiquidatingEnhancedIndexingStrategy"
            strategy_kwargs: dict[str, Any] = {
                "risk_data": self._risk_data(D, pd, predictions, start_time, end_time),
                "optimizer_kwargs": {
                    "lamb": self.config.risk_aversion,
                    "delta": self.config.turnover_limit,
                    "b_dev": self.config.benchmark_deviation,
                },
                "optimization_log": optimization_log,
                "risk_degree": self.config.risk_degree,
            }
        else:
            strategy_class = "LiquidatingTopkDropoutStrategy"
            strategy_kwargs = {
                "topk": self.config.topk,
                "n_drop": self.config.n_drop,
                "risk_degree": 1.0,
                "hold_thresh": 1,
                "only_tradable": True,
                "forbid_all_trade_at_limit": False,
            }
        port_analysis_config = {
            "strategy": {
                "class": strategy_class,
                "module_path": "rquant.qlib_ext.strategy",
                "kwargs": {"signal": "<PRED>", "rebalance_days": self.config.rebalance_days, **strategy_kwargs},
            },
            "executor": {
                "class": "ZeroOrderSafeSimulatorExecutor",
                "module_path": "rquant.qlib_ext.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True, "verbose": False},
            },
            "backtest": {
                "start_time": None,
                "end_time": None,
                "account": self.config.initial_capital,
                "benchmark": self.config.benchmark,
                "exchange_kwargs": {"exchange": exchange},
            },
        }
        with R.start(experiment_name=BACKTEST_EXPERIMENT, recorder_name=self.config.strategy):
            recorder = R.get_recorder()
            # The artifacts SignalRecord saves; SigAnaRecord and PortAnaRecord read them back.
            recorder.save_objects(**{"pred.pkl": signal.to_frame("score")})
            # As in SignalRecord, a label that cannot be built (no horizon) is saved as None; nothing is scored.
            labels = _labels(D, pd, signal, self.horizon) if self.horizon is not None else None
            recorder.save_objects(**{"label.pkl": labels})
            signal_summary = None
            if labels is not None:
                signal_summary = self._write_signal_analysis(pd, SigAnaRecord(recorder).generate())
            artifacts = PortAnaRecord(
                recorder,
                config=port_analysis_config,
                risk_analysis_freq="day",
                indicator_analysis_freq="day",
            ).generate()
            recorder_id = recorder.id
        release_run_tracking(store)
        if not artifacts or "report_normal_1day.pkl" not in artifacts:
            raise DataContractError("Qlib's PortAnaRecord did not return daily portfolio metrics")
        report = artifacts["report_normal_1day.pkl"]
        positions = artifacts["positions_normal_1day.pkl"]
        indicator_frame = artifacts["indicators_normal_1day.pkl"]
        indicator_object = artifacts["indicators_normal_1day_obj.pkl"]

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
                optimization_log.rows,
                columns=["datetime", "universe", "holdings", "turnover", "active_share", "tracking_error"],
            ).to_parquet(optimizations_path, index=False)

        manifest = {
            "status": "complete",
            "engine": "qlib.workflow.record_temp.PortAnaRecord",
            "strategy": strategy_class,
            "start": str(pd.Timestamp(start_time).date()),
            "end": str(pd.Timestamp(end_time).date()),
            "config": asdict(self.config),
            "signal_days": signal_days,
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
            "metrics_definition": METRICS_DEFINITION,
            "metrics": risk_metrics(risk_analysis, report, artifacts.get("port_analysis_1day.pkl")),
            "indicator_analysis": _indicator_analysis(artifacts.get("indicator_analysis_1day.pkl")),
            "mlflow": {"tracking_uri": tracking_uri, "experiment": BACKTEST_EXPERIMENT, "recorder_id": recorder_id},
            "artifacts": {
                "portfolio": sha256_file(portfolio_path),
                "positions": sha256_file(holdings_path),
                "trades": sha256_file(trades_path),
                "trade_indicators": sha256_file(indicators_path),
                **({"optimizations": sha256_file(optimizations_path)} if optimizations_path is not None else {}),
            },
        }
        if signal_summary is not None:
            manifest["signal"] = {key: signal_summary[key] for key in ("label", "overall")}
        atomic_write_json(self.run_directory / "backtest.json", manifest)
        return manifest

    def _write_signal_analysis(self, pd: Any, artifacts: dict[str, Any] | None) -> dict[str, Any] | None:
        """Write SigAnaRecord's daily IC and Rank IC, and their overall and calendar-year summary."""
        from rquant.workflow.signal_analysis import summarize_ic

        if not artifacts:
            return None
        daily = pd.DataFrame({"ic": artifacts["ic.pkl"], "rank_ic": artifacts["ric.pkl"]})
        daily = daily.rename_axis("datetime").reset_index()
        daily.to_parquet(self.run_directory / "signal_ic.parquet", index=False)
        assert self.horizon is not None
        summary = {
            "status": "complete",
            "engine": "qlib.workflow.record_temp.SigAnaRecord",
            "horizon": self.horizon,
            "label": LABEL_EXPRESSIONS[self.horizon],
            **summarize_ic(daily),
        }
        atomic_write_json(self.run_directory / "signal_analysis.json", summary)
        return summary

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


BACKTEST_EXPERIMENT = "rquant-backtest"

METRICS_DEFINITION = (
    "qlib.contrib.evaluate.risk_analysis of daily returns at freq 'day' (N = 238 trading days a year). "
    "Default mode 'sum', Qlib's convention: annualized_return = mean daily return * 238, max_drawdown on the "
    "cumulative sum of daily returns, information_ratio = mean / std * sqrt(238) (the Sharpe ratio at zero risk-free "
    "rate for net_return). Mode 'product' compounds: annualized_return = final NAV ^ (238 / days) - 1, max_drawdown "
    "on the NAV. net_return is return - cost; excess returns subtract the benchmark, as PortAnaRecord computes them."
)


def risk_metrics(risk_analysis: Any, report: Any, port_analysis: Any = None) -> dict[str, dict[str, float | None]]:
    """``risk_analysis`` of the net and benchmark returns in both modes, and PortAnaRecord's excess returns."""
    net = report["return"].astype(float) - report["cost"].astype(float)

    def analysed(returns: Any, mode: str = "sum") -> dict[str, float | None]:
        frame = risk_analysis(returns, freq="day", mode=mode)
        return {name: _finite(float(value)) for name, value in frame["risk"].items()}

    metrics = {
        "net_return": analysed(net),
        "net_return_compounded": analysed(net, "product"),
        "benchmark_return": analysed(report["bench"].astype(float)),
        "benchmark_return_compounded": analysed(report["bench"].astype(float), "product"),
    }
    if port_analysis is not None:
        for name in ("excess_return_with_cost", "excess_return_without_cost"):
            metrics[name] = {key: _finite(float(value)) for key, value in port_analysis.loc[name, "risk"].items()}
    else:
        metrics["excess_return_with_cost"] = analysed(net - report["bench"].astype(float))
    return metrics


def _indicator_analysis(frame: Any) -> dict[str, float | None] | None:
    if frame is None:
        return None
    return {str(name): _finite(float(value)) for name, value in frame["value"].items()}


def _labels(D: Any, pd: Any, signal: Any, horizon: str) -> Any:
    """The raw label of ``horizon`` on the signal's rows, as SignalRecord saves it (a <datetime, instrument> frame)."""
    days = signal.index.get_level_values("datetime")
    labels = D.features(
        sorted(signal.index.get_level_values("instrument").unique()),
        [LABEL_EXPRESSIONS[horizon]],
        start_time=days.min(),
        end_time=days.max(),
        freq="day",
    )
    labels.columns = ["LABEL0"]
    labels = labels.swaplevel().sort_index()
    labels.index = labels.index.set_names(["datetime", "instrument"])
    return labels.reindex(signal.index)


def _weighting_description(config: PortfolioConfig) -> str:
    if config.strategy == "topk-dropout":
        return f"equal weight across the top {config.topk}"
    return (
        "Qlib EnhancedIndexingOptimizer against the universe float-cap index: "
        f"risk aversion {config.risk_aversion}, turnover limit {config.turnover_limit}, "
        f"per-stock deviation {config.benchmark_deviation}, {config.risk_factors} PCA risk factors over "
        f"{config.risk_lookback_days} days, risk degree {config.risk_degree}"
    )


def _rebalance_description(config: PortfolioConfig) -> str:
    cadence = "daily" if config.rebalance_days == 1 else f"every {config.rebalance_days} trading days"
    if config.strategy == "enhanced-indexing":
        return f"{cadence}, re-optimised weights"
    if config.n_drop == config.topk:
        return f"{cadence}, full rebalance into the latest top {config.topk}"
    return f"{cadence}, drop at most {config.n_drop} of the top {config.topk}"


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
