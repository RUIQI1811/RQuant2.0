"""Qlib's daily simulator executor with trade indicators that stay defined on a step without orders."""

from __future__ import annotations

from typing import Any

try:
    from qlib.backtest.executor import SimulatorExecutor as _SimulatorExecutor
    from qlib.backtest.report import Indicator as _Indicator
except ImportError:  # pragma: no cover - Qlib is a locked dependency

    class _SimulatorExecutor:  # type: ignore[no-redef]
        pass

    class _Indicator:  # type: ignore[no-redef]
        pass


class ZeroOrderSafeIndicator(_Indicator):  # type: ignore[misc,valid-type]
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


class ZeroOrderSafeSimulatorExecutor(_SimulatorExecutor):  # type: ignore[misc,valid-type]
    def reset_common_infra(self, common_infra: Any, copy_trade_account: bool = False) -> None:
        super().reset_common_infra(common_infra, copy_trade_account)
        self.trade_account.indicator = ZeroOrderSafeIndicator()
