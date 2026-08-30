"""Stable input contract and strategy-to-engine adapter for backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..configuration import StrategyConfig
from ..time.session import session_close
from .engine import BacktestConfig, run_backtest


SIGNAL_COLUMNS = {
    "asof", "ts_code", "candidate_rank", "selected",
    "selection_decision_id", "policy_id",
}
PRICE_COLUMNS = {
    "trade_date", "ts_code", "raw_open", "raw_high", "raw_low", "raw_close",
    "economic_open", "economic_high", "economic_low", "economic_close", "adj_factor",
}


@dataclass(frozen=True)
class BacktestInputs:
    """The only data objects the event-driven engine is allowed to consume."""

    signals: pd.DataFrame
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame | None = None
    minute_prices: pd.DataFrame | None = None
    data_manifest: dict[str, Any] | None = None

    def validate(self) -> None:
        missing_signal = SIGNAL_COLUMNS - set(self.signals.columns)
        if missing_signal:
            raise ValueError(f"signals missing columns: {sorted(missing_signal)}")
        missing_price = PRICE_COLUMNS - set(self.prices.columns)
        if missing_price:
            raise ValueError(f"prices missing columns: {sorted(missing_price)}")
        if self.signals.duplicated(["asof", "ts_code"]).any():
            raise ValueError("signals must have unique asof/ts_code keys")
        if self.prices.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("prices must have unique trade_date/ts_code keys")
        if not self.signals["selected"].isin([True, False]).all():
            raise ValueError("signals.selected must be boolean")
        ranks = pd.to_numeric(self.signals["candidate_rank"], errors="coerce")
        if ranks.isna().any() or (ranks < 1).any():
            raise ValueError("signals.candidate_rank must be positive")
        signal_dates = pd.to_datetime(self.signals["asof"], errors="raise", utc=True)
        price_dates = pd.to_datetime(self.prices["trade_date"], errors="raise", utc=True)
        if signal_dates.isna().any() or price_dates.isna().any():
            raise ValueError("signal and price dates must be non-null")
        if "available_time" in self.signals.columns:
            available = pd.to_datetime(self.signals["available_time"], errors="raise", utc=True)
            cutoff = signal_dates.map(session_close)
            if bool((available > cutoff).any()):
                raise AssertionError("signal ledger contains data unavailable at signal close")


def backtest_config_from_strategy(strategy: StrategyConfig, *, slippage_each_side: float | None = None) -> BacktestConfig:
    """Build the existing engine config without duplicating strategy values."""
    execution = strategy.execution
    selection = strategy.selection
    slippage = slippage_each_side
    if slippage is None:
        configured = execution.get("slippage_each_side", 0.0)
        slippage = float(configured[0] if isinstance(configured, (tuple, list)) else configured)
    return BacktestConfig(
        max_positions=int(selection.get("max_positions", execution.get("max_positions", 1))),
        hold_sessions=int(execution["hold_sessions"]),
        stop_loss_pct=(None if execution.get("stop_loss_pct") is None else float(execution["stop_loss_pct"])),
        take_profit_pct=(None if execution.get("take_profit_pct") is None else float(execution["take_profit_pct"])),
        trailing_arm_pct=(None if execution.get("trailing_arm_pct") is None else float(execution["trailing_arm_pct"])),
        trailing_drawdown_pct=(None if execution.get("trailing_drawdown_pct") is None else float(execution["trailing_drawdown_pct"])),
        stop_loss_mode=str(execution.get("stop_loss_mode", "close_next_session_open")),
        accounting_price_basis=str(execution.get("accounting_price_basis", "raw")),
        corporate_actions_mode=str(execution.get("corporate_actions_mode", "auto")),
        buy_commission=float(execution.get("buy_commission", 0.0003)),
        sell_commission=float(execution.get("sell_commission", 0.0003)),
        stamp_duty=float(execution.get("stamp_duty", 0.0005)),
        lot_size=int(execution.get("lot_size", 100)),
        buy_slippage=float(slippage),
        sell_slippage=float(slippage),
        max_order_to_adv20=(None if execution.get("max_order_to_adv20") is None else float(execution["max_order_to_adv20"])),
    )


def run_from_strategy(inputs: BacktestInputs, strategy: StrategyConfig, *, slippage_each_side: float | None = None) -> dict[str, pd.DataFrame]:
    """Run one immutable input bundle with the execution section of a strategy."""
    inputs.validate()
    config = backtest_config_from_strategy(strategy, slippage_each_side=slippage_each_side)
    return run_backtest(
        inputs.signals,
        inputs.prices,
        corporate_actions=inputs.corporate_actions,
        minute_prices=inputs.minute_prices,
        config=config,
    )


__all__ = ["BacktestInputs", "backtest_config_from_strategy", "run_from_strategy"]
