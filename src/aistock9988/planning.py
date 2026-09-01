"""Compile experiment configs and an exchange calendar into an immutable plan."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .configuration import StrategyConfig


@dataclass(frozen=True)
class RunRequest:
    signal_start: str
    signal_end: str
    execution_end: str
    output_dir: str
    run_name: str


@dataclass(frozen=True)
class RunPlan:
    run_name: str
    output_dir: str
    signal_start: str
    signal_end: str
    execution_end: str
    feature_start: str
    signal_sessions: tuple[str, ...]
    execution_sessions: tuple[str, ...]
    maximum_feature_lookback_sessions: int
    hold_sessions_from_fill: int
    strategy_id: str
    strategy_hash: str
    mode: str
    required_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_run_plan(
    strategy: StrategyConfig,
    request: RunRequest,
    sessions: Iterable[object],
    *,
    require_complete_horizon: bool = True,
) -> RunPlan:
    calendar = pd.DatetimeIndex(pd.to_datetime(list(sessions), errors="raise", utc=True)).normalize().unique().sort_values()
    if calendar.empty:
        raise ValueError("trading calendar is empty")
    start = _day(request.signal_start)
    signal_end = _day(request.signal_end)
    execution_end = _day(request.execution_end)
    if not start <= signal_end <= execution_end:
        raise ValueError("require signal_start <= signal_end <= execution_end")
    if require_complete_horizon and signal_end >= execution_end:
        raise ValueError("a complete backtest requires signal_end < execution_end")
    signal_days = calendar[(calendar >= start) & (calendar <= signal_end)]
    execution_days = calendar[(calendar >= start) & (calendar <= execution_end)]
    if signal_days.empty or execution_days.empty:
        raise ValueError("requested range has no exchange sessions")
    if str(strategy.decision["frequency"]) == "weekly":
        grouped = pd.Series(signal_days, index=signal_days).groupby(signal_days.to_period("W-FRI"), sort=True)
        signal_days = pd.DatetimeIndex([group.iloc[-1] for _, group in grouped])
    lookback = max(_feature_windows(strategy.features), default=1)
    first_index = calendar.get_indexer([signal_days[0]])[0]
    if first_index < lookback:
        raise ValueError(f"calendar does not contain {lookback} lookback sessions before first signal")
    feature_start = calendar[first_index - lookback]
    hold = int(strategy.execution["hold_sessions_from_fill"])
    last_index = calendar.get_indexer([signal_days[-1]])[0]
    execution_index = calendar.get_indexer([execution_days[-1]])[0]
    required_exit_index = last_index + int(strategy.decision["entry_delay_sessions"]) + hold
    if require_complete_horizon and required_exit_index > execution_index:
        raise ValueError("execution_end does not cover the final signal entry and holding horizon")
    dense = strategy.data_policy["dense_required"]
    sources = {"trade_cal_ts", "stock_basic_ts"}
    for stage in ("selection", "training", "execution"):
        sources.update(str(value) for value in dense[stage])
    sources.update(str(value) for value in strategy.data_policy.get("sparse_event", ()))
    sources.update(str(value) for value in strategy.data_policy.get("optional_enrichment", ()))
    return RunPlan(
        run_name=request.run_name,
        output_dir=str(Path(request.output_dir).resolve()),
        signal_start=str(signal_days[0].date()),
        signal_end=str(signal_days[-1].date()),
        execution_end=str(execution_days[-1].date()),
        feature_start=str(feature_start.date()),
        signal_sessions=tuple(str(value.date()) for value in signal_days),
        execution_sessions=tuple(str(value.date()) for value in execution_days),
        maximum_feature_lookback_sessions=lookback,
        hold_sessions_from_fill=hold,
        strategy_id=strategy.strategy_id,
        strategy_hash=strategy.config_hash,
        mode=strategy.mode,
        required_sources=tuple(sorted(sources)),
    )


def _feature_windows(features: Any) -> list[int]:
    windows: list[int] = []
    if isinstance(features, Mapping):
        for key, value in features.items():
            if key == "window_sessions":
                windows.append(int(value))
            else:
                windows.extend(_feature_windows(value))
    elif isinstance(features, (list, tuple)):
        for value in features:
            windows.extend(_feature_windows(value))
    return windows


def _day(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value)
    return day.tz_localize("UTC").normalize() if day.tzinfo is None else day.tz_convert("UTC").normalize()


__all__ = ["RunRequest", "RunPlan", "compile_run_plan"]
