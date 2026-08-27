"""Conservative intraday stop execution from minute OHLC bars."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IntradayStopResult:
    status: str
    trigger_time: pd.Timestamp | None
    trigger_economic_price: float | None
    execution_time: pd.Timestamp | None
    execution_raw_price: float | None
    reason: str


def find_stop_execution(minutes: pd.DataFrame, *, entry_economic_price: float,
                        stop_loss_pct: float, start_time: object) -> IntradayStopResult:
    """Find the first conservative stop fill after ``start_time``.

    If the bar opens through the stop, the raw open is used. Otherwise the
    economic stop level is converted back with that bar's adjustment factor.
    A locked limit-down bar remains pending; minute OHLC cannot prove queue
    priority, so it is never treated as filled.
    """
    if entry_economic_price <= 0 or not -1 < stop_loss_pct < 0:
        raise ValueError("entry price must be positive and stop_loss_pct must be in (-1, 0)")
    required = {"trade_time", "open", "low", "high", "adj_factor", "economic_low",
                "up_limit", "down_limit", "is_locked_limit_down", "available_time"}
    missing = required - set(minutes.columns)
    if missing:
        raise ValueError(f"minute stop panel missing columns: {sorted(missing)}")
    start = pd.Timestamp(start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    level = entry_economic_price * (1.0 + stop_loss_pct)
    bars = minutes[pd.to_datetime(minutes["trade_time"], utc=True) > start].sort_values("trade_time", kind="mergesort")
    for row in bars.to_dict("records"):
        trigger_time = pd.Timestamp(row["trade_time"])
        available_time = pd.Timestamp(row["available_time"])
        if available_time.tzinfo is None:
            available_time = available_time.tz_localize("UTC")
        if available_time > trigger_time:
            continue
        if float(row["economic_low"]) > level:
            continue
        if bool(row["is_locked_limit_down"]):
            return IntradayStopResult("PENDING", trigger_time, level, None, None, "locked_limit_down")
        raw_stop = level / float(row["adj_factor"])
        raw_open = float(row["open"])
        fill = raw_open if raw_open <= raw_stop else raw_stop
        if fill <= float(row["down_limit"]):
            return IntradayStopResult("PENDING", trigger_time, level, None, None, "limit_down")
        return IntradayStopResult("FILLED", trigger_time, level, trigger_time, fill, "intraday_stop")
    return IntradayStopResult("NOT_TRIGGERED", None, None, None, None, "no_bar_crossed_stop")
