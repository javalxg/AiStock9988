from __future__ import annotations

from datetime import datetime

import pandas as pd


def enforce_available_time(frame: pd.DataFrame, *, decision_time: datetime,
                           available_column: str = "available_time") -> pd.DataFrame:
    """Return a stable copy containing only information visible at decision_time."""
    if available_column not in frame.columns:
        raise ValueError(f"missing PIT column: {available_column}")
    available = pd.to_datetime(frame[available_column], errors="raise")
    cutoff = pd.Timestamp(decision_time)
    mask = available <= cutoff
    result = frame.loc[mask].copy()
    result[available_column] = available.loc[mask]
    return result.sort_values(list(result.columns), kind="mergesort").reset_index(drop=True)


def assert_no_future(frame: pd.DataFrame, *, decision_time: datetime,
                     available_column: str = "available_time") -> None:
    if available_column not in frame.columns:
        raise ValueError(f"missing PIT column: {available_column}")
    future = pd.to_datetime(frame[available_column], errors="raise") > pd.Timestamp(decision_time)
    if bool(future.any()):
        raise AssertionError(f"PIT violation: {int(future.sum())} rows are not available at decision time")
