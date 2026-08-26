"""Frozen q70 endpoint label construction."""
from __future__ import annotations

import pandas as pd

from .maturity import LabelProfile, build_endpoint_labels


def build_q70_t10_labels(panel: pd.DataFrame, *, profile: LabelProfile,
                         session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    required = {"ts_code", "event_time", "economic_open"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"q70 label panel missing columns: {sorted(missing)}")
    if profile.entry_delay_sessions != 1 or profile.horizon_sessions != 10:
        raise ValueError("q70_t10 requires entry_delay_sessions=1 and horizon_sessions=10")
    ordered = panel.sort_values(["ts_code", "event_time"], kind="mergesort").copy()
    grouped = ordered.groupby("ts_code", sort=False)
    ordered["entry_time"] = grouped["event_time"].shift(-profile.entry_delay_sessions)
    ordered["exit_time"] = grouped["event_time"].shift(-(profile.entry_delay_sessions + profile.horizon_sessions))
    ordered["entry_economic_open"] = grouped["economic_open"].shift(-profile.entry_delay_sessions)
    ordered["exit_economic_open"] = grouped["economic_open"].shift(-(profile.entry_delay_sessions + profile.horizon_sessions))
    candidates = ordered.dropna(subset=["entry_time", "exit_time", "entry_economic_open", "exit_economic_open"])
    labels = build_endpoint_labels(
        candidates.rename(columns={"event_time": "signal_time"}),
        profile=profile,
        signal_column="signal_time", entry_column="entry_time", exit_column="exit_time",
        entry_price_column="entry_economic_open", exit_price_column="exit_economic_open",
        session_dates=session_dates,
    )
    labels = labels.rename(columns={"signal_time": "event_time"})
    labels["available_time"] = labels["exit_time"] + pd.Timedelta(hours=15)
    return labels.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)
