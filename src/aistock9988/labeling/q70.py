"""Frozen q70 endpoint label construction."""
from __future__ import annotations

import pandas as pd

from .maturity import LabelProfile, build_endpoint_labels
from ..time.session import session_open


def build_q70_endpoint_labels(panel: pd.DataFrame, *, profile: LabelProfile,
                              session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    required = {"ts_code", "event_time", "economic_open"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"q70 label panel missing columns: {sorted(missing)}")
    if profile.entry_delay_sessions <= 0 or profile.horizon_sessions <= 0:
        raise ValueError("q70 endpoint labels require positive entry delay and horizon")
    ordered = panel.sort_values(["ts_code", "event_time"], kind="mergesort").copy()
    sessions = pd.DatetimeIndex(session_dates)
    if sessions.tz is None:
        sessions = sessions.tz_localize("UTC")
    else:
        sessions = sessions.tz_convert("UTC")
    sessions = sessions.normalize().drop_duplicates().sort_values()
    session_positions = {day: i for i, day in enumerate(sessions)}
    signal_days = pd.to_datetime(ordered["event_time"], utc=True).dt.normalize()

    def target_days(offset: int) -> pd.Series:
        values = []
        for day in signal_days:
            position = session_positions.get(day)
            values.append(sessions[position + offset] if position is not None and position + offset < len(sessions) else pd.NaT)
        return pd.Series(values, index=ordered.index, dtype="datetime64[ns, UTC]")

    # Join by the global session calendar, not by row offset. A missing
    # security/session row must never turn T+11 into the next available row.
    ordered["entry_time"] = target_days(profile.entry_delay_sessions)
    ordered["exit_time"] = target_days(profile.entry_delay_sessions + profile.horizon_sessions)
    prices = ordered[["ts_code", "event_time", "economic_open"]].copy()
    prices["event_time"] = pd.to_datetime(prices["event_time"], utc=True).dt.normalize()
    entry_prices = prices.rename(columns={"event_time": "entry_time", "economic_open": "entry_economic_open"})
    exit_prices = prices.rename(columns={"event_time": "exit_time", "economic_open": "exit_economic_open"})
    ordered = ordered.merge(entry_prices[["ts_code", "entry_time", "entry_economic_open"]],
                            on=["ts_code", "entry_time"], how="left", validate="many_to_one")
    ordered = ordered.merge(exit_prices[["ts_code", "exit_time", "exit_economic_open"]],
                            on=["ts_code", "exit_time"], how="left", validate="many_to_one")
    candidates = ordered.dropna(subset=["entry_time", "exit_time", "entry_economic_open", "exit_economic_open"])
    labels = build_endpoint_labels(
        candidates.rename(columns={"event_time": "signal_time"}),
        profile=profile,
        signal_column="signal_time", entry_column="entry_time", exit_column="exit_time",
        entry_price_column="entry_economic_open", exit_price_column="exit_economic_open",
        session_dates=session_dates,
    )
    labels = labels.rename(columns={"signal_time": "event_time"})
    labels["available_time"] = labels["exit_time"].map(session_open)
    return labels.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)


def build_q70_t10_labels(panel: pd.DataFrame, *, profile: LabelProfile,
                         session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the production q70 T+10 label profile."""
    if profile.entry_delay_sessions != 1 or profile.horizon_sessions != 10:
        raise ValueError("q70_t10 requires entry_delay_sessions=1 and horizon_sessions=10")
    return build_q70_endpoint_labels(panel, profile=profile, session_dates=session_dates)
