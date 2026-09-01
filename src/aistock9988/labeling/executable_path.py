"""Executable path labels aligned with the canonical backtest engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..time.session import session_open


@dataclass(frozen=True)
class ExecutablePathLabelProfile:
    id: str = "label.executable_path_open_open_t10_base.v1"
    entry_delay_sessions: int = 1
    hold_sessions_from_fill: int = 10
    stop_threshold_pct: float = -0.08
    buy_slippage: float = 0.001
    sell_slippage: float = 0.001
    buy_commission: float = 0.0003
    sell_commission: float = 0.0003
    stamp_duty: float = 0.0005

    def validate(self) -> None:
        if self.entry_delay_sessions <= 0 or self.hold_sessions_from_fill <= 0:
            raise ValueError("entry delay and holding horizon must be positive")
        if not -1.0 < self.stop_threshold_pct < 0.0:
            raise ValueError("stop threshold must be in (-1, 0)")
        costs = (
            self.buy_slippage,
            self.sell_slippage,
            self.buy_commission,
            self.sell_commission,
            self.stamp_duty,
        )
        if any(value < 0.0 or value >= 1.0 for value in costs):
            raise ValueError("label costs must be in [0, 1)")


BUY_BLOCKED = {
    "MISSING_REQUIRED_DATA",
    "SUSPENDED",
    "LIMIT_UP",
    "ZERO_VOLUME",
    "OUT_OF_UNIVERSE",
}
SELL_BLOCKED = {
    "MISSING_REQUIRED_DATA",
    "SUSPENDED",
    "LIMIT_DOWN",
    "ZERO_VOLUME",
    "OUT_OF_UNIVERSE",
}


def _keys(dates: pd.Series, codes: pd.Series) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates, utc=True).dt.normalize(), codes.astype(str)],
        names=["trade_date", "ts_code"],
    )


def _numeric_lookup(
    indexed: pd.DataFrame,
    dates: pd.Series,
    codes: pd.Series,
    column: str,
) -> np.ndarray:
    return pd.to_numeric(indexed[column].reindex(_keys(dates, codes)), errors="coerce").to_numpy(
        dtype=float
    )


def _object_lookup(
    indexed: pd.DataFrame,
    dates: pd.Series,
    codes: pd.Series,
    column: str,
) -> np.ndarray:
    return indexed[column].reindex(_keys(dates, codes)).astype("object").to_numpy()


def build_executable_path_labels(
    signal_keys: pd.DataFrame,
    execution_panel: pd.DataFrame,
    session_dates: pd.DatetimeIndex,
    *,
    profile: ExecutablePathLabelProfile | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build per-candidate returns using the engine's actual exit semantics.

    A blocked T+1 buy has no label because that candidate could not enter. A
    close-known stop exits at the next sellable open, while a time exit starts
    at ``entry + hold_sessions_from_fill`` and retries until sellable. Returns
    include the profile's slippage, commissions, and stamp duty and are never
    clamped to the stop threshold.
    """
    profile = profile or ExecutablePathLabelProfile()
    profile.validate()
    required_keys = {"event_time", "ts_code"}
    required_execution = {
        "trade_date",
        "ts_code",
        "economic_open",
        "economic_close",
        "execution_data_eligible",
        "execution_status",
    }
    if missing := sorted(required_keys - set(signal_keys.columns)):
        raise ValueError(f"signal keys missing columns: {missing}")
    if missing := sorted(required_execution - set(execution_panel.columns)):
        raise ValueError(f"execution panel missing columns: {missing}")

    sessions = pd.DatetimeIndex(pd.to_datetime(session_dates, utc=True)).normalize()
    sessions = sessions.drop_duplicates().sort_values()
    if sessions.empty:
        raise ValueError("session calendar is empty")
    positions = pd.Series(np.arange(len(sessions), dtype="int64"), index=sessions)

    source = signal_keys[["event_time", "ts_code"]].drop_duplicates().copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True).dt.normalize()
    source["ts_code"] = source["ts_code"].astype(str)
    source["signal_index"] = source["event_time"].map(positions)
    source = source.dropna(subset=["signal_index"]).copy()
    source["signal_index"] = source["signal_index"].astype("int64")
    source["entry_index"] = source["signal_index"] + profile.entry_delay_sessions
    source = source[source["entry_index"].lt(len(sessions))].copy()
    source["entry_date"] = source["entry_index"].map(lambda value: sessions[int(value)])

    execution = execution_panel[list(required_execution)].copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"], utc=True).dt.normalize()
    execution["ts_code"] = execution["ts_code"].astype(str)
    if execution.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate stock-session keys")
    indexed = execution.set_index(["trade_date", "ts_code"]).sort_index()

    entry_open = _numeric_lookup(indexed, source["entry_date"], source["ts_code"], "economic_open")
    entry_eligible = _object_lookup(
        indexed, source["entry_date"], source["ts_code"], "execution_data_eligible"
    )
    entry_status = _object_lookup(
        indexed, source["entry_date"], source["ts_code"], "execution_status"
    )
    entry_ok = pd.Series(entry_eligible).fillna(False).astype(bool).to_numpy()
    entry_ok &= ~pd.Series(entry_status).fillna("MISSING_REQUIRED_DATA").isin(BUY_BLOCKED).to_numpy()
    entry_ok &= np.isfinite(entry_open) & (entry_open > 0.0)
    rejected_entry_rows = int((~entry_ok).sum())
    source = source.loc[entry_ok].reset_index(drop=True)
    entry_open = entry_open[entry_ok]
    source["entry_economic_open"] = entry_open
    source["entry_economic_fill"] = entry_open * (1.0 + profile.buy_slippage)

    count = len(source)
    stop_offset = np.full(count, -1, dtype="int64")
    stop_crossing_return = np.full(count, np.nan, dtype=float)
    active = np.ones(count, dtype=bool)
    for offset in range(profile.hold_sessions_from_fill):
        path_index = source["entry_index"].to_numpy(dtype="int64") + offset
        in_calendar = path_index < len(sessions)
        path_dates = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns, UTC]")
        path_dates.loc[in_calendar] = sessions.take(path_index[in_calendar])
        close = _numeric_lookup(indexed, path_dates, source["ts_code"], "economic_close")
        eligible = _object_lookup(
            indexed, path_dates, source["ts_code"], "execution_data_eligible"
        )
        valid_close = pd.Series(eligible).fillna(False).astype(bool).to_numpy()
        valid_close &= np.isfinite(close) & (close > 0.0) & in_calendar
        path_return = np.full(count, np.nan, dtype=float)
        np.divide(
            close,
            source["entry_economic_fill"].to_numpy(dtype=float),
            out=path_return,
            where=valid_close,
        )
        path_return -= 1.0
        hit = active & valid_close & (path_return <= profile.stop_threshold_pct)
        stop_offset[hit] = offset
        stop_crossing_return[hit] = path_return[hit]
        active[hit] = False

    entry_index = source["entry_index"].to_numpy(dtype="int64")
    desired_exit_index = entry_index + profile.hold_sessions_from_fill
    stop_rows = stop_offset >= 0
    desired_exit_index[stop_rows] = entry_index[stop_rows] + stop_offset[stop_rows] + 1
    source["trigger_type"] = np.where(stop_rows, "STOP_LOSS", "TIME_EXIT")
    source["stop_crossing_return"] = stop_crossing_return
    trigger_index = np.where(stop_rows, desired_exit_index - 1, desired_exit_index)
    source["trigger_date"] = pd.Series(
        pd.NaT, index=source.index, dtype="datetime64[ns, UTC]"
    )
    trigger_in_calendar = trigger_index < len(sessions)
    source.loc[trigger_in_calendar, "trigger_date"] = sessions.take(
        trigger_index[trigger_in_calendar]
    ).to_numpy()

    exit_index = np.full(count, -1, dtype="int64")
    exit_open = np.full(count, np.nan, dtype=float)
    exit_retry_sessions = np.full(count, -1, dtype="int64")
    unresolved = desired_exit_index < len(sessions)
    retry = 0
    while unresolved.any():
        indices = desired_exit_index + retry
        in_calendar = unresolved & (indices < len(sessions))
        if not in_calendar.any():
            break
        dates = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns, UTC]")
        dates.loc[in_calendar] = sessions.take(indices[in_calendar])
        opens = _numeric_lookup(indexed, dates, source["ts_code"], "economic_open")
        eligible = _object_lookup(indexed, dates, source["ts_code"], "execution_data_eligible")
        status = _object_lookup(indexed, dates, source["ts_code"], "execution_status")
        sellable = pd.Series(eligible).fillna(False).astype(bool).to_numpy()
        sellable &= ~pd.Series(status).fillna("MISSING_REQUIRED_DATA").isin(SELL_BLOCKED).to_numpy()
        sellable &= np.isfinite(opens) & (opens > 0.0) & in_calendar
        resolved_now = unresolved & sellable
        exit_index[resolved_now] = indices[resolved_now]
        exit_open[resolved_now] = opens[resolved_now]
        exit_retry_sessions[resolved_now] = retry
        unresolved[resolved_now] = False
        unresolved[indices >= len(sessions) - 1] = False
        retry += 1

    resolved = exit_index >= 0
    labels = source.loc[resolved, [
        "event_time",
        "ts_code",
        "entry_date",
        "trigger_date",
        "trigger_type",
        "entry_economic_open",
        "entry_economic_fill",
        "stop_crossing_return",
    ]].copy()
    labels["exit_date"] = sessions.take(exit_index[resolved])
    labels["exit_retry_sessions"] = exit_retry_sessions[resolved]
    labels["exit_economic_open"] = exit_open[resolved]
    labels["exit_economic_fill"] = labels["exit_economic_open"] * (1.0 - profile.sell_slippage)
    entry_cost = labels["entry_economic_fill"] * (1.0 + profile.buy_commission)
    exit_proceeds = labels["exit_economic_fill"] * (
        1.0 - profile.sell_commission - profile.stamp_duty
    )
    labels["economic_return"] = (
        labels["exit_economic_fill"] / labels["entry_economic_fill"] - 1.0
    )
    labels["label_return"] = exit_proceeds / entry_cost - 1.0
    labels["available_time"] = labels["exit_date"].map(session_open)
    labels = labels.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)

    stop_labels = labels[labels["trigger_type"].eq("STOP_LOSS")]
    audit = {
        "profile": asdict(profile),
        "requested_rows": int(len(signal_keys[["event_time", "ts_code"]].drop_duplicates())),
        "calendar_resolved_signal_rows": int(len(source) + rejected_entry_rows),
        "entry_rejected_rows": rejected_entry_rows,
        "label_rows": int(len(labels)),
        "unresolved_exit_rows": int((~resolved).sum()),
        "retried_exit_rows": int(labels["exit_retry_sessions"].gt(0).sum()),
        "maximum_exit_retry_sessions": (
            int(labels["exit_retry_sessions"].max()) if len(labels) else None
        ),
        "stop_rows": int(len(stop_labels)),
        "time_exit_rows": int(labels["trigger_type"].eq("TIME_EXIT").sum()),
        "mean_label_return": float(labels["label_return"].mean()) if len(labels) else None,
        "mean_stop_label_return": (
            float(stop_labels["label_return"].mean()) if len(stop_labels) else None
        ),
        "mean_stop_crossing_return": (
            float(stop_labels["stop_crossing_return"].mean()) if len(stop_labels) else None
        ),
        "minimum_label_return": float(labels["label_return"].min()) if len(labels) else None,
        "maximum_label_return": float(labels["label_return"].max()) if len(labels) else None,
        "labels_clamped": False,
    }
    return labels, audit


__all__ = ["ExecutablePathLabelProfile", "build_executable_path_labels"]
