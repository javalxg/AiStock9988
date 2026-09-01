"""Frozen dragon-tiger pullback/reclaim event-state selector."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def build_pullback_reclaim_ledgers(
    events: pd.DataFrame,
    execution: pd.DataFrame,
    signal_sessions: tuple[str, ...],
    *,
    entries_per_decision: int,
) -> dict[str, pd.DataFrame]:
    required_events = {
        "event_date", "ts_code", "up_reason", "institution_net_buy",
        "institution_positive", "top_list_daily_amount", "amount_consistent",
    }
    missing = sorted(required_events - set(events.columns))
    if missing:
        raise ValueError(f"dragon-tiger events missing columns: {missing}")
    required_execution = {
        "trade_date", "ts_code", "universe_pass", "selection_data_eligible",
        "economic_open", "economic_high", "economic_close", "amount",
    }
    missing = sorted(required_execution - set(execution.columns))
    if missing:
        raise ValueError(f"execution panel missing event-state columns: {missing}")

    panel = execution.sort_values(["trade_date", "ts_code"], kind="mergesort").copy()
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate stock-day keys")
    by_key = panel.set_index(["trade_date", "ts_code"])
    sessions = pd.DatetimeIndex(panel["trade_date"].drop_duplicates().sort_values())
    session_index = {day: index for index, day in enumerate(sessions)}
    observing_until: dict[str, int] = {}
    state_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for event in events.sort_values(["event_date", "ts_code"], kind="mergesort").itertuples(index=False):
        day = pd.Timestamp(event.event_date)
        code = str(event.ts_code)
        row: dict[str, object] = {
            "event_date": day,
            "ts_code": code,
            "status": "PENDING",
            "pullback_date": pd.NaT,
            "reclaim_date": pd.NaT,
            "gap_return": np.nan,
            "institution_intensity": np.nan,
            "reclaim_volume_ratio": np.nan,
        }
        if not bool(event.up_reason):
            row["status"] = "NOT_UP_REASON"
            state_rows.append(row)
            continue
        if not bool(event.institution_positive):
            row["status"] = "INSTITUTION_NOT_POSITIVE"
            state_rows.append(row)
            continue
        amount = float(event.top_list_daily_amount)
        if not bool(event.amount_consistent) or not np.isfinite(amount) or amount <= 0:
            row["status"] = "TOP_LIST_AMOUNT_INVALID"
            state_rows.append(row)
            continue
        if day not in session_index:
            row["status"] = "EVENT_SESSION_MISSING"
            state_rows.append(row)
            continue
        event_index = session_index[day]
        if event_index <= observing_until.get(code, -1):
            row["status"] = "OVERLAPPING_EVENT_SKIPPED"
            state_rows.append(row)
            continue
        event_row = _eligible_row(by_key, day, code)
        if event_row is None:
            row["status"] = "EVENT_STOCK_DAY_INELIGIBLE"
            state_rows.append(row)
            continue
        if event_index + 1 >= len(sessions):
            row["status"] = "T1_UNAVAILABLE"
            state_rows.append(row)
            continue
        t1 = sessions[event_index + 1]
        t1_row = _eligible_row(by_key, t1, code)
        if t1_row is None:
            row["status"] = "T1_STOCK_DAY_INELIGIBLE"
            state_rows.append(row)
            continue
        event_close = float(event_row.economic_close)
        t1_open = float(t1_row.economic_open)
        row["gap_return"] = t1_open / event_close - 1.0
        if not t1_open > event_close:
            row["status"] = "NON_POSITIVE_GAP"
            state_rows.append(row)
            continue
        observing_until[code] = min(event_index + 5, len(sessions) - 1)

        pullback_index: int | None = None
        for index in range(event_index + 1, min(event_index + 4, len(sessions) - 1) + 1):
            observed = _eligible_row(by_key, sessions[index], code)
            if observed is not None and float(observed.economic_close) <= event_close:
                pullback_index = index
                row["pullback_date"] = sessions[index]
                break
        if pullback_index is None:
            row["status"] = "NO_PULLBACK_BY_T4"
            state_rows.append(row)
            continue

        reclaim_index: int | None = None
        for index in range(pullback_index + 1, min(event_index + 5, len(sessions) - 1) + 1):
            current = _eligible_row(by_key, sessions[index], code)
            previous = _eligible_row(by_key, sessions[index - 1], code)
            if current is None or previous is None:
                continue
            current_close = float(current.economic_close)
            previous_high = float(previous.economic_high)
            current_amount = float(current.amount)
            previous_amount = float(previous.amount)
            if (
                current_close > event_close
                and current_close > previous_high
                and current_amount > previous_amount
            ):
                reclaim_index = index
                row["reclaim_date"] = sessions[index]
                row["reclaim_volume_ratio"] = current_amount / previous_amount
                break
        if reclaim_index is None:
            row["status"] = "NO_RECLAIM_BY_T5"
            state_rows.append(row)
            continue

        intensity = float(event.institution_net_buy) / amount
        row["status"] = "CONFIRMED"
        row["institution_intensity"] = intensity
        candidate_rows.append({
            "asof": sessions[reclaim_index],
            "event_date": day,
            "pullback_date": sessions[pullback_index],
            "ts_code": code,
            "institution_intensity": intensity,
            "reclaim_volume_ratio": float(row["reclaim_volume_ratio"]),
            "candidate_status": "IN_VIEW",
        })
        state_rows.append(row)

    state = pd.DataFrame(state_rows)
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        candidates = pd.DataFrame(columns=[
            "asof", "event_date", "pullback_date", "ts_code",
            "institution_intensity", "reclaim_volume_ratio", "candidate_status",
            "candidate_rank", "candidate_snapshot_id",
        ])
    else:
        candidates = candidates.sort_values(
            ["asof", "institution_intensity", "reclaim_volume_ratio", "ts_code"],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        candidates["candidate_rank"] = candidates.groupby("asof", sort=False).cumcount() + 1
        snapshots = {
            day: _snapshot_id(group)
            for day, group in candidates.groupby("asof", sort=True)
        }
        candidates["candidate_snapshot_id"] = candidates["asof"].map(snapshots)
    if candidates.duplicated(["asof", "ts_code"]).any():
        raise AssertionError("confirmed candidates contain duplicate stock-day keys")

    signal_days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    selection = pd.DataFrame({"asof": signal_days})
    selection["decision_id"] = selection["asof"].map(
        lambda value: f"dtr-v1-{value.date()}"
    )
    selection["desired_entries"] = int(entries_per_decision)
    selection["target_weight_each"] = 0.20
    selection["target_weight_basis"] = "decision_nav"
    selection["cash_fraction_policy"] = 1.0
    return {"state": state, "candidate": candidates, "selection": selection}


def _eligible_row(
    by_key: pd.DataFrame,
    day: pd.Timestamp,
    code: str,
) -> object | None:
    try:
        row = by_key.loc[(day, code)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"duplicate execution row: {day.date()} {code}")
    if not bool(row.universe_pass) or not bool(row.selection_data_eligible):
        return None
    values = np.asarray([
        row.economic_open, row.economic_high, row.economic_close, row.amount
    ], dtype=float)
    if not np.isfinite(values).all() or not (values[:3] > 0).all() or values[3] <= 0:
        return None
    return row


def _snapshot_id(group: pd.DataFrame) -> str:
    values = group[[
        "asof", "event_date", "pullback_date", "ts_code",
        "institution_intensity", "reclaim_volume_ratio",
    ]].copy()
    raw = values.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = ["build_pullback_reclaim_ledgers"]
