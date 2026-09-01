"""Read-only, PIT-timed dragon-tiger event source for formal experiments."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .quantdb import readonly_connection


@dataclass(frozen=True)
class DragonTigerEvents:
    events: pd.DataFrame
    manifest: dict[str, Any]


def load_dragon_tiger_cutoffs() -> dict[str, str]:
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT 'top_list_ts' source_name, MAX(trade_date) max_date FROM top_list_ts "
            "UNION ALL SELECT 'top_inst_ts', MAX(trade_date) FROM top_inst_ts",
            connection,
        )
    if frame["max_date"].isna().any() or set(frame["source_name"]) != {"top_list_ts", "top_inst_ts"}:
        raise ValueError("dragon-tiger source cutoff query is incomplete")
    return {
        str(row.source_name): str(pd.Timestamp(row.max_date).date())
        for row in frame.itertuples(index=False)
    }


def load_dragon_tiger_events(start: str, end: str) -> DragonTigerEvents:
    """Load and aggregate event rows without persisting source business data."""
    with readonly_connection() as connection:
        top = pd.read_sql_query(
            "SELECT trade_date, ts_code, amount, reason FROM top_list_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code, reason",
            connection,
            params=(start, end),
        )
        institution = pd.read_sql_query(
            "SELECT trade_date, ts_code, side, buy, sell, net_buy, reason "
            "FROM top_inst_ts WHERE trade_date BETWEEN %s AND %s "
            "AND exalter='机构专用' ORDER BY trade_date, ts_code, reason, side",
            connection,
            params=(start, end),
        )
    if top.empty:
        raise ValueError("top_list_ts returned no rows for the requested event range")

    for frame in (top, institution):
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="raise", utc=True
        ).dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    top["reason"] = top["reason"].fillna("").astype(str)
    top["amount"] = pd.to_numeric(top["amount"], errors="coerce")
    institution["net_buy"] = pd.to_numeric(
        institution["net_buy"], errors="coerce"
    ).fillna(0.0)

    grouped = top.groupby(["trade_date", "ts_code"], sort=True)
    rows: list[dict[str, Any]] = []
    inconsistent_amount_stock_days = 0
    for (trade_date, ts_code), frame in grouped:
        reasons = tuple(sorted(set(frame["reason"])))
        amounts = frame["amount"].dropna().to_numpy(dtype=float)
        amount = float(amounts[0]) if len(amounts) else np.nan
        consistent = bool(
            len(amounts)
            and np.isfinite(amounts).all()
            and np.allclose(amounts, amount, rtol=1e-9, atol=0.01)
        )
        inconsistent_amount_stock_days += int(not consistent)
        rows.append({
            "event_date": trade_date,
            "ts_code": ts_code,
            "reason_set": reasons,
            "reason_count": len(reasons),
            "up_reason": any("涨幅" in value for value in reasons),
            "top_list_daily_amount": amount if consistent else np.nan,
            "amount_consistent": consistent,
        })
    events = pd.DataFrame(rows)

    inst = (
        institution.groupby(["trade_date", "ts_code"], as_index=False, sort=True)
        .agg(institution_net_buy=("net_buy", "sum"), institution_row_count=("net_buy", "size"))
        .rename(columns={"trade_date": "event_date"})
    )
    events = events.merge(
        inst, on=["event_date", "ts_code"], how="left", validate="one_to_one"
    )
    events["institution_net_buy"] = events["institution_net_buy"].fillna(0.0)
    events["institution_row_count"] = events["institution_row_count"].fillna(0).astype(int)
    events["institution_positive"] = events["institution_net_buy"].gt(0.0)
    events = events.sort_values(["event_date", "ts_code"], kind="mergesort").reset_index(drop=True)
    if events.duplicated(["event_date", "ts_code"]).any():
        raise AssertionError("aggregated dragon-tiger events contain duplicate stock-day keys")

    manifest = {
        "source": "quant_db",
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "top_list_rows": int(len(top)),
        "top_list_stock_days": int(len(events)),
        "top_list_trade_dates": int(top["trade_date"].nunique()),
        "true_institution_rows": int(len(institution)),
        "true_institution_stock_days": int(len(inst)),
        "inconsistent_amount_stock_days": int(inconsistent_amount_stock_days),
        "event_min_date": str(events["event_date"].min().date()),
        "event_max_date": str(events["event_date"].max().date()),
        "top_list_sha256": _frame_hash(top),
        "true_institution_sha256": _frame_hash(institution),
        "aggregation_contract": {
            "institution": "sum net_buy only where exalter=机构专用",
            "top_list_reason": "sorted distinct set per stock-day",
            "top_list_net_amount": "not loaded and never summed",
            "daily_amount": "single stock-day value; duplicate reasons must agree",
        },
        "credentials_persisted": False,
        "business_data_persisted": False,
    }
    return DragonTigerEvents(events=events, manifest=manifest)


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].fillna("").astype(str)
    payload = {
        "columns": list(normalized.columns),
        "dtypes": [str(normalized[column].dtype) for column in normalized.columns],
        "rows_hash": hashlib.sha256(
            pd.util.hash_pandas_object(normalized, index=False).to_numpy().tobytes()
        ).hexdigest(),
        "rows": len(normalized),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "DragonTigerEvents", "load_dragon_tiger_cutoffs", "load_dragon_tiger_events"
]
