from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .quantdb import readonly_connection


def load_f0_panel(start: str, end: str) -> pd.DataFrame:
    """Load the frozen F0 ingredients from quant_db without mutating it."""
    spec = json.loads((Path(__file__).resolve().parents[3] / "configs/feature_sets/f0_123_columns.json").read_text())
    technical = spec["technical"]
    fundamental = spec["fundamental"]
    tech_sql = ", ".join(technical)
    fund_sql = ", ".join(fundamental)
    with readonly_connection() as conn:
        factor = pd.read_sql_query(
            f"SELECT ts_code, trade_date, update_time, {tech_sql}, open, close "
            "FROM stock_factor_pro_ts WHERE trade_date >= %s AND trade_date <= %s "
            "ORDER BY trade_date, ts_code", conn, params=(start, end))
        basic = pd.read_sql_query(
            f"SELECT ts_code, trade_date, update_time, {fund_sql} FROM daily_basic_ts "
            "WHERE trade_date >= %s AND trade_date <= %s ORDER BY trade_date, ts_code",
            conn, params=(start, end))
        industry = pd.read_sql_query(
            "SELECT ts_code, industry FROM stock_basic_ts WHERE industry IS NOT NULL AND industry <> ''",
            conn)
    factor["event_time"] = pd.to_datetime(factor.pop("trade_date"), utc=True)
    factor["source_ingested_time"] = pd.to_datetime(factor.pop("update_time"), utc=True)
    basic["event_time"] = pd.to_datetime(basic.pop("trade_date"), utc=True)
    basic["basic_source_ingested_time"] = pd.to_datetime(basic.pop("update_time"), utc=True)
    merged = factor.merge(basic, on=["ts_code", "event_time"], how="inner", validate="one_to_one")
    merged = merged.merge(industry, on="ts_code", how="left", validate="many_to_one")
    # Securities without an as-of industry cannot receive the 57 sector-relative columns.
    # Exclude them deterministically; the caller records the dropped count in the snapshot audit.
    merged = merged.dropna(subset=["industry"]).copy()
    numeric_cols = [*technical, *fundamental, "open", "close"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    # Fundamental data is visible only when both source rows are available.
    # Daily factors are treated as available at the signal session close, while ingestion times
    # remain separate audit fields and never determine historical visibility.
    merged["available_time"] = merged["event_time"] + pd.Timedelta(hours=15)
    sector_cols = []
    for col in technical:
        name = f"{col}_sector_rel"
        merged[name] = merged[col] - merged.groupby(["event_time", "industry"], sort=False)[col].transform("mean")
        sector_cols.append(name)
    merged = merged.drop(columns=["industry"])
    cols = ["ts_code", "event_time", "available_time", "open", "close", *technical, *fundamental, *sector_cols]
    return merged[cols].sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)
