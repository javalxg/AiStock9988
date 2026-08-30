from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .quantdb import readonly_connection
from .industry_pit import resolve_industry_map
from ..time.session import parse_source_time, session_close


def load_f0_panel(start: str, end: str, *, return_audit: bool = False,
                  sector_relative_statistic: str = "mean",
                  ts_codes: list[str] | None = None,
                  include_industry: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Load the frozen F0 ingredients from quant_db without mutating it."""
    if sector_relative_statistic not in {"mean", "median"}:
        raise ValueError("sector_relative_statistic must be mean or median")
    spec = json.loads((Path(__file__).resolve().parents[3] / "configs/feature_sets/f0_123_columns.json").read_text())
    technical = spec["technical"]
    fundamental = spec["fundamental"]
    tech_sql = ", ".join(technical)
    fund_sql = ", ".join(fundamental)
    code_sql = ""
    code_params: tuple[object, ...] = ()
    if ts_codes:
        code_sql = " AND f.ts_code IN (" + ",".join(["%s"] * len(ts_codes)) + ")"
        code_params = tuple(ts_codes)
    with readonly_connection() as conn:
        factor = pd.read_sql_query(
            f"SELECT f.ts_code, f.trade_date, f.update_time, {', '.join('f.' + c for c in technical)}, "
            "f.open, f.close, a.adj_factor, a.update_time AS adj_update_time "
            "FROM stock_factor_pro_ts f JOIN adj_factor_ts a "
            "ON a.ts_code=f.ts_code AND a.trade_date=f.trade_date "
            "WHERE f.trade_date >= %s AND f.trade_date <= %s " + code_sql +
            "ORDER BY f.trade_date, f.ts_code", conn, params=(start, end, *code_params))
        basic = pd.read_sql_query(
            f"SELECT ts_code, trade_date, update_time, {fund_sql} FROM daily_basic_ts "
            "WHERE trade_date >= %s AND trade_date <= %s " +
            (" AND ts_code IN (" + ",".join(["%s"] * len(ts_codes)) + ")" if ts_codes else "") +
            " ORDER BY trade_date, ts_code", conn, params=(start, end, *code_params))
        membership = pd.read_sql_query(
            "SELECT index_code, con_code, name, in_date, out_date, update_time "
            "FROM index_member_all_ts "
            "WHERE in_date <= %s AND (out_date IS NULL OR out_date > %s) "
            "ORDER BY con_code, in_date DESC, index_code ASC",
            conn, params=(end, start))
    factor["event_time"] = pd.to_datetime(factor.pop("trade_date"), utc=True)
    factor["source_ingested_time"] = parse_source_time(factor.pop("update_time"))
    factor["adj_source_ingested_time"] = parse_source_time(factor.pop("adj_update_time"))
    basic["event_time"] = pd.to_datetime(basic.pop("trade_date"), utc=True)
    basic["basic_source_ingested_time"] = parse_source_time(basic.pop("update_time"))
    merged = factor.merge(basic, on=["ts_code", "event_time"], how="inner", validate="one_to_one")
    merged["ts_code"] = merged["ts_code"].astype(str)
    membership["con_code"] = membership["con_code"].astype(str)
    numeric_cols = [*technical, *fundamental, "open", "close", "adj_factor"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    # These tables are a frozen historical snapshot: update_time records the
    # batch ingestion of the snapshot, not a historical vendor publication
    # timestamp.  Use the exchange session close as the replay availability
    # time and retain the raw ingestion timestamps for auditability.
    merged["available_time"] = merged["event_time"].map(session_close)
    merged["economic_open"] = merged["open"] * merged["adj_factor"]
    merged["economic_close"] = merged["close"] * merged["adj_factor"]
    sector_cols = []
    resolved_industry = []
    visible_groups = []
    for event_time, group in merged.groupby("event_time", sort=True):
        decision_time = session_close(event_time)
        visible = group[group["available_time"] <= decision_time].copy()
        mapping, audit = resolve_industry_map(
            # index_member_all_ts.update_time is a batch-ingestion timestamp in
            # this frozen snapshot, not a historical vendor publication time.
            # Applying it as a cutoff would erase all pre-ingestion history.
            membership, signal_date=event_time, decision_time=None,
            universe_codes=visible["ts_code"].astype(str).tolist(),
        )
        resolved_industry.append(asdict(audit))
        visible["industry"] = visible["ts_code"].map(mapping)
        visible_groups.append(visible)
    merged = pd.concat(visible_groups, ignore_index=True) if visible_groups else merged.iloc[0:0].copy()
    # Securities without an as-of PIT industry cannot receive sector-relative features.
    merged = merged.dropna(subset=["industry"]).copy()
    for col in technical:
        name = f"{col}_sector_rel"
        merged[name] = merged[col] - merged.groupby(["event_time", "industry"], sort=False)[col].transform(sector_relative_statistic)
        sector_cols.append(name)
    if not include_industry:
        merged = merged.drop(columns=["industry"])
    cols = ["ts_code", "event_time", "available_time", "economic_open", "economic_close",
            *technical, *fundamental, *sector_cols]
    if include_industry:
        cols.insert(2, "industry")
    result = merged[cols].sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)
    if return_audit:
        return result, {"industry_resolution": resolved_industry,
                        "membership_rows_loaded": int(len(membership)),
                        "sector_relative_statistic": sector_relative_statistic,
                        "source_id": "quant_db"}
    return result
