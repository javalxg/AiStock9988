"""PIT company-action source for accounting backtests.

``adj_factor`` is intentionally not treated as an accounting ledger.  This
module converts the dividend feed into explicit ex-date actions so the engine
can adjust shares, cost basis, and cash separately from economic prices.
"""
from __future__ import annotations

import pandas as pd

from ..execution.corporate_actions import CorporateAction
from .quantdb import readonly_connection
from ..time.session import parse_source_time


def normalize_corporate_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Tushare dividend columns to the engine contract.

    Tushare's per-10-share fields are converted to per-share values.  Rights
    issues are not silently modeled as free shares; they must be added later
    with subscription-price cash-flow semantics.
    """
    required = {"ts_code", "ex_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"corporate action source missing columns: {sorted(missing)}")
    out = frame.copy()
    out["ex_date"] = pd.to_datetime(out["ex_date"], errors="coerce", utc=True).dt.normalize()
    if out["ex_date"].isna().any():
        raise ValueError("corporate action source contains invalid ex_date")

    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(out[name], errors="coerce").fillna(0.0) if name in out else pd.Series(0.0, index=out.index)

    # Tushare dividend fields are normally quoted per 10 shares.
    cash = numeric("div_cash") / 10.0
    bonus = (numeric("stk_div") + numeric("stk_bo_rate") + numeric("stk_co_rate")) / 10.0
    out["cash_dividend"] = cash
    out["split_ratio"] = 1.0 + bonus
    if (out["cash_dividend"] < 0).any() or (out["split_ratio"] <= 0).any():
        raise ValueError("corporate action values are invalid")
    if "div_proc" not in out:
        raise ValueError("corporate action source must include div_proc implementation status")
    status = out["div_proc"].astype(str)
    out = out[status.str.contains("实施", na=False)].copy()
    if "update_time" not in out:
        raise ValueError("corporate action source must include update_time for PIT")
    out["action_type"] = out["div_proc"]
    out["available_time"] = parse_source_time(out["update_time"])
    if out["available_time"].isna().any():
        raise ValueError("corporate action source has null update_time")
    # A provider may retain several revisions of the same implemented event.
    # Keep the latest PIT version once; applying every revision would multiply
    # dividends and split ratios in the accounting ledger.
    out = out.sort_values(["ts_code", "ex_date", "available_time"], kind="mergesort")
    out = out.drop_duplicates(["ts_code", "ex_date"], keep="last")
    cols = ["ts_code", "ex_date", "split_ratio", "cash_dividend", "available_time", "action_type"]
    return out[cols].sort_values(["ex_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def load_corporate_actions(start: str, end: str, *, ts_codes: list[str] | None = None) -> pd.DataFrame:
    """Load completed dividend/bonus events from the read-only database.

    The loader discovers the installed dividend table and its available
    columns, because deployments use both ``dividend_ts`` and
    ``stk_dividend_ts`` names.
    """
    candidates = ("dividend_ts", "stk_dividend_ts", "dividend")
    with readonly_connection() as conn:
        table_frame = pd.read_sql_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name IN (%s,%s,%s)",
            conn, params=candidates,
        )
        table_column = next((column for column in table_frame.columns if column.lower() == "table_name"), None)
        if table_column is None:
            raise RuntimeError("information_schema response has no table_name column")
        tables = table_frame[table_column].tolist()
        if not tables:
            raise RuntimeError("no dividend table found; company actions are required for accounting backtests")
        table = str(tables[0])
        column_frame = pd.read_sql_query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=DATABASE() AND table_name=%s", conn, params=(table,)
        )
        column_name = next((column for column in column_frame.columns if column.lower() == "column_name"), None)
        if column_name is None:
            raise RuntimeError("information_schema response has no column_name column")
        columns = column_frame[column_name].tolist()
        wanted = ["ts_code", "ex_date", "div_cash", "stk_div", "stk_bo_rate", "stk_co_rate",
                  "div_proc", "update_time"]
        selected = [c for c in wanted if c in columns]
        if not {"ts_code", "ex_date"}.issubset(selected):
            raise RuntimeError(f"{table} lacks ts_code/ex_date required for PIT company actions")
        params: list[object] = [start, end]
        code_filter = ""
        if ts_codes:
            code_filter = " AND ts_code IN (" + ",".join(["%s"] * len(ts_codes)) + ")"
            params.extend(ts_codes)
        query = ("SELECT " + ", ".join(selected) + " FROM `" + table + "` "
                 "WHERE ex_date >= %s AND ex_date <= %s" + code_filter)
        frame = pd.read_sql_query(query, conn, params=tuple(params))
    return normalize_corporate_actions(frame)
