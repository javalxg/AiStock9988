"""Production execution-price loader with explicit raw/economic separation."""
from __future__ import annotations

import pandas as pd

from ..execution.prices import validate_execution_panel
from .quantdb import readonly_connection
from ..time.session import parse_source_time, session_close, session_open


def normalize_execution_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "adj_factor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"execution source missing columns: {sorted(missing)}")
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], utc=True).dt.normalize()
    for col in ("open", "high", "low", "close", "adj_factor"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    if (out[["open", "high", "low", "close", "adj_factor"]] <= 0).any().any():
        raise ValueError("execution source prices and adj_factor must be positive")
    for raw, economic in (("open", "economic_open"), ("high", "economic_high"),
                          ("low", "economic_low"), ("close", "economic_close")):
        out[economic] = out[raw] * out["adj_factor"]
    if "up_limit" in out and "down_limit" in out:
        out["up_limit"] = pd.to_numeric(out["up_limit"], errors="coerce")
        out["down_limit"] = pd.to_numeric(out["down_limit"], errors="coerce")
        if out[["up_limit", "down_limit"]].isna().any().any():
            raise ValueError("execution source has missing daily limit prices")
        out["is_limit_up"] = out["open"] >= out["up_limit"]
        out["is_limit_down"] = out["open"] <= out["down_limit"]
    if "available_time" not in out:
        if "update_time" not in out:
            raise ValueError("execution source requires available_time or update_time")
        out["available_time"] = parse_source_time(out["update_time"])
    else:
        out["available_time"] = pd.to_datetime(out["available_time"], errors="raise", utc=True)
    if "open_available_time" not in out:
        out["open_available_time"] = out["available_time"]
    if "close_available_time" not in out:
        out["close_available_time"] = out["available_time"]
    if "is_suspended" not in out:
        if "amount" not in out:
            raise ValueError("execution source requires explicit is_suspended or amount")
        amount = pd.to_numeric(out["amount"], errors="coerce")
        if amount.isna().any():
            raise ValueError("execution source amount is required to derive suspension state")
        out["is_suspended"] = amount <= 0
    out = out.rename(columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"})
    return validate_execution_panel(out)


def load_execution_panel(start: str, end: str, *, ts_codes: list[str] | None = None) -> pd.DataFrame:
    """Read raw daily prices, PIT adj factors and daily limit prices read-only."""
    code_filter = ""
    params: list[object] = [start, end]
    if ts_codes:
        code_filter = " AND m.ts_code IN (" + ",".join(["%s"] * len(ts_codes)) + ")"
        params.extend(ts_codes)
    with readonly_connection() as conn:
        frame = pd.read_sql_query(
            "SELECT m.ts_code, m.trade_date, m.open, m.high, m.low, m.close, m.pct_chg, m.amount, "
            "m.update_time AS market_update_time, a.adj_factor, a.update_time AS adj_update_time, "
            "l.up_limit, l.down_limit, l.update_time AS limit_update_time, "
            "(SELECT MAX(s.update_time) FROM suspend_d_ts s "
            "WHERE s.ts_code=m.ts_code AND s.suspend_date <= m.trade_date "
            "AND (s.resume_date IS NULL OR s.resume_date > m.trade_date)) AS suspension_update_time, "
            "CASE WHEN EXISTS (SELECT 1 FROM suspend_d_ts s "
            "WHERE s.ts_code=m.ts_code AND s.suspend_date <= m.trade_date "
            "AND (s.resume_date IS NULL OR s.resume_date > m.trade_date)) "
            "THEN 1 ELSE 0 END AS is_suspended "
            "FROM market_daily_ts m "
            "JOIN adj_factor_ts a ON a.ts_code=m.ts_code AND a.trade_date=m.trade_date "
            "JOIN stk_limit_ts l ON l.ts_code=m.ts_code AND l.trade_date=m.trade_date "
            "WHERE m.source = 'daily' AND m.trade_date >= %s AND m.trade_date <= %s " + code_filter +
            " ORDER BY m.trade_date, m.ts_code",
            conn, params=tuple(params),
        )
    update_cols = [c for c in ("market_update_time", "adj_update_time", "limit_update_time",
                               "suspension_update_time") if c in frame]
    if frame.empty:
        raise ValueError("no execution rows for requested range")
    availability = pd.concat([parse_source_time(frame[c]) for c in update_cols], axis=1)
    frame["source_ingested_time"] = availability.max(axis=1)
    if frame["source_ingested_time"].isna().any():
        raise ValueError("execution source has null available_time")
    trade_days = pd.to_datetime(frame["trade_date"], errors="raise", utc=True).dt.normalize()
    # Raw open/limit/tradability fields are observable at the exchange open;
    # raw/economic close fields become observable at the exchange close.  Keep
    # ingestion time separately so the snapshot remains auditable.
    source_time = frame["source_ingested_time"]
    frame["open_available_time"] = trade_days.map(session_open)
    frame["close_available_time"] = trade_days.map(session_close)
    frame["available_time"] = frame["close_available_time"]
    return normalize_execution_panel(frame)


def load_market_context_panel(start: str, end: str) -> pd.DataFrame:
    """Load PIT-auditable daily history needed by SelectionPolicy."""
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("market context end must not precede start")
    if not start or not end:
        return pd.DataFrame(columns=["ts_code", "trade_date", "raw_close", "pct_chg", "amount"])
    with readonly_connection() as conn:
        frame = pd.read_sql_query(
            "SELECT m.ts_code, m.trade_date, m.close AS raw_close, m.pct_chg, m.amount, "
            "m.update_time AS market_update_time, l.up_limit, l.down_limit, "
            "l.update_time AS limit_update_time "
            "FROM market_daily_ts m JOIN stk_limit_ts l "
            "ON l.ts_code=m.ts_code AND l.trade_date=m.trade_date "
            "WHERE m.source='daily' AND m.trade_date >= %s AND m.trade_date <= %s "
            "ORDER BY m.trade_date, m.ts_code", conn, params=(start, end),
        )
    if frame.empty:
        raise ValueError("no market context rows for requested range")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame["raw_close"] = pd.to_numeric(frame["raw_close"], errors="raise")
    frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="raise")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    # The DB is a frozen historical snapshot; source update_time is batch
    # ingestion metadata, not historical publication time.
    frame["available_time"] = frame["trade_date"].map(session_close)
    frame["is_limit_up"] = frame["raw_close"] >= pd.to_numeric(frame["up_limit"], errors="raise")
    frame["is_limit_down"] = frame["raw_close"] <= pd.to_numeric(frame["down_limit"], errors="raise")
    return frame
