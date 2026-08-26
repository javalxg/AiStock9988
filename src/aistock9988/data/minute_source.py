"""Canonical minute execution bars for intraday backtests."""
from __future__ import annotations

import re
import hashlib

import numpy as np
import pandas as pd

from .quantdb import readonly_connection
from ..time.session import parse_source_time


_FREQ = re.compile(r"^(5min|15min|30min|60min)$")


def normalize_minute_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_time", "open", "high", "low", "close", "adj_factor",
                "up_limit", "down_limit", "available_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"minute source missing columns: {sorted(missing)}")
    out = frame.copy()
    out["trade_time"] = pd.to_datetime(out["trade_time"], errors="raise", utc=True)
    out["available_time"] = pd.to_datetime(out["available_time"], errors="raise", utc=True)
    if out["available_time"].isna().any():
        raise ValueError("minute source available_time must be non-null")
    out["trade_date"] = out["trade_time"].dt.normalize()
    for col in ("open", "high", "low", "close", "adj_factor", "up_limit", "down_limit"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    if out[["open", "high", "low", "close", "adj_factor", "up_limit", "down_limit"]].isna().any().any():
        raise ValueError("minute source contains null prices, adjustment factors or limits")
    if not np.isfinite(out[["open", "high", "low", "close", "adj_factor", "up_limit", "down_limit"]].to_numpy(dtype=float)).all():
        raise ValueError("minute source contains non-finite prices, adjustment factors or limits")
    if (out[["open", "high", "low", "close", "adj_factor", "up_limit", "down_limit"]] <= 0).any().any():
        raise ValueError("minute source prices, adjustment factors and limits must be positive")
    if out.duplicated(["ts_code", "trade_time"]).any():
        raise ValueError("minute source contains duplicate ts_code/trade_time")
    for raw, economic in (("open", "economic_open"), ("high", "economic_high"),
                          ("low", "economic_low"), ("close", "economic_close")):
        out[economic] = out[raw] * out["adj_factor"]
    out["is_limit_up"] = out["open"] >= out["up_limit"]
    out["is_limit_down"] = out["open"] <= out["down_limit"]
    # A minute is treated as locked when its entire OHLC range is at the limit.
    out["is_locked_limit_up"] = out["low"] >= out["up_limit"]
    out["is_locked_limit_down"] = out["high"] <= out["down_limit"]
    return out.sort_values(["trade_time", "ts_code"], kind="mergesort").reset_index(drop=True)


def load_minute_execution_panel(start: str, end: str, *, freq: str = "5min",
                                ts_codes: list[str] | None = None) -> pd.DataFrame:
    if not _FREQ.fullmatch(freq):
        raise ValueError("freq must be one of 5min, 15min, 30min, 60min")
    with readonly_connection() as conn:
        tables = _minute_tables(conn, freq)
        if not tables:
            raise RuntimeError(f"no minute storage tables discovered for {freq}")
        frames = []
        for table in tables:
            codes = [c for c in (ts_codes or []) if _bucket_table_for_code(c, tables) == table]
            if ts_codes and not codes:
                continue
            code_filter = ""
            # Date arguments are inclusive for the public loader.  Use a
            # half-open SQL interval so ``end='YYYY-MM-DD'`` includes the
            # complete trading session instead of only midnight.
            start_bound, end_bound = _minute_bounds(start, end)
            params: list[object] = [start_bound, end_bound]
            if codes:
                code_filter = " AND m.ts_code IN (" + ",".join(["%s"] * len(codes)) + ")"
                params.extend(codes)
            frame = pd.read_sql_query(
                "SELECT m.ts_code, m.trade_time, m.open, m.high, m.low, m.close, "
                "m.update_time AS minute_update_time, a.adj_factor, "
                "a.update_time AS adj_update_time, l.up_limit, l.down_limit, "
                "l.update_time AS limit_update_time "
                f"FROM `{table}` m "
                "JOIN market_daily_ts d ON d.ts_code=m.ts_code AND d.trade_date=DATE(m.trade_time) "
                "JOIN adj_factor_ts a ON a.ts_code=m.ts_code AND a.trade_date=DATE(m.trade_time) "
                "JOIN stk_limit_ts l ON l.ts_code=m.ts_code AND l.trade_date=DATE(m.trade_time) "
                "WHERE m.trade_time >= %s AND m.trade_time < %s "
                "AND d.open > 0 AND d.high > 0 AND d.low > 0 AND d.close > 0 "
                "AND COALESCE(d.amount, 0) > 0" + code_filter +
                " ORDER BY m.trade_time, m.ts_code", conn, params=tuple(params))
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise ValueError(f"no minute bars for freq={freq} and requested universe")
    availability = pd.concat([
        parse_source_time(frame["minute_update_time"]),
        parse_source_time(frame["adj_update_time"]),
        parse_source_time(frame["limit_update_time"]),
    ], axis=1)
    frame["available_time"] = availability.max(axis=1)
    if frame["available_time"].isna().any():
        raise ValueError("minute source has null available_time")
    return normalize_minute_panel(frame)


def _minute_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_bound = pd.Timestamp(start)
    end_bound = pd.Timestamp(end)
    if start_bound.tzinfo is None:
        start_bound = start_bound.tz_localize("UTC")
    else:
        start_bound = start_bound.tz_convert("UTC")
    if end_bound.tzinfo is None:
        end_bound = end_bound.tz_localize("UTC")
    else:
        end_bound = end_bound.tz_convert("UTC")
    # The project API passes trading dates.  For an explicit timestamp, keep
    # the requested instant as the exclusive boundary.
    if len(str(end)) <= 10:
        end_bound += pd.Timedelta(days=1)
    return start_bound, end_bound


def _minute_tables(conn, freq: str) -> list[str]:
    suffix = {"5min": "5m", "15min": "15m", "30min": "30m", "60min": "60m"}[freq]
    rows = pd.read_sql_query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema=DATABASE() AND (table_name=%s OR table_name LIKE %s) "
        "ORDER BY table_name", conn, params=(f"market_stk_mins_{suffix}_ts", f"market_stk_mins_{suffix}_b%"))
    table_column = next((column for column in rows.columns if column.lower() == "table_name"), None)
    if table_column is None:
        raise RuntimeError("information_schema response has no table_name column")
    names = rows[table_column].astype(str).tolist()
    buckets = sorted([n for n in names if re.fullmatch(rf"market_stk_mins_{suffix}_b\d+", n)],
                     key=lambda n: int(n.rsplit("b", 1)[1]))
    return buckets or ([f"market_stk_mins_{suffix}_ts"] if f"market_stk_mins_{suffix}_ts" in names else [])


def _bucket_table_for_code(ts_code: str, tables: list[str]) -> str:
    buckets = [t for t in tables if re.search(r"_b\d+$", t)]
    if not buckets:
        return tables[0]
    bucket_count = len(buckets)
    bucket = int(hashlib.sha256(str(ts_code).encode("utf-8")).hexdigest(), 16) % bucket_count
    for table in buckets:
        if table.endswith(f"_b{bucket}"):
            return table
    raise RuntimeError(f"minute bucket table b{bucket} is missing")
