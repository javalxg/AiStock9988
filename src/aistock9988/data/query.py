from __future__ import annotations

import hashlib
import json
import re

import pandas as pd


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def query_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def load_daily_panel(connection, *, table: str, columns: list[str], start: str, end: str,
                     decision_time: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a bounded panel with parameterized dates; no writes and no SELECT *."""
    table = _identifier(table)
    cols = [_identifier(c) for c in columns]
    if not cols:
        raise ValueError("at least one column is required")
    if decision_time is not None and "available_time" not in cols:
        raise ValueError("available_time must be selected when decision_time is provided")
    selected = ", ".join(cols)
    sql = f"SELECT {selected} FROM {table} WHERE trade_date >= ? AND trade_date <= ?"
    params: tuple[object, ...] = (start, end)
    if decision_time is not None:
        sql += " AND available_time <= ?"
        params += (decision_time,)
    sql += " ORDER BY trade_date, ts_code"
    # sqlite uses '?'; production adapters can translate placeholders without changing the spec.
    frame = pd.read_sql_query(sql, connection, params=params)
    spec = {"table": table, "columns": cols, "start": start, "end": end, "decision_time": decision_time}
    return frame, {"query": spec, "query_hash": query_hash(spec), "row_count": len(frame)}
