import sqlite3

import pandas as pd
import pytest

from aistock9988.data.loaders import LoadRequest, SQLLoader


def _db():
    conn = sqlite3.connect(":memory:")
    pd.DataFrame([
        {"ts_code": "000002.SZ", "event_time": "2026-08-21", "available_time": "2026-08-21T15:00:00Z", "pct_chg": -1.0},
        {"ts_code": "000001.SZ", "event_time": "2026-08-21", "available_time": "2026-08-21T15:00:00Z", "pct_chg": 1.0},
        {"ts_code": "000001.SZ", "event_time": "2026-08-22", "available_time": "2026-08-22T15:00:00Z", "pct_chg": 2.0},
    ]).to_sql("market_daily", conn, index=False)
    return conn


def test_sql_loader_is_read_only_pit_filtered_and_stable():
    conn = _db()
    request = LoadRequest(required_columns=("ts_code", "event_time", "available_time", "pct_chg"),
                           decision_time=pd.Timestamp("2026-08-21T16:00:00Z"))
    out = SQLLoader(conn, "market_daily").load(request)
    assert out["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert len(out) == 2
    assert conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0] == 3


def test_sql_loader_rejects_missing_pit_column():
    conn = sqlite3.connect(":memory:")
    pd.DataFrame([{"event_time": "2026-08-21", "ts_code": "000001.SZ"}]).to_sql("bad", conn, index=False)
    with pytest.raises(ValueError, match="available_time"):
        SQLLoader(conn, "bad").load(LoadRequest(required_columns=("event_time", "ts_code")))
