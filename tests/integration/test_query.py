import sqlite3

import pandas as pd
import pytest

from aistock9988.data.query import load_daily_panel


def test_bounded_panel_query_is_parameterized_and_hashed():
    conn = sqlite3.connect(":memory:")
    pd.DataFrame([{"trade_date": "2026-08-20", "ts_code": "A", "f": 1.0},
                  {"trade_date": "2026-08-21", "ts_code": "B", "f": 2.0},
                  {"trade_date": "2026-08-22", "ts_code": "C", "f": 3.0}]).to_sql("factor_daily", conn, index=False)
    out, meta = load_daily_panel(conn, table="factor_daily", columns=["trade_date", "ts_code", "f"],
                                 start="2026-08-20", end="2026-08-21", decision_time="2026-08-21T16:00:00Z")
    assert out.ts_code.tolist() == ["A", "B"]
    assert len(meta["query_hash"]) == 64 and meta["row_count"] == 2


def test_unsafe_identifier_is_rejected():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match="unsafe"):
        load_daily_panel(conn, table="factor_daily;DROP", columns=["f"], start="2026-01-01", end="2026-01-02")
