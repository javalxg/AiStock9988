import pandas as pd
import pytest

from aistock9988.data.minute_source import _bucket_table_for_code, normalize_minute_panel
from aistock9988.execution.intraday import find_stop_execution


def _bars(locked=False):
    return normalize_minute_panel(pd.DataFrame({
        "ts_code": ["A", "A"],
        "trade_time": ["2026-01-02 09:31:00+00:00", "2026-01-02 09:32:00+00:00"],
        "open": [8.0, 7.0], "high": [10.0, 7.0], "low": [8.0, 7.0], "close": [9.0, 7.0],
        "adj_factor": [2.0, 2.0], "up_limit": [12.0, 12.0], "down_limit": [7.0, 7.0],
    }))


def test_minute_panel_builds_economic_prices_and_locked_state():
    bars = _bars()
    assert bars.iloc[0].economic_close == 18
    assert bool(bars.iloc[1].is_locked_limit_down) is True


def test_intraday_stop_uses_raw_open_when_gap_crosses_stop():
    bars = _bars().iloc[[0]].copy()
    result = find_stop_execution(bars, entry_economic_price=20, stop_loss_pct=-0.08,
                                 start_time="2026-01-02 09:30:00Z")
    assert result.status == "FILLED"
    assert result.execution_raw_price == 8.0


def test_intraday_stop_does_not_fill_locked_limit_down():
    bars = _bars().iloc[[1]].copy()
    result = find_stop_execution(bars, entry_economic_price=20, stop_loss_pct=-0.08,
                                 start_time="2026-01-02 09:30:00Z")
    assert result.status == "PENDING" and result.reason == "locked_limit_down"


def test_minute_panel_rejects_missing_limits():
    with pytest.raises(ValueError, match="missing columns"):
        normalize_minute_panel(pd.DataFrame({"ts_code": ["A"], "trade_time": ["2026-01-02"],
                                              "open": [1], "high": [1], "low": [1], "close": [1],
                                              "adj_factor": [1]}))


def test_minute_bucket_router_uses_actual_bucket_count():
    tables = [f"market_stk_mins_5m_b{i}" for i in range(16)]
    table = _bucket_table_for_code("000001.SZ", tables)
    bucket = int(table.rsplit("b", 1)[1])
    import hashlib
    expected = int(hashlib.sha256(b"000001.SZ").hexdigest(), 16) % 16
    assert bucket == expected
