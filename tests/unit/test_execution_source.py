import pandas as pd
import pytest

from aistock9988.data.execution_source import normalize_execution_panel


def test_execution_source_derives_economic_prices_and_limit_states():
    panel = normalize_execution_panel(pd.DataFrame({
        "ts_code": ["A", "A"], "trade_date": ["2026-01-01", "2026-01-02"],
        "open": [10, 9], "high": [11, 10], "low": [9, 8], "close": [10, 9],
        "adj_factor": [2, 2], "up_limit": [10, 10], "down_limit": [9, 9],
    }))
    assert panel.iloc[0].economic_close == 20
    assert bool(panel.iloc[0].is_limit_up) is True
    assert bool(panel.iloc[1].is_limit_down) is True


def test_execution_source_rejects_missing_adjustment_factor():
    with pytest.raises(ValueError, match="adj_factor"):
        normalize_execution_panel(pd.DataFrame({"ts_code": ["A"], "trade_date": ["2026-01-01"],
                                                  "open": [10], "high": [11], "low": [9], "close": [10]}))


def test_execution_source_rejects_missing_limit_prices():
    with pytest.raises(ValueError, match="limit prices"):
        normalize_execution_panel(pd.DataFrame({
            "ts_code": ["A"], "trade_date": ["2026-01-01"],
            "open": [10], "high": [11], "low": [9], "close": [10],
            "adj_factor": [1], "up_limit": [None], "down_limit": [9],
        }))
