import pandas as pd
import pytest

from aistock9988.data.corporate_actions_source import normalize_corporate_actions


def test_tushare_per_ten_fields_become_explicit_per_share_action():
    out = normalize_corporate_actions(pd.DataFrame([{
        "ts_code": "000001.SZ", "ex_date": "2026-08-21", "div_cash": 3.0,
        "stk_div": 2.0, "stk_bo_rate": 1.0, "stk_co_rate": 0.0,
        "update_time": "2026-08-20T10:00:00Z", "div_proc": "实施",
    }]))
    assert out.loc[0, "cash_dividend"] == pytest.approx(0.3)
    assert out.loc[0, "split_ratio"] == pytest.approx(1.3)
    assert out.loc[0, "action_type"] == "实施"
