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


def test_announcement_date_is_used_instead_of_late_snapshot_time():
    out = normalize_corporate_actions(pd.DataFrame([{
        "ts_code": "000001.SZ", "ex_date": "2026-08-21", "div_cash": 1.0,
        "imp_ann_date": "2026-08-19", "update_time": "2026-08-26 15:00:00",
        "div_proc": "实施",
    }]))
    assert out.loc[0, "available_time"] == pd.Timestamp("2026-08-19T07:00:00Z")


def test_late_snapshot_without_announcement_date_is_rejected():
    with pytest.raises(ValueError, match="not PIT-visible"):
        normalize_corporate_actions(pd.DataFrame([{
            "ts_code": "000001.SZ", "ex_date": "2026-08-21", "div_cash": 1.0,
            "update_time": "2026-08-26 15:00:00", "div_proc": "实施",
        }]))


def test_corporate_action_revisions_are_applied_once():
    out = normalize_corporate_actions(pd.DataFrame([
            {"ts_code": "000001.SZ", "ex_date": "2026-08-21", "div_cash": 3.0,
         "update_time": "2026-08-19T01:00:00Z", "div_proc": "实施"},
        {"ts_code": "000001.SZ", "ex_date": "2026-08-21", "div_cash": 3.0,
         "update_time": "2026-08-20T01:00:00Z", "div_proc": "实施"},
    ]))
    assert len(out) == 1
    assert out.iloc[0].cash_dividend == pytest.approx(0.3)


def test_conflicting_corporate_action_revisions_are_rejected():
    with pytest.raises(ValueError, match="conflicting economic terms"):
        normalize_corporate_actions(pd.DataFrame([
            {"ts_code": "A.SZ", "ex_date": "2026-08-21", "div_cash": 3.0,
             "stk_div": 0.0, "stk_bo_rate": 0.0, "stk_co_rate": 0.0,
             "div_proc": "实施", "ann_date": "2026-08-01", "update_time": "2026-08-02"},
            {"ts_code": "A.SZ", "ex_date": "2026-08-21", "div_cash": 4.0,
             "stk_div": 0.0, "stk_bo_rate": 0.0, "stk_co_rate": 0.0,
             "div_proc": "实施", "ann_date": "2026-08-01", "update_time": "2026-08-03"},
        ]))


def test_non_numeric_corporate_action_value_is_rejected():
    with pytest.raises(ValueError, match="non-numeric"):
        normalize_corporate_actions(pd.DataFrame([{
            "ts_code": "A.SZ", "ex_date": "2026-08-21", "div_cash": "bad",
            "div_proc": "实施", "update_time": "2026-08-01",
        }]))
