import pandas as pd
import pytest

from aistock9988.selection.q70_policy import GATES, build_q70_selection_ledger


def _candidates(asof="2026-01-20"):
    rows = []
    for rank, code in enumerate(("A.SZ", "B.SZ"), start=1):
        row = {"asof": asof, "ts_code": code, "candidate_rank": rank, "score": 3 - rank}
        row.update({gate: 1.0 for gate in GATES})
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_history():
    dates = pd.date_range("2026-01-01", "2026-01-20", tz="UTC")
    rows = []
    for code in ("A.SZ", "B.SZ"):
        for day in dates:
            rows.append({"ts_code": code, "trade_date": day, "pct_chg": 0.2,
                         "raw_close": 10.0, "amount": 1000.0,
                         "available_time": day + pd.Timedelta(hours=6, minutes=59)})
    # This apparent limit-down observation was ingested after the decision
    # time and must not affect the selection.
    rows.append({"ts_code": "A.SZ", "trade_date": dates[-1], "pct_chg": -20.0,
                 "raw_close": 8.0, "amount": 1000.0,
                 "available_time": dates[-1] + pd.Timedelta(hours=8)})
    return pd.DataFrame(rows)


def test_q70_policy_uses_only_pit_visible_daily_history_and_freezes_weights():
    first = build_q70_selection_ledger(_candidates(), _daily_history(), asof="2026-01-20",
                                       alpha_weight=True, alpha_power=1.0)
    assert first["selected"].tolist() == [True, True]
    assert first["target_weight"].sum() == pytest.approx(1.0)
    assert first.iloc[0].target_weight > first.iloc[1].target_weight
    assert "recent_limit_down_gate" not in first.iloc[0].rejection_reason
    assert first["selection_decision_id"].nunique() == 1

    changed = build_q70_selection_ledger(_candidates(), _daily_history(), asof="2026-01-20",
                                         alpha_weight=True, alpha_power=2.0)
    assert changed.iloc[0].selection_decision_id != first.iloc[0].selection_decision_id


def test_q70_policy_rejects_mismatched_candidate_date():
    with pytest.raises(ValueError, match="does not match"):
        build_q70_selection_ledger(_candidates("2026-01-19"), _daily_history(), asof="2026-01-20")
