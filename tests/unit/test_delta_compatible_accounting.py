import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest


def _prices():
    days = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-22"], utc=True)
    return pd.DataFrame({
        "trade_date": days, "ts_code": ["A"] * 3,
        "raw_open": [10., 10., 10.], "raw_high": [10., 10., 10.], "raw_low": [10., 10., 10.],
        "raw_close": [10., 10., 12.], "economic_open": [100., 100., 100.],
        "economic_high": [100., 100., 100.], "economic_low": [100., 100., 100.],
        "economic_close": [100., 100., 120.], "adj_factor": [10., 10., 10.],
        "open_available_time": days + pd.Timedelta(hours=1, minutes=30),
        "close_available_time": days + pd.Timedelta(hours=7),
        "available_time": days + pd.Timedelta(hours=7),
        "is_suspended": [False] * 3, "is_limit_up": [False] * 3, "is_limit_down": [False] * 3,
    })


def test_economic_accounting_basis_changes_cash_and_nav():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    result = run_backtest(signals, _prices(), config=BacktestConfig(
        initial_cash=1000, hold_sessions=5, accounting_price_basis="economic"))
    assert result["trades"].iloc[0].price == 100.0
    assert result["trades"].iloc[-1].price == 120.0
    assert result["nav"].iloc[-1].nav > 1000


def test_economic_accounting_does_not_apply_company_actions_twice():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    actions = pd.DataFrame([{
        "ts_code": "A", "ex_date": "2026-08-21", "split_ratio": 2.0,
        "cash_dividend": 1.0, "available_time": "2026-08-20T08:00:00Z",
    }])
    result = run_backtest(signals, _prices(), corporate_actions=actions, config=BacktestConfig(
        initial_cash=1000, hold_sessions=1, accounting_price_basis="economic"))
    assert result["corporate_actions"].empty
    assert result["trades"].iloc[-1].shares == result["trades"].iloc[0].shares


def test_economic_accounting_rejects_explicit_action_application():
    try:
        run_backtest(pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                                   "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]}),
                     _prices(), config=BacktestConfig(accounting_price_basis="economic",
                                                       corporate_actions_mode="apply"))
    except ValueError as exc:
        assert "twice" in str(exc)
    else:
        raise AssertionError("economic accounting must reject explicit corporate-action application")


def test_gap_return_uses_matching_economic_reference_price():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _prices().copy()
    prices.loc[prices.index[-1], "economic_open"] = 120.0
    result = run_backtest(signals, prices, config=BacktestConfig(
        initial_cash=1000, hold_sessions=1, accounting_price_basis="economic"))
    sell = result["trades"].iloc[-1]
    assert sell.gap_return_raw == 0.0
    assert abs(sell.gap_return_economic - 0.2) < 1e-12
    assert sell.gap_return == sell.gap_return_economic
