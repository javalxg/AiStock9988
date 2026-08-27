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
