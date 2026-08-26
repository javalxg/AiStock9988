import pandas as pd

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.reporting.metrics import summarize_backtest


def test_backtest_uses_next_open_and_final_liquidation():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1]})
    prices = pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"],
                           "ts_code": ["A"] * 3, "raw_open": [10.0, 11.0, 13.0],
                           "raw_high": [10.0, 12.0, 14.0], "raw_low": [10.0, 11.0, 13.0],
                           "raw_close": [10.0, 12.0, 14.0], "economic_open": [10.0, 11.0, 13.0],
                           "economic_high": [10.0, 12.0, 14.0], "economic_low": [10.0, 11.0, 13.0],
                           "economic_close": [10.0, 12.0, 14.0], "adj_factor": [1.0, 1.0, 1.0]})
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, hold_sessions=5))
    trades = result["trades"]
    assert trades.iloc[0].side == "BUY" and trades.iloc[0].price == 11.0
    assert trades.iloc[-1].reason == "end_of_test_liquidation"
    assert result["nav"].iloc[-1].nav > 1000
    metrics = summarize_backtest(result["nav"], trades, initial_cash=1000)
    assert metrics["total_return"] > 0


def test_stop_loss_uses_economic_price_but_executes_raw_price():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1]})
    prices = pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"],
                           "ts_code": ["A"] * 3, "raw_open": [10.0, 10.0, 8.0],
                           "raw_high": [10.0, 10.0, 8.0], "raw_low": [10.0, 10.0, 8.0],
                           "raw_close": [10.0, 10.0, 8.0], "economic_open": [100.0, 100.0, 90.0],
                           "economic_high": [100.0, 100.0, 90.0], "economic_low": [100.0, 100.0, 90.0],
                           "economic_close": [100.0, 100.0, 90.0], "adj_factor": [10.0, 10.0, 10.0]})
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, stop_loss_pct=-0.08))
    assert result["trades"].iloc[-1].side == "SELL"
    assert result["trades"].iloc[-1].price == 8.0
