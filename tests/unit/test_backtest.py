import pandas as pd
import pytest

from aistock9988.backtest.engine import BacktestConfig, run_backtest
from aistock9988.reporting.metrics import summarize_backtest


def _complete_prices(frame):
    days = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame["open_available_time"] = days + pd.Timedelta(hours=1, minutes=30)
    frame["close_available_time"] = days + pd.Timedelta(hours=7)
    frame["available_time"] = frame["close_available_time"]
    for column in ("is_suspended", "is_limit_up", "is_limit_down"):
        if column not in frame:
            frame[column] = False
    return frame


def test_backtest_uses_next_open_and_final_liquidation():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1], "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"],
                           "ts_code": ["A"] * 3, "raw_open": [10.0, 11.0, 13.0],
                           "raw_high": [10.0, 12.0, 14.0], "raw_low": [10.0, 11.0, 13.0],
                           "raw_close": [10.0, 12.0, 14.0], "economic_open": [10.0, 11.0, 13.0],
                           "economic_high": [10.0, 12.0, 14.0], "economic_low": [10.0, 11.0, 13.0],
                           "economic_close": [10.0, 12.0, 14.0], "adj_factor": [1.0, 1.0, 1.0]}))
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, hold_sessions=5))
    trades = result["trades"]
    assert trades.iloc[0].side == "BUY" and trades.iloc[0].price == 11.0
    assert trades.iloc[-1].reason == "end_of_test_liquidation"
    assert result["nav"].iloc[-1].nav > 1000
    metrics = summarize_backtest(result["nav"], trades, initial_cash=1000)
    assert metrics["total_return"] > 0
    assert "equal_trade_return_ratio" in metrics and "economic_trade_return_ratio" in metrics


def test_missing_execution_price_is_retained_as_pending_order():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["MISSING"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["rcqt"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21"], "ts_code": ["OTHER", "OTHER"],
                           "raw_open": [10.0, 10.0], "raw_high": [10.0, 10.0], "raw_low": [10.0, 10.0], "raw_close": [10.0, 10.0],
                           "economic_open": [10.0, 10.0], "economic_high": [10.0, 10.0], "economic_low": [10.0, 10.0],
                           "economic_close": [10.0, 10.0], "adj_factor": [1.0, 1.0]}))
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000))
    order = result["orders"].iloc[0]
    assert order.ts_code == "MISSING"
    assert order.status == "PENDING"
    assert order.final_reason == "pending_missing_price"


def test_stop_loss_uses_economic_price_but_executes_raw_price():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1], "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23"],
                           "ts_code": ["A"] * 4, "raw_open": [10.0, 10.0, 8.0, 8.0],
                           "raw_high": [10.0, 10.0, 8.0, 8.0], "raw_low": [10.0, 10.0, 8.0, 8.0],
                           "raw_close": [10.0, 10.0, 10.0, 8.0], "economic_open": [100.0, 100.0, 90.0, 90.0],
                           "economic_high": [100.0, 100.0, 90.0, 90.0], "economic_low": [100.0, 100.0, 90.0, 90.0],
                           "economic_close": [100.0, 100.0, 90.0, 90.0], "adj_factor": [10.0, 10.0, 10.0, 10.0]}))
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, stop_loss_pct=-0.08))
    assert result["trades"].iloc[-1].side == "SELL"
    assert result["trades"].iloc[-1].price == 8.0
    stop_order = result["orders"].query("trigger_type == 'STOP_LOSS'").iloc[0]
    assert stop_order.status == "FILLED"
    assert stop_order.trigger_price == 90.0
    assert stop_order.execution_price == 8.0
    assert stop_order.gap_return == pytest.approx(-0.2)
    assert result["trades"].iloc[-1].trigger_type == "STOP_LOSS"
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1000)
    assert metrics["stop_loss_count"] == 1
    assert metrics["economic_trade_return_ratio"] == pytest.approx(-0.1)
    assert metrics["stop_loss_gap_loss"] == pytest.approx(-0.2)


def test_stop_loss_is_pending_at_limit_down_and_expired_with_reason():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1], "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23"],
                           "ts_code": ["A"] * 4, "raw_open": [10.0, 10.0, 8.0, 8.0],
                           "raw_high": [10.0, 10.0, 8.0, 8.0], "raw_low": [10.0, 10.0, 8.0, 8.0],
                           "raw_close": [10.0, 10.0, 8.0, 8.0], "economic_open": [100.0, 100.0, 90.0, 90.0],
                           "economic_high": [100.0, 100.0, 90.0, 90.0], "economic_low": [100.0, 100.0, 90.0, 90.0],
                           "economic_close": [100.0, 100.0, 90.0, 90.0], "adj_factor": [10.0, 10.0, 10.0, 10.0],
                           "is_limit_down": [False, False, True, True]}))
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, stop_loss_pct=-0.08))
    stop_order = result["orders"].query("trigger_type == 'STOP_LOSS'").iloc[0]
    assert stop_order.status == "EXPIRED"
    assert stop_order.final_reason == "unclosed_non_tradable"
    assert stop_order.last_attempt_reason == "limit_down"


def test_backtest_uses_intraday_stop_before_daily_close():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1], "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22", "2026-08-23"],
                           "ts_code": ["A"] * 4, "raw_open": [10., 10., 10., 10.],
                           "raw_high": [10., 10., 10., 10.], "raw_low": [10., 10., 10., 10.],
                           "raw_close": [10., 10., 10., 10.], "economic_open": [100., 100., 100., 100.],
                           "economic_high": [100., 100., 100., 100.], "economic_low": [100., 100., 100., 100.],
                           "economic_close": [100., 100., 100., 100.], "adj_factor": [10., 10., 10., 10.]}))
    minute = pd.DataFrame({"ts_code": ["A"], "trade_time": ["2026-08-22 01:31:00Z"],
                           "open": [8.], "high": [10.], "low": [8.], "close": [9.],
                           "adj_factor": [10.], "up_limit": [12.], "down_limit": [7.],
                           "available_time": ["2026-08-22T01:31:00Z"]})
    result = run_backtest(signals, prices, minute_prices=minute,
                          config=BacktestConfig(initial_cash=1000, stop_loss_pct=-0.08, stop_loss_mode="intraday_5min"))
    assert result["trades"].iloc[-1].reason == "intraday_stop_loss"
    assert result["trades"].iloc[-1].trade_date == pd.Timestamp("2026-08-22", tz="UTC")


def test_backtest_rejects_minute_bar_that_was_not_yet_available():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({
        "trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"], "ts_code": ["A"] * 3,
        "raw_open": [10.] * 3, "raw_high": [10.] * 3, "raw_low": [10.] * 3, "raw_close": [10.] * 3,
        "economic_open": [100.] * 3, "economic_high": [100.] * 3,
        "economic_low": [100.] * 3, "economic_close": [100.] * 3, "adj_factor": [10.] * 3,
    }))
    minute = pd.DataFrame({
        "ts_code": ["A"], "trade_time": ["2026-08-22T01:31:00Z"],
        "open": [8.], "high": [10.], "low": [8.], "close": [9.], "adj_factor": [10.],
        "up_limit": [12.], "down_limit": [7.], "available_time": ["2026-08-22T06:00:00Z"],
    })
    result = run_backtest(signals, prices, minute_prices=minute,
                          config=BacktestConfig(initial_cash=1000, stop_loss_pct=-0.08,
                                                stop_loss_mode="intraday_5min"))
    assert "intraday_stop_loss" not in result["trades"].get("reason", pd.Series(dtype=str)).tolist()


def test_nav_does_not_use_close_that_was_not_visible_by_session_close():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({
        "trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"], "ts_code": ["A"] * 3,
        "raw_open": [10., 10., 10.], "raw_high": [10., 10., 100.], "raw_low": [10.] * 3,
        "raw_close": [10., 10., 100.], "economic_open": [10.] * 3,
        "economic_high": [10., 10., 100.], "economic_low": [10.] * 3,
        "economic_close": [10., 10., 100.], "adj_factor": [1.] * 3,
    }))
    prices.loc[2, "close_available_time"] = "2026-08-22T08:00:00Z"
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, hold_sessions=10))
    assert result["positions"].iloc[0].raw_mark_price == 10.0
    assert result["nav"].iloc[-1].market_value == 990.0


def test_terminal_liquidation_respects_a_share_t1():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1],
                            "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({
        "trade_date": ["2026-08-20", "2026-08-21"], "ts_code": ["A"] * 2,
        "raw_open": [10., 10.], "raw_high": [10., 10.], "raw_low": [10., 10.], "raw_close": [10., 10.],
        "economic_open": [10., 10.], "economic_high": [10., 10.], "economic_low": [10., 10.],
        "economic_close": [10., 10.], "adj_factor": [1., 1.],
    }))
    result = run_backtest(signals, prices, config=BacktestConfig(initial_cash=1000, hold_sessions=10))
    assert result["trades"]["side"].tolist() == ["BUY"]
    assert result["positions"].iloc[0].shares == 99


def test_corporate_action_changes_shares_and_dividend_cash():
    signals = pd.DataFrame({"asof": ["2026-08-20"], "ts_code": ["A"], "candidate_rank": [1], "selected": [True], "selection_decision_id": ["d1"], "policy_id": ["p1"]})
    prices = _complete_prices(pd.DataFrame({"trade_date": ["2026-08-20", "2026-08-21", "2026-08-22"],
                           "ts_code": ["A"] * 3, "raw_open": [10.0, 10.0, 5.0],
                           "raw_high": [10.0, 10.0, 5.0], "raw_low": [10.0, 10.0, 5.0],
                           "raw_close": [10.0, 5.0, 5.0], "economic_open": [10.0, 10.0, 10.0],
                           "economic_high": [10.0, 10.0, 10.0], "economic_low": [10.0, 10.0, 10.0],
                           "economic_close": [10.0, 10.0, 10.0], "adj_factor": [1.0, 2.0, 2.0]}))
    actions = pd.DataFrame({"ts_code": ["A"], "ex_date": ["2026-08-22"],
                            "split_ratio": [2.0], "cash_dividend": [0.5],
                            "available_time": ["2026-08-22T01:00:00Z"]})
    result = run_backtest(signals, prices, corporate_actions=actions,
                          config=BacktestConfig(initial_cash=1000, hold_sessions=10))
    assert result["corporate_actions"].iloc[0].cash_dividend == 49.5
    assert result["positions"].empty
