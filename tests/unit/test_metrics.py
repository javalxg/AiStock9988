import pandas as pd

from aistock9988.reporting.metrics import summarize_backtest


def test_summary_reports_period_risk_cost_and_extreme_dependence_metrics():
    nav = pd.DataFrame({
        "trade_date": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
        "nav": [100.0, 110.0, 108.0, 90.0, 95.0, 97.0, 96.0, 120.0, 118.0, 115.0],
    })
    trades = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
        "order_id": ["b", "s"], "ts_code": ["A", "A"], "side": ["BUY", "SELL"],
        "price": [10.0, 12.0], "shares": [10.0, 10.0], "gross_value": [100.0, 120.0],
        "commission": [1.0, 1.0], "stamp_duty": [0.0, 0.1],
        "gap_flag": [False, True], "gap_return": [None, -0.2],
        "economic_return": [None, 0.2],
    })
    result = summarize_backtest(nav.assign(cash=nav.nav, market_value=0.0), trades, initial_cash=100.0)
    assert result["weekly_mean"] is not None
    assert result["annual_returns"]
    assert result["fees_and_taxes"] == 2.1
    assert result["gap_fill_count"] == 1
    assert result["gap_loss"] == -0.2
    assert result["trade_return_excluding_top3_profit"] == 0.0
