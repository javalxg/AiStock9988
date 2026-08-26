from __future__ import annotations

import pandas as pd


def summarize_backtest(nav: pd.DataFrame, trades: pd.DataFrame, *, initial_cash: float) -> dict[str, float | int | None]:
    if nav.empty:
        return {"initial_cash": initial_cash, "final_nav": initial_cash, "total_return": 0.0,
                "max_drawdown": 0.0, "trade_count": 0, "trade_win_rate": None,
                "portfolio_profit_factor": None}
    values = pd.to_numeric(nav["nav"], errors="raise")
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    wins = None
    pf = None
    if not sells.empty:
        # Pairing is FIFO by security and sufficient for the one-position-per-security engine.
        buys = trades[trades["side"] == "BUY"].groupby("ts_code", sort=False)["gross_value"].sum()
        proceeds = sells.groupby("ts_code", sort=False)["gross_value"].sum()
        pnl = proceeds.subtract(buys, fill_value=0.0)
        wins = float((pnl > 0).mean()) if len(pnl) else None
        gains = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        pf = float(gains / losses) if losses else None
    return {"initial_cash": float(initial_cash), "final_nav": float(values.iloc[-1]),
            "total_return": float(values.iloc[-1] / initial_cash - 1.0),
            "max_drawdown": float(drawdown.min()), "trade_count": int(len(sells)),
            "trade_win_rate": wins, "portfolio_profit_factor": pf}
