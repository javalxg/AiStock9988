from __future__ import annotations

import pandas as pd


def summarize_backtest(nav: pd.DataFrame, trades: pd.DataFrame, *, initial_cash: float) -> dict[str, float | int | None]:
    if nav.empty:
        return {"initial_cash": initial_cash, "final_nav": initial_cash, "total_return": 0.0,
                "max_drawdown": 0.0, "trade_count": 0, "trade_win_rate": None,
                "portfolio_profit_factor": None, "equal_trade_return_ratio": None,
                "economic_trade_return_ratio": None, "stop_loss_count": 0,
                "stop_loss_gap_loss": 0.0}
    values = pd.to_numeric(nav["nav"], errors="raise")
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    wins = None
    pf = None
    equal_trade_return = None
    economic_trade_return = None
    stop_count = 0
    stop_gap_loss = 0.0
    if not sells.empty:
        # FIFO pairing is deterministic and preserves fee/tax accounting.
        lots: dict[str, list[dict[str, float]]] = {}
        realized: list[float] = []
        returns: list[float] = []
        for row in trades.sort_values(["trade_date", "order_id"], kind="mergesort").to_dict("records"):
            code = str(row["ts_code"])
            shares = float(row["shares"])
            if row["side"] == "BUY":
                lots.setdefault(code, []).append({"shares": shares, "cost": float(row["gross_value"] + row["commission"])})
                continue
            proceeds = float(row["gross_value"] - row["commission"] - row["stamp_duty"])
            remaining = shares
            allocated_cost = 0.0
            while remaining > 0 and lots.get(code):
                lot = lots[code][0]
                take = min(remaining, lot["shares"])
                allocated_cost += lot["cost"] * take / lot["shares"]
                lot["shares"] -= take
                lot["cost"] -= lot["cost"] * take / (lot["shares"] + take)
                remaining -= take
                if lot["shares"] <= 1e-12:
                    lots[code].pop(0)
            realized.append(proceeds - allocated_cost)
            if allocated_cost > 0:
                returns.append(proceeds / allocated_cost - 1.0)
            if row.get("economic_return") is not None:
                economic_trade_return = (economic_trade_return or []) + [float(row["economic_return"])]
            if row.get("trigger_type") == "STOP_LOSS":
                stop_count += 1
                gap = float(row.get("gap_return") or 0.0)
                stop_gap_loss += min(0.0, gap)
        pnl = pd.Series(realized, dtype=float)
        wins = float((pnl > 0).mean()) if len(pnl) else None
        gains = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        pf = float(gains / losses) if losses else None
        equal_trade_return = float(pd.Series(returns).mean()) if returns else None
        if economic_trade_return is not None:
            economic_trade_return = float(pd.Series(economic_trade_return).mean())
    return {"initial_cash": float(initial_cash), "final_nav": float(values.iloc[-1]),
            "total_return": float(values.iloc[-1] / initial_cash - 1.0),
            "max_drawdown": float(drawdown.min()), "trade_count": int(len(sells)),
            "trade_win_rate": wins, "portfolio_profit_factor": pf,
            "equal_trade_return_ratio": equal_trade_return,
            "economic_trade_return_ratio": economic_trade_return,
            "stop_loss_count": stop_count, "stop_loss_gap_loss": float(stop_gap_loss)}
