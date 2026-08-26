from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize_backtest(nav: pd.DataFrame, trades: pd.DataFrame, *, initial_cash: float) -> dict[str, object]:
    if nav.empty:
        return _empty_metrics(initial_cash)
    values = pd.to_numeric(nav["nav"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("NAV contains non-finite values")
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    wins = None
    pf = None
    equal_trade_return = None
    economic_trade_return = None
    stop_count = 0
    stop_gap_loss = 0.0
    realized_returns: list[float] = []
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
                trade_return = proceeds / allocated_cost - 1.0
                returns.append(trade_return)
                realized_returns.append(trade_return)
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
    base = {"initial_cash": float(initial_cash), "final_nav": float(values.iloc[-1]),
            "total_return": float(values.iloc[-1] / initial_cash - 1.0),
            "max_drawdown": float(drawdown.min()), "trade_count": int(len(sells)),
            "trade_win_rate": wins, "portfolio_profit_factor": pf,
            "equal_trade_return_ratio": equal_trade_return,
            "economic_trade_return_ratio": economic_trade_return,
            "stop_loss_count": stop_count, "stop_loss_gap_loss": float(stop_gap_loss)}
    base.update(_period_metrics(nav, values, initial_cash=initial_cash))
    base.update(_cost_metrics(trades, initial_cash=initial_cash))
    if realized_returns:
        base["trade_return_excluding_top3_profit"] = _compound_excluding_top_positive(realized_returns, 3)
    else:
        base["trade_return_excluding_top3_profit"] = None
    return base


def _empty_metrics(initial_cash: float) -> dict[str, float | int | None]:
    return {
        "initial_cash": float(initial_cash), "final_nav": float(initial_cash), "total_return": 0.0,
        "max_drawdown": 0.0, "trade_count": 0, "trade_win_rate": None,
        "portfolio_profit_factor": None, "equal_trade_return_ratio": None,
        "economic_trade_return_ratio": None, "stop_loss_count": 0, "stop_loss_gap_loss": 0.0,
        "annualized_return": 0.0, "weekly_mean": None, "weekly_median": None,
        "weekly_std": None, "weekly_positive_ratio": None, "sharpe": None,
        "sortino": None, "calmar": None, "worst_week": None,
        "max_consecutive_losing_weeks": 0, "return_excluding_best_week": None,
        "trade_return_excluding_top3_profit": None, "gross_turnover": 0.0,
        "fees_and_taxes": 0.0, "gap_fill_count": 0, "gap_loss": 0.0,
    }


def _period_metrics(nav: pd.DataFrame, values: pd.Series, *, initial_cash: float) -> dict[str, object]:
    dates = pd.to_datetime(nav["trade_date"], errors="raise", utc=True)
    ordered = pd.DataFrame({"date": dates, "nav": values.to_numpy(dtype=float)}).sort_values("date")
    daily = ordered["nav"].pct_change()
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    sessions = max(1, len(ordered) - 1)
    annualized = float((ordered["nav"].iloc[-1] / initial_cash) ** (252.0 / sessions) - 1.0)
    weekly_nav = ordered.assign(period=ordered["date"].dt.tz_localize(None).dt.to_period("W-SUN")).groupby("period", sort=True)["nav"].last()
    weekly = weekly_nav.pct_change()
    if len(weekly):
        weekly.iloc[0] = weekly_nav.iloc[0] / initial_cash - 1.0
    weekly = weekly.astype(float)
    weekly_mean = float(weekly.mean()) if len(weekly) else None
    weekly_median = float(weekly.median()) if len(weekly) else None
    weekly_std = float(weekly.std(ddof=1)) if len(weekly) > 1 else None
    weekly_positive = float((weekly > 0).mean()) if len(weekly) else None
    weekly_sharpe = float(weekly.mean() / weekly.std(ddof=1) * math.sqrt(52)) if len(weekly) > 1 and weekly.std(ddof=1) > 0 else None
    downside = weekly[weekly < 0]
    weekly_sortino = float(weekly.mean() / downside.std(ddof=1) * math.sqrt(52)) if len(downside) > 1 and downside.std(ddof=1) > 0 else None
    drawdown = ordered["nav"] / ordered["nav"].cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = float(annualized / abs(max_dd)) if max_dd < 0 else None
    losses = weekly < 0
    max_losing = current = 0
    for loss in losses:
        current = current + 1 if loss else 0
        max_losing = max(max_losing, current)
    excluded = weekly.drop(weekly.idxmax()) if len(weekly) else weekly
    return {
        "annualized_return": annualized, "weekly_mean": weekly_mean,
        "weekly_median": weekly_median, "weekly_std": weekly_std,
        "weekly_positive_ratio": weekly_positive, "sharpe": weekly_sharpe,
        "sortino": weekly_sortino, "calmar": calmar,
        "worst_week": float(weekly.min()) if len(weekly) else None,
        "max_consecutive_losing_weeks": int(max_losing),
        "return_excluding_best_week": float((1.0 + excluded).prod() - 1.0) if len(excluded) else 0.0,
        "annual_returns": {
            str(period): float(group.iloc[-1] / (ordered.loc[ordered["date"] < group.index[0], "nav"].iloc[-1]
                                                if not ordered.loc[ordered["date"] < group.index[0]].empty else initial_cash) - 1.0)
            for period, group in ordered.set_index("date")["nav"].groupby(ordered.set_index("date").index.year)
        },
    }


def _cost_metrics(trades: pd.DataFrame, *, initial_cash: float) -> dict[str, object]:
    if trades.empty:
        return {"gross_turnover": 0.0, "fees_and_taxes": 0.0, "gap_fill_count": 0, "gap_loss": 0.0}
    gross = pd.to_numeric(trades.get("gross_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    commission = pd.to_numeric(trades.get("commission", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    duty = pd.to_numeric(trades.get("stamp_duty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(trades.get("gap_return", pd.Series(dtype=float)), errors="coerce")
    flags = trades.get("gap_flag", pd.Series(False, index=trades.index)).fillna(False).astype(bool)
    return {"gross_turnover": float(gross.sum() / initial_cash),
            "fees_and_taxes": float((commission + duty).sum()),
            "gap_fill_count": int(flags.sum()),
            "gap_loss": float(gap[flags].clip(upper=0).sum()) if flags.any() else 0.0}


def _compound_excluding_top_positive(returns: list[float], count: int) -> float:
    excluded = set(sorted(range(len(returns)), key=lambda i: returns[i], reverse=True)[:count])
    return float(np.prod([1.0 + value for index, value in enumerate(returns) if index not in excluded]) - 1.0)
