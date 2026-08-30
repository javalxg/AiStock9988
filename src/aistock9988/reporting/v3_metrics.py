"""Portfolio metrics derived from complete V3 fills and daily NAV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def summarize_v3(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    initial_cash: float,
    positions: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, object]:
    if nav.empty:
        raise ValueError("V3 NAV ledger is empty")
    ordered = nav.sort_values("trade_date", kind="mergesort").copy()
    values = pd.to_numeric(ordered["nav"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("NAV contains non-finite values")
    drawdown = values / values.cummax() - 1.0
    sells = fills[fills["side"].eq("SELL")].copy() if not fills.empty else pd.DataFrame()
    pnl = pd.to_numeric(sells.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    dates = pd.to_datetime(ordered["trade_date"], errors="raise", utc=True)
    weekly_nav = ordered.assign(
        period=dates.dt.tz_localize(None).dt.to_period("W-SUN")
    ).groupby("period", sort=True)["nav"].last()
    weekly = weekly_nav.pct_change()
    if len(weekly):
        weekly.iloc[0] = weekly_nav.iloc[0] / initial_cash - 1.0
    without_best = weekly.drop(weekly.idxmax()) if len(weekly) else weekly
    positive_pnl = pnl[pnl > 0]
    top3_index = positive_pnl.nlargest(3).index if len(positive_pnl) else pd.Index([])
    pnl_excluding_top3 = pnl.drop(index=top3_index) if len(top3_index) else pnl
    sessions = max(1, len(ordered) - 1)
    total_return = float(values.iloc[-1] / initial_cash - 1.0)
    fees = 0.0 if fills.empty else float(
        pd.to_numeric(fills["commission"], errors="raise").sum()
        + pd.to_numeric(fills["stamp_duty"], errors="raise").sum()
    )
    metrics = {
        "initial_cash": float(initial_cash),
        "final_nav": float(values.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252.0 / sessions) - 1.0),
        "max_drawdown": float(drawdown.min()),
        "portfolio_profit_factor": gains / losses if losses > 0 else None,
        "trade_count": int(len(sells)),
        "trade_win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "realized_net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "trade_pnl_excluding_top3_profit": float(pnl_excluding_top3.sum()) if len(pnl_excluding_top3) else 0.0,
        "return_excluding_top3_profit": float(pnl_excluding_top3.sum() / initial_cash) if len(pnl_excluding_top3) else 0.0,
        "return_excluding_best_week": float((1.0 + without_best).prod() - 1.0) if len(without_best) else 0.0,
        "best_week": float(weekly.max()) if len(weekly) else None,
        "worst_week": float(weekly.min()) if len(weekly) else None,
        "weekly_positive_ratio": float((weekly > 0).mean()) if len(weekly) else None,
        "weekly_ge_5_ratio": float((weekly >= 0.05).mean()) if len(weekly) else None,
        "weekly_ge_5_count": int((weekly >= 0.05).sum()) if len(weekly) else 0,
        "fees_and_taxes": fees,
        "gross_turnover": 0.0 if fills.empty else float(pd.to_numeric(fills["gross_value"], errors="raise").sum() / initial_cash),
        "average_gross_exposure": float(pd.to_numeric(ordered["gross_exposure"], errors="coerce").mean()),
        "max_open_positions": int(pd.to_numeric(ordered["open_positions"], errors="raise").max()),
        "sharpe_weekly": (
            float(weekly.mean() / weekly.std(ddof=1) * math.sqrt(52))
            if len(weekly) > 1 and weekly.std(ddof=1) > 0 else None
        ),
    }
    if positions is not None and corporate_actions is not None:
        replay = _replay_without_top3(
            ordered, fills, positions, corporate_actions, top3_index, initial_cash
        )
        # Keep the scalar field used by acceptance while retaining the full
        # replay audit in a separate object.
        metrics["return_excluding_top3_profit"] = (
            float(replay["return"]) if replay.get("available") else float("nan")
        )
        metrics["top3_replay"] = replay
    else:
        metrics["top3_replay"] = {
            "available": False,
            "reason": "position_ledger_not_supplied",
        }
    return metrics


def _replay_without_top3(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    top3_index: pd.Index,
    initial_cash: float,
) -> dict[str, object]:
    """Replay NAV after removing the three largest profitable trade keys."""
    if not len(top3_index):
        return {
            "available": True,
            "removed_trade_count": 0,
            "final_nav": float(nav["nav"].iloc[-1]),
            "return": float(nav["nav"].iloc[-1] / initial_cash - 1.0),
        }
    required_fills = {
        "decision_id", "ts_code", "side", "trade_date", "gross_value",
        "commission", "stamp_duty",
    }
    required_positions = {"decision_id", "ts_code", "trade_date", "market_value"}
    if not required_fills.issubset(fills.columns) or not required_positions.issubset(positions.columns):
        return {"available": False, "reason": "trade_identity_columns_missing"}
    winners = fills.loc[top3_index]
    keys = set(zip(winners["decision_id"].astype(str), winners["ts_code"].astype(str)))
    fill_frame = fills.copy()
    fill_frame["trade_key"] = list(zip(fill_frame["decision_id"].astype(str), fill_frame["ts_code"].astype(str)))
    removed_fills = fill_frame[fill_frame["trade_key"].isin(keys)]
    position_frame = positions.copy()
    position_frame["trade_key"] = list(zip(position_frame["decision_id"].astype(str), position_frame["ts_code"].astype(str)))
    removed_positions = position_frame[position_frame["trade_key"].isin(keys)]
    action_frame = corporate_actions.copy()
    if not action_frame.empty and {"decision_id", "ts_code", "trade_date", "cash_dividend"}.issubset(action_frame.columns):
        action_frame["trade_key"] = list(zip(action_frame["decision_id"].astype(str), action_frame["ts_code"].astype(str)))
        removed_actions = action_frame[action_frame["trade_key"].isin(keys)]
    else:
        removed_actions = pd.DataFrame(columns=["trade_date", "cash_dividend"])

    dates = pd.to_datetime(nav["trade_date"], errors="raise", utc=True).dt.normalize()
    contribution = pd.Series(0.0, index=nav.index)
    for _, fill in removed_fills.iterrows():
        day = pd.Timestamp(fill["trade_date"])
        gross = float(fill["gross_value"])
        fees = float(fill.get("commission", 0.0)) + float(fill.get("stamp_duty", 0.0))
        signed = -(gross + fees) if str(fill["side"]) == "BUY" else gross - fees
        contribution.loc[dates >= day] += signed
    for _, action in removed_actions.iterrows():
        day = pd.Timestamp(action["trade_date"])
        contribution.loc[dates >= day] += float(action.get("cash_dividend", 0.0))
    if not removed_positions.empty:
        marked = removed_positions.groupby("trade_date", sort=False)["market_value"].sum()
        marked.index = pd.to_datetime(marked.index, errors="raise", utc=True).normalize()
        for day, value in marked.items():
            contribution.loc[dates == day] += float(value)
    replay_nav = pd.to_numeric(nav["nav"], errors="raise") - contribution
    if not np.isfinite(replay_nav.to_numpy(dtype=float)).all():
        return {"available": False, "reason": "non_finite_replayed_nav"}
    return {
        "available": True,
        "removed_trade_count": int(len(keys)),
        "final_nav": float(replay_nav.iloc[-1]),
        "return": float(replay_nav.iloc[-1] / initial_cash - 1.0),
        "max_drawdown": float((replay_nav / replay_nav.cummax() - 1.0).min()),
    }


__all__ = ["summarize_v3"]
