"""Configured q70 selection gates and immutable selection decisions."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..market.context import build_context
from ..time.session import session_close


POLICY_ID = "selection.q70_gated_top20_to_top2.v1"
GATES = ("xsii_td3_bfq_sector_rel", "expma_12_bfq_sector_rel", "boll_mid_bfq_sector_rel")


def build_q70_selection_ledger(candidates: pd.DataFrame, daily: pd.DataFrame, *,
                               asof: str, max_positions: int = 2,
                               breadth_min: float = 0.40, factor_floor: float = 0.8,
                               weak_breadth_positions: int = 2,
                               volatility_window_sessions: int = 20,
                               volatility_max: float = 0.07,
                               recent_limit_down_window_sessions: int = 20,
                               recent_limit_down_threshold: float = -0.098,
                               peak_drawdown_window_sessions: int = 5,
                               peak_drawdown_threshold: float = -0.10,
                               exclude_beijing: bool = True,
                               alpha_weight: bool = False, alpha_power: float = 1.0) -> pd.DataFrame:
    required = {"asof", "ts_code", "candidate_rank", *GATES}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"q70 candidate ledger missing columns: {sorted(missing)}")
    if candidates.empty or candidates["asof"].astype(str).nunique() != 1:
        raise ValueError("q70 candidates must contain one non-empty decision cross-section")
    day = pd.Timestamp(asof).normalize()
    candidate_day = pd.to_datetime(candidates["asof"], errors="raise", utc=True).dt.normalize()
    expected_day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    if not candidate_day.eq(expected_day).all():
        raise ValueError("candidate ledger asof does not match SelectionPolicy decision date")
    market = daily.copy()
    required_market = {"ts_code", "trade_date", "pct_chg", "raw_close", "available_time"}
    if missing_market := required_market - set(market.columns):
        raise ValueError(f"market context missing columns: {sorted(missing_market)}")
    market["trade_date"] = pd.to_datetime(market["trade_date"], utc=True).dt.normalize()
    market["available_time"] = pd.to_datetime(market["available_time"], errors="raise", utc=True)
    market = market[(market["trade_date"] <= expected_day) &
                    (market["available_time"] <= session_close(expected_day))].copy()
    context_rows = market[market["trade_date"] == expected_day]
    if context_rows.empty:
        raise ValueError("no PIT-visible market context for selection date")
    context = build_context(context_rows, asof=day.date())
    context_columns = [column for column in ("ts_code", "trade_date", "pct_chg", "raw_close", "amount",
                                              "available_time", "is_limit_up", "is_limit_down")
                       if column in market.columns]
    context_payload = market[context_columns].sort_values(["trade_date", "ts_code"], kind="mergesort").to_csv(
        index=False, lineterminator="\n"
    )
    context_hash = hashlib.sha256(context_payload.encode()).hexdigest()
    out = candidates.sort_values(["candidate_rank", "ts_code"], kind="mergesort").head(20).copy()
    policy_config = {
        "max_positions": max_positions, "breadth_min": breadth_min, "factor_floor": factor_floor,
        "weak_breadth_positions": weak_breadth_positions,
        "volatility_window_sessions": volatility_window_sessions, "volatility_max": volatility_max,
        "recent_limit_down_window_sessions": recent_limit_down_window_sessions,
        "recent_limit_down_threshold": recent_limit_down_threshold,
        "peak_drawdown_window_sessions": peak_drawdown_window_sessions,
        "peak_drawdown_threshold": peak_drawdown_threshold, "exclude_beijing": exclude_beijing,
        "alpha_weight": alpha_weight, "alpha_power": alpha_power,
    }
    out["policy_id"] = POLICY_ID
    out["context_breadth_ratio"] = context.breadth_ratio
    out["context_hash"] = context_hash
    out["selected"] = False
    out["rejection_reason"] = ""
    for idx, row in out.iterrows():
        code = str(row["ts_code"])
        reasons: list[str] = []
        if exclude_beijing and code.endswith(".BJ"):
            reasons.append("beijing_excluded")
        if any(pd.isna(row[col]) or float(row[col]) < factor_floor for col in GATES):
            reasons.append("sector_factor_gate")
        history = market[market["ts_code"].astype(str) == code].sort_values("trade_date")
        volatility_history = history.tail(volatility_window_sessions)
        limit_history = history.tail(recent_limit_down_window_sessions)
        volatility_returns = pd.to_numeric(volatility_history.get("pct_chg", pd.Series(dtype=float)),
                                           errors="coerce").dropna() / 100.0
        limit_returns = pd.to_numeric(limit_history.get("pct_chg", pd.Series(dtype=float)),
                                      errors="coerce").dropna() / 100.0
        if len(volatility_returns) >= 2 and float(volatility_returns.std(ddof=1)) > volatility_max:
            reasons.append("volatility_gate")
        if len(limit_returns) and float(limit_returns.min()) <= recent_limit_down_threshold:
            reasons.append("recent_limit_down_gate")
        closes = pd.to_numeric(history["raw_close"], errors="coerce").dropna().tail(peak_drawdown_window_sessions)
        if len(closes) and float(closes.iloc[-1] / closes.max() - 1.0) <= peak_drawdown_threshold:
            reasons.append("peak_drawdown_gate")
        out.at[idx, "rejection_reason"] = ";".join(reasons)
    eligible = out[out["rejection_reason"] == ""]
    selected_n = min(max_positions, weak_breadth_positions if context.breadth_ratio < breadth_min else max_positions)
    out.loc[eligible.head(selected_n).index, "selected"] = True
    out["target_weight"] = 0.0
    selected_index = out.index[out["selected"]]
    if len(selected_index):
        raw_weights = ((1.0 / out.loc[selected_index, "candidate_rank"].astype(float).pow(alpha_power))
                       if alpha_weight else pd.Series(1.0, index=selected_index))
        out.loc[selected_index, "target_weight"] = raw_weights / raw_weights.sum()
    decision_payload = {
        "policy_id": POLICY_ID, "asof": str(day.date()), "config": policy_config,
        "context_breadth_ratio": context.breadth_ratio, "context_hash": context_hash,
        "candidates": out[["ts_code", "candidate_rank", *GATES, "selected", "rejection_reason"]].to_dict("records"),
    }
    decision_key = json.dumps(decision_payload, ensure_ascii=False, sort_keys=True, default=str)
    out["selection_decision_id"] = hashlib.sha256(decision_key.encode()).hexdigest()[:16]
    return out.reset_index(drop=True)
