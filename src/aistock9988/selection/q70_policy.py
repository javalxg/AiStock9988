"""Configured q70 selection gates and immutable selection decisions."""
from __future__ import annotations

import hashlib

import pandas as pd

from ..market.context import build_context


POLICY_ID = "selection.q70_gated_top20_to_top2.v1"
GATES = ("xsii_td3_bfq_sector_rel", "expma_12_bfq_sector_rel", "boll_mid_bfq_sector_rel")


def build_q70_selection_ledger(candidates: pd.DataFrame, daily: pd.DataFrame, *,
                               asof: str, max_positions: int = 2,
                               breadth_min: float = 0.40, factor_floor: float = 0.8) -> pd.DataFrame:
    required = {"asof", "ts_code", "candidate_rank", *GATES}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"q70 candidate ledger missing columns: {sorted(missing)}")
    if candidates.empty or candidates["asof"].astype(str).nunique() != 1:
        raise ValueError("q70 candidates must contain one non-empty decision cross-section")
    day = pd.Timestamp(asof).normalize()
    market = daily.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], utc=True).dt.normalize()
    context_rows = market[market["trade_date"] == day]
    context = build_context(context_rows, asof=day.date())
    out = candidates.sort_values(["candidate_rank", "ts_code"], kind="mergesort").head(20).copy()
    decision_key = f"{POLICY_ID}|{day.date()}|{','.join(out['ts_code'].astype(str))}"
    out["selection_decision_id"] = hashlib.sha256(decision_key.encode()).hexdigest()[:16]
    out["policy_id"] = POLICY_ID
    out["context_breadth_ratio"] = context.breadth_ratio
    out["selected"] = False
    out["rejection_reason"] = ""
    for idx, row in out.iterrows():
        code = str(row["ts_code"])
        reasons: list[str] = []
        if code.endswith(".BJ"):
            reasons.append("beijing_excluded")
        if any(pd.isna(row[col]) or float(row[col]) < factor_floor for col in GATES):
            reasons.append("sector_factor_gate")
        history = market[(market["ts_code"].astype(str) == code) & (market["trade_date"] <= day)].sort_values("trade_date").tail(20)
        returns = pd.to_numeric(history.get("pct_chg", pd.Series(dtype=float)), errors="coerce").dropna() / 100.0
        if len(returns) >= 2 and float(returns.std(ddof=1)) > 0.07:
            reasons.append("volatility_gate")
        if len(returns) and float(returns.min()) <= -0.098:
            reasons.append("recent_limit_down_gate")
        closes = pd.to_numeric(history.get("raw_close", history.get("close", pd.Series(dtype=float))), errors="coerce").dropna().tail(5)
        if len(closes) and float(closes.iloc[-1] / closes.max() - 1.0) <= -0.10:
            reasons.append("peak_drawdown_gate")
        out.at[idx, "rejection_reason"] = ";".join(reasons)
    eligible = out[out["rejection_reason"] == ""]
    selected_n = min(max_positions, 2 if context.breadth_ratio < breadth_min else max_positions)
    out.loc[eligible.head(selected_n).index, "selected"] = True
    return out.reset_index(drop=True)
