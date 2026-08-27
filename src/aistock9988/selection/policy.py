from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..market.context import MarketContext


@dataclass(frozen=True)
class SelectionDecision:
    asof: str
    selected: tuple[str, ...]
    requested_n: int
    breadth_ratio: float
    reason: str
    policy_id: str


def direct_topk(candidates: pd.DataFrame, context: MarketContext, *, max_positions: int = 2,
               breadth_min: float = 0.40, low_breadth_top_n: int = 2,
               policy_id: str = "selection.direct_topk.v1") -> SelectionDecision:
    """Select from frozen Top20 only; market context is as-of and cannot inspect outcomes."""
    if max_positions <= 0 or low_breadth_top_n <= 0:
        raise ValueError("position limits must be positive")
    required = {"ts_code", "candidate_rank", "asof"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate ledger missing columns: {sorted(missing)}")
    if str(context.asof) != str(candidates["asof"].iloc[0]):
        raise ValueError("candidate and market context asof mismatch")
    if candidates["asof"].nunique() != 1 or candidates["ts_code"].duplicated().any():
        raise ValueError("candidate ledger must contain one unique cross-section")
    ranks = sorted(candidates["candidate_rank"].tolist())
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("candidate_rank must be unique, positive and contiguous")
    ordered = candidates.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
    requested = min(max_positions, low_breadth_top_n) if context.breadth_ratio < breadth_min else max_positions
    selected = tuple(ordered.head(requested)["ts_code"].tolist())
    reason = "low_breadth_reduce_exposure" if context.breadth_ratio < breadth_min else "normal_breadth_topk"
    return SelectionDecision(str(context.asof), selected, requested, context.breadth_ratio, reason, policy_id)
