"""Deterministic, auditable risk decisions for the execution engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopLossDecision:
    """A close-triggered stop decision for daily data.

    The trigger is observed at the prior session close and is therefore only
    executable at the next session's first tradable opportunity.
    """

    triggered: bool
    trigger_return: float
    trigger_price: float
    trigger_session: object
    reason: str


def evaluate_close_stop_loss(*, entry_economic_price: float,
                             mark_economic_price: float,
                             stop_loss_pct: float | None,
                             trigger_session: object) -> StopLossDecision:
    if entry_economic_price <= 0 or mark_economic_price <= 0:
        raise ValueError("economic prices must be positive")
    if stop_loss_pct is not None and (stop_loss_pct >= 0 or stop_loss_pct <= -1):
        raise ValueError("stop_loss_pct must be negative, greater than -1, and expressed as a ratio")
    current_return = mark_economic_price / entry_economic_price - 1.0
    triggered = stop_loss_pct is not None and current_return <= stop_loss_pct
    return StopLossDecision(
        triggered=bool(triggered),
        trigger_return=float(current_return),
        trigger_price=float(mark_economic_price),
        trigger_session=trigger_session,
        reason="stop_loss_close_trigger" if triggered else "not_triggered",
    )
