from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorporateAction:
    """PIT-visible company action effective on ex_date."""

    ts_code: str
    ex_date: str
    split_ratio: float = 1.0
    cash_dividend: float = 0.0
    available_time: str | None = None

    def __post_init__(self) -> None:
        if self.split_ratio <= 0 or self.cash_dividend < 0:
            raise ValueError("split_ratio must be positive and cash_dividend non-negative")


def apply_action(position: dict, action: CorporateAction) -> float:
    """Apply one action to a position and return cash received from dividends.

    Cost basis is kept in total-cost form so split adjustments preserve the
    economic value of the position while changing share count and per-share cost.
    """
    shares_before = float(position["shares"])
    position["shares"] = shares_before * action.split_ratio
    position["entry_price"] = position["entry_price"] / action.split_ratio
    cash = shares_before * action.cash_dividend
    position["corporate_actions"] = int(position.get("corporate_actions", 0)) + 1
    return cash
