from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class MarketContext:
    asof: date
    universe_count: int
    advancing_count: int
    declining_count: int
    unchanged_count: int
    breadth_ratio: float
    limit_up_count: int
    limit_down_count: int
    turnover_total: float | None
    index_trend: str
    captured_only: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(daily: pd.DataFrame, *, asof: date, index_close: float | None = None,
                  index_ma20: float | None = None, index_ma60: float | None = None,
                  captured_only: bool = True) -> MarketContext:
    """Build a same-day market snapshot from already PIT-filtered raw daily rows.

    Expected columns: pct_chg and optional is_limit_up/is_limit_down/amount.
    No future row may be passed to this function; the provider is responsible for PIT filtering.
    """
    if "pct_chg" not in daily.columns:
        raise ValueError("market context requires pct_chg")
    pct = pd.to_numeric(daily["pct_chg"], errors="coerce").dropna()
    adv = int((pct > 0).sum())
    dec = int((pct < 0).sum())
    unchanged = int((pct == 0).sum())
    n = len(pct)
    breadth = adv / n if n else 0.0
    limit_up = int(daily.get("is_limit_up", pd.Series(False, index=daily.index)).fillna(False).astype(bool).sum())
    limit_down = int(daily.get("is_limit_down", pd.Series(False, index=daily.index)).fillna(False).astype(bool).sum())
    amount = pd.to_numeric(daily["amount"], errors="coerce") if "amount" in daily else pd.Series(dtype=float)
    if index_close is not None and index_ma20 is not None and index_ma60 is not None:
        trend = "bull" if index_close > index_ma60 and index_ma20 > index_ma60 else "non_bull"
    else:
        trend = "unknown"
    return MarketContext(date.fromisoformat(str(asof)), n, adv, dec, unchanged, breadth,
                         limit_up, limit_down, float(amount.sum()) if not amount.empty else None,
                         trend, captured_only)
