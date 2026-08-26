from __future__ import annotations

import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "trade_date", "ts_code", "raw_open", "raw_high", "raw_low", "raw_close",
    "economic_open", "economic_high", "economic_low", "economic_close", "adj_factor", "available_time",
    "is_suspended", "is_limit_up", "is_limit_down",
}
NUMERIC_PRICE_COLUMNS = REQUIRED_PRICE_COLUMNS - {"trade_date", "ts_code", "available_time",
                                                  "is_suspended", "is_limit_up", "is_limit_down"}


def validate_execution_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate the explicit raw/economic price contract without filling data silently."""
    missing = REQUIRED_PRICE_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"execution panel missing columns: {sorted(missing)}")
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], utc=True).dt.normalize()
    out["available_time"] = pd.to_datetime(out["available_time"], errors="raise", utc=True)
    if out["available_time"].isna().any():
        raise ValueError("execution panel available_time must be non-null")
    for column in NUMERIC_PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
        if (out[column] <= 0).any():
            raise ValueError(f"{column} must be positive")
    if out.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate security/session keys")
    for column in ("is_suspended", "is_limit_up", "is_limit_down"):
        out[column] = out[column].fillna(False).astype(bool)
    return out.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def economic_return(entry_economic_price: float, mark_economic_price: float) -> float:
    if entry_economic_price <= 0 or mark_economic_price <= 0:
        raise ValueError("economic prices must be positive")
    return mark_economic_price / entry_economic_price - 1.0
