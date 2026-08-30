"""Point-in-time fields for the T0 shock -> T1 reversal contract."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from .engine import build_feature_ledger


def build_t1_reversal_feature_ledger(bundle, strategy) -> pd.DataFrame:
    out = build_feature_ledger(bundle, strategy).copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    out = out.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)

    # The shared feature engine intentionally returns derived features only.
    # This event rule also needs the raw T0/T1 execution fields, so join the
    # same frozen execution panel explicitly instead of relying on incidental
    # columns or a second database read.
    execution = bundle.execution.copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"], utc=True).dt.normalize()
    execution["ts_code"] = execution["ts_code"].astype(str).str.upper()
    execution = execution[[
        "trade_date", "ts_code", "raw_open", "raw_close", "down_limit",
        "amount", "economic_open",
    ]].rename(columns={"trade_date": "asof"})
    execution = execution.drop_duplicates(["asof", "ts_code"], keep="last")
    out = out.merge(execution, on=["asof", "ts_code"], how="left", validate="one_to_one")

    grouped = out.groupby("ts_code", sort=False)
    amount = pd.to_numeric(out["amount"], errors="coerce")
    out["prior20_amount_median"] = grouped["amount"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").shift(1).rolling(20, min_periods=20).median()
    )
    out["t0_amount_ratio"] = amount / out["prior20_amount_median"]
    drop_max = float(strategy.features.get("t0_intraday_return_max", -0.05))
    ratio_min = float(strategy.features.get("t0_amount_ratio_min", 1.5))
    out["t0_shock"] = (
        pd.to_numeric(out["raw_open"], errors="coerce").gt(0)
        & (pd.to_numeric(out["raw_close"], errors="coerce") / pd.to_numeric(out["raw_open"], errors="coerce") - 1.0).le(drop_max)
        & pd.to_numeric(out["raw_close"], errors="coerce").gt(pd.to_numeric(out["down_limit"], errors="coerce"))
        & out["t0_amount_ratio"].ge(ratio_min)
    ).astype(float)
    # T+1 values are joined by the event runner; keep the feature contract
    # explicit so no future field can accidentally enter Stage-1 scoring.
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "provider": "t1_reversal_confirm_v1"}, sort_keys=True).encode()
    ).hexdigest()
    out["t1_reversal_confirmed"] = 0.0
    out["t1_ret1"] = np.nan
    numeric = out[[
        "prior20_amount_median", "t0_amount_ratio", "t0_shock", "amount",
        "raw_open", "raw_close", "down_limit", "economic_open", "economic_close",
    ]].apply(pd.to_numeric, errors="coerce")
    out["feature_ready"] = out["feature_ready"].astype(bool) & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    out.loc[~out["feature_ready"] & out["feature_rejection_reason"].eq(""), "feature_rejection_reason"] = "T1_REVERSAL_FEATURE_NOT_MATURE"
    return out.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


__all__ = ["build_t1_reversal_feature_ledger"]
