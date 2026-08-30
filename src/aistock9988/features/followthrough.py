"""Causal T+1 follow-through feature provider for the preregistered V1 rule."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig
from ..data.bundle import DataBundle
from .engine import build_feature_ledger


def build_followthrough_feature_ledger(
    bundle: DataBundle, strategy: StrategyConfig
) -> pd.DataFrame:
    """Create a watchlist on T and signal only after T+1 close confirmation.

    The returned ``asof`` is the confirmation session.  The common V3 engine
    therefore enters on the following session (T+2 from the original watchlist).
    No label or post-entry value is used here.
    """
    out = build_feature_ledger(bundle, strategy).copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True, errors="raise").dt.normalize()
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    out = out.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)

    execution = bundle.execution.copy()
    execution["asof"] = pd.to_datetime(execution["trade_date"], utc=True, errors="raise").dt.normalize()
    execution["ts_code"] = execution["ts_code"].astype(str).str.upper()
    execution = execution[["asof", "ts_code", "raw_close"]]
    out = out.merge(execution, on=["asof", "ts_code"], how="left", validate="one_to_one")

    universe = bundle.universe.copy()
    universe["asof"] = pd.to_datetime(universe["asof"], utc=True, errors="raise").dt.normalize()
    universe["ts_code"] = universe["ts_code"].astype(str).str.upper()
    universe["list_date"] = pd.to_datetime(universe["list_date"], utc=True, errors="coerce").dt.normalize()
    out = out.merge(universe[["asof", "ts_code", "list_date"]], on=["asof", "ts_code"], how="left", validate="one_to_one")

    # Prior 20-session raw return spikes are excluded from the setup.  The
    # shift ensures the current day's move cannot leak into this setup test.
    grouped = out.groupby("ts_code", sort=False)
    raw_close = pd.to_numeric(out["raw_close"], errors="coerce")
    raw_return1 = raw_close / grouped["raw_close"].shift(1) - 1.0
    real_session = out["execution_status"].eq("TRADABLE") & raw_close.gt(0)
    previous_real_session = (
        grouped["execution_status"].shift(1).eq("TRADABLE")
        & grouped["raw_close"].shift(1).gt(0)
    )
    out["watchlist_tradable"] = real_session
    # Carried marks on suspended/zero-volume rows are not observations.
    raw_return1 = raw_return1.where(real_session & previous_real_session)
    out["raw_return1"] = raw_return1
    raw_gain_window = int(strategy.features["raw_gain20_max"].get("window_sessions", 20))
    out["raw_gain20_max"] = grouped["raw_return1"].transform(
        lambda values: values.shift(1).rolling(raw_gain_window, min_periods=raw_gain_window).max()
    )
    out["raw_gain_window_complete"] = grouped["raw_return1"].transform(
        lambda values: values.shift(1).rolling(raw_gain_window, min_periods=raw_gain_window).count()
    ).eq(raw_gain_window)

    vol_median = out.groupby("asof", sort=False)["vol20"].transform("median")
    amount_multiplier = float(strategy.execution.get("amount_unit_multiplier", 1000.0))
    min_amount_yuan = float(strategy.universe.get("min_median_amount_yuan", 0.0))
    min_listed_sessions = int(strategy.universe.get("min_listed_sessions", 0))
    sessions = pd.DatetimeIndex(pd.to_datetime(bundle.calendar["session"], utc=True)).normalize()
    asof_pos = sessions.searchsorted(out["asof"].to_numpy(), side="left")
    list_pos = sessions.searchsorted(out["list_date"].fillna(pd.Timestamp("2262-04-11", tz="UTC")).to_numpy(), side="left")
    listed_session_count = asof_pos - list_pos
    amount_ok = out["liq20"].mul(amount_multiplier).ge(min_amount_yuan)

    setup_conditions = [
        out["feature_ready"].astype(bool),
        pd.Series(listed_session_count, index=out.index).ge(min_listed_sessions),
        amount_ok,
        real_session,
        out["dist_ma60"].between(0.0, 0.12, inclusive="both"),
        out["ret20"].between(0.02, 0.20, inclusive="both"),
        out["ret60"].between(0.0, 0.35, inclusive="both"),
        out["dd20"].ge(-0.10),
        out["vol20"].le(vol_median),
        out["volume_ratio_20"].between(0.70, 2.00, inclusive="both"),
        out["economic_close"].ge(out["ma5"]),
        out["economic_close"].ge(out["prev3_high"]),
        out["ret1"].ge(-0.02),
        out["raw_gain20_max"].lt(0.095),
        out["raw_gain_window_complete"],
    ]
    out["watchlist_pass"] = np.logical_and.reduce(setup_conditions).astype(float)

    previous_watchlist = grouped["watchlist_pass"].shift(1).fillna(0.0)
    tradable_next = real_session
    out["confirm_tradable"] = tradable_next
    out["followthrough_pass"] = (
        previous_watchlist.eq(1.0)
        & out["feature_ready"].astype(bool)
        & out["ret1"].gt(0.0)
        & out["economic_close"].ge(out["ma5"])
        & tradable_next
    ).astype(float)
    out["volume_stability20"] = (out["volume_ratio_20"] - 1.0).abs()

    custom = [
        "raw_return1", "raw_gain20_max", "volume_stability20",
        "raw_gain_window_complete", "watchlist_tradable", "confirm_tradable",
        "watchlist_pass", "followthrough_pass",
    ]
    numeric = out[custom].apply(pd.to_numeric, errors="coerce")
    out["feature_ready"] = out["feature_ready"].astype(bool) & np.isfinite(
        numeric[["raw_gain20_max", "volume_stability20"]].to_numpy(dtype=float)
    ).all(axis=1)
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps(
            {"strategy_hash": strategy.config_hash, "provider": "followthrough_confirm_v1", "custom": custom},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return out.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


__all__ = ["build_followthrough_feature_ledger"]
