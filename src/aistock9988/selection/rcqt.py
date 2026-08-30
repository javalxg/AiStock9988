"""Transparent Reset-Confirmation + Quiet Trend selection rules."""
from __future__ import annotations

import pandas as pd
import hashlib
import json


def _rank(series: pd.Series) -> pd.Series:
    order = sorted(series.index, key=lambda key: (series.loc[key], str(key)))
    values = pd.Series(index=order, data=(range(1, len(order) + 1)), dtype=float)
    return values.reindex(series.index) / max(len(series), 1)


def score_rcqt(frame: pd.DataFrame, *, require_right_confirmation: bool = True) -> pd.DataFrame:
    """Score one as-of cross-section; all columns are T-close features."""
    required = {"ts_code", "asof", "ret1", "dist_ma60", "ret20", "ret60", "dd20", "dd60", "vol20", "liq20", "volume_ratio_20", "close", "ma5", "prev3_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RCQT frame missing columns: {sorted(missing)}")
    if frame.empty:
        return frame.copy()
    if frame["asof"].nunique(dropna=False) > 1:
        parts = [score_rcqt(group.copy(), require_right_confirmation=require_right_confirmation) for _, group in frame.groupby("asof", sort=True)]
        return pd.concat(parts, ignore_index=True)
    out = frame.copy()
    if out.duplicated(["asof", "ts_code"]).any():
        raise ValueError("RCQT frame must contain unique asof/ts_code")
    numeric = sorted(required - {"ts_code", "asof"})
    coerced = out[numeric].apply(pd.to_numeric, errors="coerce")
    if coerced.isna().any().any():
        raise ValueError("RCQT frame contains missing or non-numeric features")
    out[numeric] = coerced
    out["right_confirmed"] = (
        (out["close"] >= out["ma5"])
        & (out["close"] >= out["prev3_high"])
        & (out["ret1"] >= -0.02)
        & out["volume_ratio_20"].between(0.70, 2.50)
    )
    vol85 = out["vol20"].quantile(0.85)
    out["reset_eligible"] = (
        out["dist_ma60"].le(0.05) & out["ret20"].le(0.08)
        & out["ret60"].le(0.35) & out["dd60"].le(-0.12)
        & out["vol20"].le(vol85) & (out["right_confirmed"] if require_right_confirmation else True)
    )
    out["quiet_eligible"] = (
        out["ret20"].gt(0) & out["ret60"].gt(0) & out["ret60"].le(0.35)
        & out["dist_ma60"].gt(0) & out["dist_ma60"].le(0.15)
        & out["dd20"].ge(-0.10) & out["ret20"].le(0.25)
    )
    if "max_single_day_return_20d" in out.columns:
        out["quiet_eligible"] &= out["max_single_day_return_20d"].lt(0.095)
    out["confirmation_strength"] = (
        0.25 * (out["close"] / out["ma5"] - 1).clip(lower=0)
        + 0.25 * (out["close"] / out["prev3_high"] - 1).clip(lower=0)
        + 0.25 * (out["ret1"] + 0.02).clip(lower=0)
        + 0.25 * (1 - (out["volume_ratio_20"] - 1).abs().clip(upper=1))
    )
    out["reset_score"] = (
        0.30 * _rank(-out["dd60"]) + 0.25 * _rank(-out["dist_ma60"])
        + 0.20 * _rank(-out["ret20"]) + 0.10 * _rank(-out["vol20"])
        + 0.10 * _rank(out["liq20"]) + 0.05 * out["confirmation_strength"]
    )
    out["quiet_score"] = (
        0.30 * _rank(out["ret60"]) + 0.25 * _rank(out["dd20"])
        + 0.20 * _rank(-out["vol20"]) + 0.15 * _rank(out["liq20"])
        + 0.10 * _rank(-abs(out["dist_ma60"] - 0.05))
    )
    return out


def select_rcqt(scored: pd.DataFrame, *, reset_slots: int = 4, quiet_slots: int = 2,
                single_weight_cap: float = 0.15, sector_weight_cap: float = 0.30,
                equity_cap: float = 0.72) -> pd.DataFrame:
    """Select deterministic 4+2 slots, preferring reset when a code overlaps."""
    if reset_slots <= 0 or quiet_slots < 0 or not (0 < single_weight_cap <= 1) or not (0 < sector_weight_cap <= 1) or not (0 < equity_cap <= 1):
        raise ValueError("invalid RCQT slot counts")
    required = {"ts_code", "reset_eligible", "quiet_eligible", "reset_score", "quiet_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"RCQT scored frame missing columns: {sorted(missing)}")
    if scored.empty:
        return scored.copy()
    if "asof" in scored.columns and scored["asof"].nunique(dropna=False) > 1:
        return pd.concat([select_rcqt(group, reset_slots=reset_slots, quiet_slots=quiet_slots,
                                      single_weight_cap=single_weight_cap, sector_weight_cap=sector_weight_cap,
                                      equity_cap=equity_cap)
                          for _, group in scored.groupby("asof", sort=True)], ignore_index=True)
    reset = scored[scored["reset_eligible"]].sort_values(["reset_score", "ts_code"], ascending=[False, True], kind="mergesort").head(reset_slots)
    used = set(reset["ts_code"])
    quiet = scored[scored["quiet_eligible"] & ~scored["ts_code"].isin(used)].sort_values(["quiet_score", "ts_code"], ascending=[False, True], kind="mergesort").head(quiet_slots)
    out = pd.concat([reset.assign(sleeve="recovery"), quiet.assign(sleeve="quiet")], ignore_index=True)
    out["selected"] = True
    out["target_weight"] = min(0.12, single_weight_cap)
    if "industry" in out.columns:
        sector_sum = out.groupby("industry")["target_weight"].transform("sum")
        out["target_weight"] = (out["target_weight"] * (sector_weight_cap / sector_sum).clip(upper=1.0)).clip(upper=single_weight_cap)
    total = float(out["target_weight"].sum())
    if total > equity_cap:
        out["target_weight"] *= equity_cap / total
    out["policy_id"] = "selection.rcqt.v1"
    context = {"policy_id": "selection.rcqt.v1", "reset_slots": reset_slots,
               "quiet_slots": quiet_slots,
               "asof": str(out["asof"].iloc[0]) if "asof" in out.columns and len(out) else None}
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out["context_hash"] = context_hash
    out["selection_decision_id"] = "rcqt-" + context_hash[:16]
    out["selection_rank"] = range(1, len(out) + 1)
    out["candidate_rank"] = out.groupby("sleeve", sort=False).cumcount() + 1
    return out
