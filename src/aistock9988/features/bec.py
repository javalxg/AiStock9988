"""Breadth-expansion continuation (BEC-V1) point-in-time features.

The provider implements one frozen mechanism: expanding market/industry breadth,
moderate industry-relative momentum, stable turnover, and a same-day right-side
confirmation.  It contains no label or post-entry field.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..data.industry_pit import resolve_industry_map
from ..time.session import session_close
from .engine import build_feature_ledger


def _window(strategy, name: str, default: int) -> int:
    spec = strategy.features.get(name, {})
    try:
        value = int(spec.get("window_sessions", default))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"feature {name} must declare a positive window_sessions") from exc
    if value <= 0:
        raise ValueError(f"feature {name} must declare a positive window_sessions")
    return value


def _industry_labels(bundle, days: pd.DatetimeIndex, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = bundle.enrichments.get("index_member_all", pd.DataFrame()).copy()
    if membership.empty:
        raise ValueError("BEC requires a frozen index_member_all_ts snapshot")
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for day in days:
        mapping, audit = resolve_industry_map(
            membership,
            signal_date=day,
            decision_time=None,
            universe_codes=codes,
        )
        rows.extend({"asof": day, "ts_code": code, "industry": label} for code, label in mapping.items())
        audits.append({
            "asof": str(day.date()),
            "industry_universe_count": audit.universe_count,
            "industry_covered_count": audit.covered_count,
            "industry_coverage_ratio": audit.coverage_ratio,
            "industry_conflict_security_count": audit.conflict_security_count,
            "industry_active_membership_count": audit.active_membership_count,
        })
    return pd.DataFrame(rows, columns=["asof", "ts_code", "industry"]), pd.DataFrame(audits)


def build_bec_feature_ledger(bundle, strategy) -> pd.DataFrame:
    """Build the BEC-V1 feature ledger from frozen bundle inputs."""
    out = build_feature_ledger(bundle, strategy).copy()
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    out = out.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)

    daily_basic = bundle.enrichments.get("daily_basic", pd.DataFrame()).copy()
    if daily_basic.empty:
        raise ValueError("BEC requires a daily_basic snapshot")
    daily_basic["ts_code"] = daily_basic["ts_code"].astype(str).str.upper()
    daily_basic["asof"] = pd.to_datetime(daily_basic["trade_date"], utc=True, errors="raise").dt.normalize()

    daily_basic = daily_basic.drop_duplicates(["ts_code", "asof"], keep="last").sort_values(["ts_code", "asof"])
    source_policy = strategy.data_policy.get("source_availability", {})
    if source_policy.get("daily_basic_ts") != "eod_trade_date_close":
        raise ValueError("BEC requires daily_basic_ts availability at EOD trade-date close")
    if source_policy.get("index_member_all_ts") != "interval_in_out_trade_date":
        raise ValueError("BEC requires interval PIT policy for index_member_all_ts")
    daily_basic["daily_basic_available_time"] = daily_basic["asof"].map(session_close)
    daily_basic["turnover_rate_f"] = pd.to_numeric(daily_basic["turnover_rate_f"], errors="coerce")
    turnover_window = _window(strategy, "turnover_ratio_20", 20)
    daily_basic["turnover_med20"] = daily_basic.groupby("ts_code", sort=False)["turnover_rate_f"].transform(
        lambda values: values.shift(1).rolling(turnover_window, min_periods=turnover_window).median()
    )
    daily_basic["turnover_ratio_20"] = daily_basic["turnover_rate_f"] / daily_basic["turnover_med20"]
    daily_basic["turnover_stability20"] = (daily_basic["turnover_ratio_20"] - 1.0).abs()

    out = out.merge(
        daily_basic[["ts_code", "asof", "turnover_rate_f", "turnover_ratio_20",
                     "turnover_stability20", "daily_basic_available_time"]],
        on=["ts_code", "asof"], how="left", validate="one_to_one",
    )
    grouped = out.groupby("ts_code", sort=False)
    ret10_window = _window(strategy, "ret10", 10)
    out["ret10"] = out["economic_close"] / grouped["economic_close"].shift(ret10_window) - 1.0
    ma20_window = _window(strategy, "ma20", 20)
    out["ma20"] = grouped["economic_close"].transform(
        lambda values: values.rolling(ma20_window, min_periods=ma20_window).mean()
    )

    days = pd.DatetimeIndex(sorted(out["asof"].drop_duplicates()))
    codes = sorted(out["ts_code"].dropna().astype(str).unique().tolist())
    labels, industry_audit = _industry_labels(bundle, days, codes)
    out = out.merge(labels, on=["asof", "ts_code"], how="left", validate="one_to_one")

    numeric_close = pd.to_numeric(out["economic_close"], errors="coerce")
    numeric_ma5 = pd.to_numeric(out["ma5"], errors="coerce")
    out["_above_ma5"] = numeric_close.ge(numeric_ma5)
    market_eligible = (
        out["universe_pass"].astype(bool)
        & out["selection_data_eligible"].astype(bool)
        & numeric_close.notna()
        & numeric_ma5.notna()
    )
    eligible = market_eligible & out["industry"].notna()
    out["breadth_ma5"] = out["_above_ma5"].where(market_eligible).groupby(out["asof"]).transform("mean")
    breadth_by_day = out.groupby("asof", sort=True)["breadth_ma5"].first()
    out["breadth_ma5_delta5"] = out["breadth_ma5"] - out["asof"].map(
        breadth_by_day.shift(_window(strategy, "breadth_ma5_delta5", 5))
    )

    # Industry state and return are computed from the same-day PIT universe.
    industry_breadth = (
        out.assign(_eligible_above=out["_above_ma5"].where(eligible))
        .groupby(["industry", "asof"], sort=True)["_eligible_above"].mean()
        .rename("industry_breadth_ma5")
        .reset_index()
    )
    out = out.merge(industry_breadth, on=["industry", "asof"], how="left", validate="many_to_one")
    industry_stats = (
        out.where(eligible)
        .groupby(["asof", "industry"], sort=False)["ret10"]
        .agg(industry_median_ret10="median")
        .reset_index()
    )
    out = out.merge(industry_stats, on=["asof", "industry"], how="left", validate="many_to_one")
    out["industry_excess_ret10"] = out["ret10"] - out["industry_median_ret10"]

    ready_for_vol = out["feature_ready"].astype(bool) & eligible
    vol85 = out.loc[ready_for_vol].groupby("asof", sort=True)["vol20"].quantile(0.85)
    out["vol20_p85"] = out["asof"].map(vol85)
    out["available_time"] = pd.concat(
        [pd.to_datetime(out["available_time"], utc=True),
         pd.to_datetime(out["daily_basic_available_time"], utc=True)], axis=1,
    ).max(axis=1)
    required = [
        "ret10", "industry_median_ret10", "industry_excess_ret10",
        "breadth_ma5", "breadth_ma5_delta5", "industry_breadth_ma5",
        "turnover_rate_f", "turnover_ratio_20", "turnover_stability20",
        "vol20_p85",
    ]
    numeric = out[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    out["feature_ready"] = out["feature_ready"].astype(bool) & eligible & finite
    out.loc[~out["feature_ready"] & out["feature_rejection_reason"].eq(""), "feature_rejection_reason"] = "BEC_FEATURE_NOT_MATURE"
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "provider": "bec_v1", "features": required},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Keep Parquet metadata JSON-serializable; the runner materializes this
    # audit as a separate ledger after the feature file is sealed.
    out.attrs["industry_audit"] = industry_audit.to_dict("records")
    return out.drop(columns=["_above_ma5"], errors="ignore").sort_values(
        ["asof", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = ["build_bec_feature_ledger"]
