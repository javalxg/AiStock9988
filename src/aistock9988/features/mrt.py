"""Features for the preregistered momentum-reset (MRT) rule.

The provider deliberately keeps the state sequence observable at T close:
relative momentum is measured before the same-day shock, while the amount
baseline excludes T.  No label or T+1 field is used here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig
from ..data.industry_pit import resolve_industry_map
from ..time.session import session_close
from .engine import build_feature_ledger


def _window(strategy: StrategyConfig, name: str, default: int) -> int:
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
        raise ValueError("MRT requires a frozen index_member_all_ts snapshot")
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for day in days:
        mapping, audit = resolve_industry_map(
            membership,
            signal_date=day,
            decision_time=None,
            universe_codes=codes,
        )
        audits.append({
            "asof": day,
            "industry_universe_count": audit.universe_count,
            "industry_covered_count": audit.covered_count,
            "industry_coverage_ratio": audit.coverage_ratio,
            "industry_conflict_security_count": audit.conflict_security_count,
            "industry_active_membership_count": audit.active_membership_count,
        })
        rows.extend({"asof": day, "ts_code": code, "industry": label} for code, label in mapping.items())
    labels = pd.DataFrame(rows, columns=["asof", "ts_code", "industry"])
    audit_frame = pd.DataFrame(audits)
    return labels, audit_frame


def build_mrt_feature_ledger(bundle, strategy: StrategyConfig) -> pd.DataFrame:
    """Build the PIT feature ledger consumed by ``mrt_v1_runner``."""
    base = build_feature_ledger(bundle, strategy)
    base["asof"] = pd.to_datetime(base["asof"], utc=True).dt.normalize()
    base["ts_code"] = base["ts_code"].astype(str).str.upper()
    base = base.sort_values(["ts_code", "asof"], kind="mergesort").reset_index(drop=True)

    execution = bundle.execution.copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"], utc=True).dt.normalize()
    execution["ts_code"] = execution["ts_code"].astype(str).str.upper()
    execution = execution[[
        "trade_date", "ts_code", "raw_open", "raw_close", "down_limit", "pct_chg", "amount",
    ]].rename(columns={"trade_date": "asof"})
    execution = execution.drop_duplicates(["asof", "ts_code"], keep="last")
    out = base.merge(execution, on=["asof", "ts_code"], how="left", validate="one_to_one")

    grouped = out.groupby("ts_code", sort=False)
    ret10_window = _window(strategy, "ret10", 10)
    adv_window = _window(strategy, "adv20_prior", 20)
    out["ret10"] = out["economic_close"] / grouped["economic_close"].shift(ret10_window) - 1.0
    out["adv20_prior"] = grouped["amount"].transform(
        lambda values: values.shift(1).rolling(adv_window, min_periods=adv_window).median()
    )
    out["shock_amount_ratio"] = out["amount"] / out["adv20_prior"]
    out["shock_close_lt_open"] = (
        pd.to_numeric(out["raw_close"], errors="coerce") < pd.to_numeric(out["raw_open"], errors="coerce")
    ).astype(float)
    out["shock_open_ok"] = (
        pd.to_numeric(out["raw_open"], errors="coerce") > pd.to_numeric(out["down_limit"], errors="coerce")
    ).astype(float)
    out["shock_close_ok"] = (
        pd.to_numeric(out["raw_close"], errors="coerce") > pd.to_numeric(out["down_limit"], errors="coerce")
    ).astype(float)

    # Market-relative ret10 uses only the same-day cross-section.
    market_median = out.groupby("asof", sort=False)["ret10"].transform("median")
    out["market_excess_ret10"] = out["ret10"] - market_median
    out["relative_strength10"] = out["market_excess_ret10"]
    # Percentile denominator is the PIT-eligible feature universe, not rows
    # that are already missing a required daily source.
    out["vol20_pct"] = out["vol20"].where(out["feature_ready"].astype(bool)).groupby(
        out["asof"], sort=False
    ).rank(method="average", pct=True)

    days = pd.DatetimeIndex(sorted(out["asof"].drop_duplicates()))
    codes = sorted(out["ts_code"].dropna().astype(str).unique().tolist())
    labels, industry_audit = _industry_labels(bundle, days, codes)
    out = out.merge(labels, on=["asof", "ts_code"], how="left", validate="one_to_one")
    out["industry_excess_ret10"] = out["ret10"] - out.groupby(["asof", "industry"], sort=False)["ret10"].transform("median")

    new_required = [
        "ret10", "market_excess_ret10", "industry_excess_ret10", "adv20_prior",
        "shock_amount_ratio", "shock_close_lt_open", "shock_open_ok", "shock_close_ok",
        "vol20_pct", "raw_open", "raw_close", "down_limit", "pct_chg",
    ]
    numeric = out[new_required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    out["feature_ready"] = out["feature_ready"].astype(bool) & finite & out["industry"].notna()
    out.loc[~out["feature_ready"] & out["feature_rejection_reason"].eq(""), "feature_rejection_reason"] = "MRT_FEATURE_NOT_MATURE"
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "provider": "mrt_v1", "windows": new_required}, sort_keys=True).encode()
    ).hexdigest()
    out.attrs["industry_audit"] = industry_audit
    return out.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


__all__ = ["build_mrt_feature_ledger"]
