"""Calendar-session feature engine for the first V3 rule strategy."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig
from ..data.bundle import DataBundle
from ..time.session import session_close


REQUIRED_QUIET_FEATURES = (
    "economic_close", "ma5", "ma60", "ret1", "ret20", "ret60", "dd20",
    "vol20", "liq20", "volume_ratio_20", "prev3_high", "dist_ma60",
    "dist_ma60_abs_005",
)


def _window(strategy: StrategyConfig, name: str, default: int) -> int:
    spec = strategy.features.get(name, {})
    try:
        value = int(spec.get("window_sessions", default))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"feature {name} must declare a positive window_sessions") from exc
    if value <= 0:
        raise ValueError(f"feature {name} must declare a positive window_sessions")
    return value


def build_feature_ledger(bundle: DataBundle, strategy: StrategyConfig) -> pd.DataFrame:
    frame = bundle.execution.copy().sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    group = frame.groupby("ts_code", sort=False, group_keys=False)
    close = pd.to_numeric(frame["economic_close"], errors="coerce")
    high = pd.to_numeric(frame["economic_high"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    ma5_window = _window(strategy, "ma5", 5)
    ma60_window = _window(strategy, "ma60", 60)
    ret1_window = _window(strategy, "ret1", 1)
    ret20_window = _window(strategy, "ret20", 20)
    ret60_window = _window(strategy, "ret60", 60)
    dd20_window = _window(strategy, "dd20", 20)
    vol20_window = _window(strategy, "vol20", 20)
    liq20_window = _window(strategy, "liq20", 20)
    prev3_window = _window(strategy, "prev3_high", 3)
    frame["ma5"] = group["economic_close"].transform(lambda values: values.rolling(ma5_window, min_periods=ma5_window).mean())
    frame["ma60"] = group["economic_close"].transform(lambda values: values.rolling(ma60_window, min_periods=ma60_window).mean())
    frame["ret1"] = close / group["economic_close"].shift(ret1_window) - 1.0
    frame["ret20"] = close / group["economic_close"].shift(ret20_window) - 1.0
    frame["ret60"] = close / group["economic_close"].shift(ret60_window) - 1.0
    frame["dd20"] = high / group["economic_high"].transform(
        lambda values: values.rolling(dd20_window, min_periods=dd20_window).max()
    ) - 1.0
    frame["vol20"] = frame.groupby("ts_code", sort=False)["ret1"].transform(
        lambda values: values.rolling(vol20_window, min_periods=vol20_window).std()
    )
    frame["liq20"] = group["amount"].transform(lambda values: values.rolling(liq20_window, min_periods=liq20_window).median())
    frame["volume_ratio_20"] = amount / frame["liq20"]
    frame["prev3_high"] = group["economic_close"].transform(
        lambda values: values.shift(1).rolling(prev3_window, min_periods=prev3_window).max()
    )
    frame["dist_ma60"] = close / frame["ma60"] - 1.0
    frame["dist_ma60_abs_005"] = (frame["dist_ma60"] - 0.05).abs()
    required = list(REQUIRED_QUIET_FEATURES)
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    frame["feature_ready"] = (
        frame["universe_pass"].astype(bool)
        & frame["selection_data_eligible"].astype(bool)
        & finite
    )
    frame["feature_rejection_reason"] = np.select(
        [
            ~frame["universe_pass"].astype(bool),
            ~frame["selection_data_eligible"].astype(bool),
            ~finite,
        ],
        [
            "UNIVERSE_REJECTED",
            frame["selection_data_rejection_reason"],
            "FEATURE_NOT_MATURE",
        ],
        default="",
    )
    feature_payload = {
        "strategy_hash": strategy.config_hash,
        "features": strategy.to_dict()["features"],
        "window_basis": "exchange_calendar_sessions",
    }
    feature_hash = hashlib.sha256(
        json.dumps(feature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frame["feature_set_hash"] = feature_hash
    frame["bundle_id"] = bundle.bundle_id
    frame = frame.rename(columns={"trade_date": "asof"})
    columns = [
        "asof", "ts_code", "bundle_id", "feature_set_hash", "available_time",
        "universe_pass", "selection_data_eligible", "training_data_eligible",
        "execution_data_eligible", "missing_required_selection", "missing_required_training",
        "missing_required_execution", "missing_optional", "execution_status", "feature_ready",
        "feature_rejection_reason",
        *required,
    ]
    return frame[columns].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


def build_flow_relative_feature_ledger(bundle: DataBundle, strategy: StrategyConfig) -> pd.DataFrame:
    """Extend the audited base features with causal flow/relative-strength fields."""
    out = build_feature_ledger(bundle, strategy)
    db = bundle.enrichments.get("daily_basic", pd.DataFrame()).copy()
    mf = bundle.enrichments.get("moneyflow", pd.DataFrame()).copy()
    idx = bundle.enrichments.get("index_daily", pd.DataFrame()).copy()
    if db.empty or idx.empty:
        raise ValueError("flow-relative strategy requires daily_basic and index_daily snapshots")

    for frame in (db, mf, idx):
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
        frame["asof"] = pd.to_datetime(frame["trade_date"], utc=True, errors="raise").dt.normalize()
    db = db.drop_duplicates(["ts_code", "asof"], keep="last").sort_values(["ts_code", "asof"])
    db["turnover_rate_f"] = pd.to_numeric(db["turnover_rate_f"], errors="coerce")
    turnover_window = _window(strategy, "turnover_ratio_20", 20)
    db["turnover_med20"] = db.groupby("ts_code")["turnover_rate_f"].transform(
        lambda s: s.rolling(turnover_window, min_periods=turnover_window).median()
    )
    db["turnover_ratio_20"] = db["turnover_rate_f"] / db["turnover_med20"]
    db["volume_ratio_basic"] = pd.to_numeric(db["volume_ratio"], errors="coerce")
    db["total_mv"] = pd.to_numeric(db["total_mv"], errors="coerce")

    mf = mf.drop_duplicates(["ts_code", "asof"], keep="last").sort_values(["ts_code", "asof"])
    for col in ("buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount",
                "buy_elg_amount", "sell_elg_amount", "net_mf_amount"):
        mf[col] = pd.to_numeric(mf[col], errors="coerce")
    mf["large_net"] = (
        mf["buy_lg_amount"] - mf["sell_lg_amount"]
        + mf["buy_elg_amount"] - mf["sell_elg_amount"]
    )
    amount = bundle.execution[["ts_code", "trade_date", "amount"]].copy()
    amount["ts_code"] = amount["ts_code"].astype(str).str.upper()
    amount["asof"] = pd.to_datetime(amount["trade_date"], utc=True, errors="raise").dt.normalize()
    amount["amount"] = pd.to_numeric(amount["amount"], errors="coerce")
    amount = amount.sort_values(["ts_code", "asof"])
    amount_window = _window(strategy, "liq20", 20)
    amount["amount_med20"] = amount.groupby("ts_code")["amount"].transform(
        lambda s: s.rolling(amount_window, min_periods=amount_window).median()
    )
    mf = mf.merge(amount[["ts_code", "asof", "amount_med20"]],
                  on=["ts_code", "asof"], how="left", validate="one_to_one")
    grouped = mf.groupby("ts_code", sort=False)
    mf["flow3_large_ratio"] = grouped["large_net"].transform(
        lambda s: s.rolling(3, min_periods=3).sum()
    ) / mf["amount_med20"]
    mf["flow5_large_ratio"] = grouped["large_net"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    ) / mf["amount_med20"]
    mf["flow5_total_ratio"] = grouped["net_mf_amount"].transform(
        lambda s: s.rolling(5, min_periods=5).sum()
    ) / mf["amount_med20"]
    mf["flow5_positive_days"] = grouped["large_net"].transform(
        lambda s: s.gt(0).rolling(5, min_periods=5).sum()
    )
    mf = mf[["ts_code", "asof", "flow3_large_ratio", "flow5_large_ratio",
             "flow5_total_ratio", "flow5_positive_days"]]

    # Use the broad SSE index as a common benchmark. Its close is EOD-visible.
    benchmark = idx[idx["ts_code"].eq("000001.SH")].copy()
    benchmark = benchmark.sort_values("asof").drop_duplicates("asof", keep="last")
    benchmark["index_ret20"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change(20)
    benchmark = benchmark[["asof", "index_ret20"]]

    out = out.merge(db[["ts_code", "asof", "turnover_rate_f", "turnover_ratio_20",
                        "volume_ratio_basic", "total_mv"]], on=["ts_code", "asof"],
                    how="left", validate="one_to_one")
    out = out.merge(mf, on=["ts_code", "asof"], how="left", validate="one_to_one")
    out = out.merge(benchmark, on="asof", how="left", validate="many_to_one")
    out["relative_strength20"] = out["ret20"] - out["index_ret20"]
    out["flow_divergence"] = np.where(
        out["ret1"].gt(0) & out["flow3_large_ratio"].lt(0), -1.0, 0.0
    )
    enriched = [
        "turnover_rate_f", "turnover_ratio_20", "volume_ratio_basic", "total_mv",
        "flow3_large_ratio", "flow5_large_ratio", "flow5_total_ratio",
        "flow5_positive_days", "index_ret20", "relative_strength20", "flow_divergence",
    ]
    required_enriched = ["turnover_rate_f", "turnover_ratio_20", "volume_ratio_basic",
                         "total_mv", "index_ret20", "relative_strength20"]
    numeric = out[required_enriched].apply(pd.to_numeric, errors="coerce")
    out["feature_ready"] = out["feature_ready"].astype(bool) & np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all(axis=1)
    out.loc[~out["feature_ready"] & out["feature_rejection_reason"].eq(""), "feature_rejection_reason"] = "ENRICHMENT_NOT_MATURE"
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "enriched": enriched},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base_columns = [column for column in out.columns if column not in enriched]
    # Keep the original contract columns first, followed by deterministic additions.
    return out[base_columns + enriched].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


def build_breadth_relative_feature_ledger(bundle: DataBundle, strategy: StrategyConfig) -> pd.DataFrame:
    """Build the pre-registered breadth/relative-strength rule feature set.

    This provider deliberately does not use moneyflow or any post-entry label.
    Broad-market breadth is a same-day state; the per-stock ranking feature
    blends that state with each stock's own relative strength and confirmation.
    """
    out = build_feature_ledger(bundle, strategy)
    db = bundle.enrichments.get("daily_basic", pd.DataFrame()).copy()
    idx = bundle.enrichments.get("index_daily", pd.DataFrame()).copy()
    if db.empty or idx.empty:
        raise ValueError("breadth-relative strategy requires daily_basic and index_daily snapshots")

    for frame in (db, idx):
        frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
        frame["asof"] = pd.to_datetime(frame["trade_date"], utc=True, errors="raise").dt.normalize()
    db = db.drop_duplicates(["ts_code", "asof"], keep="last").sort_values(["ts_code", "asof"])
    # These are EOD fact tables. update_time is batch-ingestion provenance;
    # the configured source policy makes the trade-date close availability explicit.
    source_policy = strategy.data_policy.get("source_availability", {})
    if source_policy.get("daily_basic_ts") != "eod_trade_date_close" or source_policy.get("index_daily_ts") != "eod_trade_date_close":
        raise ValueError("breadth-relative strategy requires explicit EOD source availability policy")
    db["daily_basic_available_time"] = db["asof"].map(session_close)
    db["turnover_rate_f"] = pd.to_numeric(db["turnover_rate_f"], errors="coerce")
    turnover_window = _window(strategy, "turnover_ratio_20", 20)
    db["turnover_med20"] = db.groupby("ts_code")["turnover_rate_f"].transform(
        lambda s: s.rolling(turnover_window, min_periods=turnover_window).median()
    )
    db["turnover_ratio_20"] = db["turnover_rate_f"] / db["turnover_med20"]
    db["turnover_stability20"] = (db["turnover_ratio_20"] - 1.0).abs()

    benchmark = idx[idx["ts_code"].eq("000001.SH")].copy()
    benchmark = benchmark.sort_values("asof").drop_duplicates("asof", keep="last")
    benchmark = benchmark.set_index("asof").reindex(pd.DatetimeIndex(bundle.calendar["session"])).rename_axis("asof").reset_index()
    benchmark["index_available_time"] = benchmark["asof"].map(session_close)
    benchmark["index_close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    index_window = _window(strategy, "index_ret10", 10)
    benchmark["index_ret10"] = benchmark["index_close"].pct_change(index_window, fill_method=None)
    benchmark = benchmark[["asof", "index_ret10", "index_available_time"]]

    out = out.merge(
        db[["ts_code", "asof", "turnover_rate_f", "turnover_ratio_20", "turnover_stability20", "daily_basic_available_time"]],
        on=["ts_code", "asof"], how="left", validate="one_to_one",
    )
    out = out.merge(benchmark, on="asof", how="left", validate="many_to_one")
    cutoff = out["asof"].map(session_close)
    enrichment_pit = (
        out["daily_basic_available_time"].le(cutoff)
        & out["index_available_time"].le(cutoff)
    )
    out["available_time"] = pd.concat(
        [pd.to_datetime(out["available_time"], utc=True),
         pd.to_datetime(out["daily_basic_available_time"], utc=True),
         pd.to_datetime(out["index_available_time"], utc=True)], axis=1
    ).max(axis=1)
    grouped = out.groupby("ts_code", sort=False)
    ret_window = _window(strategy, "ret10", 10)
    out["ret10"] = out["economic_close"] / grouped["economic_close"].shift(ret_window) - 1.0
    out["relative_strength10"] = out["ret10"] - out["index_ret10"]

    # Breadth denominator is the PIT-eligible universe with finite close/MA20.
    ma20_window = _window(strategy, "market_breadth_ma20", 20)
    out["ma20"] = out.groupby("ts_code", sort=False)["economic_close"].transform(
        lambda values: values.rolling(ma20_window, min_periods=ma20_window).mean()
    )
    eligible = (
        out["universe_pass"].astype(bool)
        & out["selection_data_eligible"].astype(bool)
        & pd.to_numeric(out["economic_close"], errors="coerce").notna()
        & pd.to_numeric(out["ma20"], errors="coerce").notna()
    )
    above_ma20 = pd.to_numeric(out["economic_close"], errors="coerce").ge(
        pd.to_numeric(out["ma20"], errors="coerce")
    )
    breadth_by_day = above_ma20.where(eligible).groupby(out["asof"]).mean()
    breadth = out["asof"].map(breadth_by_day)
    out["market_breadth_ma20"] = breadth
    breadth_lag5 = breadth_by_day.shift(_window(strategy, "market_breadth_delta5", 5))
    out["market_breadth_delta5"] = breadth - out["asof"].map(breadth_lag5)
    # State-conditioned strength: broad participation favors 10-session
    # relative strength; narrow participation favors fresh one-session proof.
    out["breadth_adjusted_strength10"] = (
        out["relative_strength10"] * out["market_breadth_ma20"]
        + out["ret1"] * (1.0 - out["market_breadth_ma20"])
    )

    required_enriched = [
        "turnover_rate_f", "turnover_ratio_20", "turnover_stability20",
        "index_ret10", "ret10", "relative_strength10", "market_breadth_ma20",
        "market_breadth_delta5", "breadth_adjusted_strength10", "ma20",
    ]
    numeric = out[required_enriched].apply(pd.to_numeric, errors="coerce")
    valid_turnover = numeric["turnover_rate_f"].ge(0)
    valid_breadth = numeric["market_breadth_ma20"].between(0.0, 1.0, inclusive="both")
    out["feature_ready"] = out["feature_ready"].astype(bool) & enrichment_pit & np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all(axis=1) & valid_turnover & valid_breadth
    out.loc[~out["feature_ready"] & out["feature_rejection_reason"].eq(""), "feature_rejection_reason"] = "ENRICHMENT_NOT_MATURE"
    enriched = required_enriched
    out["feature_set_hash"] = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "enriched": enriched},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base_columns = [column for column in out.columns if column not in enriched]
    return out[base_columns + enriched].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "REQUIRED_QUIET_FEATURES", "build_feature_ledger", "build_flow_relative_feature_ledger",
    "build_breadth_relative_feature_ledger",
]
