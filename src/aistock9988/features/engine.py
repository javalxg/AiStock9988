"""Calendar-session features for the registered rules strategies."""
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


def _validate_pipc_feature_contract(strategy: StrategyConfig) -> dict[str, float | int]:
    """Bind the registered PIPC providers to the only implementation we execute."""
    expected = {
        "turnover_rate_f": {
            "provider": "passthrough", "source": "daily_basic.turnover_rate_f",
            "window_sessions": 1,
        },
        "turnover_rate": {
            "provider": "passthrough", "source": "daily_basic.turnover_rate",
            "window_sessions": 1,
        },
        "turnover_raw": {
            "provider": "coalesce", "sources": ("turnover_rate_f", "turnover_rate"),
            "window_sessions": 1,
        },
        "turnover_5": {
            "provider": "rolling_median", "source": "turnover_raw", "window_sessions": 5,
        },
        "turnover_prior5": {
            "provider": "prior_rolling_median", "source": "turnover_raw",
            "window_sessions": 5, "lag_sessions": 5,
        },
        "turnover_impulse": {
            "provider": "ratio_minus_one", "numerator": "turnover_5",
            "denominator": "turnover_prior5",
        },
        "turnover_5_p75": {
            "provider": "cross_sectional_quantile", "source": "turnover_5",
            "quantile": 0.75,
        },
        "prev5_high": {
            "provider": "prior_rolling_max", "source": "economic_close",
            "window_sessions": 5,
        },
    }
    for feature_name, required in expected.items():
        actual = strategy.features.get(feature_name)
        if actual is None:
            raise ValueError(f"PIPC feature contract is missing {feature_name}")
        for key, expected_value in required.items():
            actual_value = actual.get(key)
            if key == "sources":
                actual_value = tuple(actual_value or ())
            if actual_value != expected_value:
                raise ValueError(
                    f"PIPC feature contract drift for {feature_name}.{key}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
    return {
        "current_window": int(expected["turnover_5"]["window_sessions"]),
        "prior_window": int(expected["turnover_prior5"]["window_sessions"]),
        "prior_lag": int(expected["turnover_prior5"]["lag_sessions"]),
        "quantile": float(expected["turnover_5_p75"]["quantile"]),
        "price_window": int(expected["prev5_high"]["window_sessions"]),
    }


def _merge_passthrough_enrichments(
    frame: pd.DataFrame,
    bundle: DataBundle,
    strategy: StrategyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge explicitly registered PIT enrichment columns into the feature panel."""
    requested: dict[str, dict[str, str]] = {}
    for feature_name, raw_spec in strategy.features.items():
        if not isinstance(raw_spec, dict) and not hasattr(raw_spec, "get"):
            continue
        if str(raw_spec.get("provider", "")) != "passthrough":
            continue
        source = str(raw_spec.get("source", ""))
        if "." not in source:
            continue
        enrichment_name, source_column = source.split(".", 1)
        requested.setdefault(enrichment_name, {})[feature_name] = source_column

    merged = frame
    feature_names: list[str] = []
    for enrichment_name, columns in sorted(requested.items()):
        source_name = f"{enrichment_name}_ts"
        policy = strategy.data_policy.get("source_availability", {})
        if policy.get(source_name) != "eod_trade_date_close":
            raise ValueError(
                f"passthrough enrichment {source_name} requires eod_trade_date_close availability"
            )
        if enrichment_name not in bundle.enrichments:
            raise ValueError(f"registered feature enrichment is absent from bundle: {enrichment_name}")
        source = bundle.enrichments[enrichment_name].copy()
        missing = sorted({"trade_date", "ts_code", *columns.values()} - set(source.columns))
        if missing:
            raise ValueError(f"{enrichment_name} passthrough columns missing: {missing}")
        source["trade_date"] = pd.to_datetime(source["trade_date"], errors="raise", utc=True).dt.normalize()
        source["ts_code"] = source["ts_code"].astype(str).str.upper()
        if source.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"{enrichment_name} contains duplicate security/session keys")
        selected = source[["trade_date", "ts_code", *columns.values()]].rename(
            columns={source_column: feature_name for feature_name, source_column in columns.items()}
        )
        merged = merged.merge(
            selected,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        feature_names.extend(columns)
    return merged, feature_names


def build_feature_ledger(bundle: DataBundle, strategy: StrategyConfig) -> pd.DataFrame:
    frame = bundle.execution.copy().sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    frame, passthrough_features = _merge_passthrough_enrichments(frame, bundle, strategy)
    close = pd.to_numeric(frame["economic_close"], errors="coerce")
    high = pd.to_numeric(frame["economic_high"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    valid = (
        frame["execution_status"].eq("TRADABLE")
        & frame["selection_data_eligible"].astype(bool)
        & amount.gt(0)
    )
    valid_close = close.where(valid)
    valid_high = high.where(valid)
    valid_amount = amount.where(valid)
    ma5_window = _window(strategy, "ma5", 5)
    ma60_window = _window(strategy, "ma60", 60)
    ret1_window = _window(strategy, "ret1", 1)
    ret20_window = _window(strategy, "ret20", 20)
    ret60_window = _window(strategy, "ret60", 60)
    dd20_window = _window(strategy, "dd20", 20)
    vol20_window = _window(strategy, "vol20", 20)
    liq20_window = _window(strategy, "liq20", 20)
    prev3_window = _window(strategy, "prev3_high", 3)
    frame["ma5"] = valid_close.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(ma5_window, min_periods=ma5_window).mean()
    )
    frame["ma60"] = valid_close.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(ma60_window, min_periods=ma60_window).mean()
    )
    frame["ret1"] = valid_close / valid_close.groupby(frame["ts_code"], sort=False).shift(ret1_window) - 1.0
    frame["ret20"] = valid_close / valid_close.groupby(frame["ts_code"], sort=False).shift(ret20_window) - 1.0
    frame["ret60"] = valid_close / valid_close.groupby(frame["ts_code"], sort=False).shift(ret60_window) - 1.0
    frame["dd20"] = valid_high / valid_high.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(dd20_window, min_periods=dd20_window).max()
    ) - 1.0
    frame["vol20"] = frame.groupby("ts_code", sort=False)["ret1"].transform(
        lambda values: values.rolling(vol20_window, min_periods=vol20_window).std()
    )
    frame["liq20"] = valid_amount.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.rolling(liq20_window, min_periods=liq20_window).median()
    )
    frame["volume_ratio_20"] = valid_amount / frame["liq20"]
    frame["prev3_high"] = valid_close.groupby(frame["ts_code"], sort=False).transform(
        lambda values: values.shift(1).rolling(prev3_window, min_periods=prev3_window).max()
    )
    frame["dist_ma60"] = valid_close / frame["ma60"] - 1.0
    frame["dist_ma60_abs_005"] = (frame["dist_ma60"] - 0.05).abs()
    # Equal-weight market return uses only tradable, PIT-eligible rows on each
    # session.  The lagged 20-session change is available after the session
    # close and is therefore safe for a next-session entry decision.
    sessions = pd.DatetimeIndex(pd.to_datetime(bundle.calendar["session"], utc=True)).normalize()
    # Reindex all market aggregates to the exchange calendar before shifting.
    # Otherwise a fully missing session disappears and ``shift(20)`` silently
    # becomes a 20-observed-row lookback, which is not a 20-session PIT feature.
    market_close = valid_close.groupby(frame["trade_date"], sort=True).mean().reindex(sessions)
    market_ret20 = market_close / market_close.shift(20) - 1.0
    universe_count = frame.groupby("trade_date", sort=True)["universe_pass"].sum().reindex(sessions)
    valid_count = valid_close.groupby(frame["trade_date"], sort=True).count().reindex(sessions)
    coverage = valid_count / universe_count.replace(0, np.nan)
    market_ret20 = market_ret20.where(coverage.ge(0.80))
    frame["mkt_ret_20d"] = frame["trade_date"].map(market_ret20)
    # Persist the denominator audit separately from the derived market return.
    # A NaN market return can mean either an immature lookback or a market-wide
    # data coverage failure; the runner must not treat the latter as a valid
    # event-driven abstention.
    frame["market_coverage"] = frame["trade_date"].map(coverage)
    frame["volume_ratio_abs_115"] = (frame["volume_ratio_20"] - 1.15).abs()
    vol_p85 = frame.groupby("trade_date", sort=True)["vol20"].quantile(0.85)
    frame["vol20_p85"] = frame["trade_date"].map(vol_p85)
    if strategy.strategy_id == "participation_impulse_preconfirmation_v1":
        pipc = _validate_pipc_feature_contract(strategy)
        turnover_f = pd.to_numeric(frame["turnover_rate_f"], errors="coerce")
        turnover = pd.to_numeric(frame["turnover_rate"], errors="coerce")
        frame["turnover_raw"] = turnover_f.where(turnover_f.notna(), turnover)
        valid_turnover = frame["turnover_raw"].where(
            frame["universe_pass"].astype(bool)
            & frame["selection_data_eligible"].astype(bool)
            & frame["turnover_raw"].gt(0)
        )
        turnover_grouped = valid_turnover.groupby(frame["ts_code"], sort=False)
        frame["turnover_5"] = turnover_grouped.transform(
            lambda values: values.rolling(
                pipc["current_window"], min_periods=pipc["current_window"]
            ).median()
        )
        frame["turnover_prior5"] = turnover_grouped.transform(
            lambda values: values.shift(pipc["prior_lag"]).rolling(
                pipc["prior_window"], min_periods=pipc["prior_window"]
            ).median()
        )
        frame["turnover_impulse"] = (
            frame["turnover_5"] / frame["turnover_prior5"] - 1.0
        ).where(frame["turnover_prior5"].gt(0))
        frame["prev5_high"] = valid_close.groupby(frame["ts_code"], sort=False).transform(
            lambda values: values.shift(1).rolling(
                pipc["price_window"], min_periods=pipc["price_window"]
            ).max()
        )
        turnover_level = frame["turnover_5"].where(
            frame["universe_pass"].astype(bool)
            & frame["selection_data_eligible"].astype(bool)
        )
        cap = turnover_level.groupby(frame["trade_date"], sort=True).quantile(pipc["quantile"])
        frame["turnover_5_p75"] = frame["trade_date"].map(cap)
        required = [
            "economic_close", "liq20", "turnover_raw", "turnover_5",
            "turnover_prior5", "turnover_impulse", "turnover_5_p75", "prev5_high",
        ]
    elif strategy.strategy_id.startswith("reset_weak_confirm_v"):
        required = [
            "economic_close", "ma5", "ret1", "ret20", "dd20", "vol20",
            "liq20", "volume_ratio_20", "prev3_high", "dist_ma60", "mkt_ret_20d",
            "volume_ratio_abs_115", "vol20_p85",
        ]
    else:
        required = list(REQUIRED_QUIET_FEATURES)
    feature_columns = list(dict.fromkeys([*required, *passthrough_features]))
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
    ranking_terms = {str(term["feature"]) for term in strategy.ranking["terms"]}
    ranking_passthrough = [
        feature for feature in passthrough_features if feature in ranking_terms
    ]
    if ranking_passthrough:
        ranking_numeric = frame[ranking_passthrough].apply(pd.to_numeric, errors="coerce")
        ranking_finite = np.isfinite(ranking_numeric.to_numpy(dtype=float)).all(axis=1)
    else:
        ranking_finite = np.ones(len(frame), dtype=bool)
    frame["ranking_feature_ready"] = frame["feature_ready"].astype(bool) & ranking_finite
    frame["ranking_feature_rejection_reason"] = np.select(
        [
            ~frame["feature_ready"].astype(bool),
            ~ranking_finite,
        ],
        [
            frame["feature_rejection_reason"],
            "RANKING_ENRICHMENT_MISSING",
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
    # Keep the pre-seal audit field expected by the V3 runner.  This strategy's
    # market-derived features are all EOD values from the same execution
    # snapshot, so their availability is the panel's available_time.
    frame["market_available_time"] = frame["available_time"]
    frame = frame.rename(columns={"trade_date": "asof"})
    columns = [
        "asof", "ts_code", "bundle_id", "feature_set_hash", "available_time", "market_available_time",
        "universe_pass", "selection_data_eligible", "training_data_eligible",
        "execution_data_eligible", "missing_required_selection", "missing_required_training",
        "missing_required_execution", "missing_optional", "execution_status", "feature_ready",
        "feature_rejection_reason", "ranking_feature_ready",
        "ranking_feature_rejection_reason", "market_coverage",
        *feature_columns,
    ]
    return frame[columns].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)

__all__ = ["REQUIRED_QUIET_FEATURES", "build_feature_ledger"]
