#!/usr/bin/env python3
"""Run the preregistered full-market F0=123 weekly ranker."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRanker

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import (
    build_data_bundle,
    load_source_max_dates,
    load_trading_calendar,
)
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.quantdb import readonly_connection
from aistock9988.features.f0_cross_section import prepare_f0_cross_sections
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.executable_path import (
    ExecutablePathLabelProfile,
    build_executable_path_labels,
)
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.time.session import session_close, session_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/f0_123_full_market_top5_v1.yaml"
DEFAULT_MODEL = ROOT / "configs/model_profiles/f0_123_full_market_weekly_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828" / "F0_123_FULL_MARKET_WEEKLY_TOP5_20260901"
)


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(repr)
    raw = pd.util.hash_pandas_object(normalized, index=False).to_numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _load_model_config(path: Path, feature_set: FeatureSet) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model profile must be a mapping")
    feature = raw["feature_set"]
    expected = {
        "id": feature_set.id,
        "expected_columns": len(feature_set.columns),
        "column_order_hash": feature_set.order_hash,
        "transform": "daily_cross_section_percentile_zscore",
        "minimum_non_null_features": 61,
        "maximum_training_rows_per_date": 1500,
        "missing_values": "xgboost_native_nan_no_imputation",
    }
    if any(feature.get(key) != value for key, value in expected.items()):
        raise ValueError("F0 full-market feature contract drift")
    if raw["evaluation"].get("parameter_sweep") is not False:
        raise ValueError("parameter_sweep must be false")
    if raw["selection"].get("factor_gate") != "none":
        raise ValueError("the preregistered baseline forbids factor gates")
    return raw


def _latest_session_on_or_before(calendar: pd.DataFrame, value: str) -> str:
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    eligible = sessions[sessions <= pd.Timestamp(value, tz="UTC")]
    if eligible.empty:
        raise ValueError(f"calendar has no session on or before {value}")
    return str(eligible[-1].date())


def _compile_plan(
    strategy: StrategyConfig,
    config: dict[str, Any],
    output: Path,
) -> tuple[Any, pd.DataFrame, dict[str, str]]:
    timeline = config["timeline"]
    dynamic = timeline.get("signal_end_policy") == "latest_common_source_session"
    if dynamic:
        signal_sources = tuple(timeline["signal_cutoff_sources"])
        execution_sources = tuple(timeline["execution_cutoff_sources"])
        cutoffs = load_source_max_dates(set(signal_sources) | set(execution_sources))
        raw_execution_end = min(cutoffs[name] for name in execution_sources)
        calendar = load_trading_calendar("2024-01-01", raw_execution_end)
        prediction_end = _latest_session_on_or_before(
            calendar, min(cutoffs[name] for name in signal_sources)
        )
        execution_end = _latest_session_on_or_before(calendar, raw_execution_end)
    else:
        prediction_end = timeline["prediction_end"]
        execution_end = timeline["execution_end"]
        cutoffs = load_source_max_dates({
            "stock_factor_pro_ts",
            "daily_basic_ts",
            "market_daily_ts",
            "adj_factor_ts",
            "stk_limit_ts",
        })
        calendar = load_trading_calendar("2024-01-01", execution_end)
    plan = compile_run_plan(
        strategy,
        RunRequest(
            timeline["prediction_start"],
            prediction_end,
            execution_end,
            str(output),
            "f0_123_full_market_weekly_top5_2026",
        ),
        calendar["session"],
        require_complete_horizon=bool(timeline.get("require_complete_horizon", True)),
    )
    if plan.signal_end != prediction_end:
        raise ValueError("weekly signal boundary differs from preregistration")
    return plan, calendar, cutoffs


def _load_label_prices(start: str, end: str) -> pd.DataFrame:
    """Load only the two dense tables needed to construct training labels."""
    with readonly_connection() as connection:
        frame = pd.read_sql_query(
            "SELECT d.trade_date, d.ts_code, d.open, d.close, a.adj_factor "
            "FROM market_daily_ts d JOIN adj_factor_ts a "
            "ON a.trade_date=d.trade_date AND a.ts_code=d.ts_code "
            "WHERE d.source='daily' AND d.trade_date BETWEEN %s AND %s "
            "ORDER BY d.trade_date, d.ts_code",
            connection,
            params=(start, end),
        )
    for column in ("open", "close", "adj_factor"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["economic_open"] = frame["open"] * frame["adj_factor"]
    frame["economic_close"] = frame["close"] * frame["adj_factor"]
    frame["execution_data_eligible"] = (
        np.isfinite(frame["economic_open"])
        & np.isfinite(frame["economic_close"])
        & frame["economic_open"].gt(0)
        & frame["economic_close"].gt(0)
    )
    return frame[[
        "trade_date", "ts_code", "economic_open", "economic_close", "execution_data_eligible"
    ]]


def _filter_f0_universe(
    panel: pd.DataFrame,
    strategy: StrategyConfig,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the same listing/ST/suffix universe before cross-sectional ranks."""
    start = str(pd.to_datetime(panel["event_time"], utc=True).min().date())
    end = str(pd.to_datetime(panel["event_time"], utc=True).max().date())
    with readonly_connection() as connection:
        master = pd.read_sql_query(
            "SELECT ts_code, list_date, delist_date FROM stock_basic_ts ORDER BY ts_code",
            connection,
        )
        st = pd.read_sql_query(
            "SELECT ts_code, trade_date FROM stock_st_ts "
            "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date, ts_code",
            connection,
            params=(start, end),
        )
    source = panel.copy()
    source["ts_code"] = source["ts_code"].astype(str).str.upper()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True).dt.normalize()
    master["ts_code"] = master["ts_code"].astype(str).str.upper()
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce", utc=True).dt.normalize()
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce", utc=True).dt.normalize()
    source = source.merge(master, on="ts_code", how="left", validate="many_to_one")

    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize().unique().sort_values()
    event_ordinal = pd.Series(np.arange(len(sessions), dtype="int64"), index=sessions)
    signal_ordinals = source["event_time"].map(event_ordinal)
    list_ordinals = source["list_date"].map(
        lambda value: int(sessions.searchsorted(value, side="left")) if pd.notna(value) else len(sessions)
    )
    listed = source["list_date"].notna() & source["list_date"].le(source["event_time"])
    not_delisted = source["delist_date"].isna() | source["event_time"].lt(source["delist_date"])
    listing_age = signal_ordinals.to_numpy(dtype="int64") - list_ordinals.to_numpy(dtype="int64") + 1
    old_enough = listing_age >= int(strategy.universe.get("min_listed_sessions", 0))
    excluded_suffixes = tuple(
        str(value).upper() for value in strategy.universe.get("exclude_suffixes", ())
    )
    suffix_pass = ~source["ts_code"].map(
        lambda code: any(code.endswith(suffix) for suffix in excluded_suffixes)
    )
    st["ts_code"] = st["ts_code"].astype(str).str.upper()
    st["trade_date"] = pd.to_datetime(st["trade_date"], utc=True).dt.normalize()
    st_keys = pd.MultiIndex.from_frame(st[["trade_date", "ts_code"]])
    source_keys = pd.MultiIndex.from_frame(
        source[["event_time", "ts_code"]].rename(columns={"event_time": "trade_date"})
    )
    non_st = ~source_keys.isin(st_keys)
    keep = listed & not_delisted & old_enough & suffix_pass & non_st
    audit = {
        "source_rows": int(len(source)),
        "eligible_rows": int(keep.sum()),
        "not_listed_rows": int((~listed).sum()),
        "delisted_rows": int((~not_delisted).sum()),
        "listing_age_rows": int((~old_enough).sum()),
        "excluded_suffix_rows": int((~suffix_pass).sum()),
        "pit_st_rows": int((~non_st).sum()),
    }
    return source.loc[keep, panel.columns].copy(), audit


def _sample_training_rows(
    prepared: pd.DataFrame,
    signal_days: set[pd.Timestamp],
    maximum_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for day, group in prepared.groupby("event_time", sort=True):
        if day in signal_days:
            prediction_parts.append(group)
        training_parts.append(
            group.sample(n=maximum_rows, random_state=42)
            if len(group) > maximum_rows
            else group
        )
    training = pd.concat(training_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    prediction = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    return training, prediction


def _lookup(
    indexed: pd.DataFrame,
    dates: pd.Series,
    codes: pd.Series,
    column: str,
) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates, utc=True).dt.normalize(), codes.astype(str)],
        names=["trade_date", "ts_code"],
    )
    return pd.to_numeric(indexed[column].reindex(keys), errors="coerce").to_numpy(dtype=float)


def _build_path_labels(
    keys: pd.DataFrame,
    execution: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    entry_delay: int,
    horizon: int,
    stop_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = keys[["event_time", "ts_code"]].drop_duplicates().copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True).dt.normalize()
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True)).normalize().unique().sort_values()
    positions = {day: index for index, day in enumerate(sessions)}
    source["signal_index"] = source["event_time"].map(positions)
    source = source.dropna(subset=["signal_index"]).copy()
    source["signal_index"] = source["signal_index"].astype(int)
    complete_horizon = source["signal_index"] + entry_delay + horizon < len(sessions)
    source = source.loc[complete_horizon].copy()
    source["entry_date"] = source["signal_index"].map(lambda index: sessions[index + entry_delay])
    source["exit_date"] = source["signal_index"].map(
        lambda index: sessions[index + entry_delay + horizon]
    )

    prices = execution.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], utc=True).dt.normalize()
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate label keys")
    indexed = prices.set_index(["trade_date", "ts_code"]).sort_index()
    entry_open = _lookup(indexed, source["entry_date"], source["ts_code"], "economic_open")
    exit_open = _lookup(indexed, source["exit_date"], source["ts_code"], "economic_open")
    entry_ok = _lookup(
        indexed, source["entry_date"], source["ts_code"], "execution_data_eligible"
    ).astype(bool)
    exit_ok = _lookup(
        indexed, source["exit_date"], source["ts_code"], "execution_data_eligible"
    ).astype(bool)
    valid = entry_ok & exit_ok & np.isfinite(entry_open) & np.isfinite(exit_open)
    valid &= (entry_open > 0) & (exit_open > 0)
    stop_hit = np.zeros(len(source), dtype=bool)
    path_complete = np.ones(len(source), dtype=bool)
    for offset in range(entry_delay, entry_delay + horizon):
        path_dates = source["signal_index"].map(lambda index: sessions[index + offset])
        close = _lookup(indexed, path_dates, source["ts_code"], "economic_close")
        eligible = _lookup(
            indexed, path_dates, source["ts_code"], "execution_data_eligible"
        ).astype(bool)
        complete = eligible & np.isfinite(close) & (close > 0)
        path_complete &= complete
        stop_hit |= complete & (close / entry_open - 1.0 <= stop_pct)
    valid &= path_complete

    labels = source.loc[valid, ["event_time", "ts_code", "exit_date"]].copy()
    endpoint = np.clip(exit_open[valid] / entry_open[valid] - 1.0, -0.5, 0.5)
    labels["label_return"] = np.where(stop_hit[valid], stop_pct, endpoint)
    labels["available_time"] = labels["exit_date"].map(session_open)
    labels = labels.drop(columns="exit_date").sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    return labels, {
        "requested_rows": int(len(keys[["event_time", "ts_code"]].drop_duplicates())),
        "calendar_horizon_rows": int(len(source)),
        "label_rows": int(len(labels)),
        "excluded_price_or_path_rows": int(len(source) - len(labels)),
        "path_stop_rows": int(stop_hit[valid].sum()),
        "mean_label": float(labels["label_return"].mean()),
        "positive_label_rate": float(labels["label_return"].gt(0).mean()),
    }


def _fit_model(
    training: pd.DataFrame,
    feature_set: FeatureSet,
    params: dict[str, Any],
    model_id: str,
) -> tuple[XGBRanker, dict[str, Any]]:
    counts = training.groupby("event_time", sort=True).size()
    valid_days = counts[counts.ge(50)].index
    frame = training[training["event_time"].isin(valid_days)].sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    )
    if frame["event_time"].nunique() < 60:
        raise ValueError(f"{model_id} has fewer than 60 valid training dates")
    X = frame[list(feature_set.columns)]
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{model_id} contains infinite features")
    y = frame["label_return"].to_numpy(dtype=float)
    qid = pd.factorize(frame["event_time"], sort=True)[0]
    model = XGBRanker(**params)
    model.fit(X, y, qid=qid)
    with tempfile.TemporaryDirectory(prefix=f"aistock-{model_id}-") as temp:
        model_path = Path(temp) / "model.json"
        model.save_model(model_path)
        model_hash = _sha(model_path)
    return model, {
        "model_id": model_id,
        "training_rows": int(len(frame)),
        "training_groups": int(frame["event_time"].nunique()),
        "training_start": str(frame["event_time"].min().date()),
        "training_end": str(frame["event_time"].max().date()),
        "maximum_label_available_time": frame["label_available_time"].max().isoformat(),
        "model_sha256": model_hash,
    }


def _walk_forward_predictions(
    training_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: FeatureSet,
    config: dict[str, Any],
    signal_days: list[pd.Timestamp],
    *,
    model_id_prefix: str = "f0_123_full_market",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    training = training_features.merge(
        labels.rename(columns={"available_time": "label_available_time"}),
        on=["event_time", "ts_code"],
        validate="one_to_one",
    )
    params = dict(config["model"])
    params.pop("family")
    models: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    model: XGBRanker | None = None
    active_month: str | None = None
    for day in signal_days:
        month = day.strftime("%Y-%m")
        if month != active_month:
            cutoff_time = session_close(day)
            window_start = day - pd.DateOffset(months=int(config["timeline"]["train_window_months"]))
            fit_rows = training[
                training["event_time"].gt(window_start)
                & training["event_time"].lt(day)
                & pd.to_datetime(training["available_time"], utc=True).le(cutoff_time)
                & pd.to_datetime(training["label_available_time"], utc=True).le(cutoff_time)
            ].copy()
            model_id = f"{model_id_prefix}_{day.strftime('%Y%m%d')}"
            model, metadata = _fit_model(fit_rows, feature_set, params, model_id)
            metadata.update({
                "model_signal_date": str(day.date()),
                "training_window_start_exclusive": str(window_start.date()),
            })
            models.append(metadata)
            active_month = month
        assert model is not None
        prediction = prediction_features[prediction_features["event_time"].eq(day)].copy()
        if prediction.empty:
            raise ValueError(f"weekly prediction cross-section is empty: {day.date()}")
        if pd.to_datetime(prediction["available_time"], utc=True).gt(session_close(day)).any():
            raise AssertionError(f"weekly prediction contains unavailable F0: {day.date()}")
        prediction["model_score"] = model.predict(prediction[list(feature_set.columns)])
        prediction["model_id"] = models[-1]["model_id"]
        predictions.append(prediction)
        models[-1].setdefault("prediction_sessions", 0)
        models[-1].setdefault("prediction_rows", 0)
        models[-1]["prediction_sessions"] += 1
        models[-1]["prediction_rows"] += int(len(prediction))
    return pd.concat(predictions, ignore_index=True), models


def _build_ledgers(
    predictions: pd.DataFrame,
    signal_days: list[pd.Timestamp],
    strategy: StrategyConfig,
    *,
    policy_id: str = "full_market_f0_123_top20_to_top5",
) -> dict[str, pd.DataFrame]:
    ranked = predictions.sort_values(
        ["event_time", "model_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["candidate_rank"] = ranked.groupby("event_time", sort=False).cumcount() + 1
    ranked = ranked[ranked["candidate_rank"].le(int(strategy.portfolio["candidate_view_size"]))].copy()
    ranked = ranked.rename(columns={"event_time": "asof"})
    ranked["candidate_status"] = "IN_VIEW"
    snapshots: dict[pd.Timestamp, str] = {}
    for day, group in ranked.groupby("asof", sort=True):
        payload = "|".join(
            f"{row.ts_code}:{int(row.candidate_rank)}" for row in group.itertuples()
        )
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    ranked["candidate_snapshot_id"] = ranked["asof"].map(snapshots)
    policy_hash = hashlib.sha256(
        f"{policy_id}|{strategy.config_hash}".encode()
    ).hexdigest()
    selection = pd.DataFrame({"asof": signal_days})
    selection["candidate_snapshot_id"] = selection["asof"].map(snapshots)
    if selection["candidate_snapshot_id"].isna().any():
        raise ValueError("a weekly signal has no candidate snapshot")
    selection["decision_id"] = selection.apply(
        lambda row: hashlib.sha256(
            f"{policy_hash}|{row['asof'].date()}|{row['candidate_snapshot_id']}".encode()
        ).hexdigest(),
        axis=1,
    )
    selection["desired_entries"] = int(strategy.portfolio["entries_per_decision"])
    selection["target_weight_each"] = float(strategy.portfolio["sizing"]["value"])
    selection["primary_rank_end"] = int(strategy.portfolio["entries_per_decision"])
    selection["replacement_rank_end"] = int(strategy.portfolio["candidate_view_size"])
    selection["policy_id"] = policy_id
    selection["policy_hash"] = policy_hash
    selection["context_hash"] = selection["asof"].map(
        lambda day: hashlib.sha256(f"{day.date()}|{policy_hash}".encode()).hexdigest()
    )
    return {"candidate": ranked, "selection": selection}


def _run_portfolios(
    ledgers: dict[str, pd.DataFrame],
    bundle: Any,
    strategy: StrategyConfig,
    execution_sessions: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    metrics: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=execution_sessions,
            scenario_name=scenario,
        )
        summary = summarize(
            result["nav"],
            result["fills"],
            initial_cash=float(strategy.execution["initial_cash"]),
            positions=result["positions"],
            corporate_actions=result["corporate_actions"],
        )
        summary.update({
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(result["execution_decisions"]["chosen"].sum()),
            "open_positions_at_end": int(len(result["open_positions"])),
            "active_signal_days": int(ledgers["candidate"]["asof"].nunique()),
        })
        metrics[scenario] = summary
        results[scenario] = result
    return metrics, results


def _acceptance(portfolio: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["evaluation"]
    cap1 = {"base": rules["cap1_base_return"], "stress": rules["cap1_stress_return"]}
    scenarios: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        item = portfolio[scenario]
        tests = {
            "pf_minimum": (item["portfolio_profit_factor"] or 0.0) >= float(rules["portfolio_profit_factor_min"]),
            "maxdd_limit": abs(item["max_drawdown"]) <= float(rules["max_drawdown_abs_max"]),
            "excluding_best_week_positive": item["return_excluding_best_week"] > 0.0,
            "excluding_top3_positive": item["return_excluding_top3_profit"] > 0.0,
            "minimum_closed_trades": item["trade_count"] >= int(rules["minimum_closed_trades"]),
            "position_cap": item["max_open_positions"] <= int(rules["maximum_positions"]),
        }
        if bool(rules.get("paired_control_required", True)):
            tests["higher_than_cap1"] = item["total_return"] > float(cap1[scenario])
        if "trade_win_rate_min" in rules:
            tests["trade_win_rate_minimum"] = (
                item["trade_win_rate"] is not None
                and item["trade_win_rate"] >= float(rules["trade_win_rate_min"])
            )
        scenarios[scenario] = {"passed": all(tests.values()), "tests": tests}
    return {"passed": all(row["passed"] for row in scenarios.values()), "scenarios": scenarios}


def _diagnostics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: FeatureSet,
) -> dict[str, Any]:
    ranked = predictions.sort_values(
        ["event_time", "model_score", "ts_code"], ascending=[True, False, True], kind="mergesort"
    ).copy()
    ranked["rank"] = ranked.groupby("event_time", sort=False).cumcount() + 1
    top20 = ranked[ranked["rank"].le(20)].merge(
        labels[["event_time", "ts_code", "label_return"]],
        on=["event_time", "ts_code"], how="inner", validate="one_to_one",
    )
    buckets = []
    for name, lower, upper in (("top1", 1, 1), ("top2", 1, 2), ("rank3_5", 3, 5), ("rank6_10", 6, 10), ("rank11_20", 11, 20)):
        group = top20[top20["rank"].between(lower, upper)]
        buckets.append({
            "bucket": name,
            "rows": int(len(group)),
            "mean_label": float(group["label_return"].mean()),
            "positive_rate": float(group["label_return"].gt(0).mean()),
            "ge_10pct_rate": float(group["label_return"].ge(0.10).mean()),
        })
    winners = top20[top20["label_return"].gt(0)]
    losers = top20[top20["label_return"].le(0)]
    contrasts = []
    for column in feature_set.columns:
        winner_mean = float(pd.to_numeric(winners[column], errors="coerce").mean())
        loser_mean = float(pd.to_numeric(losers[column], errors="coerce").mean())
        contrasts.append({
            "feature": column,
            "winner_mean": winner_mean,
            "loser_mean": loser_mean,
            "difference": winner_mean - loser_mean,
        })
    contrasts.sort(key=lambda row: abs(row["difference"]), reverse=True)
    return {
        "top20_labeled_rows": int(len(top20)),
        "rank_buckets": buckets,
        "winner_loser_largest_absolute_differences": contrasts[:25],
    }


def _verification(
    models: list[dict[str, Any]],
    signal_days: list[pd.Timestamp],
    prediction_sessions: int,
    portfolio: dict[str, dict[str, Any]],
    results: dict[str, dict[str, pd.DataFrame]],
    execution_end: str,
) -> dict[str, Any]:
    expected_months = {day.strftime("%Y-%m") for day in signal_days}
    checks: dict[str, bool] = {
        "one_model_per_signal_month": len(models) == len(expected_months),
        "no_skipped_signal_week": prediction_sessions == len(signal_days),
        "all_model_labels_mature": all(
            pd.Timestamp(row["maximum_label_available_time"]) <= session_close(row["model_signal_date"])
            for row in models
        ),
    }
    for scenario in ("base", "stress"):
        result = results[scenario]
        nav = result["nav"]
        checks[f"{scenario}_nav_identity"] = bool(
            np.allclose(nav["cash"] + nav["market_value"], nav["nav"], rtol=0, atol=1e-8)
        )
        checks[f"{scenario}_cash_nonnegative"] = bool((nav["cash"] >= -1e-8).all())
        checks[f"{scenario}_position_cap"] = portfolio[scenario]["max_open_positions"] <= 5
        checks[f"{scenario}_execution_end"] = str(nav["trade_date"].max().date()) == execution_end
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def _write_result(
    output: Path,
    sample: dict[str, Any],
    portfolio: dict[str, dict[str, Any]],
    acceptance: dict[str, Any],
    *,
    label_name: str,
    signal_start: str,
    signal_end: str,
    execution_end: str,
) -> None:
    lines = [
        "# F0-123 Full-Market Weekly Top5", "",
        "## Contract", "",
        "- Frozen F0=123 only; daily cross-sectional percentile/z-score; at least 61 values per row.",
        "- Daily broad-market training (deterministic cap 1500/date), monthly grouped XGBRanker, weekly full-market scoring.",
        f"- Training label: `{label_name}`; Top20 to Top5, H10, maximum five positions and canonical Base/Stress execution.",
        "- No auxiliary data, factor gate, feature selection, threshold scan, fallback, or business-data cache.", "",
        "## Sample", "",
        f"- 2026 signal sessions: `{signal_start}` through `{signal_end}`; execution through database cutoff `{execution_end}`.",
        f"- Prepared training rows: `{sample['training_feature_rows']}` across `{sample['training_feature_sessions']}` sessions.",
        f"- Weekly full-market prediction rows: `{sample['prediction_feature_rows']}` across `{sample['signal_sessions']}` signal weeks.",
        f"- Mature executable labels: `{sample['label_audit']['label_rows']}`; stop labels: `{sample['label_audit'].get('stop_rows', sample['label_audit'].get('path_stop_rows'))}`.", "",
        "## Portfolio", "",
        "| Cost | Return | Win rate | PF | MaxDD | Ex-best | Ex-top3 | Trades | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in ("base", "stress"):
        item = portfolio[scenario]
        lines.append(
            f"| {scenario} | {item['total_return']:+.2%} | {item['trade_win_rate']:.2%} | {item['portfolio_profit_factor']:.3f} | "
            f"{item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | "
            f"{item['return_excluding_top3_profit']:+.2%} | {item['trade_count']} | "
            f"{acceptance['scenarios'][scenario]['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(
        "The F0=123 full-market baseline advances unchanged to forward registration."
        if acceptance["passed"]
        else "The exact baseline is rejected. Continue only with the preregistered 123-factor winner/loser and rank-bucket diagnosis; auxiliary data remains unauthorized."
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    config = _load_model_config(args.model_profile, feature_set)
    plan, calendar, cutoffs = _compile_plan(strategy, config, output)
    if min(cutoffs["stock_factor_pro_ts"], cutoffs["daily_basic_ts"]) < plan.signal_end:
        raise ValueError(f"F0 sources do not cover the final weekly signal: {cutoffs}")

    signal_days = list(pd.to_datetime(plan.signal_sessions, utc=True))
    signal_day_set = set(signal_days)
    print("phase=f0_load", flush=True)
    raw_f0 = load_f0_panel(config["timeline"]["train_start"], plan.signal_end)
    raw_f0, universe_audit = _filter_f0_universe(raw_f0, strategy, calendar)
    print(f"phase=f0_prepare source_rows={len(raw_f0)}", flush=True)
    prepared, prep_audit = prepare_f0_cross_sections(
        raw_f0,
        feature_set,
        minimum_non_null_features=int(config["feature_set"]["minimum_non_null_features"]),
        maximum_rows_per_date=int(config["feature_set"]["maximum_training_rows_per_date"]),
        sample_seed=42,
        uncapped_dates=signal_days,
    )
    del raw_f0
    gc.collect()
    training_features, prediction_features = _sample_training_rows(
        prepared,
        signal_day_set,
        int(config["feature_set"]["maximum_training_rows_per_date"]),
    )
    del prepared
    gc.collect()

    label_keys = pd.concat([
        training_features[["event_time", "ts_code"]],
        prediction_features[["event_time", "ts_code"]],
    ], ignore_index=True).drop_duplicates()
    label_name = str(config["label"]["profile"])
    bundle = None
    if label_name == "label.executable_path_open_open_t10_base.v1":
        print("phase=execution_bundle_load_for_executable_labels", flush=True)
        bundle = build_data_bundle(plan, strategy, output)
        base_costs = strategy.execution["cost_scenarios"]["base"]
        labels, label_audit = build_executable_path_labels(
            label_keys,
            bundle.execution,
            pd.DatetimeIndex(calendar["session"]),
            profile=ExecutablePathLabelProfile(
                entry_delay_sessions=int(strategy.decision["entry_delay_sessions"]),
                hold_sessions_from_fill=int(strategy.execution["hold_sessions_from_fill"]),
                stop_threshold_pct=float(strategy.execution["stop"]["threshold_pct"]),
                buy_slippage=float(base_costs["slippage_each_side"]),
                sell_slippage=float(base_costs["slippage_each_side"]),
                buy_commission=float(base_costs["buy_commission"]),
                sell_commission=float(base_costs["sell_commission"]),
                stamp_duty=float(base_costs["stamp_duty"]),
            ),
        )
    elif label_name == "label.path_stop_open_open_t10.v1":
        print("phase=label_price_load", flush=True)
        label_prices = _load_label_prices(config["timeline"]["train_start"], plan.execution_end)
        labels, label_audit = _build_path_labels(
            label_keys,
            label_prices,
            pd.DatetimeIndex(calendar["session"]),
            entry_delay=int(config["label"]["signal_to_entry_sessions"]),
            horizon=int(config["label"]["entry_to_exit_sessions"]),
            stop_pct=float(config["label"]["path_stop_pct"]),
        )
        del label_prices
    else:
        raise ValueError(f"unsupported F0 training label profile: {label_name}")
    del label_keys
    gc.collect()
    print(
        f"phase=model_train training_rows={len(training_features)} signal_weeks={len(signal_days)}",
        flush=True,
    )
    predictions, models = _walk_forward_predictions(
        training_features,
        prediction_features,
        labels,
        feature_set,
        config,
        signal_days,
        model_id_prefix=str(config["identity"]["model_profile_id"]),
    )
    completed_prediction_sessions = int(predictions["event_time"].nunique())
    print("phase=model_train_complete", flush=True)
    ledgers = _build_ledgers(
        predictions,
        signal_days,
        strategy,
        policy_id=f"{config['identity']['model_profile_id']}_top20_to_top5",
    )
    diagnostics = _diagnostics(predictions, labels, feature_set)
    training_key_hash = _frame_hash(training_features[["event_time", "ts_code"]])
    prediction_key_hash = _frame_hash(prediction_features[["event_time", "ts_code"]])
    training_rows = int(len(training_features))
    training_sessions = int(training_features["event_time"].nunique())
    prediction_rows = int(len(prediction_features))
    del training_features, prediction_features, predictions, labels
    gc.collect()

    if bundle is None:
        print("phase=execution_bundle_load", flush=True)
        bundle = build_data_bundle(plan, strategy, output)
    print("phase=backtest", flush=True)
    portfolio, results = _run_portfolios(ledgers, bundle, strategy, plan.execution_sessions)
    acceptance = _acceptance(portfolio, config)
    verification = _verification(
        models,
        signal_days,
        completed_prediction_sessions,
        portfolio,
        results,
        plan.execution_end,
    )
    if not verification["passed"]:
        raise AssertionError(f"verification failed: {verification['failed']}")

    sample = {
        "preprocessing": asdict(prep_audit),
        "universe_filter": universe_audit,
        "training_feature_rows": training_rows,
        "training_feature_sessions": training_sessions,
        "prediction_feature_rows": prediction_rows,
        "signal_sessions": int(len(signal_days)),
        "prediction_rows": prediction_rows,
        "label_audit": label_audit,
        "training_features_sha256": training_key_hash,
        "prediction_features_sha256": prediction_key_hash,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir()
    (output / "manifests").mkdir()
    shutil.copyfile(args.strategy, output / "configs/strategy.yaml")
    shutil.copyfile(args.model_profile, output / "configs/model_profile.yaml")
    _write_json(output / "RUN_STATUS.json", {
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "model_profile_id": config["identity"]["model_profile_id"],
        "verification_passed": verification["passed"],
        "acceptance_passed": acceptance["passed"],
        "credentials_persisted": False,
        "business_data_persisted": False,
    })
    _write_json(output / "plan.json", {
        "run_plan": plan.to_dict(),
        "source_cutoffs": cutoffs,
        "feature_set_id": feature_set.id,
        "feature_order_hash": feature_set.order_hash,
        "bundle_manifest": bundle.manifest,
    })
    _write_json(output / "sample_audit.json", sample)
    _write_json(output / "model_manifest.json", models)
    _write_json(output / "portfolio_metrics.json", portfolio)
    _write_json(output / "acceptance.json", acceptance)
    _write_json(output / "verification.json", verification)
    _write_json(output / "f0_diagnostics.json", diagnostics)
    _write_result(
        output,
        sample,
        portfolio,
        acceptance,
        label_name=label_name,
        signal_start=plan.signal_start,
        signal_end=plan.signal_end,
        execution_end=plan.execution_end,
    )
    code_paths = [
        ROOT / "src/aistock9988/data/q70_source.py",
        ROOT / "src/aistock9988/features/f0_cross_section.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        Path(__file__).resolve(),
    ]
    _write_json(output / "manifests/code_manifest.json", {
        str(path.relative_to(ROOT)): _sha(path) for path in code_paths
    })
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests/artifact_manifest.json", artifacts)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"run_complete={run(args)}")


if __name__ == "__main__":
    main()
