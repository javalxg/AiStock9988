#!/usr/bin/env python3
"""Run a preregistered F0-123 ranker inside one fixed-rule Stage1 pool."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRanker

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.quantdb import readonly_connection
from aistock9988.features.engine import build_feature_ledger
from aistock9988.features.registry import FeatureSet
from aistock9988.labeling.executable_path import (
    ExecutablePathLabelProfile,
    build_executable_path_labels,
)
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.selection.pipeline import build_rule_ledgers
from aistock9988.time.session import session_close, session_open


ROOT = Path(__file__).resolve().parents[1]
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
    profile_id = raw["identity"]["model_profile_id"]
    if profile_id not in {
        "f0_123_conditional_reset_v1",
        "f0_123_cap1_executable_v1",
    }:
        raise ValueError("unexpected model profile")
    configured = raw["feature_set"]
    if (
        configured["id"] != feature_set.id
        or int(configured["expected_columns"]) != len(feature_set.columns)
        or configured["column_order_hash"] != feature_set.order_hash
        or configured.get("missing_values") != "xgboost_native_nan_no_imputation"
    ):
        raise ValueError("F0-123 feature manifest drift")
    if raw["evaluation"].get("parameter_sweep") is not False:
        raise ValueError("parameter_sweep must be false")
    timeline = raw["timeline"]
    if profile_id == "f0_123_conditional_reset_v1":
        timeline.setdefault("train_end", "2025-12-31")
        timeline.setdefault("training_execution_end", "2026-01-19")
    required_timeline = {
        "train_start",
        "train_end",
        "training_execution_end",
        "prediction_start",
        "prediction_end",
        "execution_end",
        "train_window_months",
        "retrain",
        "prediction",
    }
    if set(timeline) != required_timeline:
        raise ValueError("model timeline fields differ from the registered contract")
    if (
        int(timeline["train_window_months"]) != 12
        or timeline["retrain"] != "monthly_previous_month_end"
        or timeline["prediction"] != "daily"
    ):
        raise ValueError("model timeline semantics differ from preregistration")
    if profile_id == "f0_123_cap1_executable_v1" and timeline != {
        "train_start": "2025-01-02",
        "train_end": "2025-12-31",
        "training_execution_end": "2026-01-19",
        "prediction_start": "2026-01-05",
        "prediction_end": "2026-08-28",
        "execution_end": "2026-09-01",
        "train_window_months": 12,
        "retrain": "monthly_previous_month_end",
        "prediction": "daily",
    }:
        raise ValueError("CAP1 executable model timeline differs from preregistration")
    label_profile = raw["label"].get("profile")
    if label_profile not in {
        "label.endpoint_open_open_t10.v1",
        "label.executable_path_open_open_t10_base.v1",
        "label.executable_path_open_open_t10_base.trailing_previous_close.v1",
    }:
        raise ValueError("unsupported conditional ranker label profile")
    if profile_id == "f0_123_cap1_executable_v1" and raw["label"] != {
        "profile": "label.executable_path_open_open_t10_base.trailing_previous_close.v1",
        "signal_to_entry_sessions": 1,
        "entry_to_time_exit_sessions": 10,
        "stop_threshold_pct": -0.08,
        "stop_mode": "trailing_from_last_close",
        "stop_execute": "next_sellable_open",
        "clamp_return": False,
    }:
        raise ValueError("CAP1 executable label differs from preregistration")
    return raw


def _source_cutoffs() -> dict[str, str]:
    tables = (
        "stock_factor_pro_ts", "daily_basic_ts", "market_daily_ts",
        "adj_factor_ts", "stk_limit_ts",
    )
    out: dict[str, str] = {}
    with readonly_connection() as connection:
        for table in tables:
            value = pd.read_sql_query(
                f"SELECT trade_date FROM {table} ORDER BY trade_date DESC LIMIT 1",
                connection,
            ).iloc[0, 0]
            out[table] = str(pd.Timestamp(value).date())
    return out


def _period_plan(
    strategy: StrategyConfig,
    *,
    signal_start: str,
    signal_end: str,
    execution_end: str,
    output: Path,
    name: str,
    require_complete_horizon: bool = True,
) -> tuple[Any, pd.DataFrame]:
    calendar_start = str((pd.Timestamp(signal_start) - pd.Timedelta(days=500)).date())
    calendar = load_trading_calendar(calendar_start, execution_end)
    plan = compile_run_plan(
        strategy,
        RunRequest(signal_start, signal_end, execution_end, str(output), name),
        calendar["session"],
        require_complete_horizon=require_complete_horizon,
    )
    return plan, calendar


def _build_labels(
    pool: pd.DataFrame,
    execution: pd.DataFrame,
    calendar: pd.DataFrame,
    label_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_profile = str(label_config["profile"])
    if label_profile.startswith("label.executable_path_open_open_t10_base"):
        profile = ExecutablePathLabelProfile(
            id=label_profile,
            stop_threshold_pct=float(label_config["stop_threshold_pct"]),
            stop_mode=str(label_config.get("stop_mode", "from_entry")),
        )
        labels, audit = build_executable_path_labels(
            pool[["asof", "ts_code"]].rename(columns={"asof": "event_time"}),
            execution,
            pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)),
            profile=profile,
        )
        return labels[
            [
                "event_time",
                "ts_code",
                "label_return",
                "economic_return",
                "trigger_type",
                "exit_date",
                "available_time",
            ]
        ], audit
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    session_index = {day: index for index, day in enumerate(sessions)}
    source = pool[["asof", "ts_code"]].copy()
    source["signal_index"] = source["asof"].map(session_index)
    if source["signal_index"].isna().any():
        raise ValueError("Stage-1 signal absent from exchange calendar")
    source["signal_index"] = source["signal_index"].astype(int)
    if (source["signal_index"] + 11 >= len(sessions)).any():
        raise ValueError("label horizon exceeds loaded exchange calendar")
    source["entry_date"] = source["signal_index"].map(lambda index: sessions[index + 1])
    source["exit_date"] = source["signal_index"].map(lambda index: sessions[index + 11])

    prices = execution[[
        "trade_date", "ts_code", "economic_open", "execution_data_eligible"
    ]].copy()
    entry = prices.rename(columns={
        "trade_date": "entry_date",
        "economic_open": "entry_open",
        "execution_data_eligible": "entry_eligible",
    })
    exit_ = prices.rename(columns={
        "trade_date": "exit_date",
        "economic_open": "exit_open",
        "execution_data_eligible": "exit_eligible",
    })
    labels = source.merge(entry, on=["entry_date", "ts_code"], how="left", validate="one_to_one")
    labels = labels.merge(exit_, on=["exit_date", "ts_code"], how="left", validate="one_to_one")
    entry_open = pd.to_numeric(labels["entry_open"], errors="coerce")
    exit_open = pd.to_numeric(labels["exit_open"], errors="coerce")
    valid = (
        labels["entry_eligible"].fillna(False).astype(bool)
        & labels["exit_eligible"].fillna(False).astype(bool)
        & entry_open.gt(0)
        & exit_open.gt(0)
        & np.isfinite(entry_open)
        & np.isfinite(exit_open)
    )
    labels = labels.loc[valid, ["asof", "ts_code", "exit_date"]].copy()
    labels["label_return"] = (exit_open[valid] / entry_open[valid] - 1.0).to_numpy()
    labels["available_time"] = labels["exit_date"].map(session_open)
    labels = labels.rename(columns={"asof": "event_time"}).drop(columns="exit_date")
    labels = labels.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True)
    return labels, {
        "stage1_rows": int(len(source)),
        "mature_price_rows": int(len(labels)),
        "missing_or_ineligible_label_rows": int(len(source) - len(labels)),
        "mean_label": float(labels["label_return"].mean()) if len(labels) else None,
        "positive_label_rate": float(labels["label_return"].gt(0).mean()) if len(labels) else None,
    }


def _build_stage1_period(
    strategy: StrategyConfig,
    plan: Any,
    output: Path,
    *,
    retain_bundle: bool,
    label_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Any | None]:
    bundle = build_data_bundle(plan, strategy, output)
    features = build_feature_ledger(bundle, strategy)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    score = ledgers["score"]
    pool = score.loc[
        score["stage1_pass"].astype(bool),
        [
            "asof", "ts_code", "bundle_id", "feature_set_hash", "rule_score",
            "execution_data_eligible",
        ],
    ].copy()
    execution_status = bundle.execution[["trade_date", "ts_code", "execution_status"]]
    pool = pool.merge(
        execution_status.rename(columns={"trade_date": "asof"}),
        on=["asof", "ts_code"],
        validate="one_to_one",
    )
    labels, label_audit = _build_labels(
        pool, bundle.execution, bundle.calendar, label_config
    )
    audit = {
        "bundle_id": bundle.bundle_id,
        "signal_start": plan.signal_start,
        "signal_end": plan.signal_end,
        "execution_end": plan.execution_end,
        "signal_sessions": int(len(plan.signal_sessions)),
        "active_stage1_sessions": int(pool["asof"].nunique()),
        "maximum_daily_stage1": int(pool.groupby("asof").size().max()) if len(pool) else 0,
        "median_daily_stage1": float(pool.groupby("asof").size().median()) if len(pool) else 0.0,
        **label_audit,
    }
    del features, ledgers, score
    gc.collect()
    return pool, labels, audit, bundle if retain_bundle else None


def _load_candidate_f0(
    pool: pd.DataFrame,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = pool[["asof", "ts_code"]].rename(columns={"asof": "event_time"}).copy()
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True).dt.normalize()
    key_index = pd.MultiIndex.from_frame(keys[["event_time", "ts_code"]])
    start = keys["event_time"].min().tz_localize(None).to_period("M")
    end = keys["event_time"].max().tz_localize(None).to_period("M")
    parts: list[pd.DataFrame] = []
    monthly: list[dict[str, Any]] = []
    for period in pd.period_range(start, end, freq="M"):
        month_start = max(keys["event_time"].min(), pd.Timestamp(period.start_time, tz="UTC"))
        month_end = min(keys["event_time"].max(), pd.Timestamp(period.end_time.date(), tz="UTC"))
        source = load_f0_panel(str(month_start.date()), str(month_end.date()))
        source["event_time"] = pd.to_datetime(source["event_time"], utc=True).dt.normalize()
        source_index = pd.MultiIndex.from_frame(source[["event_time", "ts_code"]])
        selected = source.loc[source_index.isin(key_index)].copy()
        values = selected[list(feature_set.columns)].apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        eligible = values.notna().any(axis=1)
        selected = selected.loc[
            eligible, ["ts_code", "event_time", "available_time", *feature_set.columns]
        ].copy()
        selected.loc[:, list(feature_set.columns)] = values.loc[eligible].to_numpy()
        parts.append(selected)
        monthly.append({
            "month": str(period),
            "source_rows": int(len(source)),
            "stage1_matches": int(len(values)),
            "f0_eligible_rows": int(len(selected)),
            "candidate_f0_sha256": _frame_hash(selected),
        })
        del source, selected, values, eligible
        gc.collect()
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if result.duplicated(["event_time", "ts_code"]).any():
        raise ValueError("F0 candidate panel contains duplicate keys")
    return result.sort_values(["event_time", "ts_code"], kind="mergesort").reset_index(drop=True), {
        "stage1_rows": int(len(pool)),
        "f0_eligible_rows": int(len(result)),
        "f0_eligible_ratio": float(len(result) / len(pool)) if len(pool) else None,
        "months": monthly,
        "candidate_f0_sha256": _frame_hash(result),
    }


def _fit_ranker_twice(
    frame: pd.DataFrame,
    feature_set: FeatureSet,
    params: dict[str, Any],
    model_id: str,
) -> tuple[XGBRanker, dict[str, Any]]:
    counts = frame.groupby("event_time", sort=True).size()
    valid_dates = counts[counts.ge(2)].index
    train = frame[frame["event_time"].isin(valid_dates)].sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    )
    if train.empty or train["event_time"].nunique() < 2:
        raise ValueError(f"{model_id} has no valid ranking dataset")
    X = train[list(feature_set.columns)]
    y = train["label_return"].to_numpy(dtype=float)
    qid = pd.factorize(train["event_time"], sort=True)[0]

    models: list[XGBRanker] = []
    hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"aistock-{model_id}-") as temp:
        for run_no in (1, 2):
            model = XGBRanker(**params)
            model.fit(X, y, qid=qid)
            path = Path(temp) / f"model-{run_no}.json"
            model.save_model(path)
            hashes.append(_sha(path))
            models.append(model)
    if hashes[0] != hashes[1]:
        raise AssertionError(f"{model_id} model bytes are not deterministic")
    return models[0], {
        "model_id": model_id,
        "training_rows": int(len(train)),
        "training_groups": int(train["event_time"].nunique()),
        "training_start": str(pd.Timestamp(train["event_time"].min()).date()),
        "training_end": str(pd.Timestamp(train["event_time"].max()).date()),
        "maximum_label_available_time": pd.Timestamp(train["label_available_time"].max()).isoformat(),
        "model_sha256_run1": hashes[0],
        "model_sha256_run2": hashes[1],
        "deterministic": True,
    }


def _monthly_predictions(
    f0: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: FeatureSet,
    config: dict[str, Any],
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    joined = f0.rename(columns={"available_time": "feature_available_time"}).merge(
        labels.rename(columns={"available_time": "label_available_time"}),
        on=["event_time", "ts_code"],
        validate="one_to_one",
    )
    prediction_start = pd.Timestamp(config["timeline"]["prediction_start"], tz="UTC")
    prediction_end = pd.Timestamp(config["timeline"]["prediction_end"], tz="UTC")
    predictions: list[pd.DataFrame] = []
    models: list[dict[str, Any]] = []
    params = dict(config["model"])
    params.pop("family")
    expected_months = pd.period_range(
        prediction_start.tz_localize(None).to_period("M"),
        prediction_end.tz_localize(None).to_period("M"),
        freq="M",
    )
    for period in expected_months:
        month_start = pd.Timestamp(period.start_time, tz="UTC")
        month_end = pd.Timestamp(period.end_time.date(), tz="UTC")
        prior_sessions = sessions[sessions < month_start]
        if prior_sessions.empty:
            raise ValueError(f"no training cutoff before {period}")
        cutoff = prior_sessions[-1]
        window_start = cutoff - pd.DateOffset(months=int(config["timeline"]["train_window_months"]))
        training = joined[
            joined["event_time"].gt(window_start)
            & joined["event_time"].le(cutoff)
            & pd.to_datetime(joined["label_available_time"], utc=True).le(session_close(cutoff))
        ].copy()
        model_id = (
            f"{config['identity']['model_profile_id']}_{period.strftime('%Y%m')}"
        )
        model, metadata = _fit_ranker_twice(training, feature_set, params, model_id)
        metadata.update({
            "training_cutoff": str(cutoff.date()),
            "window_start_exclusive": str(pd.Timestamp(window_start).date()),
        })
        prediction = f0[
            f0["event_time"].between(max(month_start, prediction_start), min(month_end, prediction_end))
        ].copy()
        if not prediction.empty:
            if pd.to_datetime(prediction["available_time"], utc=True).gt(
                prediction["event_time"].map(session_close)
            ).any():
                raise AssertionError(f"{model_id} prediction contains unavailable features")
            prediction["model_score"] = model.predict(prediction[list(feature_set.columns)])
            prediction["model_id"] = model_id
            predictions.append(prediction[["event_time", "ts_code", "model_score", "model_id"]])
        metadata["prediction_rows"] = int(len(prediction))
        metadata["prediction_sessions"] = int(prediction["event_time"].nunique()) if len(prediction) else 0
        models.append(metadata)
        del model, training, prediction
        gc.collect()
    if len(models) != len(expected_months):
        raise AssertionError("monthly model count mismatch")
    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return result, models


def _candidate_ledgers(
    pool: pd.DataFrame,
    predictions: pd.DataFrame,
    signal_sessions: tuple[str, ...],
    *,
    rank_column: str,
    policy_id: str,
    strategy: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    frame = pool.merge(
        predictions[["event_time", "ts_code", "model_score", "model_id"]].rename(
            columns={"event_time": "asof"}
        ),
        on=["asof", "ts_code"],
        validate="one_to_one",
    )
    frame = frame.sort_values(
        ["asof", rank_column, "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    frame["candidate_rank"] = frame.groupby("asof", sort=False).cumcount() + 1
    frame = frame[frame["candidate_rank"].le(int(strategy.portfolio["candidate_view_size"]))].copy()
    frame["candidate_status"] = "IN_VIEW"
    snapshots: dict[pd.Timestamp, str] = {}
    for day, group in frame.groupby("asof", sort=True):
        payload = "|".join(
            f"{row.ts_code}:{int(row.candidate_rank)}" for row in group.itertuples()
        )
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    empty = hashlib.sha256(b"").hexdigest()
    frame["candidate_snapshot_id"] = frame["asof"].map(snapshots)
    policy_hash = hashlib.sha256(
        f"{policy_id}|{strategy.config_hash}|{rank_column}".encode()
    ).hexdigest()
    days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    selection = pd.DataFrame({"asof": days})
    selection["candidate_snapshot_id"] = selection["asof"].map(snapshots).fillna(empty)
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
    return {"candidate": frame, "selection": selection}


def _run_portfolios(
    ledgers: dict[str, pd.DataFrame],
    bundle: Any,
    strategy: StrategyConfig,
    execution_sessions: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    portfolio: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, pd.DataFrame]] = {}
    active_days = int(ledgers["candidate"]["asof"].nunique())
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
        metrics = summarize(
            result["nav"],
            result["fills"],
            initial_cash=float(strategy.execution["initial_cash"]),
            positions=result["positions"],
            corporate_actions=result["corporate_actions"],
        )
        metrics.update({
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if len(result["execution_decisions"]) else 0,
            "open_positions_at_end": int(len(result["open_positions"])),
            "active_signal_days": active_days,
        })
        portfolio[scenario] = metrics
        results[scenario] = result
    return portfolio, results


def _acceptance(
    control: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["evaluation"]
    scenarios: dict[str, Any] = {}
    benchmarks = {
        "base": float(rules["cap1_base_return"]),
        "stress": float(rules["cap1_stress_return"]),
    }
    for scenario in ("base", "stress"):
        item = challenger[scenario]
        tests = {
            "higher_than_same_pool_control": item["total_return"] > control[scenario]["total_return"],
            "higher_than_cap1": item["total_return"] > benchmarks[scenario],
            "pf_minimum": (item["portfolio_profit_factor"] or 0.0) >= float(rules["portfolio_profit_factor_min"]),
            "maxdd_limit": abs(item["max_drawdown"]) <= float(rules["max_drawdown_abs_max"]),
            "excluding_best_week_positive": item["return_excluding_best_week"] > 0.0,
            "excluding_top3_positive": item["return_excluding_top3_profit"] > 0.0,
            "minimum_closed_trades": item["trade_count"] >= int(rules["minimum_closed_trades"]),
            "position_cap": item["max_open_positions"] <= int(rules["maximum_positions"]),
        }
        if "trade_win_rate_min" in rules:
            tests["trade_win_rate_minimum"] = (
                item["trade_win_rate"] is not None
                and item["trade_win_rate"] >= float(rules["trade_win_rate_min"])
            )
        scenarios[scenario] = {"passed": all(tests.values()), "tests": tests}
    return {"passed": all(row["passed"] for row in scenarios.values()), "scenarios": scenarios}


def _engine_label_parity(
    result: dict[str, pd.DataFrame], labels: pd.DataFrame
) -> dict[str, Any]:
    fills = result["fills"]
    decisions = result["execution_decisions"]
    sells = fills[fills["side"].eq("SELL")][
        ["decision_id", "ts_code", "trade_date", "economic_return", "reason"]
    ].copy()
    chosen = decisions[decisions["chosen"].astype(bool)][
        ["decision_id", "ts_code", "signal_session"]
    ].copy()
    trades = sells.merge(
        chosen,
        on=["decision_id", "ts_code"],
        how="left",
        validate="one_to_one",
        indicator="decision_match",
    )
    reference = labels[
        [
            "event_time",
            "ts_code",
            "economic_return",
            "trigger_type",
            "exit_date",
        ]
    ].rename(
        columns={
            "event_time": "signal_session",
            "economic_return": "label_economic_return",
        }
    )
    aligned = trades.merge(
        reference,
        on=["signal_session", "ts_code"],
        how="left",
        validate="one_to_one",
        indicator="label_match",
    )
    errors = (
        pd.to_numeric(aligned["economic_return"], errors="coerce")
        - pd.to_numeric(aligned["label_economic_return"], errors="coerce")
    ).abs()
    maximum_error = float(errors.max()) if len(errors) else 0.0
    unmatched_decisions = int(trades["decision_match"].ne("both").sum())
    missing = int(aligned["label_match"].ne("both").sum())
    trigger_mismatches = int(
        aligned["reason"].astype(str).ne(aligned["trigger_type"].astype(str)).sum()
    )
    exit_date_mismatches = int(
        pd.to_datetime(aligned["trade_date"], utc=True)
        .dt.normalize()
        .ne(pd.to_datetime(aligned["exit_date"], utc=True).dt.normalize())
        .sum()
    )
    return {
        "closed_control_sells": int(len(sells)),
        "decision_matched_trades": int(len(trades) - unmatched_decisions),
        "matched_labels": int(len(aligned) - missing),
        "unmatched_decisions": unmatched_decisions,
        "missing_labels": missing,
        "trigger_type_mismatches": trigger_mismatches,
        "exit_date_mismatches": exit_date_mismatches,
        "maximum_absolute_economic_return_error": maximum_error,
        "passed": bool(
            len(sells) > 0
            and len(trades) == len(sells)
            and unmatched_decisions == 0
            and missing == 0
            and trigger_mismatches == 0
            and exit_date_mismatches == 0
            and maximum_error <= 1e-10
        ),
    }


def _verify(
    models: list[dict[str, Any]],
    portfolios: dict[str, dict[str, dict[str, Any]]],
    results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    execution_end: str,
    ledgers: dict[str, dict[str, pd.DataFrame]],
    signal_sessions: tuple[str, ...],
    model_profile_id: str,
    label_parity: dict[str, Any],
) -> dict[str, Any]:
    control_candidate = ledgers["control"]["candidate"]
    challenger_candidate = ledgers["challenger"]["candidate"]
    checks: dict[str, bool] = {
        "eight_monthly_models": len(models) == 8,
        "no_skipped_months": all(row["training_rows"] > 0 for row in models),
        "model_bytes_deterministic": all(row["deterministic"] for row in models),
        "all_training_labels_mature": all(
            pd.Timestamp(row["maximum_label_available_time"])
            <= session_close(row["training_cutoff"])
            for row in models
        ),
        "control_rule_score_descending": bool(
            control_candidate.groupby("asof", sort=False)["rule_score"]
            .apply(lambda values: values.is_monotonic_decreasing)
            .all()
        ),
        "challenger_model_score_descending": bool(
            challenger_candidate.groupby("asof", sort=False)["model_score"]
            .apply(lambda values: values.is_monotonic_decreasing)
            .all()
        ),
        "control_all_signal_dates_present": int(
            ledgers["control"]["selection"]["asof"].nunique()
        )
        == len(signal_sessions),
        "challenger_all_signal_dates_present": int(
            ledgers["challenger"]["selection"]["asof"].nunique()
        )
        == len(signal_sessions),
        "control_candidate_keys_unique": not control_candidate.duplicated(
            ["asof", "ts_code"]
        ).any(),
        "challenger_candidate_keys_unique": not challenger_candidate.duplicated(
            ["asof", "ts_code"]
        ).any(),
        "control_month_model_identity": bool(
            (
                control_candidate["model_id"]
                == control_candidate["asof"].map(
                    lambda day: f"{model_profile_id}_{pd.Timestamp(day).strftime('%Y%m')}"
                )
            ).all()
        ),
        "challenger_month_model_identity": bool(
            (
                challenger_candidate["model_id"]
                == challenger_candidate["asof"].map(
                    lambda day: f"{model_profile_id}_{pd.Timestamp(day).strftime('%Y%m')}"
                )
            ).all()
        ),
        "base_control_executable_label_parity": bool(label_parity["passed"]),
    }
    for arm in ("control", "challenger"):
        for scenario in ("base", "stress"):
            result = results[arm][scenario]
            nav = result["nav"]
            checks[f"{arm}_{scenario}_nav_identity"] = bool(
                np.allclose(nav["cash"] + nav["market_value"], nav["nav"], rtol=0, atol=1e-8)
            )
            checks[f"{arm}_{scenario}_cash_nonnegative"] = bool((nav["cash"] >= -1e-8).all())
            checks[f"{arm}_{scenario}_position_cap"] = portfolios[arm][scenario]["max_open_positions"] <= 5
            checks[f"{arm}_{scenario}_execution_end"] = (
                str(pd.Timestamp(nav["trade_date"].max()).date()) == execution_end
            )
            decisions = result["execution_decisions"]
            checks[f"{arm}_{scenario}_next_session_entries"] = bool(
                decisions.empty
                or (decisions["execution_session"] > decisions["signal_session"]).all()
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def _write_result(
    output: Path,
    sample: dict[str, Any],
    control: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
    acceptance: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    lines = [
        "# F0-123 Conditional Stage1 Ranker", "",
        "## Contract", "",
        "- Configured fixed-rule Stage-1, then frozen F0=123 XGBRanker inside that pool.",
        "- All 123 columns are retained; feature-level NaN is passed to XGBoost without imputation.",
        "- Monthly trailing-12-month training, daily 2026 prediction, no skipped month or fallback.",
        "- Same canonical portfolio engine, Base/Stress costs, H10, trailing stop, 20% sizing and five-position cap.",
        "- No wide-table 202 factors, extra data source, feature selection, threshold scan, or business-data cache.", "",
        "## Sample", "",
        f"- Stage-1 rows: `{sample['combined_stage1_rows']}`; F0-eligible rows: `{sample['f0_eligible_rows']}` ({sample['f0_eligible_ratio']:.2%}).",
        f"- Training labels: `{sample['training_label_rows']}`; 2026 prediction rows: `{sample['prediction_rows']}`.", "",
        "## Portfolio", "",
        "| Cost | Transparent control | F0-123 challenger | Delta | Win rate | PF | MaxDD | Ex-best | Ex-top3 | Weekly >=5% | Trades | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in ("base", "stress"):
        base = control[scenario]
        item = challenger[scenario]
        lines.append(
            f"| {scenario} | {base['total_return']:+.2%} | {item['total_return']:+.2%} | "
            f"{item['total_return'] - base['total_return']:+.2%} | "
            f"{item['trade_win_rate']:.1%} | {item['portfolio_profit_factor']:.3f} | "
            f"{item['max_drawdown']:.2%} | "
            f"{item['return_excluding_best_week']:+.2%} | {item['return_excluding_top3_profit']:+.2%} | "
            f"{item['weekly_ge_5_count']} ({item['weekly_ge_5_ratio']:.1%}) | "
            f"{item['trade_count']} | {acceptance['scenarios'][scenario]['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    if verification["passed"] and acceptance["passed"]:
        lines.append("The conditional F0=123 ranker advances unchanged to forward registration.")
    else:
        lines.append(
            "The conditional F0=123 ranker is rejected unchanged. This result will not be repaired with feature, model, Stage-1, TopN, or execution tuning."
        )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    prereg = args.prereg.resolve()
    if not prereg.is_file():
        raise FileNotFoundError(f"preregistration is missing: {prereg}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    model_config = _load_model_config(args.model_profile, feature_set)
    if (
        model_config["identity"]["model_profile_id"]
        == "f0_123_cap1_executable_v1"
        and strategy.strategy_id != "reset_weak_confirm_v3_cap1_20"
    ):
        raise ValueError("CAP1 executable model requires the exact CAP1-20 strategy")
    strategy_binding = model_config.get("strategy", {})
    if model_config["identity"]["model_profile_id"] == "f0_123_cap1_executable_v1" and (
        strategy_binding.get("strategy_id") != strategy.strategy_id
        or strategy_binding.get("config_hash") != strategy.config_hash
    ):
        raise ValueError("CAP1 executable model strategy hash differs from preregistration")
    cutoffs = _source_cutoffs()
    if min(cutoffs["stock_factor_pro_ts"], cutoffs["daily_basic_ts"]) < model_config["timeline"]["prediction_end"]:
        raise ValueError(f"F0 source cutoff does not cover prediction end: {cutoffs}")

    timeline = model_config["timeline"]
    label_config = model_config["label"]
    train_plan, _ = _period_plan(
        strategy,
        signal_start=timeline["train_start"],
        signal_end=timeline["train_end"],
        execution_end=timeline["training_execution_end"],
        output=output,
        name="conditional_reset_training_2025",
    )
    train_pool, train_labels, train_audit, _ = _build_stage1_period(
        strategy,
        train_plan,
        output,
        retain_bundle=False,
        label_config=label_config,
    )
    gc.collect()

    prediction_plan, prediction_calendar = _period_plan(
        strategy,
        signal_start=timeline["prediction_start"],
        signal_end=timeline["prediction_end"],
        execution_end=timeline["execution_end"],
        output=output,
        name="conditional_f0_123_prediction_2026",
        require_complete_horizon=False,
    )
    prediction_pool, prediction_labels, prediction_audit, execution_bundle = _build_stage1_period(
        strategy,
        prediction_plan,
        output,
        retain_bundle=True,
        label_config=label_config,
    )
    combined_pool = pd.concat([train_pool, prediction_pool], ignore_index=True)
    combined_labels = pd.concat([train_labels, prediction_labels], ignore_index=True)
    f0, f0_audit = _load_candidate_f0(combined_pool, feature_set)
    prediction_f0 = f0[
        f0["event_time"].between(
            pd.Timestamp(timeline["prediction_start"], tz="UTC"),
            pd.Timestamp(timeline["prediction_end"], tz="UTC"),
        )
    ].copy()
    sessions = pd.DatetimeIndex(pd.to_datetime(prediction_calendar["session"], utc=True)).normalize()
    predictions, models = _monthly_predictions(
        f0, combined_labels, feature_set, model_config, sessions
    )
    if len(predictions) != len(prediction_f0):
        raise AssertionError("not every F0-eligible 2026 Stage-1 row received a prediction")

    eligible_pool = prediction_pool.merge(
        prediction_f0[["event_time", "ts_code"]].rename(columns={"event_time": "asof"}),
        on=["asof", "ts_code"],
        validate="one_to_one",
    )
    control_ledgers = _candidate_ledgers(
        eligible_pool,
        predictions,
        prediction_plan.signal_sessions,
        rank_column="rule_score",
        policy_id="conditional_reset_transparent_control",
        strategy=strategy,
    )
    challenger_ledgers = _candidate_ledgers(
        eligible_pool,
        predictions,
        prediction_plan.signal_sessions,
        rank_column="model_score",
        policy_id="conditional_reset_f0_123_ranker",
        strategy=strategy,
    )
    control_portfolio, control_results = _run_portfolios(
        control_ledgers, execution_bundle, strategy, prediction_plan.execution_sessions
    )
    challenger_portfolio, challenger_results = _run_portfolios(
        challenger_ledgers, execution_bundle, strategy, prediction_plan.execution_sessions
    )
    acceptance = _acceptance(control_portfolio, challenger_portfolio, model_config)
    portfolios = {"control": control_portfolio, "challenger": challenger_portfolio}
    results = {"control": control_results, "challenger": challenger_results}
    label_parity = _engine_label_parity(control_results["base"], prediction_labels)
    verification = _verify(
        models,
        portfolios,
        results,
        timeline["execution_end"],
        {"control": control_ledgers, "challenger": challenger_ledgers},
        prediction_plan.signal_sessions,
        str(model_config["identity"]["model_profile_id"]),
        label_parity,
    )
    if not verification["passed"]:
        raise AssertionError(f"verification failed: {verification['failed']}")

    sample = {
        "train_period": train_audit,
        "prediction_period": prediction_audit,
        "combined_stage1_rows": int(len(combined_pool)),
        "f0_eligible_rows": int(len(f0)),
        "f0_eligible_ratio": float(len(f0) / len(combined_pool)),
        "training_label_rows": int(len(combined_labels)),
        "prediction_rows": int(len(predictions)),
        "prediction_sessions": int(predictions["event_time"].nunique()),
        "f0_audit": f0_audit,
        "base_control_executable_label_parity": label_parity,
    }
    comparison = {
        scenario: {
            field: {
                "control": control_portfolio[scenario][field],
                "challenger": challenger_portfolio[scenario][field],
                "delta": challenger_portfolio[scenario][field] - control_portfolio[scenario][field],
            }
            for field in (
                "total_return", "portfolio_profit_factor", "max_drawdown",
                "trade_win_rate", "return_excluding_best_week",
                "return_excluding_top3_profit", "trade_count",
            )
        }
        for scenario in ("base", "stress")
    }

    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests"):
        (output / name).mkdir()
    shutil.copyfile(args.strategy, output / "configs/strategy.yaml")
    shutil.copyfile(args.model_profile, output / "configs/model_profile.yaml")
    shutil.copyfile(prereg, output / "configs/preregistration.md")
    _write_json(output / "RUN_STATUS.json", {
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "model_profile_id": model_config["identity"]["model_profile_id"],
        "verification_passed": verification["passed"],
        "acceptance_passed": acceptance["passed"],
        "credentials_persisted": False,
        "business_data_persisted": False,
    })
    _write_json(output / "plan.json", {
        "training": train_plan.to_dict(),
        "prediction": prediction_plan.to_dict(),
        "source_cutoffs": cutoffs,
        "feature_set_id": feature_set.id,
        "feature_order_hash": feature_set.order_hash,
    })
    _write_json(output / "sample_audit.json", sample)
    _write_json(output / "model_manifest.json", models)
    _write_json(output / "control_metrics.json", control_portfolio)
    _write_json(output / "challenger_metrics.json", challenger_portfolio)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "acceptance.json", acceptance)
    _write_json(output / "verification.json", verification)
    _write_result(
        output, sample, control_portfolio, challenger_portfolio, acceptance, verification
    )
    code_paths = [
        ROOT / "src/aistock9988/data/q70_source.py",
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/labeling/executable_path.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        Path(__file__).resolve(),
    ]
    code_paths.append(prereg)
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
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args()
    print(f"run_complete={run(args)}")


if __name__ == "__main__":
    main()
