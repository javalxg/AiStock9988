#!/usr/bin/env python3
"""Run the preregistered causal F0=123 Top20 event selector."""
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
from xgboost import XGBClassifier

from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_source_max_dates, load_trading_calendar
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.features.f0_cross_section import prepare_f0_cross_sections
from aistock9988.features.registry import FeatureSet
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.time.session import session_close
from full_market_f0_123_ranker_runner import (
    _build_path_labels,
    _filter_f0_universe,
    _frame_hash,
    _load_label_prices,
    _run_portfolios,
    _sample_training_rows,
    _sha,
    _walk_forward_predictions,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/f0_123_full_market_top5_v1.yaml"
DEFAULT_MODEL = ROOT / "configs/model_profiles/f0_123_causal_top20_event_selector_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT / "docs/council_20260828" / "F0_123_CAUSAL_TOP20_EVENT_SELECTOR_20260901"
)


def _load_config(path: Path, feature_set: FeatureSet) -> dict[str, Any]:
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
        raise ValueError("F0 feature contract drift")
    stage2_expected = {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "event_return_minimum": 0.10,
        "probability_gate": "none",
    }
    if any(raw["stage2"].get(key) != value for key, value in stage2_expected.items()):
        raise ValueError("Stage-2 preregistration contract drift")
    if raw["evaluation"].get("parameter_sweep") is not False:
        raise ValueError("parameter_sweep must be false")
    return raw


def _weekly_sessions(calendar: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    sessions = sessions[(sessions >= pd.Timestamp(start, tz="UTC")) & (sessions <= pd.Timestamp(end, tz="UTC"))]
    grouped = pd.Series(sessions, index=sessions).groupby(sessions.tz_localize(None).to_period("W-FRI"), sort=True)
    return [pd.Timestamp(group.iloc[-1]) for _, group in grouped]


def _compile_plan(
    strategy: StrategyConfig,
    config: dict[str, Any],
    output: Path,
) -> tuple[Any, pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    timeline = config["timeline"]
    calendar = load_trading_calendar("2023-01-01", timeline["execution_end"])
    plan = compile_run_plan(
        strategy,
        RunRequest(
            timeline["prediction_start"],
            timeline["prediction_end"],
            timeline["execution_end"],
            str(output),
            "f0_123_causal_top20_event_selector_2026",
        ),
        calendar["session"],
    )
    historical = _weekly_sessions(
        calendar,
        timeline["historical_prediction_start"],
        timeline["historical_prediction_end"],
    )
    forward = list(pd.to_datetime(plan.signal_sessions, utc=True))
    if not historical or not forward:
        raise ValueError("historical or forward weekly signal calendar is empty")
    return plan, calendar, historical, forward


def _load_prepared_features(
    config: dict[str, Any],
    strategy: StrategyConfig,
    calendar: pd.DataFrame,
    feature_set: FeatureSet,
    prediction_days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare one year at a time so raw full-market panels never coexist."""
    timeline = config["timeline"]
    maximum = int(config["feature_set"]["maximum_training_rows_per_date"])
    prediction_set = set(prediction_days)
    training_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    prep_audits: list[dict[str, Any]] = []
    universe_audits: list[dict[str, Any]] = []
    year_ranges = (
        (timeline["feature_start"], "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", timeline["prediction_end"]),
    )
    for start, end in year_ranges:
        print(f"phase=f0_load start={start} end={end}", flush=True)
        raw = load_f0_panel(start, end)
        raw, universe_audit = _filter_f0_universe(raw, strategy, calendar)
        prepared, prep_audit = prepare_f0_cross_sections(
            raw,
            feature_set,
            minimum_non_null_features=int(config["feature_set"]["minimum_non_null_features"]),
            maximum_rows_per_date=maximum,
            sample_seed=42,
            uncapped_dates=[day for day in prediction_days if start <= str(day.date()) <= end],
        )
        del raw
        gc.collect()
        year_prediction_days = [day for day in prediction_days if start <= str(day.date()) <= end]
        if year_prediction_days:
            training, prediction = _sample_training_rows(prepared, prediction_set, maximum)
        else:
            training = prepared
            prediction = prepared.iloc[0:0].copy()
        training_parts.append(training)
        if not prediction.empty:
            prediction_parts.append(prediction)
        prep_audits.append({"start": start, "end": end, **asdict(prep_audit)})
        universe_audits.append({"start": start, "end": end, **universe_audit})
        del prepared, training, prediction
        gc.collect()
    training_features = pd.concat(training_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    prediction_features = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["event_time", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    return training_features, prediction_features, prep_audits, universe_audits


def _stage1_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["stage1"])
    model.pop("candidate_view_size")
    model["family"] = model.pop("family")
    return {"model": model, "timeline": config["timeline"]}


def _compress_stage1_top20(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: FeatureSet,
    candidate_view_size: int,
) -> pd.DataFrame:
    ranked = predictions.sort_values(
        ["event_time", "model_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["stage1_rank"] = ranked.groupby("event_time", sort=False).cumcount() + 1
    ranked = ranked[ranked["stage1_rank"].le(candidate_view_size)].copy()
    ranked["stage1_rank_pct"] = (ranked["stage1_rank"] - 1.0) / max(candidate_view_size - 1, 1)
    ranked = ranked.merge(
        labels[["event_time", "ts_code", "label_return", "available_time"]].rename(
            columns={"available_time": "label_available_time"}
        ),
        on=["event_time", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    return ranked.sort_values(["event_time", "stage1_rank"], kind="mergesort").reset_index(drop=True)


def _fit_stage2(
    training: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
    model_id: str,
    event_minimum: float,
) -> tuple[XGBClassifier, dict[str, Any]]:
    if training["event_time"].nunique() < 20 or len(training) < 300:
        raise ValueError(f"{model_id} has insufficient causal Top20 history")
    y = training["label_return"].ge(event_minimum).astype("int8")
    if y.nunique() != 2 or int(y.sum()) < 20:
        raise ValueError(f"{model_id} does not contain both event classes")
    X = training[feature_columns]
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{model_id} contains infinite features")
    model = XGBClassifier(**params)
    model.fit(X, y)
    with tempfile.TemporaryDirectory(prefix=f"aistock-{model_id}-") as temp:
        model_path = Path(temp) / "model.json"
        model.save_model(model_path)
        model_hash = _sha(model_path)
    return model, {
        "model_id": model_id,
        "training_rows": int(len(training)),
        "training_signal_sessions": int(training["event_time"].nunique()),
        "training_start": str(training["event_time"].min().date()),
        "training_end": str(training["event_time"].max().date()),
        "event_rows": int(y.sum()),
        "event_rate": float(y.mean()),
        "maximum_label_available_time": pd.to_datetime(
            training["label_available_time"], utc=True
        ).max().isoformat(),
        "model_sha256": model_hash,
    }


def _walk_forward_stage2(
    top20: pd.DataFrame,
    feature_set: FeatureSet,
    config: dict[str, Any],
    forward_days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    feature_columns = [*feature_set.columns, "model_score", "stage1_rank"]
    stage2 = dict(config["stage2"])
    event_minimum = float(stage2.pop("event_return_minimum"))
    stage2.pop("family")
    stage2.pop("probability_gate")
    stage2.pop("entries_per_decision")
    predictions: list[pd.DataFrame] = []
    models: list[dict[str, Any]] = []
    model: XGBClassifier | None = None
    active_month: str | None = None
    for day in forward_days:
        month = day.strftime("%Y-%m")
        if month != active_month:
            cutoff = session_close(day)
            window_start = day - pd.DateOffset(months=int(config["timeline"]["train_window_months"]))
            fit_rows = top20[
                top20["event_time"].gt(window_start)
                & top20["event_time"].lt(day)
                & pd.to_datetime(top20["available_time"], utc=True).le(cutoff)
                & pd.to_datetime(top20["label_available_time"], utc=True).le(cutoff)
                & top20["label_return"].notna()
            ].copy()
            model_id = f"f0_123_top20_event_{day.strftime('%Y%m%d')}"
            model, metadata = _fit_stage2(
                fit_rows, feature_columns, stage2, model_id, event_minimum
            )
            metadata.update({
                "model_signal_date": str(day.date()),
                "training_window_start_exclusive": str(window_start.date()),
            })
            models.append(metadata)
            active_month = month
        assert model is not None
        prediction = top20[top20["event_time"].eq(day)].copy()
        if len(prediction) != 20:
            raise ValueError(f"Stage-1 candidate count is not 20: {day.date()}")
        prediction["stage2_probability"] = model.predict_proba(prediction[feature_columns])[:, 1]
        prediction["stage2_model_id"] = models[-1]["model_id"]
        predictions.append(prediction)
        models[-1].setdefault("prediction_sessions", 0)
        models[-1].setdefault("prediction_rows", 0)
        models[-1]["prediction_sessions"] += 1
        models[-1]["prediction_rows"] += int(len(prediction))
    return pd.concat(predictions, ignore_index=True), models


def _build_paired_ledgers(
    forward_top20: pd.DataFrame,
    forward_days: list[pd.Timestamp],
    strategy: StrategyConfig,
) -> dict[str, dict[str, pd.DataFrame]]:
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    orderings = {
        "control": (["event_time", "stage1_rank", "ts_code"], [True, True, True]),
        "challenger": (
            ["event_time", "stage2_probability", "stage1_rank", "ts_code"],
            [True, False, True, True],
        ),
    }
    membership_snapshots: dict[pd.Timestamp, str] = {}
    for day, group in forward_top20.groupby("event_time", sort=True):
        payload = "|".join(sorted(group["ts_code"].astype(str)))
        membership_snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    for name, (columns, ascending) in orderings.items():
        candidate = forward_top20.sort_values(columns, ascending=ascending, kind="mergesort").copy()
        candidate["candidate_rank"] = candidate.groupby("event_time", sort=False).cumcount() + 1
        candidate = candidate.rename(columns={"event_time": "asof"})
        candidate["candidate_status"] = "IN_VIEW"
        candidate["candidate_snapshot_id"] = candidate["asof"].map(membership_snapshots)
        policy_id = f"f0_123_top20_{name}_top5"
        policy_hash = hashlib.sha256(f"{policy_id}|{strategy.config_hash}".encode()).hexdigest()
        selection = pd.DataFrame({"asof": forward_days})
        selection["candidate_snapshot_id"] = selection["asof"].map(membership_snapshots)
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
        ledgers[name] = {"candidate": candidate, "selection": selection}
    return ledgers


def _event_diagnostics(forward_top20: pd.DataFrame, event_minimum: float) -> dict[str, Any]:
    labeled = forward_top20.dropna(subset=["label_return"]).copy()
    labeled["event"] = labeled["label_return"].ge(event_minimum)
    control = labeled[labeled["stage1_rank"].le(5)]
    challenger_rank = labeled.sort_values(
        ["event_time", "stage2_probability", "stage1_rank", "ts_code"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    challenger_rank["stage2_rank"] = challenger_rank.groupby("event_time", sort=False).cumcount() + 1
    challenger = challenger_rank[challenger_rank["stage2_rank"].le(5)]
    total_events = int(labeled["event"].sum())

    def summary(frame: pd.DataFrame) -> dict[str, Any]:
        events = int(frame["event"].sum())
        return {
            "rows": int(len(frame)),
            "events": events,
            "event_rate": float(frame["event"].mean()),
            "top20_event_recall": float(events / total_events) if total_events else 0.0,
            "mean_label_return": float(frame["label_return"].mean()),
        }

    control_summary = summary(control)
    challenger_summary = summary(challenger)
    return {
        "labeled_top20_rows": int(len(labeled)),
        "top20_events": total_events,
        "top20_event_rate": float(labeled["event"].mean()),
        "control_top5": control_summary,
        "challenger_top5": challenger_summary,
        "event_recall_lift": challenger_summary["top20_event_recall"] - control_summary["top20_event_recall"],
        "event_rate_lift": challenger_summary["event_rate"] - control_summary["event_rate"],
    }


def _acceptance(
    paired_metrics: dict[str, dict[str, dict[str, Any]]],
    event_diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["evaluation"]
    scenarios: dict[str, Any] = {}
    for scenario in ("base", "stress"):
        challenger = paired_metrics["challenger"][scenario]
        control = paired_metrics["control"][scenario]
        tests = {
            "higher_than_paired_control": challenger["total_return"] > control["total_return"],
            "higher_than_cap1": challenger["total_return"] > float(rules[f"cap1_{scenario}_return"]),
            "pf_minimum": (challenger["portfolio_profit_factor"] or 0.0) >= float(rules["portfolio_profit_factor_min"]),
            "maxdd_limit": abs(challenger["max_drawdown"]) <= float(rules["max_drawdown_abs_max"]),
            "excluding_best_week_positive": challenger["return_excluding_best_week"] > 0.0,
            "excluding_top3_positive": challenger["return_excluding_top3_profit"] > 0.0,
            "minimum_closed_trades": challenger["trade_count"] >= int(rules["minimum_closed_trades"]),
            "position_cap": challenger["max_open_positions"] <= int(rules["maximum_positions"]),
        }
        scenarios[scenario] = {"passed": all(tests.values()), "tests": tests}
    recall_lift = event_diagnostics["event_recall_lift"] > 0.0
    return {
        "passed": all(row["passed"] for row in scenarios.values()) and recall_lift,
        "event_recall_lift_positive": recall_lift,
        "scenarios": scenarios,
    }


def _verification(
    stage1_models: list[dict[str, Any]],
    stage2_models: list[dict[str, Any]],
    historical_days: list[pd.Timestamp],
    forward_days: list[pd.Timestamp],
    forward_top20: pd.DataFrame,
    paired_ledgers: dict[str, dict[str, pd.DataFrame]],
    paired_metrics: dict[str, dict[str, dict[str, Any]]],
    paired_results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    causal_top20_rows: int,
    execution_end: str,
) -> dict[str, Any]:
    all_days = historical_days + forward_days
    control_membership = paired_ledgers["control"]["candidate"].groupby("asof")["ts_code"].agg(lambda values: tuple(sorted(values)))
    challenger_membership = paired_ledgers["challenger"]["candidate"].groupby("asof")["ts_code"].agg(lambda values: tuple(sorted(values)))
    checks: dict[str, bool] = {
        "one_stage1_model_per_month": len(stage1_models) == len({day.strftime("%Y-%m") for day in all_days}),
        "one_stage2_model_per_forward_month": len(stage2_models) == len({day.strftime("%Y-%m") for day in forward_days}),
        "no_skipped_stage1_week": sum(int(row.get("prediction_sessions", 0)) for row in stage1_models) == len(all_days),
        "no_skipped_stage2_week": sum(int(row.get("prediction_sessions", 0)) for row in stage2_models) == len(forward_days),
        "twenty_causal_candidates_every_week": causal_top20_rows == 20 * len(all_days),
        "all_forward_weeks_present": int(forward_top20["event_time"].nunique()) == len(forward_days),
        "twenty_candidates_every_week": bool(forward_top20.groupby("event_time").size().eq(20).all()),
        "paired_candidate_membership_identical": control_membership.equals(challenger_membership),
        "all_stage1_labels_mature": all(
            pd.Timestamp(row["maximum_label_available_time"]) <= session_close(row["model_signal_date"])
            for row in stage1_models
        ),
        "all_stage2_labels_mature": all(
            pd.Timestamp(row["maximum_label_available_time"]) <= session_close(row["model_signal_date"])
            for row in stage2_models
        ),
    }
    for policy in ("control", "challenger"):
        for scenario in ("base", "stress"):
            result = paired_results[policy][scenario]
            nav = result["nav"]
            checks[f"{policy}_{scenario}_nav_identity"] = bool(
                np.allclose(nav["cash"] + nav["market_value"], nav["nav"], rtol=0, atol=1e-8)
            )
            checks[f"{policy}_{scenario}_cash_nonnegative"] = bool((nav["cash"] >= -1e-8).all())
            checks[f"{policy}_{scenario}_position_cap"] = paired_metrics[policy][scenario]["max_open_positions"] <= 5
            checks[f"{policy}_{scenario}_execution_end"] = str(nav["trade_date"].max().date()) == execution_end
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def _write_result(
    output: Path,
    sample: dict[str, Any],
    paired_metrics: dict[str, dict[str, dict[str, Any]]],
    event_diagnostics: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    lines = [
        "# F0-123 Causal Top20 Event Selector", "",
        "## Contract", "",
        "- Stage-1: causal monthly F0=123 ranker; full-market weekly Top20 in 2025 and 2026.",
        "- Stage-2: trailing-12-month classifier trained only on mature causal Top20 rows; fixed H10 >=10% event.",
        "- Paired control/challenger use identical Top20 membership and canonical Base/Stress execution.",
        "- No auxiliary data, probability gate, threshold scan, fallback, cache, or alternate backtest engine.", "",
        "## Sample", "",
        f"- Stage-1 training rows: `{sample['training_feature_rows']}` across `{sample['training_feature_sessions']}` sessions.",
        f"- Full prediction rows: `{sample['prediction_feature_rows']}` across `{sample['all_signal_sessions']}` weekly sessions.",
        f"- Causal Top20 rows: `{sample['causal_top20_rows']}`; mature labels: `{sample['label_audit']['label_rows']}`.", "",
        "## Paired Portfolio", "",
        "| Policy | Cost | Return | PF | MaxDD | Ex-best | Ex-top3 | Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("control", "challenger"):
        for scenario in ("base", "stress"):
            item = paired_metrics[policy][scenario]
            pf = item["portfolio_profit_factor"]
            lines.append(
                f"| {policy} | {scenario} | {item['total_return']:+.2%} | "
                f"{pf:.3f} | {item['max_drawdown']:.2%} | "
                f"{item['return_excluding_best_week']:+.2%} | "
                f"{item['return_excluding_top3_profit']:+.2%} | {item['trade_count']} |"
            )
    lines.extend([
        "", "## Event Selection", "",
        f"- Control Top5 event rate: `{event_diagnostics['control_top5']['event_rate']:.2%}`; "
        f"Top20-event recall: `{event_diagnostics['control_top5']['top20_event_recall']:.2%}`.",
        f"- Challenger Top5 event rate: `{event_diagnostics['challenger_top5']['event_rate']:.2%}`; "
        f"Top20-event recall: `{event_diagnostics['challenger_top5']['top20_event_recall']:.2%}`.",
        f"- Event recall lift: `{event_diagnostics['event_recall_lift']:+.2%}`.",
        "", "## Decision", "",
        "The exact causal Top20 event selector passes its preregistered acceptance contract."
        if acceptance["passed"]
        else "The exact causal Top20 event selector is rejected under its preregistered stop rule.",
    ])
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    strategy = StrategyConfig.from_yaml(args.strategy)
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    config = _load_config(args.model_profile, feature_set)
    plan, calendar, historical_days, forward_days = _compile_plan(strategy, config, output)
    all_prediction_days = historical_days + forward_days
    cutoffs = load_source_max_dates({
        "stock_factor_pro_ts", "daily_basic_ts", "market_daily_ts", "adj_factor_ts", "stk_limit_ts"
    })
    if min(cutoffs["stock_factor_pro_ts"], cutoffs["daily_basic_ts"]) < plan.signal_end:
        raise ValueError(f"F0 sources do not cover the final weekly signal: {cutoffs}")

    training_features, prediction_features, prep_audits, universe_audits = _load_prepared_features(
        config, strategy, calendar, feature_set, all_prediction_days
    )
    print("phase=label_price_load", flush=True)
    label_prices = _load_label_prices(config["timeline"]["feature_start"], plan.execution_end)
    label_keys = pd.concat([
        training_features[["event_time", "ts_code"]],
        prediction_features[["event_time", "ts_code"]],
    ], ignore_index=True).drop_duplicates()
    labels, label_audit = _build_path_labels(
        label_keys,
        label_prices,
        pd.DatetimeIndex(calendar["session"]),
        entry_delay=int(config["label"]["signal_to_entry_sessions"]),
        horizon=int(config["label"]["entry_to_exit_sessions"]),
        stop_pct=float(config["label"]["path_stop_pct"]),
    )
    del label_prices, label_keys
    gc.collect()

    print(
        f"phase=stage1 training_rows={len(training_features)} signal_weeks={len(all_prediction_days)}",
        flush=True,
    )
    stage1_predictions, stage1_models = _walk_forward_predictions(
        training_features,
        prediction_features,
        labels,
        feature_set,
        _stage1_config(config),
        all_prediction_days,
    )
    top20 = _compress_stage1_top20(
        stage1_predictions,
        labels,
        feature_set,
        int(config["stage1"]["candidate_view_size"]),
    )
    training_key_hash = _frame_hash(training_features[["event_time", "ts_code"]])
    prediction_key_hash = _frame_hash(prediction_features[["event_time", "ts_code"]])
    training_rows = int(len(training_features))
    training_sessions = int(training_features["event_time"].nunique())
    prediction_rows = int(len(prediction_features))
    del training_features, prediction_features, stage1_predictions
    gc.collect()

    print(f"phase=stage2 causal_top20_rows={len(top20)}", flush=True)
    forward_top20, stage2_models = _walk_forward_stage2(top20, feature_set, config, forward_days)
    event_diagnostics = _event_diagnostics(
        forward_top20, float(config["stage2"]["event_return_minimum"])
    )
    paired_ledgers = _build_paired_ledgers(forward_top20, forward_days, strategy)
    top20_key_hash = _frame_hash(top20[["event_time", "ts_code", "stage1_rank"]])
    causal_top20_rows = int(len(top20))
    del top20, forward_top20, labels
    gc.collect()

    print("phase=execution_bundle_load", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    paired_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    paired_results: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for policy in ("control", "challenger"):
        print(f"phase=backtest policy={policy}", flush=True)
        metrics, results = _run_portfolios(
            paired_ledgers[policy], bundle, strategy, plan.execution_sessions
        )
        paired_metrics[policy] = metrics
        paired_results[policy] = results
    acceptance = _acceptance(paired_metrics, event_diagnostics, config)
    verification = _verification(
        stage1_models,
        stage2_models,
        historical_days,
        forward_days,
        paired_ledgers["control"]["candidate"].rename(columns={"asof": "event_time"}),
        paired_ledgers,
        paired_metrics,
        paired_results,
        causal_top20_rows,
        config["timeline"]["execution_end"],
    )
    if not verification["passed"]:
        raise AssertionError(f"verification failed: {verification['failed']}")

    sample = {
        "preprocessing": prep_audits,
        "universe_filter": universe_audits,
        "training_feature_rows": training_rows,
        "training_feature_sessions": training_sessions,
        "prediction_feature_rows": prediction_rows,
        "historical_signal_sessions": len(historical_days),
        "forward_signal_sessions": len(forward_days),
        "all_signal_sessions": len(all_prediction_days),
        "causal_top20_rows": causal_top20_rows,
        "label_audit": label_audit,
        "training_features_sha256": training_key_hash,
        "prediction_features_sha256": prediction_key_hash,
        "causal_top20_keys_sha256": top20_key_hash,
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
    _write_json(output / "stage1_model_manifest.json", stage1_models)
    _write_json(output / "stage2_model_manifest.json", stage2_models)
    _write_json(output / "paired_portfolio_metrics.json", paired_metrics)
    _write_json(output / "event_diagnostics.json", event_diagnostics)
    _write_json(output / "acceptance.json", acceptance)
    _write_json(output / "verification.json", verification)
    _write_result(output, sample, paired_metrics, event_diagnostics, acceptance)
    code_paths = [
        ROOT / "src/aistock9988/data/q70_source.py",
        ROOT / "src/aistock9988/features/f0_cross_section.py",
        ROOT / "src/aistock9988/backtest/engine.py",
        ROOT / "scripts/full_market_f0_123_ranker_runner.py",
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
