#!/usr/bin/env python3
"""Train and freeze one capacity-aligned F0 V2 forward decision."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.features.f0_cross_section import prepare_f0_cross_sections
from aistock9988.features.registry import FeatureSet
from aistock9988.forward.lockbox import ForwardLockbox
from aistock9988.labeling.executable_path import (
    ExecutablePathLabelProfile,
    build_executable_path_labels,
)
from aistock9988.planning import RunPlan
from aistock9988.time.session import session_close
from f0_123_executable_forward_preflight import _load_profile, run as run_preflight
from full_market_f0_123_ranker_runner import _filter_f0_universe, _fit_model, _frame_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/f0_123_full_market_top5_v1.yaml"
DEFAULT_PROFILE = ROOT / "configs/model_profiles/f0_123_executable_forward_v2.yaml"
DEFAULT_LOCKBOX = (
    ROOT / "docs/council_20260828" / "F0_123_EXECUTABLE_V2_FORWARD_LOCKBOX"
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_hash(strategy: StrategyConfig, profile_path: Path) -> str:
    return _hash_bytes(
        f"{strategy.config_hash}|{_hash_bytes(profile_path.read_bytes())}".encode()
    )


def _capacity_status(
    signal_day: pd.Timestamp,
    lockbox_root: Path,
    calendar: pd.DatetimeIndex,
    interval: int,
) -> dict[str, Any]:
    existing = sorted((lockbox_root / "candidate").glob("part-*.parquet"))
    if not existing:
        return {"ready": True, "anchor": str(signal_day.date()), "session_offset": 0}
    anchor = pd.Timestamp(existing[0].stem.removeprefix("part-"), tz="UTC")
    anchor_index = int(calendar.get_indexer([anchor])[0])
    signal_index = int(calendar.get_indexer([signal_day])[0])
    if anchor_index < 0 or signal_index < 0:
        raise ValueError("forward capacity date is absent from the exchange calendar")
    offset = signal_index - anchor_index
    if offset <= 0:
        return {
            "ready": False,
            "status": "WAITING_FOR_NEWER_SESSION",
            "anchor": str(anchor.date()),
            "session_offset": offset,
        }
    if offset % interval != 0:
        next_offset = ((offset // interval) + 1) * interval
        next_day = calendar[anchor_index + next_offset] if anchor_index + next_offset < len(calendar) else None
        return {
            "ready": False,
            "status": "WAITING_FOR_CAPACITY_SESSION",
            "anchor": str(anchor.date()),
            "session_offset": offset,
            "next_capacity_session": None if next_day is None else str(next_day.date()),
        }
    return {"ready": True, "anchor": str(anchor.date()), "session_offset": offset}


def _plan(
    strategy: StrategyConfig,
    signal_day: pd.Timestamp,
    lockbox_root: Path,
) -> tuple[RunPlan, pd.DataFrame, pd.DatetimeIndex, pd.Timestamp]:
    window_start = signal_day - pd.DateOffset(months=12)
    calendar_start = str((signal_day - pd.DateOffset(years=3)).date())
    calendar = load_trading_calendar(calendar_start, str(signal_day.date()))
    all_sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    execution_sessions = all_sessions[
        (all_sessions >= window_start.normalize()) & (all_sessions <= signal_day)
    ]
    required = {
        "trade_cal_ts",
        "stock_basic_ts",
        "market_daily_ts",
        "adj_factor_ts",
        "stk_limit_ts",
        "stock_st_ts",
        "suspend_d_ts",
        "stk_auction_o_ts",
        "corporate_actions",
    }
    plan = RunPlan(
        run_name=f"f0_123_executable_v2_forward_{signal_day.strftime('%Y%m%d')}",
        output_dir=str(lockbox_root.resolve()),
        signal_start=str(signal_day.date()),
        signal_end=str(signal_day.date()),
        execution_end=str(signal_day.date()),
        feature_start=str(window_start.date()),
        signal_sessions=(str(signal_day.date()),),
        execution_sessions=tuple(str(day.date()) for day in execution_sessions),
        maximum_feature_lookback_sessions=0,
        hold_sessions_from_fill=10,
        strategy_id=strategy.strategy_id,
        strategy_hash=strategy.config_hash,
        mode=strategy.mode,
        required_sources=tuple(sorted(required)),
    )
    return plan, calendar, all_sessions, window_start


def _build_forward_ledgers(
    prediction: pd.DataFrame,
    signal_day: pd.Timestamp,
    strategy: StrategyConfig,
    model_id: str,
    config_hash: str,
) -> dict[str, pd.DataFrame]:
    ranked = prediction.sort_values(
        ["model_score", "ts_code"], ascending=[False, True], kind="mergesort"
    ).copy()
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    score = ranked[["event_time", "ts_code", "available_time", "model_score"]].rename(
        columns={"event_time": "asof"}
    )
    score["model_id"] = model_id
    candidate = ranked[ranked["candidate_rank"].le(20)].copy()
    candidate = candidate.rename(columns={"event_time": "asof"})
    candidate["candidate_status"] = "IN_VIEW"
    payload = "|".join(
        f"{row.ts_code}:{int(row.candidate_rank)}" for row in candidate.itertuples()
    )
    snapshot = _hash_bytes(payload.encode())
    candidate["candidate_snapshot_id"] = snapshot
    candidate = candidate[[
        "asof", "ts_code", "available_time", "model_score", "candidate_rank",
        "candidate_status", "candidate_snapshot_id",
    ]]
    policy_id = "f0_123_executable_v2_top20_to_top5"
    policy_hash = _hash_bytes(f"{policy_id}|{config_hash}".encode())
    selection = pd.DataFrame({
        "asof": [signal_day],
        "candidate_snapshot_id": [snapshot],
        "desired_entries": [int(strategy.portfolio["entries_per_decision"])],
        "target_weight_each": [float(strategy.portfolio["sizing"]["value"])],
        "primary_rank_end": [int(strategy.portfolio["entries_per_decision"])],
        "replacement_rank_end": [int(strategy.portfolio["candidate_view_size"])],
        "policy_id": [policy_id],
        "policy_hash": [policy_hash],
    })
    selection["decision_id"] = _hash_bytes(
        f"{policy_hash}|{signal_day.date()}|{snapshot}".encode()
    )
    selection["context_hash"] = _hash_bytes(
        f"{signal_day.date()}|{policy_hash}".encode()
    )
    return {"score": score, "candidate": candidate, "selection": selection}


def _train_and_score(
    strategy: StrategyConfig,
    config: dict[str, Any],
    feature_set: FeatureSet,
    signal_day: pd.Timestamp,
    lockbox_root: Path,
    config_hash: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    plan, calendar, all_sessions, window_start = _plan(strategy, signal_day, lockbox_root)
    print("phase=execution_bundle_load", flush=True)
    bundle = build_data_bundle(plan, strategy, lockbox_root)
    print("phase=f0_load", flush=True)
    raw = load_f0_panel(str(window_start.date()), str(signal_day.date()))
    raw, universe_audit = _filter_f0_universe(raw, strategy, calendar)
    prepared, prep_audit = prepare_f0_cross_sections(
        raw,
        feature_set,
        minimum_non_null_features=int(config["feature_set"]["minimum_non_null_features"]),
        maximum_rows_per_date=int(config["feature_set"]["maximum_training_rows_per_date"]),
        sample_seed=42,
        uncapped_dates=[signal_day],
    )
    del raw
    gc.collect()
    prediction = prepared[prepared["event_time"].eq(signal_day)].copy()
    training_features = prepared[prepared["event_time"].lt(signal_day)].copy()
    del prepared
    gc.collect()
    if prediction.empty:
        raise ValueError("fully covered signal has no eligible F0 prediction rows")
    if pd.to_datetime(prediction["available_time"], utc=True).gt(session_close(signal_day)).any():
        raise AssertionError("forward F0 contains observations unavailable at signal close")

    print(f"phase=label_build training_keys={len(training_features)}", flush=True)
    labels, label_audit = build_executable_path_labels(
        training_features[["event_time", "ts_code"]],
        bundle.execution,
        all_sessions,
        profile=ExecutablePathLabelProfile(),
    )
    training = training_features.merge(
        labels[["event_time", "ts_code", "label_return", "available_time"]].rename(
            columns={"available_time": "label_available_time"}
        ),
        on=["event_time", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    cutoff = session_close(signal_day)
    training = training[
        training["event_time"].gt(window_start)
        & pd.to_datetime(training["available_time"], utc=True).le(cutoff)
        & pd.to_datetime(training["label_available_time"], utc=True).le(cutoff)
    ].copy()
    params = dict(config["model"])
    params.pop("family")
    model_id = f"f0_123_executable_v2_{signal_day.strftime('%Y%m%d')}"
    print(f"phase=model_train rows={len(training)}", flush=True)
    model, model_audit = _fit_model(training, feature_set, params, model_id)
    prediction["model_score"] = model.predict(prediction[list(feature_set.columns)])
    ledgers = _build_forward_ledgers(
        prediction, signal_day, strategy, model_id, config_hash
    )
    audit = {
        "run_plan": plan.to_dict(),
        "bundle_manifest": bundle.manifest,
        "preprocessing": asdict(prep_audit),
        "universe_filter": universe_audit,
        "label_audit": label_audit,
        "model_audit": model_audit,
        "training_rows_after_maturity": int(len(training)),
        "training_keys_sha256": _frame_hash(training[["event_time", "ts_code"]]),
        "prediction_rows": int(len(prediction)),
        "prediction_keys_sha256": _frame_hash(prediction[["event_time", "ts_code"]]),
        "business_data_persisted": False,
        "model_persisted": False,
    }
    return ledgers, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, feature_set = _load_profile(args.profile)
    strategy = StrategyConfig.from_yaml(args.strategy)
    preflight = run_preflight(args.profile)
    if preflight["status"] != "READY_TO_FREEZE_FIRST_FORWARD_SIGNAL":
        return {"status": preflight["status"], "preflight": preflight, "lockbox_written": False}
    signal_day = pd.Timestamp(preflight["first_eligible_forward_signal"], tz="UTC")
    calendar_frame = load_trading_calendar(
        config["timeline"]["forward_not_before"],
        str((signal_day + pd.DateOffset(years=2)).date()),
    )
    capacity = _capacity_status(
        signal_day,
        args.lockbox.resolve(),
        pd.DatetimeIndex(pd.to_datetime(calendar_frame["session"], utc=True)).normalize(),
        int(config["label"]["hold_sessions_from_fill"]),
    )
    if not capacity["ready"]:
        return {"status": capacity["status"], "preflight": preflight, "capacity": capacity, "lockbox_written": False}
    config_hash = _config_hash(strategy, args.profile)
    ledgers, audit = _train_and_score(
        strategy, config, feature_set, signal_day, args.lockbox.resolve(), config_hash
    )
    lockbox = ForwardLockbox(
        args.lockbox,
        experiment_id="f0_123_executable_v2_forward",
        config_sha256=config_hash,
    )
    manifest = lockbox.append(
        ledgers,
        bundle_id=str(audit["bundle_manifest"]["bundle_id"]),
        freeze_data_cutoff=str(signal_day.date()),
        metadata={
            "model_profile_id": config["identity"]["model_profile_id"],
            "feature_set_id": feature_set.id,
            "observed_through": config["timeline"]["observed_through"],
            "capacity": capacity,
            "preflight_source_cutoffs": preflight["source_cutoffs"],
            "training_audit": audit,
        },
    )
    return {
        "status": "FORWARD_SIGNAL_FROZEN",
        "signal_date": str(signal_day.date()),
        "manifest_sha256": manifest["manifest_sha256"],
        "model_sha256": audit["model_audit"]["model_sha256"],
        "prediction_rows": audit["prediction_rows"],
        "candidate_rows": int(len(ledgers["candidate"])),
        "lockbox_written": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--lockbox", type=Path, default=DEFAULT_LOCKBOX)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
