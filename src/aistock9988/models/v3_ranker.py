"""Monthly walk-forward candidate ranker for the V3 hybrid strategy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from ..configuration import ModelConfig, StrategyConfig
from ..time.session import session_close, session_open
from .trainer import train_ranker


def build_h10_label_ledger(execution_panel: pd.DataFrame, model: ModelConfig) -> pd.DataFrame:
    """Build T+1-open to H-open labels on the global exchange-session grid."""
    delay = int(model.label["entry_delay_sessions"])
    horizon = int(model.label["horizon_sessions"])
    exit_shift = delay + horizon
    frame = execution_panel.sort_values(["ts_code", "trade_date"], kind="mergesort").copy()
    grouped = frame.groupby("ts_code", sort=False)
    frame["entry_date"] = grouped["trade_date"].shift(-delay)
    frame["exit_date"] = grouped["trade_date"].shift(-exit_shift)
    frame["entry_open"] = grouped["economic_open"].shift(-delay)
    frame["exit_open"] = grouped["economic_open"].shift(-exit_shift)
    frame["entry_status"] = grouped["execution_status"].shift(-delay)
    frame["exit_status"] = grouped["execution_status"].shift(-exit_shift)
    frame["entry_data_eligible"] = grouped["execution_data_eligible"].shift(-delay)
    frame["exit_data_eligible"] = grouped["execution_data_eligible"].shift(-exit_shift)
    frame["label_return"] = frame["exit_open"] / frame["entry_open"] - 1.0
    frame["label_available_time"] = frame["exit_date"].map(
        lambda value: session_open(value) if pd.notna(value) else pd.NaT
    )
    finite = np.isfinite(frame[["entry_open", "exit_open", "label_return"]].to_numpy(dtype=float)).all(axis=1)
    frame["label_eligible"] = (
        frame["entry_data_eligible"].fillna(False).astype(bool)
        & frame["exit_data_eligible"].fillna(False).astype(bool)
        & frame["entry_status"].eq("TRADABLE")
        & frame["exit_status"].eq("TRADABLE")
        & finite
    )
    frame["label_rejection_reason"] = np.select(
        [
            frame["entry_date"].isna() | frame["exit_date"].isna(),
            ~frame["entry_data_eligible"].fillna(False).astype(bool),
            ~frame["exit_data_eligible"].fillna(False).astype(bool),
            ~frame["entry_status"].eq("TRADABLE"),
            ~frame["exit_status"].eq("TRADABLE"),
            ~finite,
        ],
        ["LABEL_NOT_MATURE", "ENTRY_DATA_INELIGIBLE", "EXIT_DATA_INELIGIBLE", "ENTRY_NOT_TRADABLE", "EXIT_NOT_TRADABLE", "NONFINITE_LABEL"],
        default="",
    )
    return frame[[
        "trade_date", "ts_code", "entry_date", "exit_date", "entry_open", "exit_open",
        "label_return", "label_available_time", "label_eligible", "label_rejection_reason",
    ]].rename(columns={"trade_date": "asof"}).sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


def monthly_walkforward_predictions(
    *,
    feature_ledger: pd.DataFrame,
    score_ledger: pd.DataFrame,
    label_ledger: pd.DataFrame,
    strategy: StrategyConfig,
    model: ModelConfig,
    prediction_sessions: tuple[str, ...],
    all_sessions: tuple[str, ...],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Train once per month and score that month's Stage1 candidates."""
    feature_names = tuple(str(value) for value in model.features["ids"])
    candidate_keys = score_ledger.loc[
        score_ledger["stage1_pass"].astype(bool),
        ["asof", "ts_code", "rule_score", "training_data_eligible"],
    ]
    candidates = candidate_keys.merge(
        feature_ledger[["asof", "ts_code", *feature_names]],
        on=["asof", "ts_code"], how="inner", validate="one_to_one",
    ).merge(label_ledger, on=["asof", "ts_code"], how="left", validate="one_to_one")
    prediction_days = pd.DatetimeIndex(pd.to_datetime(prediction_sessions, utc=True)).normalize()
    sessions = pd.DatetimeIndex(pd.to_datetime(all_sessions, utc=True)).normalize().sort_values()
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    months = prediction_days.tz_localize(None).to_period("M").unique().sort_values()

    for month in months:
        test_days = prediction_days[prediction_days.tz_localize(None).to_period("M") == month]
        test = candidates[candidates["asof"].isin(test_days)].copy()
        if test.empty:
            audits.append({"month": str(month), "status": "NO_STAGE1_CANDIDATES"})
            continue
        month_start = test_days[0]
        prior = sessions[sessions < month_start]
        if prior.empty:
            raise ValueError(f"no training cutoff session before {month}")
        cutoff = prior[-1]
        cutoff_time = session_close(cutoff)
        window_start = cutoff - pd.DateOffset(months=int(model.walk_forward["training_window_months"]))
        train = candidates[
            candidates["asof"].gt(window_start)
            & candidates["asof"].le(cutoff)
            & candidates["training_data_eligible"].astype(bool)
            & candidates["label_eligible"].fillna(False).astype(bool)
            & pd.to_datetime(candidates["label_available_time"], utc=True).le(cutoff_time)
        ].copy()
        finite_train = np.isfinite(train[list(feature_names)].to_numpy(dtype=float)).all(axis=1)
        train = train.loc[finite_train].sort_values(["asof", "ts_code"], kind="mergesort")
        group_size = train.groupby("asof", sort=False)["ts_code"].transform("size")
        train = train[group_size >= 2].reset_index(drop=True)
        if len(train) < int(model.walk_forward["minimum_training_rows"]):
            raise ValueError(f"insufficient mature Stage1 training rows for {month}: {len(train)}")
        if pd.to_datetime(train["label_available_time"], utc=True).max() > cutoff_time:
            raise AssertionError(f"immature label entered training for {month}")
        finite_test = np.isfinite(test[list(feature_names)].to_numpy(dtype=float)).all(axis=1)
        if not finite_test.all():
            raise ValueError(f"nonfinite prediction features for {month}")

        relevance = _relevance(train["label_return"], train["asof"])
        sample_hash = _sample_key_hash(train)
        model_id = f"{model.model_id}_{month.strftime('%Y%m')}_cutoff_{cutoff.strftime('%Y%m%d')}"
        artifact = train_ranker(
            train[list(feature_names)].reset_index(drop=True),
            relevance.reset_index(drop=True),
            group_dates=train["asof"].reset_index(drop=True),
            feature_set_id=model.feature_set_id,
            label_profile_id=model.label_profile_id,
            training_cutoff=str(cutoff_time),
            model_id=model_id,
            output_dir=output_dir,
            params=dict(model.params),
            metadata_extra={
                "strategy_hash": strategy.config_hash,
                "model_config_hash": model.config_hash,
                "stage1_contract_hash": _stage1_hash(strategy),
                "training_sample_key_hash": sample_hash,
                "training_population": "stage1_pass_and_data_eligible",
            },
        )
        fitted = XGBRanker()
        fitted.load_model(output_dir / f"{model_id}.json")
        test["model_score"] = fitted.predict(test[list(feature_names)])
        test["model_id"] = model_id
        test["model_cutoff"] = cutoff_time
        predictions.append(test[["asof", "ts_code", "model_score", "model_id", "model_cutoff"]])
        audits.append({
            "month": str(month), "status": "TRAINED", "model_id": model_id,
            "model_sha256": artifact.model_sha256, "cutoff": str(cutoff_time),
            "window_start_exclusive": str(window_start), "train_rows": len(train),
            "train_dates": int(train["asof"].nunique()), "train_start": str(train["asof"].min()),
            "train_end": str(train["asof"].max()),
            "max_label_available_time": str(pd.to_datetime(train["label_available_time"], utc=True).max()),
            "training_sample_key_hash": sample_hash, "test_rows": len(test),
            "test_dates": int(test["asof"].nunique()),
        })
    if not predictions:
        raise ValueError("monthly walk-forward produced no predictions")
    return pd.concat(predictions, ignore_index=True), audits


def build_hybrid_ledgers(
    rule_ledgers: dict[str, pd.DataFrame],
    predictions: pd.DataFrame,
    strategy: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    """Replace only the Stage2 ordering while preserving the frozen Stage1 pool."""
    score = rule_ledgers["score"].drop(columns=["model_score", "final_score"]).merge(
        predictions[["asof", "ts_code", "model_score", "model_id", "model_cutoff"]],
        on=["asof", "ts_code"], how="left", validate="one_to_one",
    )
    score = score.merge(
        rule_ledgers["candidate"][["asof", "ts_code", "execution_status"]],
        on=["asof", "ts_code"], how="left", validate="one_to_one",
    )
    missing = score["stage1_pass"].astype(bool) & score["model_score"].isna()
    if missing.any():
        raise ValueError(f"Stage1 candidates without model predictions: {int(missing.sum())}")
    score["final_score"] = score["model_score"]
    score["candidate_rank"] = pd.Series(pd.NA, index=score.index, dtype="Int64")
    passed = score[score["stage1_pass"]].sort_values(
        ["asof", "model_score", "rule_score", "ts_code"],
        ascending=[True, False, False, True], kind="mergesort",
    )
    score.loc[passed.index, "candidate_rank"] = (passed.groupby("asof").cumcount() + 1).astype("Int64")
    view_size = int(strategy.portfolio["candidate_view_size"])
    score["candidate_status"] = np.select(
        [~score["stage1_pass"].astype(bool), score["candidate_rank"].fillna(view_size + 1).astype(int).le(view_size)],
        ["REJECTED", "IN_VIEW"], default="BELOW_VIEW",
    )
    score["candidate_snapshot_id"] = ""
    snapshots: dict[pd.Timestamp, str] = {}
    for day, group in score[score["candidate_status"].eq("IN_VIEW")].groupby("asof", sort=True):
        ordered = group.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        payload = "|".join(f"{row.ts_code}:{int(row.candidate_rank)}" for row in ordered.itertuples())
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    score["candidate_snapshot_id"] = score["asof"].map(snapshots).fillna("")

    selection = rule_ledgers["selection"].copy()
    selection["candidate_snapshot_id"] = selection["asof"].map(snapshots).fillna("")
    policy_hash = hashlib.sha256(
        json.dumps({"strategy_hash": strategy.config_hash, "stage2": "monthly_xgb_ranker"}, sort_keys=True).encode()
    ).hexdigest()
    selection["policy_id"] = strategy.strategy_id
    selection["policy_hash"] = policy_hash
    selection["decision_id"] = selection.apply(
        lambda row: hashlib.sha256(
            f"{policy_hash}|{row['asof'].date()}|{row['candidate_snapshot_id']}".encode()
        ).hexdigest(),
        axis=1,
    )
    candidate_columns = list(rule_ledgers["candidate"].columns) + ["model_id", "model_cutoff"]
    candidate = score[candidate_columns].sort_values(
        ["asof", "candidate_rank", "ts_code"], kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    return {"score": score.reset_index(drop=True), "candidate": candidate, "selection": selection}


def _relevance(labels: pd.Series, dates: pd.Series) -> pd.Series:
    percentile = labels.groupby(dates).rank(method="first", pct=True)
    return np.minimum((percentile * 5).astype(int), 4).astype(float)


def _sample_key_hash(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row.asof.isoformat()}|{row.ts_code}|{float(row.label_return):.12g}"
        for row in frame[["asof", "ts_code", "label_return"]].itertuples(index=False)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _stage1_hash(strategy: StrategyConfig) -> str:
    payload = json.dumps(strategy.to_dict()["stage1"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["build_h10_label_ledger", "monthly_walkforward_predictions", "build_hybrid_ledgers"]
