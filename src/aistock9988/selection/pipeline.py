"""Configuration-driven rule scoring and frozen V3 selection ledgers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig


def build_rule_ledgers(
    feature_ledger: pd.DataFrame,
    strategy: StrategyConfig,
    signal_sessions: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    signal_days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    frame = feature_ledger[feature_ledger["asof"].isin(signal_days)].copy()
    if frame.empty:
        raise ValueError("feature ledger has no configured signal sessions")
    frame["stage1_pass"] = False
    ready = frame["feature_ready"].astype(bool)
    frame.loc[ready, "stage1_pass"] = evaluate_expression(frame.loc[ready], strategy.stage1["expression"])
    frame["rule_score"] = np.nan
    ranking_ready = (
        frame["ranking_feature_ready"].astype(bool)
        if "ranking_feature_ready" in frame
        else ready
    )
    rank_ready = ready & frame["stage1_pass"] & ranking_ready
    for term in strategy.ranking["terms"]:
        spec = dict(term)
        feature = str(spec["feature"])
        if feature not in frame:
            raise ValueError(f"ranking references unknown feature: {feature}")
        direction = str(spec["direction"])
        if direction not in {"asc", "desc"}:
            raise ValueError(f"invalid ranking direction: {direction}")
        term_score = frame.loc[rank_ready].groupby("asof", sort=True)[feature].rank(
            # The final score is sorted descending, so a larger percentile must
            # always mean "more preferred": asc favors smaller raw values and
            # desc favors larger raw values.
            method="average", pct=True, ascending=(direction == "desc")
        )
        current = frame.loc[rank_ready, "rule_score"].fillna(0.0)
        frame.loc[rank_ready, "rule_score"] = current + float(spec["weight"]) * term_score
    ranking_rejection = (
        frame["ranking_feature_rejection_reason"]
        if "ranking_feature_rejection_reason" in frame
        else pd.Series("RANKING_FEATURE_NOT_READY", index=frame.index)
    )
    frame["score_rejection_reason"] = np.select(
        [
            ~frame["universe_pass"].astype(bool),
            ~frame["feature_ready"].astype(bool),
            ~frame["stage1_pass"],
            ~ranking_ready,
        ],
        [
            "UNIVERSE_REJECTED",
            frame["feature_rejection_reason"],
            "STAGE1_REJECTED",
            ranking_rejection,
        ],
        default="",
    )
    frame["candidate_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    passed = frame[rank_ready].sort_values(
        ["asof", "rule_score", "ts_code"], ascending=[True, False, True], kind="mergesort"
    )
    ranks = passed.groupby("asof", sort=True).cumcount() + 1
    frame.loc[passed.index, "candidate_rank"] = ranks.astype("Int64")
    view_size = int(strategy.portfolio["candidate_view_size"])
    in_view = frame["candidate_rank"].fillna(view_size + 1).astype(int).le(view_size).to_numpy(dtype=bool)
    frame["candidate_status"] = np.select(
        [
            (~frame["stage1_pass"]).to_numpy(dtype=bool),
            (~ranking_ready).to_numpy(dtype=bool),
            in_view,
        ],
        ["REJECTED", "RANKING_DATA_MISSING", "IN_VIEW"],
        default="BELOW_VIEW",
    )
    frame["candidate_snapshot_id"] = ""
    empty_snapshot = hashlib.sha256(b"").hexdigest()
    snapshots: dict[pd.Timestamp, str] = {day: empty_snapshot for day in signal_days}
    for day, group in frame[frame["candidate_status"].eq("IN_VIEW")].groupby("asof", sort=True):
        ordered = group.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        payload = "|".join(f"{row.ts_code}:{int(row.candidate_rank)}" for row in ordered.itertuples())
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    frame["candidate_snapshot_id"] = frame["asof"].map(snapshots).fillna("")

    policy_payload = {
        "strategy_id": strategy.strategy_id,
        "strategy_hash": strategy.config_hash,
        "portfolio": strategy.to_dict()["portfolio"],
    }
    policy_hash = hashlib.sha256(
        json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decision_rows: list[dict[str, Any]] = []
    primary_end = int(strategy.portfolio["entries_per_decision"])
    for day in signal_days:
        snapshot_id = snapshots.get(day, "")
        decision_id = hashlib.sha256(f"{policy_hash}|{day.date()}|{snapshot_id}".encode()).hexdigest()
        decision_rows.append({
            "decision_id": decision_id,
            "asof": day,
            "desired_entries": primary_end,
            "target_weight_each": float(strategy.portfolio["sizing"]["value"]),
            "primary_rank_end": primary_end,
            "replacement_rank_end": view_size,
            "candidate_snapshot_id": snapshot_id,
            "policy_id": strategy.strategy_id,
            "policy_hash": policy_hash,
            "context_hash": hashlib.sha256(f"{day.date()}|{strategy.config_hash}".encode()).hexdigest(),
        })
    score_columns = [
        "asof", "ts_code", "bundle_id", "feature_set_hash", "universe_pass",
        "selection_data_eligible", "training_data_eligible", "execution_data_eligible",
        "missing_required_selection", "missing_required_training", "missing_required_execution",
        "missing_optional", "feature_ready",
        "ranking_feature_ready", "stage1_pass", "rule_score",
        "score_rejection_reason", "market_coverage",
    ]
    candidate_columns = score_columns + [
        "candidate_rank", "candidate_status", "candidate_snapshot_id", "execution_status",
    ]
    return {
        "score": frame[score_columns].sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True),
        "candidate": frame[candidate_columns].sort_values(["asof", "candidate_rank", "ts_code"], kind="mergesort", na_position="last").reset_index(drop=True),
        "selection": pd.DataFrame(decision_rows),
    }


def evaluate_expression(frame: pd.DataFrame, node: Mapping[str, Any]) -> pd.Series:
    if "all" in node:
        result = pd.Series(True, index=frame.index)
        for child in node["all"]:
            result &= evaluate_expression(frame, child)
        return result
    if "any" in node:
        result = pd.Series(False, index=frame.index)
        for child in node["any"]:
            result |= evaluate_expression(frame, child)
        return result
    if "not" in node:
        return ~evaluate_expression(frame, node["not"])
    left = pd.to_numeric(frame[str(node["left"])], errors="raise")
    op = str(node["op"])
    if op == "between":
        return left.between(float(node["lower"]), float(node["upper"]), inclusive="both")
    right = pd.to_numeric(frame[str(node["right"])], errors="raise") if "right" in node else float(node["value"])
    return {
        "gt": left.gt,
        "ge": left.ge,
        "lt": left.lt,
        "le": left.le,
    }[op](right)


__all__ = ["build_rule_ledgers", "evaluate_expression"]
