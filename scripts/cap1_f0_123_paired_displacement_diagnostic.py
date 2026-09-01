#!/usr/bin/env python3
"""Attribute the rejected CAP1 F0-123 ranker's promoted/displaced trades."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import conditional_f0_123_reset_ranker_runner as ranker
from aistock9988.configuration import StrategyConfig
from aistock9988.features.registry import FeatureSet


FORBIDDEN_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle", ".joblib", ".model", ".bin"}
METRIC_FIELDS = (
    "total_return",
    "portfolio_profit_factor",
    "max_drawdown",
    "trade_win_rate",
    "return_excluding_best_week",
    "return_excluding_top3_profit",
    "trade_count",
)
SOURCE_NAME = "CAP1_F0_123_EXECUTABLE_RANKER_V1_2026_TO_DB_CUTOFF_20260902"
SOURCE_MANIFEST_SHA256 = "fc076d6565ca32ee1a465804edde6f42b257aa220a0b2b3b2e5c044703305735"
SOURCE_STRATEGY_SHA256 = "6d0af91e975a4a4d1e28ce4f508f06c94a3bb6b9e5b6f72db2f366ed5e81404d"
SOURCE_PROFILE_SHA256 = "1f44fa586254fa15e3206a3c7a9760a4b427713c57b1c7bcb9acb9a6f28cf57f"
PREREG_NAME = "CAP1_F0_123_PAIRED_DISPLACEMENT_DIAGNOSTIC_V1_PREREG_20260902.md"
PREREG_SHA256 = "9d80f90e65cba1c48b04745bcb347782034f84b8e965efc921591eb8db516a56"
STOCK_CODE_PATTERN = re.compile(r"\b\d{6}\.(?:SZ|SH|BJ)\b", re.IGNORECASE)
ALLOWED_AGGREGATE_KEYS = frozenset({
    "status", "completed_at", "source", "scope", "reproduction",
    "selection_attribution", "paired_displacements", "prebuy_state_audit",
    "nav_bridge", "monthly_paired_displacements", "rank_quality",
    "top_winner_loser_features", "top_promoted_displaced_features", "constraints",
    "source_run", "artifact_count", "artifact_manifest_verified",
    "source_verification_passed", "source_acceptance_passed", "code_manifest_verified",
    "source_cutoffs_verified", "source_cutoffs", "stock_factor_pro_ts", "daily_basic_ts",
    "market_daily_ts", "adj_factor_ts", "stk_limit_ts", "performance_years",
    "signal_start", "signal_end", "execution_end", "training_input_start",
    "training_input_end", "historical_2024_2025_portfolio_validation", "passed",
    "failed", "maximum_absolute_metric_error", "model_hashes_match",
    "eight_monthly_models", "shared", "displaced", "promoted", "control_entries",
    "challenger_entries", "jaccard", "events", "closed", "mean_return",
    "median_return", "win_rate", "return_ge_10_rate", "stop_rate", "time_exit_rate",
    "realized_pnl", "pairs", "closed_pairs", "mean_return_delta",
    "median_return_delta", "nonadditive_paired_realized_pnl_delta",
    "challenger_pair_win_rate", "isolated_pair_metrics_are_portfolio_bridge",
    "winner_transitions", "winner_to_winner", "winner_to_loser", "loser_to_winner",
    "loser_to_loser", "coverage", "direct_same_prebuy_state",
    "downstream_path_divergence", "displaced_events", "promoted_events",
    "paired_events", "unpaired_displaced", "unpaired_promoted", "pair_coverage",
    "control_buy_signal_days", "challenger_buy_signal_days", "shared_buy_signal_days",
    "same_prebuy_state_days", "downstream_diverged_state_days",
    "maximum_prebuy_cash_difference", "control_final_nav", "challenger_final_nav",
    "final_nav_delta", "bridge_delta", "control_contribution_identity_error",
    "challenger_contribution_identity_error", "bridge_identity_error", "categories",
    "shared_sizing_and_path_effect", "direct_same_prebuy_state_displacement",
    "gross_price_pnl", "fees_taxes", "dividends", "nav_contribution", "control_trades",
    "challenger_trades", "month", "direct_pairs", "downstream_pairs", "control",
    "challenger", "all_candidates", "rank_1_5", "rank_6_10",
    "score_spearman_to_return", "feature", "winner_minus_loser_standardized",
    "promoted_minus_displaced_standardized", "valid_n", "winner_n", "loser_n",
    "winner_mean", "winner_median", "loser_mean", "loser_median", "spearman_to_return",
    "displaced_n", "promoted_n", "displaced_mean", "promoted_mean", "model_repaired",
    "strategy_changed", "parameter_scan_performed", "raw_business_data_written",
    "wide_202_factor_system_used", "stock_codes_or_individual_sessions_persisted",
    "source_artifact_manifest_sha256", "source_strategy_sha256",
    "source_model_profile_sha256", "source_preregistration_sha256",
    "diagnostic_preregistration_sha256", "diagnostic_code_sha256", "base_runner_sha256",
    "feature_manifest_sha256",
})


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _assert_aggregate_schema(
    payload: Any, *, location: str = "root", extra_keys: frozenset[str] = frozenset()
) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) not in ALLOWED_AGGREGATE_KEYS | extra_keys:
                raise AssertionError(f"aggregate output key is not allowlisted at {location}.{key}")
            _assert_aggregate_schema(value, location=f"{location}.{key}", extra_keys=extra_keys)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_aggregate_schema(value, location=f"{location}[{index}]", extra_keys=extra_keys)


def _assert_output_privacy(output: Path) -> None:
    forbidden_files = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    if forbidden_files:
        raise AssertionError(f"forbidden business-data artifacts persisted: {forbidden_files}")
    leaked_codes = []
    for path in output.rglob("*"):
        if path.is_file() and STOCK_CODE_PATTERN.search(path.read_text(encoding="utf-8")):
            leaked_codes.append(str(path.relative_to(output)))
    if leaked_codes:
        raise AssertionError(f"stock-code content leaked into aggregate output: {leaked_codes}")


def _pct(value: Any) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):+.2%}"


def _rate(value: Any) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.1%}"


def _verify_source(source: Path) -> dict[str, Any]:
    if source.name != SOURCE_NAME:
        raise ValueError(f"source run must be {SOURCE_NAME}")
    required = {
        "RUN_STATUS.json",
        "verification.json",
        "acceptance.json",
        "control_metrics.json",
        "challenger_metrics.json",
        "model_manifest.json",
        "configs/strategy.yaml",
        "configs/model_profile.yaml",
        "configs/preregistration.md",
        "manifests/artifact_manifest.json",
        "manifests/code_manifest.json",
        "plan.json",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"source run is incomplete: {missing}")
    status = _load_json(source / "RUN_STATUS.json")
    verification = _load_json(source / "verification.json")
    acceptance = _load_json(source / "acceptance.json")
    if (
        status.get("status") != "DIAGNOSTIC_COMPLETED"
        or status.get("verification_passed") is not True
        or status.get("acceptance_passed") is not False
        or status.get("business_data_persisted") is not False
        or verification.get("passed") is not True
        or acceptance.get("passed") is not False
    ):
        raise ValueError("source run is not the verified rejected CAP1 F0-123 run")
    manifest_path = source / "manifests/artifact_manifest.json"
    if _sha(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source artifact manifest differs from preregistration")
    if _sha(source / "configs/strategy.yaml") != SOURCE_STRATEGY_SHA256:
        raise ValueError("source strategy differs from preregistration")
    if _sha(source / "configs/model_profile.yaml") != SOURCE_PROFILE_SHA256:
        raise ValueError("source model profile differs from preregistration")
    manifest = _load_json(manifest_path)
    bad_hashes = []
    for relative, expected in manifest.items():
        path = source / relative
        if not path.is_file() or _sha(path) != expected["sha256"]:
            bad_hashes.append(relative)
    if bad_hashes:
        raise ValueError(f"source artifact hash mismatch: {bad_hashes}")
    code_manifest = _load_json(source / "manifests/code_manifest.json")
    bad_code = []
    for relative, expected_sha in code_manifest.items():
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected_sha:
            bad_code.append(relative)
    if bad_code:
        raise ValueError(f"current code differs from source code manifest: {bad_code}")
    return {
        "source_run": source.name,
        "artifact_count": len(manifest),
        "artifact_manifest_verified": True,
        "source_verification_passed": True,
        "source_acceptance_passed": False,
        "code_manifest_verified": True,
    }


def _verify_source_cutoffs(source: Path) -> dict[str, str]:
    frozen = _load_json(source / "plan.json")["source_cutoffs"]
    current = ranker._source_cutoffs()
    if current != frozen:
        raise ValueError(f"source cutoffs differ from frozen run: current={current} frozen={frozen}")
    return current


def _trade_frame(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    chosen = result["execution_decisions"]
    chosen = chosen[chosen["chosen"].astype(bool)][
        ["decision_id", "signal_session", "execution_session", "attempt_no", "candidate_rank", "ts_code"]
    ].copy()
    buys = result["fills"][result["fills"]["side"].eq("BUY")][
        [
            "order_id", "decision_id", "ts_code", "trade_date", "shares",
            "gross_value", "commission", "stamp_duty", "cash_after",
        ]
    ].rename(columns={
        "order_id": "buy_order_id",
        "trade_date": "buy_date",
        "gross_value": "buy_gross",
        "commission": "buy_commission",
        "stamp_duty": "buy_stamp_duty",
        "cash_after": "buy_cash_after",
    })
    sells = result["fills"][result["fills"]["side"].eq("SELL")][
        [
            "decision_id", "ts_code", "trade_date", "gross_value", "commission",
            "stamp_duty", "economic_return", "realized_pnl", "reason",
        ]
    ].rename(columns={
        "trade_date": "sell_date",
        "gross_value": "sell_gross",
        "commission": "sell_commission",
        "stamp_duty": "sell_stamp_duty",
        "reason": "exit_reason",
    })
    buy_match = chosen[["decision_id", "ts_code"]].merge(
        buys[["decision_id", "ts_code"]],
        on=["decision_id", "ts_code"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(buy_match) != len(chosen) or not buy_match["_merge"].eq("both").all():
        counts = buy_match["_merge"].value_counts().to_dict()
        raise AssertionError(f"chosen decisions and BUY fills differ: {counts}")
    trades = chosen.merge(buys, on=["decision_id", "ts_code"], validate="one_to_one")
    trades = trades.merge(sells, on=["decision_id", "ts_code"], how="left", validate="one_to_one")
    if trades.duplicated(["signal_session", "ts_code"]).any():
        raise ValueError("actual trade keys are not unique")
    trades["signal_session"] = pd.to_datetime(trades["signal_session"], utc=True).dt.normalize()
    trades["closed"] = trades["sell_date"].notna()
    actions = result["corporate_actions"]
    if actions.empty:
        trades["dividends"] = 0.0
    else:
        dividends = actions.groupby(["decision_id", "ts_code"], as_index=False)["cash_dividend"].sum()
        trades = trades.merge(dividends, on=["decision_id", "ts_code"], how="left", validate="one_to_one")
        trades["dividends"] = pd.to_numeric(trades["cash_dividend"], errors="coerce").fillna(0.0)
        trades = trades.drop(columns="cash_dividend")
    open_positions = result["open_positions"]
    if open_positions.empty:
        trades["ending_mark"] = np.nan
    else:
        marks = open_positions[["ts_code", "shares", "last_raw_close"]].copy()
        marks["ending_mark"] = (
            pd.to_numeric(marks["shares"], errors="coerce")
            * pd.to_numeric(marks["last_raw_close"], errors="coerce")
        )
        trades = trades.merge(
            marks[["ts_code", "ending_mark"]], on="ts_code", how="left", validate="many_to_one"
        )
    trades["exit_or_mark_gross"] = pd.to_numeric(trades["sell_gross"], errors="coerce").where(
        trades["closed"], pd.to_numeric(trades["ending_mark"], errors="coerce")
    )
    trades["gross_price_pnl"] = trades["exit_or_mark_gross"] - pd.to_numeric(
        trades["buy_gross"], errors="coerce"
    )
    trades["fees_taxes"] = -(
        pd.to_numeric(trades["buy_commission"], errors="coerce").fillna(0.0)
        + pd.to_numeric(trades["buy_stamp_duty"], errors="coerce").fillna(0.0)
        + pd.to_numeric(trades["sell_commission"], errors="coerce").fillna(0.0)
        + pd.to_numeric(trades["sell_stamp_duty"], errors="coerce").fillna(0.0)
    )
    trades["nav_contribution"] = (
        trades["gross_price_pnl"] + trades["fees_taxes"] + trades["dividends"]
    )
    closed_error = (
        pd.to_numeric(trades.loc[trades["closed"], "nav_contribution"], errors="coerce")
        - pd.to_numeric(trades.loc[trades["closed"], "realized_pnl"], errors="coerce")
    ).abs()
    if len(closed_error) and float(closed_error.max()) > 1e-8:
        raise AssertionError("closed-trade NAV contribution differs from realized PnL")
    return trades.sort_values(["signal_session", "candidate_rank", "ts_code"], kind="mergesort")


def _return_stats(frame: pd.DataFrame, column: str = "economic_return") -> dict[str, Any]:
    values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
    valid = frame.loc[values.notna()].copy()
    values = values[values.notna()].astype(float)
    reasons = valid.get("exit_reason", pd.Series(index=valid.index, dtype=object)).astype(str)
    return {
        "events": int(len(frame)),
        "closed": int(len(valid)),
        "mean_return": float(values.mean()) if len(values) else None,
        "median_return": float(values.median()) if len(values) else None,
        "win_rate": float(values.gt(0).mean()) if len(values) else None,
        "return_ge_10_rate": float(values.ge(0.10).mean()) if len(values) else None,
        "stop_rate": float(reasons.eq("STOP_LOSS").mean()) if len(valid) else None,
        "time_exit_rate": float(reasons.eq("TIME_EXIT").mean()) if len(valid) else None,
        "realized_pnl": float(pd.to_numeric(valid.get("realized_pnl"), errors="coerce").sum())
        if "realized_pnl" in valid else None,
    }


def _selection_sets(
    control: pd.DataFrame, challenger: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["signal_session", "ts_code"]
    membership = control[keys].merge(challenger[keys], on=keys, how="outer", indicator=True)
    shared_keys = membership[membership["_merge"].eq("both")][keys]
    control_only_keys = membership[membership["_merge"].eq("left_only")][keys]
    challenger_only_keys = membership[membership["_merge"].eq("right_only")][keys]
    shared = control.merge(shared_keys, on=keys, validate="one_to_one")
    displaced = control.merge(control_only_keys, on=keys, validate="one_to_one")
    promoted = challenger.merge(challenger_only_keys, on=keys, validate="one_to_one")
    return shared, displaced, promoted


def _prebuy_states(
    result: dict[str, pd.DataFrame], trades: pd.DataFrame
) -> dict[pd.Timestamp, dict[str, Any]]:
    positions = result["positions"].copy()
    if not positions.empty:
        positions["trade_date"] = pd.to_datetime(positions["trade_date"], utc=True).dt.normalize()
        positions["entry_date"] = pd.to_datetime(positions["entry_date"], utc=True).dt.normalize()
    fills = result["fills"].copy()
    fills["trade_date"] = pd.to_datetime(fills["trade_date"], utc=True).dt.normalize()
    actions = result["corporate_actions"].copy()
    if not actions.empty:
        actions["trade_date"] = pd.to_datetime(actions["trade_date"], utc=True).dt.normalize()
    states: dict[pd.Timestamp, dict[str, Any]] = {}
    for signal_day, group in trades.groupby("signal_session", sort=True):
        execution_days = pd.DatetimeIndex(pd.to_datetime(group["execution_session"], utc=True).unique())
        if len(execution_days) != 1:
            raise ValueError("one signal session maps to multiple execution sessions")
        execution_day = execution_days[0]
        prior = positions[positions["trade_date"].eq(signal_day)][
            ["ts_code", "shares", "entry_date", "state"]
        ].copy() if not positions.empty else pd.DataFrame(
            columns=["ts_code", "shares", "entry_date", "state"]
        )
        if not actions.empty:
            day_actions = actions[actions["trade_date"].eq(execution_day)]
            action_shares = day_actions.set_index("ts_code")["shares_after"].to_dict()
            prior["shares"] = prior.apply(
                lambda row: action_shares.get(str(row["ts_code"]), row["shares"]), axis=1
            )
        sold = set(
            fills.loc[
                fills["trade_date"].eq(execution_day) & fills["side"].eq("SELL"), "ts_code"
            ].astype(str)
        )
        prior = prior[~prior["ts_code"].astype(str).isin(sold)]
        signature = tuple(
            (str(row.ts_code), float(row.shares), str(pd.Timestamp(row.entry_date).date()), str(row.state))
            for row in prior.sort_values("ts_code", kind="mergesort").itertuples(index=False)
        )
        day_buys = fills[
            fills["trade_date"].eq(execution_day) & fills["side"].eq("BUY")
        ].sort_values("order_id", kind="mergesort")
        if day_buys.empty:
            raise ValueError("actual trade group has no buy fill")
        first_buy = day_buys.iloc[0]
        cash_before = float(
            first_buy["cash_after"] + first_buy["gross_value"] + first_buy["commission"]
            + first_buy["stamp_duty"]
        )
        states[pd.Timestamp(signal_day)] = {
            "execution_day": pd.Timestamp(execution_day),
            "holding_signature": signature,
            "slots": 5 - len(signature),
            "cash_before": cash_before,
            "buy_count": int(len(day_buys)),
        }
    return states


def _direct_state_days(
    control_result: dict[str, pd.DataFrame],
    challenger_result: dict[str, pd.DataFrame],
    control_trades: pd.DataFrame,
    challenger_trades: pd.DataFrame,
) -> tuple[set[pd.Timestamp], dict[str, Any]]:
    control = _prebuy_states(control_result, control_trades)
    challenger = _prebuy_states(challenger_result, challenger_trades)
    shared_days = sorted(set(control) & set(challenger))
    direct: set[pd.Timestamp] = set()
    maximum_cash_error = 0.0
    for day in shared_days:
        left = control[day]
        right = challenger[day]
        cash_error = abs(float(left["cash_before"]) - float(right["cash_before"]))
        maximum_cash_error = max(maximum_cash_error, cash_error)
        if (
            left["execution_day"] == right["execution_day"]
            and left["holding_signature"] == right["holding_signature"]
            and left["slots"] == right["slots"]
            and cash_error <= 1e-8
        ):
            direct.add(day)
    return direct, {
        "control_buy_signal_days": len(control),
        "challenger_buy_signal_days": len(challenger),
        "shared_buy_signal_days": len(shared_days),
        "same_prebuy_state_days": len(direct),
        "downstream_diverged_state_days": len(shared_days) - len(direct),
        "maximum_prebuy_cash_difference": maximum_cash_error,
    }


def _pair_exclusives(
    displaced: pd.DataFrame,
    promoted: pd.DataFrame,
    direct_days: set[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    unpaired_displaced = 0
    unpaired_promoted = 0
    days = sorted(set(displaced["signal_session"]) | set(promoted["signal_session"]))
    for day in days:
        left = displaced[displaced["signal_session"].eq(day)].sort_values(
            ["candidate_rank", "ts_code"], kind="mergesort"
        )
        right = promoted[promoted["signal_session"].eq(day)].sort_values(
            ["candidate_rank", "ts_code"], kind="mergesort"
        )
        pair_count = min(len(left), len(right))
        unpaired_displaced += len(left) - pair_count
        unpaired_promoted += len(right) - pair_count
        for slot, (control_row, challenger_row) in enumerate(zip(
            left.iloc[:pair_count].itertuples(index=False),
            right.iloc[:pair_count].itertuples(index=False),
        ), start=1):
            pairs.append({
                "signal_session": day,
                "pair_slot": slot,
                "causal_class": "DIRECT_SAME_PREBUY_STATE"
                if pd.Timestamp(day) in direct_days else "DOWNSTREAM_PATH_DIVERGENCE",
                "control_return": control_row.economic_return,
                "challenger_return": challenger_row.economic_return,
                "control_pnl": control_row.realized_pnl,
                "challenger_pnl": challenger_row.realized_pnl,
                "control_reason": control_row.exit_reason,
                "challenger_reason": challenger_row.exit_reason,
            })
    result = pd.DataFrame(pairs)
    if result.empty:
        return result, {
            "displaced_events": int(len(displaced)),
            "promoted_events": int(len(promoted)),
            "paired_events": 0,
            "unpaired_displaced": unpaired_displaced,
            "unpaired_promoted": unpaired_promoted,
            "pair_coverage": 0.0,
        }
    result["return_delta"] = (
        pd.to_numeric(result["challenger_return"], errors="coerce")
        - pd.to_numeric(result["control_return"], errors="coerce")
    )
    result["pnl_delta"] = (
        pd.to_numeric(result["challenger_pnl"], errors="coerce")
        - pd.to_numeric(result["control_pnl"], errors="coerce")
    )
    denominator = max(len(displaced), len(promoted))
    return result, {
        "displaced_events": int(len(displaced)),
        "promoted_events": int(len(promoted)),
        "paired_events": int(len(result)),
        "unpaired_displaced": unpaired_displaced,
        "unpaired_promoted": unpaired_promoted,
        "pair_coverage": float(len(result) / denominator) if denominator else None,
    }


def _pair_bucket(pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {"pairs": 0, "closed_pairs": 0}
    closed = pairs[pairs["control_return"].notna() & pairs["challenger_return"].notna()].copy()
    control_win = pd.to_numeric(closed["control_return"], errors="coerce").gt(0)
    challenger_win = pd.to_numeric(closed["challenger_return"], errors="coerce").gt(0)
    return {
        "pairs": int(len(pairs)),
        "closed_pairs": int(len(closed)),
        "mean_return_delta": float(closed["return_delta"].mean()) if len(closed) else None,
        "median_return_delta": float(closed["return_delta"].median()) if len(closed) else None,
        "nonadditive_paired_realized_pnl_delta": float(closed["pnl_delta"].sum())
        if len(closed) else None,
        "challenger_pair_win_rate": float(closed["return_delta"].gt(0).mean()) if len(closed) else None,
        "isolated_pair_metrics_are_portfolio_bridge": False,
        "winner_transitions": {
            "winner_to_winner": int((control_win & challenger_win).sum()),
            "winner_to_loser": int((control_win & ~challenger_win).sum()),
            "loser_to_winner": int((~control_win & challenger_win).sum()),
            "loser_to_loser": int((~control_win & ~challenger_win).sum()),
        },
    }


def _pair_summary(pairs: pd.DataFrame, coverage: dict[str, Any]) -> dict[str, Any]:
    overall = _pair_bucket(pairs)
    overall["coverage"] = coverage
    if pairs.empty:
        overall["direct_same_prebuy_state"] = {"pairs": 0, "closed_pairs": 0}
        overall["downstream_path_divergence"] = {"pairs": 0, "closed_pairs": 0}
        return overall
    overall["direct_same_prebuy_state"] = _pair_bucket(
        pairs[pairs["causal_class"].eq("DIRECT_SAME_PREBUY_STATE")]
    )
    overall["downstream_path_divergence"] = _pair_bucket(
        pairs[pairs["causal_class"].eq("DOWNSTREAM_PATH_DIVERGENCE")]
    )
    return overall


def _monthly_pairs(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    if pairs.empty:
        return []
    frame = pairs.copy()
    frame["month"] = frame["signal_session"].dt.strftime("%Y-%m")
    rows = []
    for month, group in frame.groupby("month", sort=True):
        closed = group[group["return_delta"].notna()]
        rows.append({
            "month": month,
            "pairs": int(len(group)),
            "closed_pairs": int(len(closed)),
            "mean_return_delta": float(closed["return_delta"].mean()) if len(closed) else None,
            "nonadditive_paired_realized_pnl_delta": float(closed["pnl_delta"].sum())
            if len(closed) else None,
            "challenger_pair_win_rate": float(closed["return_delta"].gt(0).mean()) if len(closed) else None,
            "direct_pairs": int(group["causal_class"].eq("DIRECT_SAME_PREBUY_STATE").sum()),
            "downstream_pairs": int(group["causal_class"].eq("DOWNSTREAM_PATH_DIVERGENCE").sum()),
        })
    return rows


def _bridge_components(control: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, Any]:
    fields = ("gross_price_pnl", "fees_taxes", "dividends", "nav_contribution")
    return {
        field: float(pd.to_numeric(challenger[field], errors="coerce").sum())
        - float(pd.to_numeric(control[field], errors="coerce").sum())
        for field in fields
    } | {
        "control_trades": int(len(control)),
        "challenger_trades": int(len(challenger)),
    }


def _nav_bridge(
    control_result: dict[str, pd.DataFrame],
    challenger_result: dict[str, pd.DataFrame],
    control: pd.DataFrame,
    challenger: pd.DataFrame,
    direct_days: set[pd.Timestamp],
    initial_cash: float,
) -> dict[str, Any]:
    if not np.isfinite(pd.to_numeric(control["nav_contribution"], errors="coerce")).all():
        raise AssertionError("control trade contribution is incomplete")
    if not np.isfinite(pd.to_numeric(challenger["nav_contribution"], errors="coerce")).all():
        raise AssertionError("challenger trade contribution is incomplete")
    keys = ["signal_session", "ts_code"]
    membership = control[keys].merge(challenger[keys], on=keys, how="outer", indicator=True)
    shared_keys = membership[membership["_merge"].eq("both")][keys]
    control_only_keys = membership[membership["_merge"].eq("left_only")][keys]
    challenger_only_keys = membership[membership["_merge"].eq("right_only")][keys]
    shared_control = control.merge(shared_keys, on=keys, validate="one_to_one")
    shared_challenger = challenger.merge(shared_keys, on=keys, validate="one_to_one")
    control_only = control.merge(control_only_keys, on=keys, validate="one_to_one")
    challenger_only = challenger.merge(challenger_only_keys, on=keys, validate="one_to_one")
    direct_control = control_only[control_only["signal_session"].isin(direct_days)]
    direct_challenger = challenger_only[challenger_only["signal_session"].isin(direct_days)]
    downstream_control = control_only[~control_only["signal_session"].isin(direct_days)]
    downstream_challenger = challenger_only[~challenger_only["signal_session"].isin(direct_days)]
    categories = {
        "shared_sizing_and_path_effect": _bridge_components(shared_control, shared_challenger),
        "direct_same_prebuy_state_displacement": _bridge_components(
            direct_control, direct_challenger
        ),
        "downstream_path_divergence": _bridge_components(
            downstream_control, downstream_challenger
        ),
    }
    control_final_nav = float(control_result["nav"].iloc[-1]["nav"])
    challenger_final_nav = float(challenger_result["nav"].iloc[-1]["nav"])
    control_contribution = float(control["nav_contribution"].sum())
    challenger_contribution = float(challenger["nav_contribution"].sum())
    final_nav_delta = challenger_final_nav - control_final_nav
    bridge_delta = sum(row["nav_contribution"] for row in categories.values())
    control_identity_error = abs(control_final_nav - initial_cash - control_contribution)
    challenger_identity_error = abs(
        challenger_final_nav - initial_cash - challenger_contribution
    )
    bridge_error = abs(final_nav_delta - bridge_delta)
    passed = max(control_identity_error, challenger_identity_error, bridge_error) <= 1e-8
    return {
        "passed": passed,
        "control_final_nav": control_final_nav,
        "challenger_final_nav": challenger_final_nav,
        "final_nav_delta": final_nav_delta,
        "bridge_delta": bridge_delta,
        "control_contribution_identity_error": control_identity_error,
        "challenger_contribution_identity_error": challenger_identity_error,
        "bridge_identity_error": bridge_error,
        "categories": categories,
    }


def _label_stats(frame: pd.DataFrame) -> dict[str, Any]:
    renamed = frame.rename(columns={"label_economic_return": "economic_return", "label_trigger_type": "exit_reason"})
    return _return_stats(renamed)


def _rank_quality(
    ledgers: dict[str, dict[str, pd.DataFrame]], labels: pd.DataFrame
) -> dict[str, Any]:
    reference = labels[
        ["event_time", "ts_code", "economic_return", "trigger_type"]
    ].rename(columns={
        "event_time": "asof",
        "economic_return": "label_economic_return",
        "trigger_type": "label_trigger_type",
    })
    output: dict[str, Any] = {}
    for arm, score_column in (("control", "rule_score"), ("challenger", "model_score")):
        candidates = ledgers[arm]["candidate"].merge(
            reference, on=["asof", "ts_code"], how="left", validate="one_to_one"
        )
        mature = candidates[candidates["label_economic_return"].notna()].copy()
        score = pd.to_numeric(mature[score_column], errors="coerce")
        returns = pd.to_numeric(mature["label_economic_return"], errors="coerce")
        valid = score.notna() & returns.notna()
        output[arm] = {
            "all_candidates": _label_stats(mature),
            "rank_1_5": _label_stats(mature[mature["candidate_rank"].le(5)]),
            "rank_6_10": _label_stats(mature[mature["candidate_rank"].between(6, 10)]),
            "score_spearman_to_return": float(score[valid].corr(returns[valid], method="spearman"))
            if int(valid.sum()) >= 3 else None,
        }
    return output


def _standardized_difference(left: pd.Series, right: pd.Series) -> float | None:
    left = pd.to_numeric(left, errors="coerce").dropna().astype(float)
    right = pd.to_numeric(right, errors="coerce").dropna().astype(float)
    if left.empty or right.empty:
        return None
    pooled = pd.concat([left, right], ignore_index=True).std(ddof=0)
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    return float((left.mean() - right.mean()) / pooled)


def _feature_diagnostics(
    prediction_f0: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: FeatureSet,
    displaced: pd.DataFrame,
    promoted: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reference = labels[["event_time", "ts_code", "economic_return"]]
    candidates = prediction_f0.merge(reference, on=["event_time", "ts_code"], validate="one_to_one")
    displaced_keys = displaced[["signal_session", "ts_code"]].rename(columns={"signal_session": "event_time"})
    promoted_keys = promoted[["signal_session", "ts_code"]].rename(columns={"signal_session": "event_time"})
    displaced_f0 = prediction_f0.merge(displaced_keys, on=["event_time", "ts_code"], validate="one_to_one")
    promoted_f0 = prediction_f0.merge(promoted_keys, on=["event_time", "ts_code"], validate="one_to_one")
    diagnostics: dict[str, Any] = {}
    for feature in feature_set.columns:
        values = pd.to_numeric(candidates[feature], errors="coerce")
        returns = pd.to_numeric(candidates["economic_return"], errors="coerce")
        valid = values.notna() & returns.notna()
        winner = values[valid & returns.gt(0)]
        loser = values[valid & returns.le(0)]
        displaced_values = pd.to_numeric(displaced_f0[feature], errors="coerce").dropna()
        promoted_values = pd.to_numeric(promoted_f0[feature], errors="coerce").dropna()
        diagnostics[feature] = {
            "valid_n": int(valid.sum()),
            "winner_n": int(len(winner)),
            "loser_n": int(len(loser)),
            "winner_mean": float(winner.mean()) if len(winner) else None,
            "winner_median": float(winner.median()) if len(winner) else None,
            "loser_mean": float(loser.mean()) if len(loser) else None,
            "loser_median": float(loser.median()) if len(loser) else None,
            "winner_minus_loser_standardized": _standardized_difference(winner, loser),
            "spearman_to_return": float(values[valid].corr(returns[valid], method="spearman"))
            if int(valid.sum()) >= 3 else None,
            "displaced_n": int(len(displaced_values)),
            "promoted_n": int(len(promoted_values)),
            "displaced_mean": float(displaced_values.mean()) if len(displaced_values) else None,
            "promoted_mean": float(promoted_values.mean()) if len(promoted_values) else None,
            "promoted_minus_displaced_standardized": _standardized_difference(
                promoted_values, displaced_values
            ),
        }

    def top_rows(field: str) -> list[dict[str, Any]]:
        rows = [
            {"feature": feature, field: values[field]}
            for feature, values in diagnostics.items()
            if values[field] is not None
        ]
        return sorted(rows, key=lambda row: (-abs(row[field]), row["feature"]))[:15]

    return (
        diagnostics,
        top_rows("winner_minus_loser_standardized"),
        top_rows("promoted_minus_displaced_standardized"),
    )


def _reproduction(
    generated: dict[str, dict[str, Any]], source: Path, models: list[dict[str, Any]]
) -> dict[str, Any]:
    failures: list[str] = []
    maximum_error = 0.0
    for arm in ("control", "challenger"):
        expected = _load_json(source / f"{arm}_metrics.json")
        for scenario in ("base", "stress"):
            for field in METRIC_FIELDS:
                left = generated[arm][scenario][field]
                right = expected[scenario][field]
                if left is None or right is None:
                    equal = left is None and right is None
                    error = 0.0 if equal else float("inf")
                else:
                    error = abs(float(left) - float(right))
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    failures.append(f"{arm}.{scenario}.{field}")
    expected_models = _load_json(source / "model_manifest.json")
    model_hash_match = [
        (row["model_id"], row["model_sha256_run1"], row["model_sha256_run2"])
        for row in models
    ] == [
        (row["model_id"], row["model_sha256_run1"], row["model_sha256_run2"])
        for row in expected_models
    ]
    if not model_hash_match:
        failures.append("model_hashes")
    return {
        "passed": not failures,
        "failed": failures,
        "maximum_absolute_metric_error": maximum_error,
        "model_hashes_match": model_hash_match,
        "eight_monthly_models": len(models) == 8,
    }


def _result_markdown(summary: dict[str, Any]) -> str:
    sets = summary["selection_attribution"]
    pairs = summary["paired_displacements"]
    rank = summary["rank_quality"]
    bridge = summary["nav_bridge"]
    direct = pairs["direct_same_prebuy_state"]
    downstream = pairs["downstream_path_divergence"]
    lines = [
        "# CAP1 F0-123 Paired Displacement Diagnostic", "",
        f"Status: `{summary['status']}`.", "",
        "## Reproduction", "",
        f"- Source rejected run reproduced: `{summary['reproduction']['passed']}`; "
        f"maximum metric error `{summary['reproduction']['maximum_absolute_metric_error']:.3e}`.",
        f"- Eight monthly model hashes matched: `{summary['reproduction']['model_hashes_match']}`.",
        "- Performance attribution uses 2026 only; 2025 is training input, not a portfolio validation period.", "",
        "## Actual Entries", "",
        f"- Shared entries: `{sets['shared']['events']}`; displaced transparent entries: "
        f"`{sets['displaced']['events']}`; promoted model entries: `{sets['promoted']['events']}`.",
        f"- Displaced closed mean return: `{_pct(sets['displaced']['mean_return'])}`; "
        f"promoted: `{_pct(sets['promoted']['mean_return'])}`.",
        f"- Direct same-state closed pairs: `{direct['closed_pairs']}`; isolated mean return delta "
        f"`{_pct(direct.get('mean_return_delta'))}`; model-side pair win rate "
        f"`{_rate(direct.get('challenger_pair_win_rate'))}`.",
        f"- Downstream-divergence closed pairs: `{downstream['closed_pairs']}`. They are descriptive, "
        "not labeled direct replacements.", "",
        "## NAV Bridge", "",
        f"- Challenger-minus-control final NAV: `{bridge['final_nav_delta']:+,.2f}`; "
        f"bridge: `{bridge['bridge_delta']:+,.2f}`; error `{bridge['bridge_identity_error']:.3e}`.",
        f"- Shared sizing/path effect: "
        f"`{bridge['categories']['shared_sizing_and_path_effect']['nav_contribution']:+,.2f}`; "
        f"direct displacement: "
        f"`{bridge['categories']['direct_same_prebuy_state_displacement']['nav_contribution']:+,.2f}`; "
        f"downstream divergence: "
        f"`{bridge['categories']['downstream_path_divergence']['nav_contribution']:+,.2f}`.", "",
        "## Rank Quality", "",
        f"- Transparent Top5 isolated mean/win: "
        f"`{_pct(rank['control']['rank_1_5']['mean_return'])}` / "
        f"`{_rate(rank['control']['rank_1_5']['win_rate'])}`.",
        f"- F0 Top5 isolated mean/win: "
        f"`{_pct(rank['challenger']['rank_1_5']['mean_return'])}` / "
        f"`{_rate(rank['challenger']['rank_1_5']['win_rate'])}`.", "",
        "## Decision", "",
        "The F0-123 ranker remains rejected. These aggregate diagnostics describe why; they do not authorize a model repair, factor filter, or threshold scan.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    source = args.source_run.resolve()
    output = args.output.resolve()
    prereg = args.prereg.resolve()
    council = (ROOT / "docs/council_20260828").resolve()
    expected_prereg = council / PREREG_NAME
    if output.parent != council:
        raise ValueError(f"output must be a direct child of {council}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must not overlap the immutable source run")
    if prereg != expected_prereg or not prereg.is_file() or _sha(prereg) != PREREG_SHA256:
        raise ValueError("diagnostic preregistration path or hash differs from review")
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise FileExistsError(f"immutable output path already exists: {output}")
    if staging.exists():
        raise FileExistsError(f"staging output path already exists: {staging}")
    source_audit = _verify_source(source)
    source_cutoffs = _verify_source_cutoffs(source)
    source_audit["source_cutoffs_verified"] = True
    source_audit["source_cutoffs"] = source_cutoffs
    strategy_path = source / "configs/strategy.yaml"
    profile_path = source / "configs/model_profile.yaml"
    strategy = StrategyConfig.from_yaml(strategy_path)
    feature_set = FeatureSet.from_f0_json(ROOT / "configs/feature_sets/f0_123_columns.json")
    model_config = ranker._load_model_config(profile_path, feature_set)
    timeline = model_config["timeline"]

    train_plan, _ = ranker._period_plan(
        strategy,
        signal_start=timeline["train_start"],
        signal_end=timeline["train_end"],
        execution_end=timeline["training_execution_end"],
        output=output,
        name="cap1_f0_displacement_training_2025",
    )
    train_pool, train_labels, _, _ = ranker._build_stage1_period(
        strategy,
        train_plan,
        output,
        retain_bundle=False,
        label_config=model_config["label"],
    )
    gc.collect()
    prediction_plan, prediction_calendar = ranker._period_plan(
        strategy,
        signal_start=timeline["prediction_start"],
        signal_end=timeline["prediction_end"],
        execution_end=timeline["execution_end"],
        output=output,
        name="cap1_f0_displacement_prediction_2026",
        require_complete_horizon=False,
    )
    prediction_pool, prediction_labels, _, execution_bundle = ranker._build_stage1_period(
        strategy,
        prediction_plan,
        output,
        retain_bundle=True,
        label_config=model_config["label"],
    )
    combined_pool = pd.concat([train_pool, prediction_pool], ignore_index=True)
    combined_labels = pd.concat([train_labels, prediction_labels], ignore_index=True)
    f0, _ = ranker._load_candidate_f0(combined_pool, feature_set)
    prediction_f0 = f0[
        f0["event_time"].between(
            pd.Timestamp(timeline["prediction_start"], tz="UTC"),
            pd.Timestamp(timeline["prediction_end"], tz="UTC"),
        )
    ].copy()
    sessions = pd.DatetimeIndex(pd.to_datetime(prediction_calendar["session"], utc=True)).normalize()
    predictions, models = ranker._monthly_predictions(
        f0, combined_labels, feature_set, model_config, sessions
    )
    eligible_pool = prediction_pool.merge(
        prediction_f0[["event_time", "ts_code"]].rename(columns={"event_time": "asof"}),
        on=["asof", "ts_code"],
        validate="one_to_one",
    )
    control_ledgers = ranker._candidate_ledgers(
        eligible_pool,
        predictions,
        prediction_plan.signal_sessions,
        rank_column="rule_score",
        policy_id="conditional_reset_transparent_control",
        strategy=strategy,
    )
    challenger_ledgers = ranker._candidate_ledgers(
        eligible_pool,
        predictions,
        prediction_plan.signal_sessions,
        rank_column="model_score",
        policy_id="conditional_reset_f0_123_ranker",
        strategy=strategy,
    )
    control_portfolio, control_results = ranker._run_portfolios(
        control_ledgers, execution_bundle, strategy, prediction_plan.execution_sessions
    )
    challenger_portfolio, challenger_results = ranker._run_portfolios(
        challenger_ledgers, execution_bundle, strategy, prediction_plan.execution_sessions
    )
    generated = {"control": control_portfolio, "challenger": challenger_portfolio}
    reproduction = _reproduction(generated, source, models)
    if not reproduction["passed"]:
        raise AssertionError(f"source reproduction failed: {reproduction['failed']}")

    control_trades = _trade_frame(control_results["base"])
    challenger_trades = _trade_frame(challenger_results["base"])
    shared, displaced, promoted = _selection_sets(control_trades, challenger_trades)
    direct_days, state_audit = _direct_state_days(
        control_results["base"],
        challenger_results["base"],
        control_trades,
        challenger_trades,
    )
    pairs, pair_coverage = _pair_exclusives(displaced, promoted, direct_days)
    pair_summary = _pair_summary(pairs, pair_coverage)
    if pair_summary["pairs"] == 0 or pair_summary["closed_pairs"] == 0:
        raise AssertionError("diagnostic has no actual promoted/displaced pair")
    nav_bridge = _nav_bridge(
        control_results["base"],
        challenger_results["base"],
        control_trades,
        challenger_trades,
        direct_days,
        float(strategy.execution["initial_cash"]),
    )
    if not nav_bridge["passed"]:
        raise AssertionError(f"NAV bridge failed: {nav_bridge}")
    ledgers = {"control": control_ledgers, "challenger": challenger_ledgers}
    rank_quality = _rank_quality(ledgers, prediction_labels)
    feature_diagnostics, winner_loser_top, promoted_displaced_top = _feature_diagnostics(
        prediction_f0, prediction_labels, feature_set, displaced, promoted
    )
    summary = {
        "status": "DIAGNOSTIC_COMPLETED_MODEL_REMAINS_REJECTED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": source_audit,
        "scope": {
            "performance_years": [2026],
            "signal_start": timeline["prediction_start"],
            "signal_end": timeline["prediction_end"],
            "execution_end": timeline["execution_end"],
            "training_input_start": timeline["train_start"],
            "training_input_end": timeline["train_end"],
            "historical_2024_2025_portfolio_validation": False,
        },
        "reproduction": reproduction,
        "selection_attribution": {
            "shared": _return_stats(shared),
            "displaced": _return_stats(displaced),
            "promoted": _return_stats(promoted),
            "control_entries": int(len(control_trades)),
            "challenger_entries": int(len(challenger_trades)),
            "jaccard": float(len(shared) / len(pd.concat([
                control_trades[["signal_session", "ts_code"]],
                challenger_trades[["signal_session", "ts_code"]],
            ]).drop_duplicates())),
        },
        "paired_displacements": pair_summary,
        "prebuy_state_audit": state_audit,
        "nav_bridge": nav_bridge,
        "monthly_paired_displacements": _monthly_pairs(pairs),
        "rank_quality": rank_quality,
        "top_winner_loser_features": winner_loser_top,
        "top_promoted_displaced_features": promoted_displaced_top,
        "constraints": {
            "model_repaired": False,
            "strategy_changed": False,
            "parameter_scan_performed": False,
            "raw_business_data_written": False,
            "wide_202_factor_system_used": False,
            "stock_codes_or_individual_sessions_persisted": False,
        },
    }
    _assert_aggregate_schema(summary)
    _assert_aggregate_schema(
        feature_diagnostics, extra_keys=frozenset(str(name) for name in feature_set.columns)
    )

    try:
        staging.mkdir(parents=True)
        _write_json(staging / "SUMMARY.json", summary)
        _write_json(staging / "FEATURE_DIAGNOSTICS.json", feature_diagnostics)
        shutil.copyfile(prereg, staging / "PREREGISTRATION.md")
        (staging / "RESULT.md").write_text(_result_markdown(summary), encoding="utf-8")
        source_manifest = {
            "source_run": source.name,
            "source_artifact_manifest_sha256": _sha(source / "manifests/artifact_manifest.json"),
            "source_strategy_sha256": _sha(strategy_path),
            "source_model_profile_sha256": _sha(profile_path),
            "source_preregistration_sha256": _sha(source / "configs/preregistration.md"),
            "diagnostic_preregistration_sha256": _sha(prereg),
            "diagnostic_code_sha256": _sha(Path(__file__).resolve()),
            "base_runner_sha256": _sha(ROOT / "scripts/conditional_f0_123_reset_ranker_runner.py"),
            "feature_manifest_sha256": _sha(ROOT / "configs/feature_sets/f0_123_columns.json"),
        }
        _assert_aggregate_schema(source_manifest)
        _write_json(staging / "SOURCE_MANIFEST.json", source_manifest)
        _assert_output_privacy(staging)
        artifacts = {
            str(path.relative_to(staging)): {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
        }
        _write_json(staging / "ARTIFACT_MANIFEST.json", artifacts)
        _assert_output_privacy(staging)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"diagnostic_complete={run(args)}")


if __name__ == "__main__":
    main()
