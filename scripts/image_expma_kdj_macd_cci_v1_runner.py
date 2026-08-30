"""Backtest the screenshot's weekly-trend/daily-reclaim rule once."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.image_timing import build_image_timing_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.pipeline import evaluate_expression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/image_expma_kdj_macd_cci_v1.yaml"
DEFAULT_MODEL = ROOT / "configs/model/disabled.yaml"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_ledgers(features: pd.DataFrame, signal_sessions: tuple[str, ...], strategy: StrategyConfig) -> dict[str, pd.DataFrame]:
    signal_days = pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize()
    frame = features[features["asof"].isin(signal_days)].copy().reset_index(drop=True)
    # The YAML expression is the source of truth; this runner only supplies
    # the custom feature columns and records every rejected row.
    frame["stage1_pass"] = False
    ready = frame["feature_ready"].astype(bool)
    frame.loc[ready, "stage1_pass"] = evaluate_expression(frame.loc[ready], strategy.stage1["expression"])
    frame["rule_score"] = np.nan
    frame.loc[frame["stage1_pass"], "rule_score"] = 1.0
    frame["model_score"] = np.nan
    frame["final_score"] = frame["rule_score"]
    frame["score_rejection_reason"] = np.select(
        [~frame["universe_pass"].astype(bool), ~frame["feature_ready"].astype(bool), ~frame["stage1_pass"].astype(bool)],
        ["UNIVERSE_REJECTED", frame.get("feature_rejection_reason", "FEATURE_NOT_MATURE"), "STAGE1_REJECTED"],
        default="",
    )
    rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for day in signal_days:
        day_frame = frame[frame["asof"].eq(day)].copy()
        ranking_terms = tuple(strategy.ranking["terms"])
        if len(ranking_terms) != 1 or str(ranking_terms[0]["feature"]) != "stable_rank":
            raise ValueError("image strategy requires the registered stable_rank ranking contract")
        ascending = str(ranking_terms[0]["direction"]) == "asc"
        group = day_frame[day_frame["stage1_pass"]].sort_values(["ts_code"], ascending=ascending, kind="mergesort").copy()
        snapshot = hashlib.sha256("|".join(group["ts_code"].astype(str)).encode()).hexdigest() if not group.empty else ""
        decision_id = hashlib.sha256(f"{strategy.config_hash}|{day.date()}|{snapshot}".encode()).hexdigest()
        rank_map = {code: rank for rank, code in enumerate(group["ts_code"].astype(str), start=1)}
        for _, row in day_frame.iterrows():
            score_rows.append({
                "asof": day, "ts_code": str(row.ts_code), "bundle_id": row.bundle_id,
                "feature_set_hash": row.get("feature_set_hash", "image_timing_v1"),
                "universe_pass": bool(row.universe_pass), "selection_data_eligible": bool(row.selection_data_eligible),
                "training_data_eligible": bool(row.training_data_eligible), "execution_data_eligible": bool(row.execution_data_eligible),
                "missing_required_selection": row.get("missing_required_selection", ""),
                "missing_required_training": row.get("missing_required_training", ""),
                "missing_required_execution": row.get("missing_required_execution", ""),
                "missing_optional": row.get("missing_optional", ""), "feature_ready": bool(row.feature_ready),
                "stage1_pass": bool(row.stage1_pass), "rule_score": row.rule_score, "model_score": np.nan,
                "final_score": row.final_score, "score_rejection_reason": row.score_rejection_reason,
            })
            rank = rank_map.get(str(row.ts_code))
            rows.append({**score_rows[-1], "candidate_rank": rank,
                "candidate_status": ("IN_VIEW" if rank is not None and rank <= int(strategy.portfolio["candidate_view_size"])
                                     else "BELOW_VIEW" if rank is not None else "REJECTED"),
                "candidate_snapshot_id": snapshot if rank is not None else "", "execution_status": row.execution_status})
    selection = []
    for day in signal_days:
        candidates = [r for r in rows if r["asof"] == day and r["candidate_status"] == "IN_VIEW"]
        snapshot = candidates[0]["candidate_snapshot_id"] if candidates else ""
        selection.append({
            "decision_id": hashlib.sha256(f"{strategy.config_hash}|{day.date()}|{snapshot}".encode()).hexdigest(),
            "asof": day, "desired_entries": int(strategy.portfolio["entries_per_decision"]),
            "target_weight_each": float(strategy.portfolio["sizing"]["value"]),
            "primary_rank_end": int(strategy.portfolio["entries_per_decision"]),
            "replacement_rank_end": int(strategy.portfolio["candidate_view_size"]),
            "candidate_snapshot_id": snapshot, "policy_id": strategy.strategy_id,
            "policy_hash": hashlib.sha256(f"{strategy.strategy_id}|{strategy.config_hash}".encode()).hexdigest(),
            "context_hash": hashlib.sha256(f"{day.date()}|{strategy.config_hash}".encode()).hexdigest(),
        })
    score_columns = [
        "asof", "ts_code", "bundle_id", "feature_set_hash", "universe_pass",
        "selection_data_eligible", "training_data_eligible", "execution_data_eligible",
        "missing_required_selection", "missing_required_training", "missing_required_execution",
        "missing_optional", "feature_ready", "stage1_pass", "rule_score", "model_score",
        "final_score", "score_rejection_reason",
    ]
    candidate_columns = score_columns + ["candidate_rank", "candidate_status", "candidate_snapshot_id", "execution_status"]
    scores = pd.DataFrame(score_rows, columns=score_columns)
    candidates = pd.DataFrame(rows, columns=candidate_columns)
    return {
        "score": scores,
        "candidate": candidates,
        "selection": pd.DataFrame(selection),
    }


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    calendar_start = str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date())
    calendar = load_trading_calendar(calendar_start, args.execution_end)
    request = RunRequest(args.signal_start, args.signal_end, args.execution_end, str(output), "image-expma-kdj-macd-cci-v1")
    plan = compile_run_plan(strategy, model, request, calendar["session"])
    bundle = build_data_bundle(plan, strategy, output)
    features, warnings = build_image_timing_feature_ledger(bundle)
    ledgers = _build_ledgers(features, plan.signal_sessions, strategy)
    features.to_parquet(output / "feature_ledger.parquet", index=False)
    warnings.to_parquet(output / "rsi_warning_ledger.parquet", index=False)
    for name, data in ledgers.items():
        data.to_parquet(output / f"{name}_ledger.parquet", index=False)
    _write_json(output / "RUN_STATUS.json", {"status": "RUNNING", "research_status": "DIAGNOSTIC_SEEN_HISTORY", "strategy_id": strategy.strategy_id, "strategy_hash": strategy.config_hash, "created_at": datetime.now(timezone.utc).isoformat(), "credentials_persisted": False})
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "data_manifest.json", bundle.manifest)
    shutil.copyfile(args.strategy, output / "strategy.yaml")
    shutil.copyfile(args.model, output / "model.yaml")
    results = {}
    for scenario in ("base", "stress"):
        result = run_v3_backtest(candidate_ledger=ledgers["candidate"], selection_ledger=ledgers["selection"], execution_panel=bundle.execution, corporate_actions=bundle.corporate_actions, strategy=strategy, execution_sessions=plan.execution_sessions, scenario_name=scenario)
        scenario_dir = output / "backtest" / scenario
        scenario_dir.mkdir(parents=True)
        for name, data in result.items():
            data.to_parquet(scenario_dir / f"{name}.parquet", index=False)
        results[scenario] = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]), positions=result["positions"], corporate_actions=result["corporate_actions"])
    warning_dates = pd.to_datetime(warnings.get("trade_date", pd.Series(dtype="datetime64[ns]")), utc=True, errors="coerce")
    signal_warning_count = int(warning_dates.between(pd.Timestamp(plan.signal_start, tz="UTC"), pd.Timestamp(plan.signal_end, tz="UTC")).sum())
    summary = {"strategy_id": strategy.strategy_id, "signal_start": plan.signal_start, "signal_end": plan.signal_end, "execution_end": plan.execution_end, "signal_count": int(ledgers["selection"].shape[0]), "candidate_count": int((ledgers["candidate"]["candidate_status"] == "IN_VIEW").sum()) if not ledgers["candidate"].empty else 0, "rsi_warning_count_signal_window": signal_warning_count, "rsi_warning_count_all_feature_rows": int(len(warnings)), "metrics": results, "data_bundle_id": bundle.bundle_id}
    _write_json(output / "summary.json", summary)
    (output / "RUN_STATUS.json").write_text(json.dumps({"status": "COMPLETE", "strategy_hash": strategy.config_hash, "model_hash": model.config_hash, **summary}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=str(DEFAULT_STRATEGY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-14")
    parser.add_argument("--execution-end", default="2026-08-28")
    parser.add_argument("--output", default=str(ROOT / "docs/council_20260828/IMAGE_EXPMA_KDJ_MACD_CCI_V1_2026_TO_0828"))
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
