"""Auditable S9 continuous winner-profile ranking backtest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.labeling.q70 import build_q70_t10_labels
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.rcqt import score_rcqt

ROOT = Path(__file__).resolve().parents[1]
LABEL_PROFILE = __import__("rcqt_stage1_quality_runner", fromlist=["LABEL_PROFILE"]).LABEL_PROFILE


def _rank_desc(values: pd.Series, codes: pd.Series) -> pd.Series:
    order = sorted(values.index, key=lambda i: (-float(values.loc[i]), str(codes.loc[i])))
    out = pd.Series(index=values.index, dtype=float)
    for rank, idx in enumerate(order, start=1):
        out.loc[idx] = 1.0 - (rank - 1) / max(len(order), 1)
    return out


def _build_features(bundle: Any, strategy: StrategyConfig) -> pd.DataFrame:
    features = build_feature_ledger(bundle, strategy)
    extra = bundle.execution[["trade_date", "ts_code", "economic_high"]].rename(columns={"trade_date": "asof"})
    extra["asof"] = pd.to_datetime(extra["asof"], utc=True).dt.normalize()
    features["asof"] = pd.to_datetime(features["asof"], utc=True).dt.normalize()
    features = features.merge(extra, on=["asof", "ts_code"], how="left", validate="one_to_one")
    features["close"] = features["economic_close"]
    features = features.sort_values(["ts_code", "asof"], kind="mergesort")
    features["dd60"] = features.groupby("ts_code", sort=False)["economic_high"].transform(
        lambda values: values / values.rolling(60, min_periods=60).max() - 1.0
    )
    return features.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True)


def _score(features: pd.DataFrame, strategy: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready = features[features["feature_ready"].astype(bool)].copy()
    liquidity_floor = float(strategy.universe.get("min_median_amount_20_stored", 0.0))
    ready = ready.loc[
        ready["liq20"].ge(liquidity_floor)
        & ready["execution_data_eligible"].astype(bool)
    ].copy()
    required = ["close", "ma5", "prev3_high", "ret1", "dist_ma60", "ret20", "ret60", "dd20", "dd60", "vol20", "liq20", "volume_ratio_20"]
    numeric = ready[required].apply(pd.to_numeric, errors="coerce")
    ready = ready.loc[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)].copy()
    if ready.empty:
        raise RuntimeError("continuous ranking universe is empty after required-data/PIT filtering")
    scored = score_rcqt(ready, require_right_confirmation=False)
    terms = [dict(term) for term in strategy.ranking["terms"]]
    for _, group in scored.groupby("asof", sort=True):
        idx = group.index
        for term_no, term in enumerate(terms):
            feature = str(term["feature"])
            values = pd.to_numeric(group[feature], errors="raise")
            direction = str(term["direction"])
            if direction == "nearest":
                score_values = -abs(values - float(term["center"]))
            elif direction == "desc":
                score_values = values
            elif direction == "asc":
                score_values = -values
            else:
                raise ValueError(f"unsupported ranking direction: {direction}")
            scored.loc[idx, f"score_term_{term_no}"] = _rank_desc(score_values, group["ts_code"])
        scored.loc[idx, "continuity_score"] = sum(
            float(term["weight"]) * scored.loc[idx, f"score_term_{term_no}"]
            for term_no, term in enumerate(terms)
        )
    scored = scored.sort_values(["asof", "continuity_score", "ts_code"], ascending=[True, False, True], kind="mergesort")
    scored["candidate_rank"] = scored.groupby("asof", sort=True).cumcount() + 1
    view_size = int(strategy.portfolio["candidate_view_size"])
    scored["candidate_status"] = np.where(scored["candidate_rank"] <= view_size, "IN_VIEW", "BELOW_VIEW")
    snapshots = {}
    for day, group in scored[scored["candidate_status"].eq("IN_VIEW")].groupby("asof", sort=True):
        payload = "|".join(f"{row.ts_code}:{int(row.candidate_rank)}" for row in group.itertuples())
        snapshots[day] = hashlib.sha256(payload.encode()).hexdigest()
    scored["candidate_snapshot_id"] = scored["asof"].map(snapshots).fillna("")
    return scored, scored[scored["candidate_status"].eq("IN_VIEW")].copy()


def _selection(view: pd.DataFrame, strategy: StrategyConfig, signal_sessions: tuple[str, ...]) -> pd.DataFrame:
    primary = int(strategy.portfolio["entries_per_decision"])
    policy_hash = hashlib.sha256(strategy.config_hash.encode()).hexdigest()
    rows = []
    for day in pd.DatetimeIndex(pd.to_datetime(signal_sessions, utc=True)).normalize():
        group = view[view["asof"].eq(day)]
        snapshot = str(group["candidate_snapshot_id"].iloc[0]) if not group.empty else ""
        rows.append({
            "decision_id": hashlib.sha256(f"{policy_hash}|{day.date()}|{snapshot}".encode()).hexdigest(),
            "asof": day, "desired_entries": primary,
            "target_weight_each": float(strategy.portfolio["sizing"]["value"]),
            "primary_rank_end": primary, "replacement_rank_end": int(strategy.portfolio["candidate_view_size"]),
            "candidate_snapshot_id": snapshot, "policy_id": strategy.strategy_id,
            "policy_hash": policy_hash,
            "context_hash": hashlib.sha256(f"{day.date()}|{strategy.config_hash}".encode()).hexdigest(),
        })
    return pd.DataFrame(rows)


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    tests = {
        "profit_factor": metrics["portfolio_profit_factor"] is not None and float(metrics["portfolio_profit_factor"]) >= float(strategy.acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(strategy.acceptance["return_excluding_best_week_min_exclusive"]),
    }
    return {"passed": all(tests.values()), "tests": tests}


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "ledgers", "backtests", "diagnostics", "logs"):
        (output / name).mkdir()
    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    calendar = load_trading_calendar(str((pd.Timestamp(args.start) - pd.Timedelta(days=500)).date()), args.execution_end)
    plan = compile_run_plan(strategy, model, RunRequest(args.start, args.signal_end, args.execution_end, str(output), args.run_name), calendar["session"])
    shutil.copyfile(args.strategy, output / "configs/strategy.yaml")
    shutil.copyfile(args.model, output / "configs/model.yaml")
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "RUN_STATUS.json", {"run_name": args.run_name, "status": "RUNNING", "created_at": datetime.now(timezone.utc).isoformat(), "strategy_id": strategy.strategy_id, "strategy_hash": strategy.config_hash, "python": sys.version})
    bundle = build_data_bundle(plan, strategy, output)
    _write_json(output / "data_manifest.json", bundle.manifest)
    bundle.universe.to_parquet(output / "ledgers/universe_ledger.parquet", index=False)
    bundle.availability.to_parquet(output / "ledgers/data_availability_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "ledgers/execution_panel.parquet", index=False)
    features = _build_features(bundle, strategy)
    features.to_parquet(output / "ledgers/feature_ledger.parquet", index=False)
    signal_days = pd.to_datetime(plan.signal_sessions, utc=True).normalize()
    scored, view = _score(features[features["asof"].isin(signal_days)].copy(), strategy)
    selection = _selection(view, strategy, plan.signal_sessions)
    scored.to_parquet(output / "ledgers/score_ledger.parquet", index=False)
    view.to_parquet(output / "ledgers/candidate_ledger.parquet", index=False)
    selection.to_parquet(output / "ledgers/selection_ledger.parquet", index=False)
    daily_view = view.groupby("asof").size().reindex(signal_days, fill_value=0)
    min_view = int(strategy.portfolio["candidate_view_size"])
    coverage_ratio = float((daily_view >= min_view).mean()) if len(daily_view) else 0.0
    if coverage_ratio < float(strategy.portfolio.get("candidate_min_daily_view_ratio", 0.0)):
        raise RuntimeError(f"candidate view coverage {coverage_ratio:.4f} below configured gate")
    label_panel = bundle.execution.rename(columns={"trade_date": "event_time"})
    label_panel = label_panel[
        label_panel["execution_status"].eq("TRADABLE")
        & pd.to_numeric(label_panel["economic_open"], errors="coerce").gt(0)
    ].copy()
    labels = build_q70_t10_labels(label_panel[["event_time", "ts_code", "economic_open"]], profile=LABEL_PROFILE, session_dates=pd.DatetimeIndex(calendar["session"]))
    labels["event_time"] = pd.to_datetime(labels["event_time"], utc=True).dt.normalize()
    candidate_event = scored.merge(labels, left_on=["asof", "ts_code"], right_on=["event_time", "ts_code"], how="left")
    candidate_event = candidate_event[candidate_event["candidate_status"].eq("IN_VIEW")].copy()
    portfolios = {}
    for scenario in ("base", "stress"):
        result = run_v3_backtest(candidate_ledger=view, selection_ledger=selection, execution_panel=bundle.execution, corporate_actions=bundle.corporate_actions, strategy=strategy, execution_sessions=plan.execution_sessions, scenario_name=scenario)
        target = output / "backtests" / scenario
        target.mkdir()
        for name, frame in result.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
        metrics.update({"scenario": scenario, "candidate_rows": int(len(candidate_event)), "view_days": int(view["asof"].nunique()), "coverage_ratio": coverage_ratio})
        metrics["acceptance"] = _acceptance(metrics, strategy)
        _write_json(target / "metrics.json", metrics)
        portfolios[scenario] = metrics
    summary = {"strategy": strategy.strategy_id, "bundle_id": bundle.bundle_id, "candidate_count": int(len(view)), "scored_count": int(len(scored)), "candidate_days": int(view["asof"].nunique()), "candidate_view_coverage_ratio": coverage_ratio, "data_coverage": bundle.manifest["coverage"], "portfolios": portfolios, "parameter_sweep": False, "decision": "ADVANCE" if all(p["acceptance"]["passed"] for p in portfolios.values()) else "ABANDON_DIAGNOSTIC_ONLY"}
    _write_json(output / "SUMMARY.json", summary)
    code_paths = [ROOT / "scripts/continuity_rank_v1_runner.py", ROOT / "src/aistock9988/data/bundle.py", ROOT / "src/aistock9988/backtest/v3_engine.py", ROOT / "src/aistock9988/features/engine.py", ROOT / "src/aistock9988/selection/rcqt.py"]
    _write_json(output / "manifests/code_manifest.json", {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths})
    _write_json(output / "RUN_STATUS.json", {"status": "DIAGNOSTIC_COMPLETED", "bundle_id": bundle.bundle_id, "overall_acceptance_passed": summary["decision"] == "ADVANCE"}, replace=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/continuity_rank_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--codes-source", type=Path, help="Deprecated; codes are pinned by strategy.universe.codes_file")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-07")
    parser.add_argument("--execution-end", default="2026-08-28")
    parser.add_argument("--run-name", default="S9_CONTINUITY_RANK_V1_2026_R1")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
