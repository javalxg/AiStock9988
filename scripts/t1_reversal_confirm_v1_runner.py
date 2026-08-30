"""Audit and (only if justified) backtest the T0 shock/T1 reversal rule."""
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
from aistock9988.features.t1_reversal import build_t1_reversal_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3

ROOT = Path(__file__).resolve().parents[1]


def _assert_frozen_contract(strategy: StrategyConfig) -> None:
    checks = strategy.stage1.get("expression", {}).get("all", ())
    expected = {("t0_shock", "ge", 1.0), ("t1_reversal_confirmed", "ge", 1.0)}
    actual = {(str(x.get("left")), str(x.get("op")), float(x.get("value"))) for x in checks if hasattr(x, "get")}
    if not expected.issubset(actual):
        raise ValueError("T1_REVERSAL_CONFIRM_V1 stage1 contract drift")
    if strategy.ranking.get("method") != "t1_ret1_desc":
        raise ValueError("T1_REVERSAL_CONFIRM_V1 ranking contract drift")
    if strategy.ranking.get("confirmation_feature") != "t1_ret1":
        raise ValueError("T1_REVERSAL_CONFIRM_V1 confirmation ranking drift")
    if strategy.ranking.get("control_method") != "t0_amount_ratio_desc":
        raise ValueError("T1_REVERSAL_CONFIRM_V1 control ranking drift")
    if strategy.ranking.get("control_feature") != "t0_amount_ratio":
        raise ValueError("T1_REVERSAL_CONFIRM_V1 control feature drift")


def _json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _ledgers(signals: pd.DataFrame, strategy: StrategyConfig, *, sort_feature: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    if sort_feature not in signals.columns:
        raise ValueError(f"ranking feature missing: {sort_feature}")
    rows = signals.copy().sort_values(["asof", sort_feature, "ts_code"], ascending=[True, False, True], kind="mergesort")
    rows["candidate_rank"] = rows.groupby("asof", sort=True).cumcount() + 1
    view = int(strategy.portfolio["candidate_view_size"])
    rows["candidate_status"] = np.where(rows["candidate_rank"] <= view, "IN_VIEW", "BELOW_VIEW")
    snapshots = {
        day: hashlib.sha256(
            "|".join(f"{r.ts_code}:{int(r.candidate_rank)}" for r in group.itertuples()).encode()
        ).hexdigest()
        for day, group in rows[rows["candidate_status"].eq("IN_VIEW")].groupby("asof", sort=True)
    }
    rows["candidate_snapshot_id"] = rows["asof"].map(snapshots).fillna("")
    # The score is the field actually used for the arm's frozen ordering.
    # In particular, the T0 control must not expose T1's realized return as
    # its rule score, even though the event ledger keeps it for diagnostics.
    rows["rule_score"] = pd.to_numeric(rows[sort_feature], errors="coerce")
    rows["model_score"] = np.nan
    rows["final_score"] = rows["rule_score"]
    rows["stage1_pass"] = True
    rows["selected"] = rows["candidate_status"].eq("IN_VIEW")
    rows["feature_ready"] = True
    rows["selection_data_eligible"] = True
    rows["training_data_eligible"] = True
    rows["execution_data_eligible"] = True
    rows["universe_pass"] = True
    for col in ("missing_required_selection", "missing_required_training", "missing_required_execution", "missing_optional", "feature_rejection_reason", "score_rejection_reason"):
        rows[col] = ""
    rows["bundle_id"] = rows.get("bundle_id", "")
    rows["feature_set_hash"] = rows.get("feature_set_hash", "")
    score_cols = ["asof", "ts_code", "bundle_id", "feature_set_hash", "universe_pass", "selection_data_eligible", "training_data_eligible", "execution_data_eligible", "missing_required_selection", "missing_required_training", "missing_required_execution", "missing_optional", "feature_ready", "stage1_pass", "rule_score", "model_score", "final_score", "score_rejection_reason"]
    # Preserve the arm's actual ordering input for audit.  This is T1 return
    # for confirmation and a T0-only field for control; no other future field
    # is promoted into the control selection ledger.
    ledger_cols = score_cols + [sort_feature, "candidate_rank", "candidate_status", "candidate_snapshot_id", "execution_status"]
    candidate = rows[list(dict.fromkeys(ledger_cols))].copy()
    decision_rows = []
    policy_hash = hashlib.sha256(f"{strategy.strategy_id}|{strategy.config_hash}".encode()).hexdigest()
    for day in sorted(rows["asof"].drop_duplicates()):
        sid = str(snapshots.get(day, ""))
        decision_rows.append({"decision_id": hashlib.sha256(f"{policy_hash}|{day.date()}|{sid}".encode()).hexdigest(), "asof": day, "desired_entries": int(strategy.portfolio["entries_per_decision"]), "target_weight_each": float(strategy.portfolio["sizing"]["value"]), "primary_rank_end": int(strategy.portfolio["entries_per_decision"]), "replacement_rank_end": view, "candidate_snapshot_id": sid, "policy_id": strategy.strategy_id, "policy_hash": policy_hash, "context_hash": hashlib.sha256(f"{day.date()}|{strategy.config_hash}".encode()).hexdigest()})
    return candidate.sort_values(["asof", "candidate_rank", "ts_code"], kind="mergesort"), pd.DataFrame(decision_rows)


def _events(features: pd.DataFrame, universe: pd.DataFrame, calendar: pd.DataFrame, signal_days: pd.DatetimeIndex, end: pd.Timestamp, strategy: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    f = features.copy().sort_values(["ts_code", "asof"], kind="mergesort")
    f["asof"] = pd.to_datetime(f["asof"], utc=True).dt.normalize()
    f["ts_code"] = f["ts_code"].astype(str).str.upper()
    u = universe.copy(); u["asof"] = pd.to_datetime(u["asof"], utc=True).dt.normalize(); u["ts_code"] = u["ts_code"].astype(str).str.upper()
    f = f.merge(u[["asof", "ts_code", "list_date", "pit_st", "universe_pass"]], on=["asof", "ts_code"], how="left", suffixes=("", "_u"), validate="one_to_one")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
    t0_drop_max = float(strategy.features.get("t0_intraday_return_max", -0.05))
    t0_ratio_min = float(strategy.features.get("t0_amount_ratio_min", 1.5))
    t1_return_min = float(strategy.features.get("t1_return_min", 0.0))
    min_listed_sessions = int(strategy.universe.get("min_listed_sessions", 120))
    amount_multiplier = float(strategy.execution.get("amount_unit_multiplier", 1000.0))
    min_median_amount = float(strategy.universe.get("min_median_amount_yuan", 100_000_000)) / amount_multiplier
    if (t0_drop_max, t0_ratio_min, t1_return_min, min_listed_sessions, min_median_amount) != (-0.05, 1.5, 0.0, 120, 100_000.0):
        raise ValueError("T1_REVERSAL_CONFIRM_V1 frozen contract drift")
    session_pos = {d: i for i, d in enumerate(sessions)}
    confirms: list[dict[str, Any]] = []; controls: list[dict[str, Any]] = []; audit: list[dict[str, Any]] = []
    for code, g in f.groupby("ts_code", sort=True):
        g = g.sort_values("asof", kind="mergesort").reset_index(drop=True)
        for i, r in g.iterrows():
            t0 = pd.Timestamp(r["asof"])
            if t0 not in signal_days:
                continue
            rec = {"ts_code": code, "t0_trade_date": t0, "t1_trade_date": pd.NaT, "t2_trade_date": pd.NaT, "status": "FAILED", "failure_reason": "", "t0_eligible": False, "t1_confirmed": False, "t0_amount_ratio": float(r.get("t0_amount_ratio", np.nan)), "t1_ret1": np.nan, "h5_return": np.nan, "h10_return": np.nan, "t0_raw_open": r.get("raw_open"), "t0_raw_close": r.get("raw_close"), "t1_raw_open": np.nan, "t1_raw_close": np.nan, "t1_economic_open": np.nan, "t1_economic_close": np.nan, "t2_raw_open": np.nan, "available_time": r.get("available_time"), "execution_status": r.get("execution_status", "MISSING_REQUIRED_DATA"), "bundle_id": r.get("bundle_id", ""), "feature_set_hash": r.get("feature_set_hash", "")}
            list_date = pd.Timestamp(r.get("list_date")) if pd.notna(r.get("list_date")) else pd.NaT
            if pd.notna(list_date) and t0 in session_pos:
                listed_idx = int(np.searchsorted(sessions.asi8, list_date.value, side="left"))
                listed_sessions = int(session_pos[t0] - listed_idx)
            else:
                listed_sessions = -1
            shock = bool(float(r.get("t0_shock", 0.0)) >= 1.0 and float(r.get("amount", np.nan)) >= min_median_amount * t0_ratio_min)
            if not bool(r.get("feature_ready", False)): rec["failure_reason"] = "T0_FEATURE_NOT_READY"
            elif not bool(r.get("universe_pass", False)) or str(code).endswith(".BJ"): rec["failure_reason"] = "UNIVERSE_REJECTED"
            elif bool(r.get("pit_st", False)): rec["failure_reason"] = "PIT_ST"
            elif listed_sessions < min_listed_sessions: rec["failure_reason"] = "LISTED_LT_120_SESSIONS"
            elif not np.isfinite(float(r.get("prior20_amount_median", np.nan))) or float(r.get("prior20_amount_median", 0.0)) < min_median_amount: rec["failure_reason"] = "MEDIAN_AMOUNT_LT_100M"
            elif not shock: rec["failure_reason"] = "T0_NOT_SHOCK"
            else:
                rec["t0_eligible"] = True
                if i + 1 >= len(g): rec["failure_reason"] = "MISSING_T1"
                else:
                    t1 = g.iloc[i + 1]; rec["t1_trade_date"] = pd.Timestamp(t1["asof"]); rec["t1_raw_open"] = t1.get("raw_open"); rec["t1_raw_close"] = t1.get("raw_close"); rec["t1_economic_open"] = t1.get("economic_open"); rec["t1_economic_close"] = t1.get("economic_close")
                    t1_ret = float(t1["economic_close"] / r["economic_close"] - 1.0) if pd.notna(t1.get("economic_close")) and pd.notna(r.get("economic_close")) and float(r["economic_close"]) > 0 else np.nan
                    rec["t1_ret1"] = t1_ret
                    # T1 may finish at the up-limit; the contract only excludes
                    # suspension, zero-volume, missing data and down-limit.
                    bad = str(t1.get("execution_status", "MISSING_REQUIRED_DATA")) in {"SUSPENDED", "LIMIT_DOWN", "ZERO_VOLUME", "MISSING_REQUIRED_DATA", "OUT_OF_UNIVERSE"}
                    confirmed = np.isfinite(t1_ret) and t1_ret > t1_return_min and float(t1.get("economic_close", np.nan)) >= float(t1.get("economic_open", np.nan)) and float(t1.get("economic_close", np.nan)) >= float(r.get("economic_close", np.nan)) and not bad
                    if not confirmed: rec["failure_reason"] = "T1_CONFIRMATION_FAILED"
                    else:
                        rec["t1_confirmed"] = True
                        if i + 2 >= len(g): rec["failure_reason"] = "MISSING_T2"
                        else:
                            t2 = g.iloc[i + 2]; rec["t2_trade_date"] = pd.Timestamp(t2["asof"]); rec["t2_raw_open"] = t2.get("raw_open")
                            if str(t2.get("execution_status", "MISSING_REQUIRED_DATA")) != "TRADABLE" or not np.isfinite(float(t2.get("raw_open", np.nan))): rec["failure_reason"] = "T2_NOT_EXECUTABLE"
                            else:
                                rec["status"] = "CONFIRMED"; rec["failure_reason"] = ""
                                for horizon, key in ((5, "h5_return"), (10, "h10_return")):
                                    j = i + 2 + horizon
                                    if j < len(g) and pd.notna(g.iloc[j].get("economic_close")):
                                        rec[key] = float(g.iloc[j]["economic_close"] / t2["economic_open"] - 1.0) if pd.notna(t2.get("economic_open")) and float(t2.get("economic_open", 0)) > 0 else np.nan
                                confirms.append(rec.copy())
            # A control enters at T1 open on every valid T0 shock with a tradable T1.
            if rec["t0_eligible"] and i + 1 < len(g):
                t1 = g.iloc[i + 1]
                if str(t1.get("execution_status", "MISSING_REQUIRED_DATA")) == "TRADABLE" and np.isfinite(float(t1.get("raw_open", np.nan))):
                    c = rec.copy(); c["status"] = "CONTROL"; c["t1_trade_date"] = pd.Timestamp(t1["asof"]); c["t2_trade_date"] = pd.NaT; controls.append(c)
            audit.append(rec)
    events = pd.DataFrame(audit)
    conf = pd.DataFrame(confirms)
    ctrl = pd.DataFrame(controls)
    return events, conf, {"control": ctrl, "event_count": len(events), "shock_count": int(sum(float(x.get("t0_shock", 0)) >= 1 for x in audit))}


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError(f"immutable output directory is not empty: {output}")
    for name in ("configs", "manifests", "ledgers", "backtests", "diagnostics", "logs"): (output / name).mkdir()
    strategy = StrategyConfig.from_yaml(args.strategy); model = ModelConfig.from_yaml(args.model)
    _assert_frozen_contract(strategy)
    calendar = load_trading_calendar(str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date()), args.execution_end)
    plan = compile_run_plan(strategy, model, RunRequest(args.signal_start, args.signal_end, args.execution_end, str(output), args.run_name), calendar["session"])
    shutil.copyfile(args.strategy, output / "configs/strategy.yaml"); shutil.copyfile(args.model, output / "configs/model.yaml"); _json(output / "plan.json", plan.to_dict())
    _json(output / "RUN_STATUS.json", {"status": "RUNNING", "run_name": args.run_name, "created_at": datetime.now(timezone.utc).isoformat(), "diagnostic_history": True})
    bundle = build_data_bundle(plan, strategy, output); _json(output / "data_manifest.json", bundle.manifest)
    bundle.universe.to_parquet(output / "ledgers/universe_ledger.parquet", index=False); bundle.execution.to_parquet(output / "ledgers/execution_panel.parquet", index=False); bundle.availability.to_parquet(output / "ledgers/data_availability_ledger.parquet", index=False)
    features = build_t1_reversal_feature_ledger(bundle, strategy); features.to_parquet(output / "ledgers/feature_ledger.parquet", index=False)
    signal_days = pd.DatetimeIndex(pd.to_datetime(plan.signal_sessions, utc=True)).normalize()
    events, confirmed, extra = _events(features, bundle.universe, bundle.calendar, signal_days, pd.Timestamp(plan.execution_end, tz="UTC"), strategy)
    events.to_parquet(output / "ledgers/event_ledger.parquet", index=False); confirmed.to_parquet(output / "ledgers/confirmed_events.parquet", index=False); extra["control"].to_parquet(output / "ledgers/control_events.parquet", index=False)
    h10_mean = float(pd.to_numeric(confirmed.get("h10_return", pd.Series(dtype=float)), errors="coerce").mean()) if not confirmed.empty else np.nan
    t0_count = int(events.get("t0_eligible", pd.Series(dtype=bool)).sum())
    confirmed_count = int(events.get("t1_confirmed", pd.Series(dtype=bool)).sum())
    audit = {"event_count": int(len(events)), "t0_shock_count": t0_count, "t0_event_rate": float(t0_count / max(1, len(events))), "confirmation_count": confirmed_count, "confirmation_rate_among_t0": float(confirmed_count / max(1, t0_count)), "t2_executable_count": int(len(confirmed)), "t2_executable_rate_among_confirmed": float(len(confirmed) / max(1, confirmed_count)), "confirmation_h5_mean": float(pd.to_numeric(confirmed.get("h5_return", pd.Series(dtype=float)), errors="coerce").mean()) if not confirmed.empty else np.nan, "confirmation_h10_mean": h10_mean, "database_cutoff": str(pd.to_datetime(bundle.execution["trade_date"], utc=True).max().date())}
    _json(output / "diagnostics/event_rate_audit.json", audit)
    if not np.isfinite(h10_mean) or h10_mean <= 0:
        summary = {"decision": "STOP_AFTER_AUDIT", "reason": "confirmation_h10_mean_nonpositive", "audit": audit}
        _json(output / "SUMMARY.json", summary); (output / "RESULT.md").write_text(f"# T1_REVERSAL_CONFIRM_V1\n\nAudit stopped before portfolio: confirmation H10 mean={h10_mean!r}.\n", encoding="utf-8"); _json(output / "RUN_STATUS.json", {"status": "DIAGNOSTIC_COMPLETED", "overall_acceptance_passed": False, "decision": "STOP_AFTER_AUDIT"}, replace=True); return output
    conf_signals = confirmed.rename(columns={"t1_trade_date": "asof"}).copy(); conf_signals["execution_status"] = "TRADABLE"; conf_signals["selected"] = True; conf_signals["target_weight"] = float(strategy.portfolio["sizing"]["value"]); conf_signals["cash_fraction"] = 1.0; conf_signals["sleeve"] = "t1_reversal_confirm"; conf_signals["available_time"] = conf_signals["asof"].map(lambda x: pd.Timestamp(x).tz_convert("UTC") + pd.Timedelta(hours=7))
    ctrl_signals = extra["control"].rename(columns={"t0_trade_date": "asof"}).copy(); ctrl_signals["execution_status"] = "TRADABLE"; ctrl_signals["selected"] = True; ctrl_signals["target_weight"] = float(strategy.portfolio["sizing"]["value"]); ctrl_signals["cash_fraction"] = 1.0; ctrl_signals["sleeve"] = "t0_control"; ctrl_signals["available_time"] = ctrl_signals["asof"].map(lambda x: pd.Timestamp(x).tz_convert("UTC") + pd.Timedelta(hours=7))
    portfolios = {}
    ranking_features = {
        "confirm": str(strategy.ranking["confirmation_feature"]),
        "control": str(strategy.ranking["control_feature"]),
    }
    for label, signals in (("confirm", conf_signals), ("control", ctrl_signals)):
        candidate, selection = _ledgers(signals, strategy, sort_feature=ranking_features[label])
        candidate.to_parquet(output / "ledgers" / f"{label}_candidate.parquet", index=False); selection.to_parquet(output / "ledgers" / f"{label}_selection.parquet", index=False)
        for scenario in ("base", "stress"):
            result = run_v3_backtest(candidate_ledger=candidate, selection_ledger=selection, execution_panel=bundle.execution, corporate_actions=bundle.corporate_actions, strategy=strategy, execution_sessions=plan.execution_sessions, scenario_name=scenario)
            target = output / "backtests" / label / scenario; target.mkdir(parents=True)
            for name, frame in result.items(): frame.to_parquet(target / f"{name}.parquet", index=False)
            metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"])); metrics["scenario"] = scenario; metrics["acceptance"] = {"profit_factor": metrics["portfolio_profit_factor"] is not None and metrics["portfolio_profit_factor"] >= 2.0, "max_drawdown": abs(metrics["max_drawdown"]) <= 0.15, "excluding_best_week": metrics["return_excluding_best_week"] > 0}; _json(target / "metrics.json", metrics); portfolios[f"{label}_{scenario}"] = metrics
    _json(output / "SUMMARY.json", {"decision": "BACKTEST_COMPLETED", "audit": audit, "portfolios": portfolios}); (output / "RESULT.md").write_text("# T1_REVERSAL_CONFIRM_V1\n\nAudit passed the non-positive-return gate; confirm and T0 control portfolios were run.\n", encoding="utf-8"); _json(output / "RUN_STATUS.json", {"status": "DIAGNOSTIC_COMPLETED", "overall_acceptance_passed": False, "decision": "BACKTEST_COMPLETED"}, replace=True); return output


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/t1_reversal_confirm_v1.yaml"); p.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml"); p.add_argument("--signal-start", default="2026-01-01"); p.add_argument("--signal-end", default="2026-08-13"); p.add_argument("--execution-end", default="2026-08-28"); p.add_argument("--run-name", default="T1_REVERSAL_CONFIRM_V1_DIAGNOSTIC_2026"); p.add_argument("--output", type=Path, default=ROOT / "docs/council_20260828/T1_REVERSAL_CONFIRM_V1_DIAGNOSTIC_2026"); print(run(p.parse_args()))


if __name__ == "__main__": main()
