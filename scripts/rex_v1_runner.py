"""Run the preregistered REX-V1 paired execution diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.pipeline import build_rule_ledgers


ROOT = Path(__file__).resolve().parents[1]


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acceptance(metrics: dict, strategy: StrategyConfig) -> dict:
    pf = metrics.get("portfolio_profit_factor")
    scenario = str(metrics.get("scenario", ""))
    checks = {
        "profit_factor": pf is not None and float(pf) >= float(strategy.acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(
            strategy.acceptance.get("return_excluding_best_week_min_exclusive", 0.0)
        ),
        "excluding_top3_replay_available": bool(
            metrics.get("top3_replay", {}).get("available", False)
        ),
        "excluding_top3_profit": float(metrics["return_excluding_top3_profit"]) > float(
            strategy.acceptance.get("return_excluding_top3_profit_min_exclusive", 0.0)
        ),
        "all_positions_closed": int(metrics.get("open_positions_at_end", 0)) == 0,
        # This check is meaningful only for the stress arm; the base arm is
        # reported with the same schema but does not satisfy this predicate.
        "stress_positive": scenario != "stress" or float(metrics["total_return"]) > 0.0,
    }
    checks["passed"] = all(checks.values())
    return checks


def _verify_bundle_dates(bundle, plan, strategy: StrategyConfig) -> dict[str, object]:
    """Fail closed when the requested terminal session is not data-mature."""
    execution = bundle.execution.copy()
    dates = pd.to_datetime(execution["trade_date"], errors="raise", utc=True).dt.normalize()
    end = pd.Timestamp(plan.execution_end, tz="UTC").normalize()
    no_future_rows = bool((dates <= end).all())
    if not no_future_rows:
        raise ValueError("execution bundle contains rows after requested execution_end")
    terminal = execution.loc[
        dates.eq(end) & execution["universe_pass"].astype(bool)
    ]
    if terminal.empty:
        raise ValueError("terminal execution session has no eligible universe rows")
    coverage = float(pd.to_numeric(terminal["execution_data_eligible"], errors="coerce").mean())
    minimum = float(strategy.data_policy.get("forward_min_coverage", 0.90))
    if not np.isfinite(coverage) or coverage < minimum:
        raise ValueError(
            f"terminal execution session is not mature: coverage={coverage:.4f} < {minimum:.4f}"
        )
    return {
        "no_future_rows_used": no_future_rows,
        "terminal_session": str(end.date()),
        "terminal_execution_coverage": coverage,
        "terminal_coverage_min": minimum,
    }


def _paired_trade_differences(control_fills: pd.DataFrame, rex_fills: pd.DataFrame) -> dict[str, object]:
    """Compare closed trades with the same selection identity.

    A trade is paired by decision_id and ts_code, both of which are created
    before execution and are independent of the eventual exit path.  Open or
    otherwise unmatched positions are excluded rather than imputed.
    """
    def sells(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["decision_id", "ts_code", "economic_return", "trade_date", "reason"])
        return frame.loc[frame["side"].eq("SELL"), [
            "decision_id", "ts_code", "economic_return", "trade_date", "reason"
        ]].copy()

    left = sells(control_fills).rename(columns={
        "economic_return": "control_return",
        "trade_date": "control_exit_date",
        "reason": "control_reason",
    })
    right = sells(rex_fills).rename(columns={
        "economic_return": "rex_return",
        "trade_date": "rex_exit_date",
        "reason": "rex_reason",
    })
    paired = left.merge(right, on=["decision_id", "ts_code"], how="inner", validate="one_to_one")
    if paired.empty:
        return {
            "matched_trade_count": 0,
            "per_trade_difference_p5": None,
            # Absence of a comparison is not evidence that the invariant
            # passed; callers must fail closed on None.
            "h10_rejected_same_sell_date": None,
        }
    paired["return_difference"] = paired["rex_return"] - paired["control_return"]
    # A rejected H10 extension must execute on exactly the same session as
    # fixed H10.  Other pairs (stop-loss or extension exits) are not tested by
    # this invariant.
    h10_mask = (
        paired["rex_reason"].astype(str).isin({
            "REX_H10_EXTENSION_REJECTED",
            "REX_H10_EXTENSION_DATA_UNAVAILABLE",
        })
        & paired["control_reason"].astype(str).eq("TIME_EXIT")
    )
    same_dates = (
        None
        if not bool(h10_mask.any())
        else bool(
            (paired.loc[h10_mask, "rex_exit_date"] == paired.loc[h10_mask, "control_exit_date"]).all()
        )
    )
    differences = pd.to_numeric(paired["return_difference"], errors="coerce").dropna()
    return {
        "matched_trade_count": int(len(paired)),
        "per_trade_difference_p5": float(differences.quantile(0.05)) if len(differences) else None,
        "h10_rejected_same_sell_date": same_dates,
    }


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict:
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    daily = candidate.groupby("asof", sort=True).agg(
        stage1_pass=("stage1_pass", "sum"),
        candidate_view=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
    )
    return {
        "signal_dates": int(score["asof"].nunique()),
        "feature_ready_rows": int(score["feature_ready"].sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"]).sum()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "daily_min_stage1": int(daily["stage1_pass"].min()),
    }


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    strategy = StrategyConfig.from_yaml(args.strategy)
    control = StrategyConfig.from_yaml(args.control_strategy)
    model = ModelConfig.from_yaml(args.model)
    if str(strategy.identity.get("research_status", "historical")) != "historical":
        raise ValueError("REX diagnostic must use the historical research_status")
    if strategy.execution.get("exit_mode") != "rex_conditional_extension_v1":
        raise ValueError("REX strategy config is missing rex_conditional_extension_v1")
    if int(strategy.execution["hold_sessions_from_fill"]) < int(strategy.execution["extension"]["max_hold_sessions"]):
        raise ValueError("REX planning hold must cover max extension hold")
    if int(control.execution["hold_sessions_from_fill"]) != int(strategy.execution["extension"]["confirmation_sessions"]):
        raise ValueError("control must use the same H10 horizon as REX confirmation")
    for key in ("entries_per_decision", "max_open_positions", "target_gross_exposure_cap"):
        if control.portfolio[key] != strategy.portfolio[key]:
            raise ValueError(f"paired portfolio mismatch: {key}")
    if control.portfolio["sizing"] != strategy.portfolio["sizing"]:
        raise ValueError("paired portfolio sizing mismatch")

    calendar_start = str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date())
    planning_calendar = load_trading_calendar(calendar_start, args.execution_end)
    request = RunRequest(
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        execution_end=args.execution_end,
        output_dir=str(output),
        run_name="rex-v1-paired-diagnostic",
    )
    plan = compile_run_plan(strategy, model, request, planning_calendar["session"])
    requested_execution_end = pd.Timestamp(args.execution_end).normalize().date().isoformat()
    if plan.execution_end != requested_execution_end:
        raise ValueError(
            "execution calendar was truncated: "
            f"requested={requested_execution_end}, actual={plan.execution_end}"
        )
    # The bundle is built once and shared by both arms.  This is the paired
    # comparison invariant: no arm can silently receive a different universe.
    bundle = build_data_bundle(plan, strategy, output)
    bundle_date_verification = _verify_bundle_dates(bundle, plan, strategy)
    features = build_feature_ledger(bundle, strategy)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    # The feature ledger intentionally spans the execution horizon so the
    # REX state machine can evaluate H10-H20 closes.  Only selection outputs
    # are constrained to the signal window.
    if bool((pd.to_datetime(ledgers["candidate"]["asof"], errors="raise", utc=True) > pd.Timestamp(plan.signal_end, tz="UTC")).any()):
        raise ValueError("candidate ledger contains rows after requested signal_end")
    selection = _selection_summary(ledgers)
    _write_json(output / "RUN_STATUS.json", {
        "status": "RUNNING",
        "research_status": "DIAGNOSTIC_SEEN_HISTORY",
        "strategy_id": strategy.strategy_id,
        "control_strategy_id": control.strategy_id,
        "strategy_hash": strategy.config_hash,
        "control_strategy_hash": control.config_hash,
        "model_hash": model.config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "credentials_persisted": False,
    })
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "data_manifest.json", bundle.manifest)
    shutil.copyfile(args.strategy, output / "rex_strategy.yaml")
    shutil.copyfile(args.control_strategy, output / "control_strategy.yaml")
    shutil.copyfile(args.model, output / "model.yaml")
    features.to_parquet(output / "feature_ledger.parquet", index=False)
    for name, frame in ledgers.items():
        frame.to_parquet(output / f"{name}_ledger.parquet", index=False)
    _write_json(output / "selection_summary.json", selection)

    arms = {"control_h10": control, "rex_v1": strategy}
    portfolio: dict[str, dict] = {}
    results_by_arm_scenario: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for arm_name, arm_strategy in arms.items():
        arm_dir = output / "backtests" / arm_name
        arm_dir.mkdir(parents=True, exist_ok=False)
        arm_metrics: dict[str, dict] = {}
        for scenario in ("base", "stress"):
            result = run_v3_backtest(
                candidate_ledger=ledgers["candidate"],
                selection_ledger=ledgers["selection"],
                execution_panel=bundle.execution,
                corporate_actions=bundle.corporate_actions,
                strategy=arm_strategy,
                execution_sessions=plan.execution_sessions,
                scenario_name=scenario,
            )
            results_by_arm_scenario[(arm_name, scenario)] = result
            scenario_dir = arm_dir / scenario
            scenario_dir.mkdir()
            for name, frame in result.items():
                frame.to_parquet(scenario_dir / f"{name}.parquet", index=False)
            metrics = summarize_v3(
                result["nav"],
                result["fills"],
                initial_cash=float(arm_strategy.execution["initial_cash"]),
                positions=result["positions"],
                corporate_actions=result["corporate_actions"],
            )
            metrics.update({
                "arm": arm_name,
                "scenario": scenario,
                "entry_attempts": int(len(result["execution_decisions"])),
                "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if not result["execution_decisions"].empty else 0,
                "open_positions_at_end": int(len(result["open_positions"])),
            })
            metrics["acceptance"] = _acceptance(metrics, arm_strategy)
            _write_json(scenario_dir / "metrics.json", metrics)
            arm_metrics[scenario] = metrics
        portfolio[arm_name] = arm_metrics
    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)

    control_base = portfolio["control_h10"]["base"]
    rex_base = portfolio["rex_v1"]["base"]
    paired_scenarios: dict[str, dict[str, object]] = {}
    improvement_min = float(strategy.acceptance.get("paired_improvement_min", 0.005))
    for scenario in ("base", "stress"):
        control_result = results_by_arm_scenario[("control_h10", scenario)]
        rex_result = results_by_arm_scenario[("rex_v1", scenario)]
        trade_pair = _paired_trade_differences(
            control_result["fills"], rex_result["fills"]
        )
        control_metrics = portfolio["control_h10"][scenario]
        rex_metrics = portfolio["rex_v1"][scenario]
        pf_delta = (
            None
            if control_metrics["portfolio_profit_factor"] is None
            or rex_metrics["portfolio_profit_factor"] is None
            else float(
                rex_metrics["portfolio_profit_factor"]
                - control_metrics["portfolio_profit_factor"]
            )
        )
        ex_best_delta = float(
            rex_metrics["return_excluding_best_week"]
            - control_metrics["return_excluding_best_week"]
        )
        p5 = trade_pair["per_trade_difference_p5"]
        checks = {
            "h10_rejected_same_sell_date": bool(
                trade_pair["h10_rejected_same_sell_date"]
            ),
            "pf_improved_by_minimum": pf_delta is not None and pf_delta >= improvement_min,
            "ex_best_week_improved_by_minimum": ex_best_delta >= improvement_min,
            "per_trade_difference_p5_improved_by_minimum": p5 is not None and float(p5) >= improvement_min,
        }
        paired_scenarios[scenario] = {
            "control_return": float(control_metrics["total_return"]),
            "rex_return": float(rex_metrics["total_return"]),
            "return_delta_rex_minus_control": float(
                rex_metrics["total_return"] - control_metrics["total_return"]
            ),
            "control_pf": control_metrics["portfolio_profit_factor"],
            "rex_pf": rex_metrics["portfolio_profit_factor"],
            "pf_delta_rex_minus_control": pf_delta,
            "ex_best_week_delta_rex_minus_control": ex_best_delta,
            "matched_trade_count": trade_pair["matched_trade_count"],
            "per_trade_difference_p5": p5,
            "improvement_min": improvement_min,
            "checks": {**checks, "passed": all(checks.values())},
        }
    paired = {
        "same_selection_ledger": True,
        "same_execution_panel": True,
        "same_corporate_actions": True,
        "scenarios": paired_scenarios,
        "base_return_delta_rex_minus_control": float(rex_base["total_return"] - control_base["total_return"]),
        "base_pf_delta_rex_minus_control": (
            None if rex_base["portfolio_profit_factor"] is None or control_base["portfolio_profit_factor"] is None
            else float(rex_base["portfolio_profit_factor"] - control_base["portfolio_profit_factor"])
        ),
        "extension_approved_event_count_by_scenario": {
            scenario: int(
                results_by_arm_scenario[("rex_v1", scenario)]["position_events"]["event_type"]
                .eq("EXTENSION_APPROVED").sum()
            )
            for scenario in ("base", "stress")
        },
    }
    paired["acceptance"] = {
        scenario: {
            "control_metrics_passed": bool(portfolio["control_h10"][scenario]["acceptance"]["passed"]),
            "rex_metrics_passed": bool(portfolio["rex_v1"][scenario]["acceptance"]["passed"]),
            "paired_improvement_passed": bool(paired_scenarios[scenario]["checks"]["passed"]),
            "passed": bool(
                portfolio["control_h10"][scenario]["acceptance"]["passed"]
                and portfolio["rex_v1"][scenario]["acceptance"]["passed"]
                and paired_scenarios[scenario]["checks"]["passed"]
            ),
        }
        for scenario in ("base", "stress")
    }
    _write_json(output / "paired_summary.json", paired)
    code_paths = [Path(__file__).resolve(), ROOT / "src/aistock9988/backtest/v3_engine.py"]
    code_paths += sorted((ROOT / "src/aistock9988").rglob("*.py"))
    _write_json(output / "code_manifest.json", {
        "schema_version": "rex-code-manifest-v1",
        "files": {str(path.relative_to(ROOT)): _sha(path) for path in dict.fromkeys(code_paths)},
    })
    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    status.update({
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "verification": {
            "same_selection_ledger": True,
            "same_execution_panel": True,
            **bundle_date_verification,
        },
    })
    _write_json(output / "RUN_STATUS.json", status)
    _write_json(output / "artifact_manifest.json", {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-start", default="2026-01-05")
    # Defaults are bounded by the current quant_db snapshot.  A caller must
    # explicitly choose a later end after verifying that the source tables
    # actually contain that session.
    # 7/30 is the latest signal session whose H20 next-open exit is still
    # observable in the current 8/28 database snapshot.
    parser.add_argument("--signal-end", default="2026-07-30")
    parser.add_argument("--execution-end", default="2026-08-28")
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/rex_v1.yaml")
    parser.add_argument("--control-strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
