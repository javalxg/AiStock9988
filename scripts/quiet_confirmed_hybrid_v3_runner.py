"""Run the V3 quiet-rule control and monthly XGBoost shadow together."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.models.v3_ranker import (
    build_h10_label_ledger,
    build_hybrid_ledgers,
    monthly_walkforward_predictions,
)
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.pipeline import build_rule_ledgers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/council_20260828/S41_QUIET_CONFIRMED_HYBRID_V3_2026"


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "ledgers", "models", "backtests", "diagnostics", "logs"):
        (output / name).mkdir()

    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    calendar_start = str((pd.Timestamp(args.training_start) - pd.Timedelta(days=500)).date())
    planning_calendar = load_trading_calendar(calendar_start, args.execution_end)
    plan = compile_run_plan(
        strategy,
        model,
        RunRequest(
            signal_start=args.training_start,
            signal_end=args.signal_end,
            execution_end=args.execution_end,
            output_dir=str(output),
            run_name=args.run_name,
        ),
        planning_calendar["session"],
    )
    prediction_sessions = tuple(
        day for day in plan.signal_sessions if pd.Timestamp(day) >= pd.Timestamp(args.prediction_start)
    )
    execution_sessions = tuple(
        day for day in plan.execution_sessions if pd.Timestamp(day) >= pd.Timestamp(args.prediction_start)
    )
    if not prediction_sessions or not execution_sessions:
        raise ValueError("prediction range has no sessions")

    shutil.copyfile(args.strategy, output / "configs/strategy.yaml")
    shutil.copyfile(args.model, output / "configs/model.yaml")
    _write_json(output / "plan.json", {
        **plan.to_dict(),
        "training_signal_start": plan.signal_start,
        "prediction_start": prediction_sessions[0],
        "prediction_sessions": prediction_sessions,
        "backtest_execution_sessions": execution_sessions,
    })
    _write_json(output / "RUN_STATUS.json", {
        "run_name": args.run_name,
        "status": "RUNNING",
        "research_status": "DIAGNOSTIC_SEEN_HISTORY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_hash": strategy.config_hash,
        "model_hash": model.config_hash,
        "python": sys.version,
        "credentials_persisted": False,
    })

    print("phase=snapshot start", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    _write_json(output / "data_manifest.json", bundle.manifest)
    bundle.universe.to_parquet(output / "ledgers/universe_ledger.parquet", index=False)
    bundle.availability.to_parquet(output / "ledgers/data_availability_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "ledgers/execution_panel.parquet", index=False)

    print("phase=features start", flush=True)
    features = build_feature_ledger(bundle, strategy)
    features.to_parquet(output / "ledgers/feature_ledger.parquet", index=False)
    all_rule_ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    eval_rule_ledgers = build_rule_ledgers(features, strategy, prediction_sessions)
    labels = build_h10_label_ledger(bundle.execution, model)
    labels.to_parquet(output / "ledgers/label_ledger.parquet", index=False)
    print(
        f"phase=features complete rows={len(features)} training_stage1={int(all_rule_ledgers['score']['stage1_pass'].sum())}",
        flush=True,
    )

    print("phase=monthly_models start", flush=True)
    all_sessions = tuple(str(day.date()) for day in pd.DatetimeIndex(bundle.calendar["session"]))
    predictions, training_audit = monthly_walkforward_predictions(
        feature_ledger=features,
        score_ledger=all_rule_ledgers["score"],
        label_ledger=labels,
        strategy=strategy,
        model=model,
        prediction_sessions=prediction_sessions,
        all_sessions=all_sessions,
        output_dir=output / "models",
    )
    hybrid_ledgers = build_hybrid_ledgers(eval_rule_ledgers, predictions, strategy)
    predictions.to_parquet(output / "ledgers/prediction_ledger.parquet", index=False)
    _write_json(output / "diagnostics/training_audit.json", training_audit)
    _write_ledgers(output, "rule", eval_rule_ledgers)
    _write_ledgers(output, "xgb", hybrid_ledgers)
    print(f"phase=monthly_models complete predictions={len(predictions)} models={len(training_audit)}", flush=True)

    summaries: dict[str, dict[str, Any]] = {}
    results_by_policy: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for policy, ledgers in (("rule", eval_rule_ledgers), ("xgb", hybrid_ledgers)):
        results_by_policy[policy] = {}
        for scenario in ("base", "stress"):
            print(f"phase=backtest policy={policy} scenario={scenario} start", flush=True)
            result = run_v3_backtest(
                candidate_ledger=ledgers["candidate"],
                selection_ledger=ledgers["selection"],
                execution_panel=bundle.execution,
                corporate_actions=bundle.corporate_actions,
                strategy=strategy,
                execution_sessions=execution_sessions,
                scenario_name=scenario,
            )
            results_by_policy[policy][scenario] = result
            target = output / f"backtests/{policy}/{scenario}"
            target.mkdir(parents=True)
            for name, frame in result.items():
                frame.to_parquet(target / f"{name}.parquet", index=False)
            metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
            metrics.update({
                "policy": policy,
                "scenario": scenario,
                "entry_attempts": int(len(result["execution_decisions"])),
                "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if not result["execution_decisions"].empty else 0,
                "open_positions_at_end": int(len(result["open_positions"])),
            })
            metrics["acceptance"] = _acceptance(metrics, strategy)
            _write_json(target / "metrics.json", metrics)
            summaries[f"{policy}_{scenario}"] = metrics
            print(
                f"phase=backtest policy={policy} scenario={scenario} return={metrics['total_return']:+.6f} "
                f"pf={metrics['portfolio_profit_factor']} maxdd={metrics['max_drawdown']:+.6f}",
                flush=True,
            )

    verification = _verify(
        bundle.execution, predictions, training_audit, eval_rule_ledgers, hybrid_ledgers,
        results_by_policy, prediction_sessions,
    )
    _write_json(output / "diagnostics/verification.json", verification)
    _write_json(output / "PORTFOLIO_SUMMARY.json", summaries)
    _write_result(output, plan, prediction_sessions, training_audit, summaries)
    _write_json(output / "manifests/config_manifest.json", {
        "strategy_hash": strategy.config_hash,
        "model_hash": model.config_hash,
        "parameter_sweep": False,
    })
    code_paths = sorted((ROOT / "src/aistock9988").rglob("*.py")) + [Path(__file__).resolve()]
    _write_json(output / "manifests/code_manifest.json", {
        str(path.relative_to(ROOT)): _sha(path) for path in code_paths
    })
    status = json.loads((output / "RUN_STATUS.json").read_text())
    status.update({
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "verification_passed": True,
        "xgb_base_acceptance_passed": summaries["xgb_base"]["acceptance"]["passed"],
    })
    _write_json(output / "RUN_STATUS.json", status, replace=True)
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests/artifact_manifest.json", artifacts)
    return output


def _write_ledgers(output: Path, prefix: str, ledgers: dict[str, pd.DataFrame]) -> None:
    for name in ("score", "candidate", "selection"):
        ledgers[name].to_parquet(output / f"ledgers/{prefix}_{name}_ledger.parquet", index=False)


def _verify(
    execution: pd.DataFrame,
    predictions: pd.DataFrame,
    training_audit: list[dict[str, Any]],
    rule_ledgers: dict[str, pd.DataFrame],
    hybrid_ledgers: dict[str, pd.DataFrame],
    results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    prediction_sessions: tuple[str, ...],
) -> dict[str, Any]:
    checks = {
        "prediction_unique": not predictions.duplicated(["asof", "ts_code"]).any(),
        "prediction_score_finite": bool(pd.to_numeric(predictions["model_score"], errors="raise").notna().all()),
        "all_months_trained": all(row["status"] == "TRAINED" for row in training_audit),
        "labels_mature_at_cutoff": all(
            pd.Timestamp(row["max_label_available_time"]) <= pd.Timestamp(row["cutoff"])
            for row in training_audit
        ),
        "rule_stage1_count_preserved": int(rule_ledgers["score"]["stage1_pass"].sum()) == int(hybrid_ledgers["score"]["stage1_pass"].sum()),
        "signal_dates_complete": int(hybrid_ledgers["selection"]["asof"].nunique()) == len(prediction_sessions),
    }
    for policy, scenarios in results.items():
        for scenario, result in scenarios.items():
            fills = result["fills"]
            if fills.empty:
                checks[f"{policy}_{scenario}_fills_data_eligible"] = True
                continue
            merged = fills[["trade_date", "ts_code"]].merge(
                execution[["trade_date", "ts_code", "execution_data_eligible"]],
                on=["trade_date", "ts_code"], how="left", validate="many_to_one",
            )
            checks[f"{policy}_{scenario}_fills_data_eligible"] = bool(merged["execution_data_eligible"].fillna(False).all())
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise AssertionError("hybrid verification failed: " + ", ".join(failed))
    return {"passed": True, "checks": checks, "failed": failed}


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    pf = metrics["portfolio_profit_factor"]
    tests = {
        "profit_factor": pf is not None and float(pf) >= float(strategy.acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(strategy.acceptance["return_excluding_best_week_min_exclusive"]),
    }
    return {"passed": all(tests.values()), "tests": tests}


def _write_result(
    output: Path,
    plan: Any,
    prediction_sessions: tuple[str, ...],
    training_audit: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# S41 Quiet-Confirmed Hybrid V3", "",
        "2026 is seen diagnostic history. XGBoost is a zero-authority shadow challenger inside the unchanged Stage1 candidate pool.", "",
        "## Contract", "",
        f"- Training candidate dates start `{plan.signal_start}`; prediction dates `{prediction_sessions[0]}` to `{prediction_sessions[-1]}`.",
        f"- Monthly models: `{len(training_audit)}`; T+1 economic open to H10 economic open labels; no parameter sweep.",
        "- Rule and XGB arms share Stage1, Top20/Top4, T+1 execution, H10, stop, sizing and base/stress costs.", "",
        "## Portfolio", "",
        "| Policy | Cost | Return | PF | MaxDD | Ex-best-week | Trades | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for policy in ("rule", "xgb"):
        for scenario in ("base", "stress"):
            item = summaries[f"{policy}_{scenario}"]
            pf = "NA" if item["portfolio_profit_factor"] is None else f"{item['portfolio_profit_factor']:.3f}"
            lines.append(
                f"| {policy} | {scenario} | {item['total_return']:+.2%} | {pf} | {item['max_drawdown']:.2%} | "
                f"{item['return_excluding_best_week']:+.2%} | {item['trade_count']} | {item['acceptance']['passed']} |"
            )
    lines.extend(["", "The XGB arm advances only if it passes the common acceptance contract and improves robustly over the rule control."])
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_hybrid_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/xgb_quiet_candidate_ranker_v1.yaml")
    parser.add_argument("--training-start", default="2025-01-01")
    parser.add_argument("--prediction-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-06")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="S41_QUIET_CONFIRMED_HYBRID_V3_2026")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
