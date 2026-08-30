"""Run the first configuration-driven V3 strategy backtest end to end."""
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
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.pipeline import build_rule_ledgers
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "council_20260828" / "S40_QUIET_CONFIRMED_V3_FULL_UNIVERSE_2026"


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "ledgers", "backtests", "diagnostics", "logs"):
        (output / name).mkdir()

    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    if str(strategy.identity.get("research_status", "historical")) == "forward_only" and not getattr(args, "diagnostic_history", False):
        raise ValueError(
            "forward-only strategies require --diagnostic-history for an explicitly labeled historical diagnostic"
        )
    calendar_start = str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date())
    planning_calendar = load_trading_calendar(calendar_start, args.execution_end)
    request = RunRequest(
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        execution_end=args.execution_end,
        output_dir=str(output),
        run_name=args.run_name,
    )
    plan = compile_run_plan(strategy, model, request, planning_calendar["session"])
    shutil.copyfile(args.strategy, output / "configs" / "strategy.yaml")
    shutil.copyfile(args.model, output / "configs" / "model.yaml")
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "RUN_STATUS.json", {
        "run_name": args.run_name,
        "status": "RUNNING",
        "research_status": "DIAGNOSTIC_SEEN_HISTORY",
        "diagnostic_history": bool(getattr(args, "diagnostic_history", False)),
        "strategy_research_status": str(strategy.identity.get("research_status", "historical")),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "strategy_hash": strategy.config_hash,
        "model_id": model.model_id,
        "model_hash": model.config_hash,
        "python": sys.version,
        "credentials_persisted": False,
    })

    print("phase=snapshot start", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    _write_json(output / "data_manifest.json", bundle.manifest)
    _write_json(output / "manifests" / "config_manifest.json", {
        "strategy_hash": strategy.config_hash,
        "model_hash": model.config_hash,
        "plan_hash": _json_hash(plan.to_dict()),
        "diagnostic_history": bool(getattr(args, "diagnostic_history", False)),
        "strategy_research_status": str(strategy.identity.get("research_status", "historical")),
    })
    bundle.universe.to_parquet(output / "ledgers" / "universe_ledger.parquet", index=False)
    bundle.availability.to_parquet(output / "ledgers" / "data_availability_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "ledgers" / "execution_panel.parquet", index=False)

    print("phase=features start", flush=True)
    features = build_feature_ledger(bundle, strategy)
    features.to_parquet(output / "ledgers" / "feature_ledger.parquet", index=False)
    print(f"phase=features complete rows={len(features)}", flush=True)

    print("phase=selection start", flush=True)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    ledgers["score"].to_parquet(output / "ledgers" / "score_ledger.parquet", index=False)
    ledgers["candidate"].to_parquet(output / "ledgers" / "candidate_ledger.parquet", index=False)
    ledgers["selection"].to_parquet(output / "ledgers" / "selection_ledger.parquet", index=False)
    selection_summary = _selection_summary(ledgers)
    _write_json(output / "diagnostics" / "selection_summary.json", selection_summary)
    print(
        f"phase=selection complete stage1={selection_summary['stage1_pass_rows']} "
        f"candidate_view={selection_summary['candidate_view_rows']}",
        flush=True,
    )

    portfolio: dict[str, dict[str, Any]] = {}
    for scenario in ("base", "stress"):
        print(f"phase=backtest scenario={scenario} start", flush=True)
        result = run_v3_backtest(
            candidate_ledger=ledgers["candidate"],
            selection_ledger=ledgers["selection"],
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=plan.execution_sessions,
            scenario_name=scenario,
        )
        target = output / "backtests" / scenario
        target.mkdir()
        for name, frame in result.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        metrics = summarize_v3(
            result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"])
        )
        metrics.update({
            "scenario": scenario,
            "entry_attempts": int(len(result["execution_decisions"])),
            "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if not result["execution_decisions"].empty else 0,
            "entry_rejection_counts": (
                {str(key): int(value) for key, value in result["execution_decisions"].loc[
                    ~result["execution_decisions"]["chosen"], "reject_reason"
                ].value_counts().sort_index().items()}
                if not result["execution_decisions"].empty else {}
            ),
            "open_positions_at_end": int(len(result["open_positions"])),
        })
        metrics["acceptance"] = _acceptance(metrics, strategy)
        _write_json(target / "metrics.json", metrics)
        portfolio[scenario] = metrics
        print(
            f"phase=backtest scenario={scenario} return={metrics['total_return']:+.6f} "
            f"pf={metrics['portfolio_profit_factor']} maxdd={metrics['max_drawdown']:+.6f}",
            flush=True,
        )

    _write_json(output / "PORTFOLIO_SUMMARY.json", portfolio)
    _write_result(output, plan.to_dict(), bundle.manifest, selection_summary, portfolio, strategy)
    verification = _verify_preseal(
        output, plan.to_dict(), bundle.availability, bundle.execution, features, ledgers, strategy
    )
    _write_json(output / "diagnostics" / "verification.json", verification)
    code_paths = [
        ROOT / "src/aistock9988/configuration.py",
        ROOT / "src/aistock9988/planning.py",
        ROOT / "src/aistock9988/data/availability.py",
        ROOT / "src/aistock9988/data/bundle.py",
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
        ROOT / "src/aistock9988/backtest/v3_engine.py",
        ROOT / "src/aistock9988/reporting/v3_metrics.py",
        Path(__file__).resolve(),
    ]
    _write_json(output / "manifests" / "code_manifest.json", {
        str(path.relative_to(ROOT)): _sha(path) for path in code_paths
    })
    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    status.update({
        "status": "DIAGNOSTIC_COMPLETED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "base_acceptance_passed": bool(portfolio["base"]["acceptance"]["passed"]),
        "stress_acceptance_passed": bool(portfolio["stress"]["acceptance"]["passed"]),
        "overall_acceptance_passed": bool(
            portfolio["base"]["acceptance"]["passed"]
            and portfolio["stress"]["acceptance"]["passed"]
        ),
        "verification_passed": bool(verification["passed"]),
    })
    _write_json(output / "RUN_STATUS.json", status, replace=True)
    artifacts = {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(output / "manifests" / "artifact_manifest.json", artifacts)
    return output


def _selection_summary(ledgers: dict[str, pd.DataFrame]) -> dict[str, Any]:
    score = ledgers["score"]
    candidate = ledgers["candidate"]
    daily = candidate.groupby("asof", sort=True).agg(
        feature_ready=("feature_ready", "sum"),
        stage1_pass=("stage1_pass", "sum"),
        candidate_view=("candidate_status", lambda values: int(values.eq("IN_VIEW").sum())),
    )
    return {
        "signal_dates": int(score["asof"].nunique()),
        "scored_rows": int(len(score)),
        "feature_ready_rows": int(score["feature_ready"].sum()),
        "selection_data_excluded_rows": int((~score["selection_data_eligible"]).sum()),
        "training_data_excluded_rows": int((~score["training_data_eligible"]).sum()),
        "stage1_pass_rows": int(score["stage1_pass"].sum()),
        "candidate_view_rows": int(candidate["candidate_status"].eq("IN_VIEW").sum()),
        "daily_min_stage1": int(daily["stage1_pass"].min()),
        "daily_median_stage1": float(daily["stage1_pass"].median()),
        "daily_max_stage1": int(daily["stage1_pass"].max()),
    }


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
    plan: dict[str, Any],
    data_manifest: dict[str, Any],
    selection: dict[str, Any],
    portfolio: dict[str, dict[str, Any]],
    strategy: StrategyConfig,
) -> None:
    strategy_dict = strategy.to_dict()
    stage1_expression = json.dumps(strategy_dict["stage1"]["expression"], sort_keys=True, default=str)
    entries = int(strategy.portfolio["entries_per_decision"])
    max_positions = int(strategy.portfolio["max_open_positions"])
    weight = float(strategy.portfolio["sizing"]["value"])
    hold = int(strategy.execution["hold_sessions_from_fill"])
    lines = [
        f"# {plan['strategy_id']} V3 Full-Universe 2026", "",
        "FORWARD_ONLY_DIAGNOSTIC: this historical replay is explicitly seen history and is not a locked out-of-sample claim.", "",
        "## Contract", "",
        f"- Signal range: `{plan['signal_start']}` to `{plan['signal_end']}`; execution through `{plan['execution_end']}`.",
        f"- Bundle: `{data_manifest['bundle_id']}`; configured full-universe codes: `{data_manifest['configured_codes']}`.",
        f"- Rules only: full universe -> Stage1 expression `{stage1_expression}` -> frozen daily Top{int(strategy.portfolio['candidate_view_size'])} -> Top{entries}/next-ranked fallback.",
        f"- T+1 raw open entry, {weight:.2%} of prior-close NAV per name, maximum {max_positions} positions, H{hold}, -8% close-trigger/next-open stop.",
        "- No threshold sweep and no XGBoost model.", "",
        "## Selection", "",
        f"- Signal dates: `{selection['signal_dates']}`; feature-ready rows: `{selection['feature_ready_rows']}`.",
        f"- Required-data exclusions on signal dates: selection `{selection['selection_data_excluded_rows']}`, training `{selection['training_data_excluded_rows']}`.",
        f"- Stage1 pass rows: `{selection['stage1_pass_rows']}`; frozen candidate-view rows: `{selection['candidate_view_rows']}`.", "",
        "## Portfolio", "",
        "| Cost | Return | PF | MaxDD | Ex-best-week | Trades | End open | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("base", "stress"):
        metrics = portfolio[name]
        pf = "NA" if metrics["portfolio_profit_factor"] is None else f"{metrics['portfolio_profit_factor']:.3f}"
        lines.append(
            f"| {name} | {metrics['total_return']:+.2%} | {pf} | {metrics['max_drawdown']:.2%} | "
            f"{metrics['return_excluding_best_week']:+.2%} | {metrics['trade_count']} | "
            f"{metrics['open_positions_at_end']} | {metrics['acceptance']['passed']} |"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(
        "The strategy advances only when PF>=2, MaxDD<=15%, and return excluding the best week remains positive. "
        "A negative result is retained unchanged as evidence; it is not repaired with a threshold scan."
    )
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_preseal(
    output: Path,
    plan: dict[str, Any],
    availability: pd.DataFrame,
    execution: pd.DataFrame,
    features: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    strategy: StrategyConfig,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["availability_unique"] = not availability.duplicated(["trade_date", "ts_code"]).any()
    checks["execution_unique"] = not execution.duplicated(["trade_date", "ts_code"]).any()
    available = pd.to_datetime(features["available_time"], errors="raise", utc=True)
    cutoff = pd.to_datetime(features["asof"], errors="raise", utc=True).map(session_close)
    checks["feature_pit"] = bool((available <= cutoff).all())
    checks["feature_unique"] = not features.duplicated(["asof", "ts_code"]).any()
    checks["feature_ready_requires_selection_data"] = bool(
        (~features["feature_ready"] | features["selection_data_eligible"]).all()
    )
    checks["score_unique"] = not ledgers["score"].duplicated(["asof", "ts_code"]).any()
    in_view = ledgers["candidate"][ledgers["candidate"]["candidate_status"].eq("IN_VIEW")]
    daily_view = in_view.groupby("asof", sort=True).size()
    checks["candidate_view_cap"] = bool((daily_view <= int(strategy.portfolio["candidate_view_size"])).all())
    min_ratio = float(strategy.portfolio.get("candidate_min_daily_view_ratio", 0.0))
    if min_ratio > 0:
        min_rows = int(np.ceil(int(strategy.portfolio["candidate_view_size"]) * min_ratio))
        signal_days = pd.DatetimeIndex(pd.to_datetime(plan["signal_sessions"], utc=True)).normalize()
        counts = daily_view.reindex(signal_days, fill_value=0)
        checks["candidate_view_minimum_coverage"] = bool(
            float((counts >= min_rows).mean()) >= min_ratio
        )
    checks["selection_signal_end"] = str(pd.to_datetime(ledgers["selection"]["asof"], utc=True).max().date()) == plan["signal_end"]
    sessions = pd.DatetimeIndex(pd.to_datetime(plan["execution_sessions"], utc=True)).normalize()
    for scenario in ("base", "stress"):
        target = output / "backtests" / scenario
        nav = pd.read_parquet(target / "nav.parquet")
        orders = pd.read_parquet(target / "orders.parquet")
        fills = pd.read_parquet(target / "fills.parquet")
        events = pd.read_parquet(target / "position_events.parquet")
        values = nav[["cash", "market_value", "nav"]].apply(pd.to_numeric, errors="raise")
        checks[f"{scenario}_nav_identity"] = bool(
            ((values["cash"] + values["market_value"] - values["nav"]).abs() <= 1e-8).all()
        )
        checks[f"{scenario}_cash_nonnegative"] = bool((values["cash"] >= -1e-8).all())
        checks[f"{scenario}_position_cap"] = bool(
            (pd.to_numeric(nav["open_positions"], errors="raise") <= int(strategy.portfolio["max_open_positions"])).all()
        )
        checks[f"{scenario}_execution_end"] = str(pd.to_datetime(nav["trade_date"], utc=True).max().date()) == plan["execution_end"]
        filled_orders = (
            set(orders.loc[orders["status"].eq("FILLED"), "order_id"].astype(str))
            if not orders.empty and "status" in orders.columns else set()
        )
        fill_ids = set(fills["order_id"].astype(str))
        checks[f"{scenario}_fill_order_bijection"] = filled_orders == fill_ids and not fills["order_id"].duplicated().any()
        if fills.empty:
            checks[f"{scenario}_fills_require_execution_data"] = True
        else:
            fill_keys = fills[["trade_date", "ts_code"]].merge(
                execution[["trade_date", "ts_code", "execution_data_eligible"]],
                on=["trade_date", "ts_code"],
                how="left",
                validate="many_to_one",
            )
            checks[f"{scenario}_fills_require_execution_data"] = bool(
                fill_keys["execution_data_eligible"].fillna(False).all()
            )
        held_gaps = (
            events[events["event_type"].eq("HELD_DATA_GAP")]
            if "event_type" in events else pd.DataFrame()
        )
        checks[f"{scenario}_held_gap_has_reason"] = bool(
            held_gaps.empty or held_gaps["reason"].astype(str).str.len().gt(0).all()
        )
        buys = (
            orders[orders["side"].eq("BUY") & orders["status"].eq("FILLED")]
            if not orders.empty and {"side", "status"}.issubset(orders.columns)
            else pd.DataFrame()
        )
        causal = True
        for row in buys.itertuples(index=False):
            decision = pd.Timestamp(row.decision_session)
            execution_session = pd.Timestamp(row.execution_session)
            index = sessions.get_indexer([decision])[0]
            causal &= index >= 0 and index + 1 < len(sessions) and sessions[index + 1] == execution_session
        checks[f"{scenario}_buy_t_plus_one"] = bool(causal)
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError("pre-seal verification failed: " + ", ".join(failed))
    return {"passed": True, "checks": checks, "failed": failed}


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-06")
    parser.add_argument("--execution-end", default="2026-08-21")
    parser.add_argument("--run-name", default="S40_QUIET_CONFIRMED_V3_FULL_UNIVERSE_2026")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostic-history",
        action="store_true",
        help="allow a forward-only strategy only as a clearly labeled seen-history diagnostic",
    )
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
